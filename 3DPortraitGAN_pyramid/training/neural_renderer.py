# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import torch
from torch_utils import persistence
from training.networks_stylegan2 import ToRGBLayer, SynthesisNetwork

from training.networks_stylegan2 import Hierarchy3DAwareGenerator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
import numpy as np


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        batch_size=1,
        explicitly_symmetry=False,
        thickness= 0.05,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        bcg_synthesis_kwargs = synthesis_kwargs.copy()
        bcg_synthesis_kwargs["channel_base"] = 32768
        bcg_synthesis_kwargs["channel_max"] = 512

        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels

        self.trigrid_channel = 12
        self.decode_channel = 32

        self.batch_size = batch_size
        self.renderer = ImportanceRenderer(w_dim = w_dim, num_ws = 14, batch_size = self.batch_size,thickness =thickness,box_warp = rendering_kwargs['box_warp'])
        self.ray_sampler = RaySampler()

        self.decoder = OSGDecoder(self.trigrid_channel, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                       'decoder_output_dim': self.decode_channel,
                                       'decoder_activation': rendering_kwargs['decoder_activation']})

        self.torgb = ToRGBLayer(self.decode_channel, 3, w_dim) if rendering_kwargs.get('use_torgb_raw', False) else None

        self.bcg_synthesis = SynthesisNetwork(w_dim, img_resolution=128,
                                              img_channels=self.decode_channel, **bcg_synthesis_kwargs) if rendering_kwargs.get('use_background', False) else None

        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None

        self.explicitly_symmetry = explicitly_symmetry

        self.avg_c = torch.tensor([[ 1.0000e+00,  1.0505e-09,  4.3685e-08, -1.1805e-07,  0.0000e+00,
                                    -9.9951e-01,  2.4033e-02, -1.1805e-07,  4.3714e-08, -2.4033e-02,
                                    -9.9951e-01,  2.6992e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                                    1.0000e+00,  6.5104e+00,  0.0000e+00,  5.0000e-01,  0.0000e+00,
                                    6.5104e+00,  5.0000e-01,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).float().cuda()

    def flip_yaw(self, matrix):
        flipped_matrix = matrix.clone()
        flipped = flipped_matrix[:, :16].reshape(-1, 4, 4)
        flipped[:, 0, 1] *= -1
        flipped[:, 0, 2] *= -1
        flipped[:, 1, 0] *= -1
        flipped[:, 2, 0] *= -1
        flipped[:, 0, 3] *= -1

        flipped = flipped.reshape(-1, 16)
        flipped_matrix[:, :16] = flipped.clone()

        return flipped_matrix



    def set_batch_size(self, batch_size):
        self.renderer.set_batch_size(batch_size)

    def render_meshes(self,shape_pose_params,resolution,cameras):

        return self.renderer.render_meshes(shape_pose_params, resolution, cameras)



    def render_planes(self, ws, planes, c, neural_rendering_resolution=None, update_emas=False, chunk = None,render_bg = True,patch_resolution=None,
                  apply_def=False, pose_params = None,ws_bcg=None,
                  **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution

        patch_info = []
        if patch_resolution is None:
            # Create a batch of rays for volume rendering
            ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
            H = W = neural_rendering_resolution
        else:
            ray_origins, ray_directions,patch_info = self.ray_sampler.patch_forward(cam2world_matrix, intrinsics,
                                                                                   patch_resolution,
                                                                                   patch_scale=patch_resolution/neural_rendering_resolution)
            H = W = patch_resolution


        N, M, _ = ray_origins.shape



        # Reshape output into three D*32-channel planes, where D=self.rendering_kwargs['triplane_depth'], defines the depth of the tri-grid
        for res_k in planes:
            # b, c, H,W
            # planes[res_k] = planes[res_k].view(len(planes[res_k]), 3, -1, planes[res_k].shape[-2], planes[res_k].shape[-1])
            if len(planes[res_k].shape) == 4:
                planes[res_k] = planes[res_k].view(len(planes[res_k]) // 3, 3, planes[res_k].shape[-3],
                                                   planes[res_k].shape[-2], planes[res_k].shape[-1])

        if chunk is not None:

            feature_list, depth_list, weight_list = list(), list(), list()
            for _ro, _rd in zip(torch.split(ray_origins, chunk, dim=1), torch.split(ray_directions, chunk, dim=1)):
                render_output = self.renderer(planes, self.decoder, _ro,
                                          _rd, self.rendering_kwargs, apply_def = apply_def, ws = ws, pose_params = pose_params )  # channels last

                _f = render_output['rgb_final']
                _d = render_output['depth_final']
                _w = render_output['weights']
                feature_list.append(_f)
                depth_list.append(_d)
                weight_list.append(_w)
            feature_samples = torch.cat(feature_list, 1)
            depth_samples = torch.cat(depth_list, 1)
            weights_samples = torch.cat(weight_list, 1)
        else:

            # Perform volume rendering
            render_output = self.renderer(planes, self.decoder, ray_origins,
                                          ray_directions, self.rendering_kwargs, apply_def = apply_def, ws = ws, pose_params = pose_params ) # channels last
            # {'rgb_final': rgb_final, 'depth_final': depth_final, 'weights': weights.sum(2)}
            feature_samples = render_output['rgb_final']
            depth_samples = render_output['depth_final']
            weights_samples = render_output['weights']


        # Reshape into 'raw' neural-rendered image

        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_samples = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if self.decoder.activation == "sigmoid":
            feature_image = feature_image * 2 - 1 # Scale to (-1, 1), taken from ray marcher
        # Generate Background
        if self.bcg_synthesis and render_bg:
            ws_bcg = ws[:,:self.bcg_synthesis.num_ws] if ws_bcg is None else ws_bcg[:,:self.bcg_synthesis.num_ws]
            if ws_bcg.size(1) < self.bcg_synthesis.num_ws:
                ws_bcg = torch.cat([ws_bcg, ws_bcg[:,-1:].repeat(1,self.bcg_synthesis.num_ws-ws_bcg.size(1),1)], 1)
            bcg_image = self.bcg_synthesis(ws_bcg, update_emas=update_emas, **synthesis_kwargs)
            bcg_image = torch.nn.functional.interpolate(bcg_image, size=feature_image.shape[2:],
                    mode='bilinear', align_corners=False, antialias=self.rendering_kwargs['sr_antialias'])
            feature_image = feature_image + (1-weights_samples) * bcg_image

        # Generate Raw image
        if self.torgb:
            rgb_image = self.torgb(feature_image, ws[:,-1], fused_modconv=False)
            rgb_image = rgb_image.to(dtype=torch.float32, memory_format=torch.contiguous_format)

        else:
            rgb_image = feature_image[:, :3]


        mask_image = weights_samples * (1 + 2 * 0.001) - 0.001

        return {'image_raw': rgb_image, 'image_depth': depth_image, "image_mask": mask_image,'patch_info':patch_info}


    def sample_trigrid(self, coordinates, directions, planes, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        # planes = planes.view(len(planes), 3, 32 * self.rendering_kwargs['triplane_depth'], planes.shape[-2],
        #                      planes.shape[-1])
        for res_k in planes:
            # b, c, H,W
            if len(planes[res_k].shape) == 4:
                planes[res_k] = planes[res_k].view(len(planes[res_k]) // 3, 3, planes[res_k].shape[-3],
                                               planes[res_k].shape[-2], planes[res_k].shape[-1])

        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)




from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 32

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        self.activation = options['decoder_activation']

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = x[..., 1:]
        sigma = x[..., 0:1]
        if self.activation == "sigmoid":
            # Original EG3D
            rgb = torch.sigmoid(rgb)*(1 + 2*0.001) - 0.001
        elif self.activation == "lrelu":
            # StyleGAN2-style, use with toRGB
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2, inplace=True) * math.sqrt(2)
        return {'rgb': rgb, 'sigma': sigma}

