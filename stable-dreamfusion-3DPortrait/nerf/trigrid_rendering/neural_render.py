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
from nerf.torch_utils import persistence
from nerf.trigrid_rendering.networks_stylegan2 import ToRGBLayer, FullyConnectedLayer

from nerf.trigrid_rendering.volumetric_rendering.renderer import ImportanceRenderer
from nerf.trigrid_rendering.volumetric_rendering.ray_sampler import RaySampler
import numpy as np

@persistence.persistent_class
class NeuralRender(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.

                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 rendering_kwargs={},
                 batch_size=1,
                 thickness=0.05,
                 apply_deformation = False
                 ):
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.trigrid_channel = 12
        self.decode_channel = 32

        self.batch_size = batch_size
        self.renderer = ImportanceRenderer(w_dim=w_dim, num_ws=14, batch_size=self.batch_size, thickness=thickness,
                                           box_warp=rendering_kwargs['box_warp'],apply_deformation = apply_deformation) # disable deformation for now
        self.ray_sampler = RaySampler()

        self.decoder = OSGDecoder(self.trigrid_channel, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                                         'decoder_output_dim': self.decode_channel,
                                                         'decoder_activation': rendering_kwargs['decoder_activation']})

        self.torgb = ToRGBLayer(self.decode_channel, 3, w_dim)

        self.rendering_kwargs = rendering_kwargs
        self.neural_rendering_resolution = 64

        self.pose_branch = GPoseBranch(z_dim=z_dim, c_dim=c_dim)

        self.avg_c = torch.tensor([[1.0000e+00, 1.0505e-09, 4.3685e-08, -1.1805e-07, 0.0000e+00,
                                    -9.9951e-01, 2.4033e-02, -1.1805e-07, 4.3714e-08, -2.4033e-02,
                                    -9.9951e-01, 2.6992e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                                    1.0000e+00, 6.5104e+00, 0.0000e+00, 5.0000e-01, 0.0000e+00,
                                    6.5104e+00, 5.0000e-01, 0.0000e+00, 0.0000e+00, 1.0000e+00]]).float().cuda()
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
    def sample_pose_params(self, c):
        assert len(c.shape) == 2 and c.shape[1] == 25
        # randomly sample z from Gaussian distribution
        z = torch.randn(c.shape[0], self.z_dim).to(c.device)

        theta = torch.atan2(c[:, [11]], c[:, [3]])  # math.atan2(z, x)
        is_left = (theta >= -np.pi / 2) & (theta <= np.pi / 2)

        flip_c = self.flip_yaw(c)
        input_c = torch.where(is_left, flip_c, c)  # if left, flip c

        pose_params = self.pose_branch(z, input_c)

        flip_pose_params = pose_params.clone()
        flip_pose_params[:, [1, 2, 4, 5]] *= -1  # flip y and z axis angles

        pose_params = torch.where(is_left, flip_pose_params, pose_params)  # if left, flip back pose_params

        return pose_params


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 32

        self.net = torch.nn.Sequential(
                FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'],
                                    lr_multiplier=options['decoder_lr_mul'])
            )
        self.activation = options['decoder_activation']




    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = x[..., 1:]
        sigma = x[..., 0:1]


        if self.activation == "sigmoid":
            # Original EG3D
            rgb = torch.sigmoid(rgb) * (1 + 2 * 0.001) - 0.001
        elif self.activation == "lrelu":
            # StyleGAN2-style, use with toRGB
            rgb = torch.nn.functional.leaky_relu(rgb, 0.2, inplace=True) * math.sqrt(2)
        return {'rgb': rgb, 'sigma': sigma}

import numpy as np
class GPoseBranch(torch.nn.Module):
    def __init__(self, z_dim, c_dim):
        super().__init__()
        hidden_dim = 64
        self.in_channel = z_dim + c_dim
        #
        # predict_betas = predict_transl = predict_scale = False
        # predict_pose = True

        out_dim = 6

        # if predict_betas:
        #     out_dim += num_betas
        # if predict_transl:
        #     out_dim += 3
        # if predict_scale:
        #     out_dim += 1
        # if predict_pose:
        #     out_dim += 6

        self.output_dim = out_dim
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(self.in_channel, 128, activation='lrelu'),
            FullyConnectedLayer(128, 32, activation='lrelu'),
            FullyConnectedLayer(32, self.output_dim)
        )


    def forward(self, z, c):
        # misc.assert_shape(feature, [None, self.in_channel])
        # misc.assert_shape(camera_parameters, [None, 25])
        feature = torch.cat([z, c], dim=1)

        pose = self.net(feature)  # (B, num_betas + 1 + 3 + 6)


        return pose