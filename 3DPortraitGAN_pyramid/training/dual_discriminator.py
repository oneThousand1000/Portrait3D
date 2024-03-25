# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discriminator architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue


@persistence.persistent_class
class SingleDiscriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 sr_upsample_factor=1,  # Ignored for SingleDiscriminator
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        img = img['image']

        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


# ----------------------------------------------------------------------------

def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear',
                                                          align_corners=False, antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(size * 2 + 2, size * 2 + 2),
                                                          mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear',
                                                          align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear',
                                                   align_corners=False, antialias=True)
        aliased = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear',
                                                  align_corners=False, antialias=False)
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered

    return ada_filtered_64


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 disc_c_noise=0,  # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))
        self.disc_c_noise = disc_c_noise

    def forward(self, img, c, update_emas=False, **block_kwargs):
        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DummyDualDiscriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))

        self.raw_fade = 1

    def forward(self, img, c, update_emas=False, **block_kwargs):
        self.raw_fade = max(0, self.raw_fade - 1 / (500000 / 32))

        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1],
                                      f=self.resample_filter) * self.raw_fade
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


# ----------------------------------------------------------------------------
from training.networks_stylegan2 import FullyConnectedLayer


class PoseShapeAwareDualDiscriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 seg_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 disc_c_noise=0,  # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
                 explicitly_symmetry=False,
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        img_channels = img_channels * 2 + seg_channels
        self.camera_param_dim = c_dim
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.pose_branch = DPoseBranch(num_betas=10, in_channel=channels_dict[4]*4*4)
        self.c_dim += self.pose_branch.output_dim

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if self.c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if self.c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=self.c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None,
                                          **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        **common_kwargs) 
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))
        self.disc_c_noise = disc_c_noise

        self.explicitly_symmetry = explicitly_symmetry

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

    def predict_pose(self, img, c,update_emas=False, **block_kwargs):


        if self.explicitly_symmetry:
            theta = torch.atan2(c[:, [11]], c[:, [3]])  # math.atan2(z, x)
            is_left = (theta >= -np.pi / 2) & (theta <= np.pi / 2)

            img_flip = torch.flip(img['image'], dims=[3])
            img_flip_raw = torch.flip(img['image_raw'], dims=[3])
            seg_flip = torch.flip(img['image_mask'], dims=[3])

            is_left_img = is_left.unsqueeze(2).unsqueeze(3)
            input_img = torch.where(is_left_img, img_flip, img['image'])  # if left, flip image
            misc.assert_shape(input_img, img_flip.shape )

            is_left_img_raw = is_left.unsqueeze(2).unsqueeze(3)
            input_img_raw = torch.where(is_left_img_raw, img_flip_raw, img['image_raw'])  # if left, flip image_raw
            misc.assert_shape(input_img_raw, img_flip_raw.shape )

            is_left_seg = is_left.unsqueeze(2).unsqueeze(3)
            input_seg = torch.where(is_left_seg, seg_flip, img['image_mask'])  # if left, flip seg
            misc.assert_shape(input_seg, seg_flip.shape )

            img = {'image': input_img, 'image_raw': input_img_raw, 'image_mask': input_seg}

            image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
            seg = filtered_resizing(img['image_mask'], size=img['image'].shape[-1], f=self.resample_filter)
            seg = 2 * seg - 1  # normalize to [-1,1]
            img = torch.cat([img['image'], image_raw, seg], 1)

            _ = update_emas  # unused
            x = None
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                x, img = block(x, img, **block_kwargs)


            pose_branch_input_feature = self.b4.get_flatten_x(x, img)
            pose_params = self.pose_branch(pose_branch_input_feature, c)

            flip_pose_params = pose_params.clone()
            flip_pose_params[:, [1, 2, 4, 5]] *= -1  # flip y and z axis angles

            pose_params = torch.where(is_left, flip_pose_params, pose_params)


        else:
            raise NotImplementedError
            image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
            seg = filtered_resizing(img['image_mask'], size=img['image'].shape[-1], f=self.resample_filter)
            seg = 2 * seg - 1  # normalize to [-1,1]
            img = torch.cat([img['image'], image_raw, seg], 1)

            _ = update_emas  # unused
            x = None
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                x, img = block(x, img, **block_kwargs)


            pose_branch_input_feature = self.b4.get_flatten_x(x, img)
            pose_params = self.pose_branch(pose_branch_input_feature, c)


        return pose_params,pose_branch_input_feature

    def forward(self, img, c, gt_pose = None, update_emas=False, **block_kwargs):

        if self.explicitly_symmetry:

            pose_params,_ = self.predict_pose(img, c, update_emas, **block_kwargs)

            image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
            seg = filtered_resizing(img['image_mask'], size=img['image'].shape[-1], f=self.resample_filter)
            seg = 2 * seg - 1  # normalize to [-1,1]
            img = torch.cat([img['image'], image_raw, seg], 1)

            _ = update_emas  # unused
            x = None
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                x, img = block(x, img, **block_kwargs)


            pose_branch_input_feature = self.b4.get_flatten_x(x, img)

        else:
            raise NotImplementedError
            pose_params, pose_branch_input_feature = self.predict_pose(img, c, update_emas, **block_kwargs)

        if gt_pose is not None:
            #raise NotImplementedError
            c = torch.cat([c, gt_pose], dim=1)
        else:
            pose_label = pose_params.detach()  # detach
            c = torch.cat([c, pose_label], dim=1)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
            # x = self.b4(x, img, cmap)
        x = self.b4(flatten_x=pose_branch_input_feature, cmap=cmap)
        return x, pose_params

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


from torch_utils import misc


class DPoseBranch(torch.nn.Module):
    def __init__(self, num_betas, in_channel):
        super().__init__()
        self.num_betas = num_betas
        hidden_dim = 64
        self.in_channel = in_channel
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

        self.in_channel += 25  # c dim

        self.output_dim = out_dim
        self.net = torch.nn.Sequential(
            # linear
            # FullyConnectedLayer(self.in_channel, hidden_dim),
            # torch.nn.LeakyReLU(0.2),
            # FullyConnectedLayer(hidden_dim, self.output_dim)  # betas, scale, transl, rots of neck and head
            FullyConnectedLayer(self.in_channel, 2048, activation='lrelu'),
            FullyConnectedLayer(2048, 512, activation='lrelu'),
            FullyConnectedLayer(512, 128, activation='lrelu'),
            FullyConnectedLayer(128, 32, activation='lrelu'),
            FullyConnectedLayer(32, self.output_dim)
        )


    def forward(self, feature, camera_parameters):
        # misc.assert_shape(feature, [None, self.in_channel])
        # misc.assert_shape(camera_parameters, [None, 25])
        feature = torch.cat([feature, camera_parameters], dim=1)

        pose = self.net(feature)  # (B, num_betas + 1 + 3 + 6)

        return pose