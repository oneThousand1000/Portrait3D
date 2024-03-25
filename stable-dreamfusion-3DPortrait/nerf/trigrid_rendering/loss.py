# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from torch_utils import misc
import copy


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_seg, real_c, real_pose, gen_z, gen_c, gen_pose,gain, cur_nimg,
                             cur_nimg_start):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, r1_gamma_seg=1000,style_mixing_prob=0, pl_weight=0,
                 density_noise_fade_kimg=0,
                 pl_batch_shrink=2, pl_decay=0.01,
                 pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0,
                 neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None,
                 neural_rendering_resolution_fade_kimg=0,
                 gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                 thickness=None,
                 pose_loss_weight = None,  input_pose_params_reg_loss_weight = None,input_pose_params_reg_loss_kimg  = None,
                 rank=None,bcg_reg_prob=0
                 ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.r1_gamma_seg = r1_gamma_seg
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.r1_gamma_init = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.density_noise_fade_kimg = density_noise_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1, 3, 3, 1], device=device)
        self.blur_raw_target = True
        self.bcg_reg_prob = bcg_reg_prob
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)


        self.thickness = thickness
        self.pose_loss_weight = pose_loss_weight
        self.input_pose_params_reg_loss_weight = input_pose_params_reg_loss_weight
        self.input_pose_params_reg_loss_kimg = input_pose_params_reg_loss_kimg


        # for snap
        self.swapping_prob = None
        self.neural_rendering_resolution = None
        self.blur_sigma = None


        self.rank = rank

    def run_G(self, z, c, pose_params, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            p_swapped = torch.roll(pose_params.clone(), 1, 0)
            rand_ = torch.rand((c.shape[0], 1), device=c.device)
            c_gen_conditioning = torch.where(rand_ < swapping_prob, c_swapped, c)
            pose_params_conditioning = torch.where(rand_ < swapping_prob, p_swapped, pose_params)
        else:
            c_gen_conditioning = torch.zeros_like(c)
            pose_params_conditioning = torch.zeros([c.shape[0],6]).to(c.device)

        ws = self.G.mapping(z, c_gen_conditioning, pose_params_conditioning,update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                     torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c,pose_params, update_emas=False)[:, cutoff:]

        if self.bcg_reg_prob > 0:
            ws_swapped = torch.roll(ws.clone(), 1, 0)
            ws_bcg = torch.where(torch.rand((ws.shape[0], 1, 1), device=ws.device) < self.bcg_reg_prob, ws_swapped, ws)
        else:
            ws_bcg = None


        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution,
                                      update_emas=update_emas,
                                      apply_def=True, pose_params=pose_params,ws_bcg = ws_bcg
                                      )
        return gen_output, ws



    def run_D(self, img, c, gt_pose=None, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(
                    blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            raise NotImplementedError
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                          torch.nn.functional.interpolate(img['image_raw'],
                                                                                          size=img['image'].shape[2:],
                                                                                          mode='bilinear',
                                                                                          antialias=True)],
                                                         dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:],
                                                               size=img['image_raw'].shape[2:], mode='bilinear',
                                                               antialias=True)

        logits, pose = self.D(img, c, gt_pose=gt_pose, update_emas=update_emas)
        return logits, pose

    def run_D_pose_prediction(self, img, c, blur_sigma=0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(
                    blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                          torch.nn.functional.interpolate(img['image_raw'],
                                                                                          size=img['image'].shape[2:],
                                                                                          mode='bilinear',
                                                                                          antialias=True)],
                                                         dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:],
                                                               size=img['image_raw'].shape[2:], mode='bilinear',
                                                               antialias=True)

        pose, _ = self.D.predict_pose(img, c)
        return pose

    def get_pose_params_D(self, real_img, real_seg, real_c, cur_nimg):
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3),
                         0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if not isinstance(real_img,dict):
            if self.neural_rendering_resolution_final is not None:
                alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
                neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (
                        1 - alpha) + self.neural_rendering_resolution_final * alpha))
            else:
                neural_rendering_resolution = self.neural_rendering_resolution_initial
            real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter,
                                             filter_mode=self.filter_mode)
            real_seg_raw = filtered_resizing(real_seg, size=neural_rendering_resolution, f=self.resample_filter,
                                             filter_mode=self.filter_mode)
            if self.blur_raw_target:
                blur_size = np.floor(blur_sigma * 3)
                if blur_size > 0:
                    f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(
                        blur_sigma).square().neg().exp2()
                    real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

            real_img = {'image': real_img, 'image_raw': real_img_raw, 'image_mask': real_seg_raw}
            
        else:
            assert 'image_raw' in real_img.keys(), 'image_raw is not in real_img.keys()'
            assert 'image' in real_img.keys(), 'image is not in real_img.keys()'


        # get pose_params from real image
        real_img_tmp_image = real_img['image'].detach().requires_grad_(True)
        real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(True)
        real_img_tmp_image_mask = real_img['image_mask'].detach().requires_grad_(True)
        real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_mask': real_img_tmp_image_mask}

        predicted_real_pose = self.run_D_pose_prediction(real_img_tmp, real_c, blur_sigma=blur_sigma)
        return predicted_real_pose

    def get_pose_params_G(self,z,c):
        predicted_pose = self.G.get_pose_params(z,c)
        return predicted_pose

    def accumulate_gradients(self, phase, real_img, real_seg, real_c, real_pose,
                             gen_z, gen_c,gen_pose,
                             gain, cur_nimg, cur_nimg_start):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3),
                         0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        self.blur_sigma = blur_sigma
        r1_gamma = self.r1_gamma
        self.G.rendering_kwargs["density_noise"] = max(1 - cur_nimg / (self.density_noise_fade_kimg * 1e3),
                                                       0) if self.density_noise_fade_kimg > 0 else 0

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None
        self.swapping_prob = swapping_prob

        if self.neural_rendering_resolution_final is not None:
            alpha = min((cur_nimg-cur_nimg_start) / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (
                        1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        self.neural_rendering_resolution = neural_rendering_resolution

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter,
                                         filter_mode=self.filter_mode)
        real_seg_raw = filtered_resizing(real_seg, size=neural_rendering_resolution, f=self.resample_filter,
                                         filter_mode=self.filter_mode)


        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(
                    blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw, 'image_mask': real_seg_raw}


        input_pose_params = self.get_pose_params_G(gen_z,gen_c)


        for i in range(input_pose_params.shape[1]):
            training_stats.report('pose_scale/input_pose_params_{}'.format(i),
                                          (input_pose_params[:, i]).abs().mean() / np.pi * 180)


        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, input_pose_params, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution)


                gen_logits, predict_gen_pose = self.run_D(gen_img, gen_c, gt_pose=None, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake_posed', gen_logits)
                training_stats.report('Loss/signs/fake_posed', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                # Lpreg
                if self.input_pose_params_reg_loss_weight>0 and cur_nimg<(self.input_pose_params_reg_loss_kimg+200) * 1e3:

                    if cur_nimg<self.input_pose_params_reg_loss_kimg  * 1e3:
                        loss_weight_scale = 1.0
                    else:
                        loss_weight_scale = 1.0 - (cur_nimg - self.input_pose_params_reg_loss_kimg  * 1e3)/(200 * 1e3)
                        loss_weight_scale = max(loss_weight_scale,0.0)

                    training_stats.report0('Progress/pose_params_reg_loss_weight_scale',loss_weight_scale)
                    loss_weight = loss_weight_scale *  self.input_pose_params_reg_loss_weight
                    input_pose_params_reg_loss = (input_pose_params - gen_pose).square().sum([1]) * loss_weight

                else:
                    input_pose_params_reg_loss = 0

                training_stats.report('Loss/D/input_pose_params_reg_loss', input_pose_params_reg_loss)

                # Lrear


                training_stats.report('Loss/G/Dloss', loss_Gmain)

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain+input_pose_params_reg_loss).mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs[
            'reg_type'] == 'l1':
            if swapping_prob is not None:
                # c_swapped = torch.roll(gen_c.clone(), 1, 0)
                # c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                p_swapped = torch.roll(input_pose_params.clone(), 1, 0)
                rand_ = torch.rand([], device=gen_c.device)
                c_gen_conditioning = torch.where( rand_< swapping_prob, c_swapped, gen_c)
                pose_params_conditioning = torch.where(rand_ < swapping_prob, p_swapped, input_pose_params)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pose_params_conditioning = torch.zeros([gen_c.shape[0],6]).to(gen_c.device)


            ws = self.G.mapping(gen_z, c_gen_conditioning, pose_params_conditioning,update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, input_pose_params,update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)[
                'sigma']
            sigma_initial = sigma[:, :sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs[
                'density_reg']
            training_stats.report('Loss/G_reg/TVloss_L1', TVloss)
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs[
            'reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                # c_swapped = torch.roll(gen_c.clone(), 1, 0)
                # c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                p_swapped = torch.roll(input_pose_params.clone(), 1, 0)
                rand_ = torch.rand([], device=gen_c.device)
                c_gen_conditioning = torch.where( rand_< swapping_prob, c_swapped, gen_c)
                pose_params_conditioning = torch.where(rand_ < swapping_prob, p_swapped, input_pose_params)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pose_params_conditioning = torch.zeros([gen_c.shape[0],6]).to(gen_c.device)

            ws = self.G.mapping(gen_z, c_gen_conditioning, pose_params_conditioning,update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)[
                'sigma']
            sigma_initial = sigma[:, :sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                # c_swapped = torch.roll(gen_c.clone(), 1, 0)
                # c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                p_swapped = torch.roll(input_pose_params.clone(), 1, 0)
                rand_ = torch.rand([], device=gen_c.device)
                c_gen_conditioning = torch.where( rand_< swapping_prob, c_swapped, gen_c)
                pose_params_conditioning = torch.where(rand_ < swapping_prob, p_swapped, input_pose_params)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pose_params_conditioning = torch.zeros([gen_c.shape[0],6]).to(gen_c.device)

            ws = self.G.mapping(gen_z, c_gen_conditioning,pose_params_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, input_pose_params,update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)[
                'sigma']
            sigma_initial = sigma[:, :sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs[
                'density_reg']
            training_stats.report('Loss/G_reg/TVloss_monotonic-detach', TVloss)
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs[
            'reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                # c_swapped = torch.roll(gen_c.clone(), 1, 0)
                # c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                p_swapped = torch.roll(input_pose_params.clone(), 1, 0)
                rand_ = torch.rand([], device=gen_c.device)
                c_gen_conditioning = torch.where( rand_< swapping_prob, c_swapped, gen_c)
                pose_params_conditioning = torch.where(rand_ < swapping_prob, p_swapped, input_pose_params)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pose_params_conditioning = torch.zeros([gen_c.shape[0],6]).to(gen_c.device)

            ws = self.G.mapping(gen_z, c_gen_conditioning, pose_params_conditioning,update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1  # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)[
                'sigma']
            sigma_initial = sigma[:, :sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()

            if swapping_prob is not None:
                # c_swapped = torch.roll(gen_c.clone(), 1, 0)
                # c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                p_swapped = torch.roll(input_pose_params.clone(), 1, 0)
                rand_ = torch.rand([], device=gen_c.device)
                c_gen_conditioning = torch.where( rand_< swapping_prob, c_swapped, gen_c)
                pose_params_conditioning = torch.where(rand_ < swapping_prob, p_swapped, input_pose_params)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pose_params_conditioning = torch.zeros([gen_c.shape[0],6]).to(gen_c.device)


            ws = self.G.mapping(gen_z, c_gen_conditioning, pose_params_conditioning,update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, input_pose_params,update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)[
                'sigma']
            sigma_initial = sigma[:, :sigma.shape[1] // 2]
            sigma_perturbed = sigma[:, sigma.shape[1] // 2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs[
                'density_reg']
            training_stats.report('Loss/G_reg/TVloss_monotonic-fixed', TVloss)
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):

                gen_img, _gen_ws = self.run_G(gen_z, gen_c, input_pose_params, swapping_prob=swapping_prob,
                                              neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits, predict_gen_pose = self.run_D(gen_img, gen_c, gt_pose=None, blur_sigma=blur_sigma,
                                                          update_emas=True)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus( gen_logits)  # -log (1 - sigmoid(gen_logits)) = log (1 + exp(gen_logits)) = softplus(gen_logits)

                pose_param_loss = (predict_gen_pose - input_pose_params).square().sum([1]) * self.pose_loss_weight
                training_stats.report('Loss/D/Poseloss', pose_param_loss)

                for i in range(predict_gen_pose.shape[1]):
                    training_stats.report('Loss_pose/fake_{}'.format(i),
                                          (predict_gen_pose[:, i] - input_pose_params[:, i]).abs().mean() / np.pi * 180)
                    training_stats.report('pose_scale/fake_{}'.format(i),
                                          (predict_gen_pose[:, i]).abs().mean() / np.pi * 180)




            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + pose_param_loss).mean().mul(gain).backward()


        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_mask = real_img['image_mask'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_mask': real_img_tmp_image_mask}

                real_logits, predicted_real_pose = self.run_D(real_img_tmp, real_c,
                                                              gt_pose=None,
                                                              blur_sigma=blur_sigma)

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())


                for i in range(predicted_real_pose.shape[1]):
                    training_stats.report('Loss_pose/real_{}'.format(i), (
                            predicted_real_pose[:, i] - real_pose[:, i]).abs().mean() / np.pi * 180)
                    training_stats.report('pose_scale/real_{}'.format(i),
                                          (predicted_real_pose[:, i]).abs().mean() / np.pi * 180)


                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(
                        -real_logits)  # - log sigmoid(real_logits) =  log (1 + exp(-real_logits)) = softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    training_stats.report('Loss/D/loss_gen', loss_Dgen)
                    training_stats.report('Loss/D/loss_real', loss_Dreal)


                    #

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()],
                                                           inputs=[real_img_tmp['image'], real_img_tmp['image_raw'], real_img_tmp['image_mask']],
                                                           create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                            r1_grads_image_mask = r1_grads[2]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                        r1_penalty_seg = r1_grads_image_mask.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_mask']],
                                                           create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_mask = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                        r1_penalty_seg = r1_grads_image_mask.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) + r1_penalty_seg * (self.r1_gamma_seg / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/r1_penalty_seg', r1_penalty_seg)
                    training_stats.report('Loss/D/reg', loss_Dr1)


            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------
