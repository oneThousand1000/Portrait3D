# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dnnlib
import PIL
from camera_utils import LookAtPoseSampler

def project(
        G,
        c,
        p,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=100,
        w_name: str,
        no_sr = False
):
    os.makedirs(f'{outdir}/{w_name}_w', exist_ok=True)
    outdir = f'{outdir}/{w_name}_w'
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    # Compute w stats.
    w_avg_path = './w_avg.npy'
    w_std_path = './w_std.npy'
    if (not os.path.exists(w_avg_path)) or (not os.path.exists(w_std_path)):
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        # c_samples = c.repeat(w_avg_samples, 1)

        # use avg look at point

        camera_lookat_point = torch.tensor([0, 0.0649, 0], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=2.7, device=device)
        focal_length = 6.5104166
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c_samples = c_samples.repeat(w_avg_samples, 1)
        p_samples = p.repeat(w_avg_samples, 1)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples, p_samples, )  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        # print('save w_avg  to ./w_avg.npy')
        # np.save('./w_avg.npy',w_avg)
        w_avg_tensor = torch.from_numpy(w_avg).cuda()
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        # np.save(w_avg_path, w_avg)
        # np.save(w_std_path, w_std)
    else:
        # w_avg = np.load(w_avg_path)
        # w_std = np.load(w_std_path)
        raise Exception(' ')

    # z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    # c_samples = c.repeat(w_avg_samples, 1)
    # w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
    # w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    # w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    # w_avg_tensor = torch.from_numpy(w_avg).cuda()
    # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    #url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    url = './models/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    p_opt = p.requires_grad_(True)

    params = [{'params': w_opt, 'lr': 0.1},
                {'params': list(noise_bufs.values()), 'lr': 0.1},
              {'params': p_opt, 'lr': 0.002}
              ]

    optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)
        # synth_images = G.synthesis(ws,c, noise_mode='const')['image']
        if no_sr:
            synth_images = G.synthesis(ws, c=c, neural_rendering_resolution = 256, noise_mode='const', apply_def=True, pose_params=p_opt)['image_raw']
            assert synth_images.shape[2] == 256
        else:
            synth_images = G.synthesis(ws, c=c, noise_mode='const', apply_def=True, pose_params=p_opt)['image']

        if step % image_log_step == 0 or step == num_steps - 1:
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

                if step == num_steps - 1:
                    frontal_c = torch.tensor([[1.0000e+00, 1.0505e-09, 4.3685e-08, -1.1805e-07, 0.0000e+00,
                      -9.9951e-01, 2.4033e-02, -1.1805e-07, 4.3714e-08, -2.4033e-02,
                      -9.9951e-01, 2.6992e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                      1.0000e+00, 6.7287e+00, 0.0000e+00, 5.0000e-01, 0.0000e+00,
                      6.7287e+00, 5.0000e-01, 0.0000e+00, 0.0000e+00, 1.0000e+00]], device=device, dtype=torch.float32)

                    synth_images = G.synthesis(ws, c=frontal_c, noise_mode='const', apply_def=False, pose_params=None)['image']
                    vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/canonical.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # if step % 10 == 0:
        #     with torch.no_grad():
        #         print({f'step {step}, first projection _{w_name}': loss.detach().cpu()})

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del G
    return w_opt,p_opt