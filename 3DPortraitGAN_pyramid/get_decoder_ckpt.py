# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic',
                     grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14,
                     cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), large_pose= False,
                     **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    camera_lookat_point = torch.tensor([0, 0.0649, 0], device=device)
    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, camera_lookat_point, radius=2.7, device=device)
    focal_length = 6.5104166 # if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)

    p = torch.zeros([len(zs), 6], device=device)

    ws = G.mapping(z=zs, c=c, p=p,truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = 'interpolation_{}_{}/'.format(all_seeds[0], all_seeds[1])
        os.makedirs(outdir, exist_ok=True)
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                if large_pose:
                    # 0 - 2pi
                    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + (frame_idx / w_frames) * 2 * np.pi,
                                                              np.pi / 2,
                                                              camera_lookat_point, radius=2.7, device=device)
                else:
                    pitch_range = 0.25
                    yaw_range = 0.35
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw_range * np.sin(2 * np.pi * frame_idx / (num_keyframes * w_frames)),
                                                            np.pi/2 -0.05 + pitch_range * np.cos(2 * np.pi * frame_idx / (num_keyframes * w_frames)),
                                                            camera_lookat_point, radius=2.7, device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                focal_length = 6.5104166 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                entangle = 'camera'
                if entangle == 'conditioning':
                    c_forward = torch.cat([LookAtPoseSampler.sample(np.pi/2,
                                                                    np.pi/2,
                                                                    camera_lookat_point,
                                                                    radius=2.7, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    img = G.synthesis(ws=w_c, c=c_forward, noise_mode='const')[image_mode][0]
                elif entangle == 'camera':
                    img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0]
                elif entangle == 'both':
                    w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    img = G.synthesis(ws=w_c, c=c[0:1], noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

                if gen_shapes:
                    # generate shapes
                    print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))

                    samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'])
                    samples = samples.to(device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm(total = samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                torch.manual_seed(0)
                                sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                sigmas[:, head:head+max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)

                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = 0
                    sigmas[-pad:] = 0
                    sigmas[:, :pad] = 0
                    sigmas[:, -pad_top:] = 0
                    sigmas[:, :, :pad] = 0
                    sigmas[:, :, -pad:] = 0

                    output_ply = True
                    if output_ply:
                        from shape_utils import convert_sdf_samples_to_ply
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                    else: # output mrc
                        with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                            mrc.data[:] = sigmas

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
def extract_decoder_ckpt(
    network_pkl: str,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """


    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # print the number of parameters of superresolution
    print('Number of parameters of superresolution: %d' % sum(p.numel() for p in G.superresolution.parameters()))
    print('Number of parameters of backbone: %d' % sum(p.numel() for p in G.backbone.parameters()))
    # exit() # sum(p.numel() for p in model.parameters())
    # save the ckpt of G.decoder
    decoder_ckpt = G.decoder.state_dict()
    decoder_state_dict = {}
    for k, v in decoder_ckpt.items():
        decoder_state_dict['decoder.'+k] = v
    ckpt = decoder_state_dict

    torgb_ckpt = G.torgb.state_dict()
    rgb_state_dict = {}
    for k, v in torgb_ckpt.items():
        rgb_state_dict['torgb.'+k] = v
    ckpt.update(rgb_state_dict)

    pose_branch_ckpt = G.pose_branch.state_dict()
    pose_branch_state_dict = {}
    for k, v in pose_branch_ckpt.items():
        pose_branch_state_dict['pose_branch.'+k] = v
    ckpt.update(pose_branch_state_dict)

    # save parameters of G.decoder


    params = {'z_dim': G.z_dim,
              'c_dim': G.c_dim,
              'w_dim': G.w_dim,
              'img_resolution': G.img_resolution,
              'img_channels': G.img_channels,
              'rendering_kwargs': G.rendering_kwargs,
              'batch_size': G.batch_size,
              'thickness': 0.25}

    print(params)
    ckpt = {'params': params,   # parameters of G.decoder
            'state_dict': ckpt}
    print('save decoder ckpt to ./models/'+os.path.basename(network_pkl).replace('.pkl', '_decoder.ckpt'))
    torch.save(ckpt, './models/'+os.path.basename(network_pkl).replace('.pkl', '_decoder.ckpt'))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_decoder_ckpt() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------