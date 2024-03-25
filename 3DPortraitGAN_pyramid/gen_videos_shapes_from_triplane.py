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
import glob
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

def gen_interp_video(G, mp4: str, trigrid=None,ws=None, shuffle_seed=None, w_frames=60*4, kind='cubic',
                     grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14,
                      image_mode='image', gen_shapes=False, device=torch.device('cuda'), large_pose= False,
                     **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    num_keyframes = 1


    camera_lookat_point = torch.tensor([0, 0.0649, 0], device=device)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, camera_lookat_point, radius=2.7, device=device)
    focal_length = 6.5104166
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(ws), 1)

    p = torch.zeros([len(ws), 6], device=device)

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
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)


    all_poses = []

    if large_pose:
        image_row = []


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
                focal_length = 6.5104166
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                img = G.render_planes(ws=w.unsqueeze(0), planes=trigrid, c=c[0:1], noise_mode='const', neural_rendering_resolution=512,chunk = 4096)[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)
                if large_pose and frame_idx % int(num_keyframes * w_frames//8) == 0:
                    image_row.append((img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8))

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)

    if large_pose:
        import PIL
        image_row = torch.cat(image_row, 1).cpu().numpy()
        PIL.Image.fromarray(image_row.astype(np.uint8)).save(mp4.replace('.mp4', '_large_pose.png'))


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
@click.option('--data_dir', help='Network pickle filename', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image_depth', 'image_raw']), required=False, metavar='STR', default='image_raw', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)

def generate_images(
    network_pkl: str,
    data_dir: str,
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    G.rendering_kwargs['ray_start'] = 2.35



    print("Reloading Modules!")
    from training.smpl_triplane import TriPlaneGenerator
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    G.set_batch_size(1)


    for res_dir in glob.glob(data_dir + '/*'):
        outdir = os.path.join(res_dir, 'results')
        os.makedirs(outdir, exist_ok=True)



        if nrr is not None: G.neural_rendering_resolution = nrr

        if truncation_cutoff == 0:
            truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
        if truncation_psi == 1.0:
            truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

        ckpt_path = os.path.join(res_dir, 'checkpoints/df.pth')
        if not os.path.exists(ckpt_path):
            continue
        print('Loading checkpoints from "%s"...' % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['model']
        trigrid = {
            8:ckpt['trigrids_8'].to(device),
            16:ckpt['trigrids_16'].to(device),
            32:ckpt['trigrids_32'].to(device),
            64:ckpt['trigrids_64'].to(device),
            128:ckpt['trigrids_128'].to(device),
            256:ckpt['trigrids_256'].to(device),
        }
        ws = ckpt['ws'].to(device)

        output = os.path.join(outdir, f'large_pose.mp4')
        print('Generating video "%s"...' % output)
        if not os.path.exists(output):
            gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames,
                         trigrid=trigrid,ws=ws, shuffle_seed=shuffle_seed, psi=truncation_psi,
                         truncation_cutoff=truncation_cutoff,  image_mode=image_mode, large_pose=True)
        output = os.path.join(outdir, f'small_pose.mp4')
        print('Generating video "%s"...' % output)
        if not os.path.exists(output):
            gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames,
                         trigrid=trigrid,ws=ws,  shuffle_seed=shuffle_seed, psi=truncation_psi,
                         truncation_cutoff=truncation_cutoff, image_mode=image_mode, large_pose=False)

        print('Generating shapes...')

        shape_res = 512
        max_batch = 1000000
        shape_format = '.mrc'

        if shape_format == '.ply':
            from shape_utils import convert_sdf_samples_to_ply
            shape_path =os.path.join(outdir, f'shape.ply')
        elif shape_format == '.mrc':  # output mrc
            shape_path = os.path.join(outdir, f'shape.mrc')

        if not os.path.exists(shape_path):

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0],
                                                               cube_length=0.9)  # .reshape(1, -1, 3)
            samples = samples.to(device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total=samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample_trigrid(samples[:, head:head + max_batch],
                                         transformed_ray_directions_expanded[:, :samples.shape[1] - head], planes = trigrid, truncation_psi=truncation_psi,
                                         truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head + max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                           os.path.join(outdir, f'shape.ply'), level=15)
            elif shape_format == '.mrc':  # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'shape.mrc'), overwrite=True, shape=sigmas.shape,
                                      mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------