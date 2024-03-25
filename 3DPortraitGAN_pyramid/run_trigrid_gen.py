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
import pickle
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

def gen_trigrids(G, output: str, latend_code_path, shuffle_seed=None, w_frames=60*4, kind='cubic',
                     grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14,
                     cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'),
                     **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]




    camera_lookat_point = torch.tensor([0, 0.0649, 0], device=device)
    # zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    ws = torch.load(latend_code_path, map_location=device)
    print('ws shape', ws.shape)

    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, camera_lookat_point, radius=2.7, device=device)
    focal_length = 6.5104166 # if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(ws), 1)

    p = torch.zeros([len(ws), 6], device=device)


    trigrids,ws = G.gen_planes(ws=ws, c=c[0:1], noise_mode='const')
    print('trigrids shape: ')
    save_files = {'ws': ws}
    for k in trigrids.keys():
        print(k, trigrids[k].shape)
        save_files[f'trigrids_{trigrids[k].shape[3]}'] = trigrids[k]
        print('save as', f'trigrids_{k}')
    # save trigrids to output pkl file
    with open(output, 'wb') as f:
        pickle.dump(save_files, f)

    print(save_files.keys())








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
@click.option('--network', 'network_pkl', help='Network pickle filename',default='./models/model.pkl')
@click.option('--test_data_dir', help='Network pickle filename',default='../test_data')

@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'AFHQ', 'Shapenet']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--inversion_name', type=str,  required=True)

def generate_images(
    network_pkl: str,
    test_data_dir: str,
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    inversion_name: str,
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


    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    G.rendering_kwargs['ray_start'] = 2.35



    G.set_batch_size(1)

    from training.smpl_triplane import TriPlaneGenerator

    if True:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    import glob

    for path in glob.glob(os.path.join(test_data_dir,f'*/samples_new_crop/{inversion_name}/*/inversion.pt')):
        outdir = os.path.dirname(path)
        latend_code = f'{outdir}/inversion.pt'

        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        if nrr is not None: G.neural_rendering_resolution = nrr

        if truncation_cutoff == 0:
            truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
        if truncation_psi == 1.0:
            truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff


        output = os.path.join(outdir, f'inversion_trigrid.pkl')
        if os.path.exists(output):
            print(f'Already exists: {output}')
            continue
        gen_trigrids(G=G, output=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames,
                             latend_code_path=latend_code, shuffle_seed=shuffle_seed, psi=truncation_psi,
                             truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------