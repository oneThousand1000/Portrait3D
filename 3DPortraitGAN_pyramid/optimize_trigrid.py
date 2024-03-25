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
import json
import legacy

from camera_utils import LookAtPoseSampler,FOV_to_intrinsics
from torch_utils import misc
import glob
import PIL
from torch.utils.data import DataLoader
import torch.nn.functional as F

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        camera_info_path = os.path.join(path, 'data', 'camera_info.json')
        with open(camera_info_path, 'r') as f:
            camera_info = json.load(f)

        self.camera_info = camera_info

        image_list = list(camera_info.keys())
        self.image_list = []
        for img_name in image_list:
            if os.path.exists(os.path.join(path, 'update_data', img_name)):
                self.image_list.append(img_name)

        self.image_dir = os.path.join(path, 'update_data')



    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]

        img_path = os.path.join(self.image_dir, img_name)

        img = imageio.imread(img_path)
        img = np.array(img).astype(np.float32)
        img = img / 255.0
        # to -1,1
        img = img * 2 - 1
        img = torch.from_numpy(img) # [H, W, C]


        camera_info = self.camera_info[img_name]
        camera_info = torch.from_numpy(np.array(camera_info)).float().squeeze()

        return img, camera_info


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

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'])
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'])

    G.rendering_kwargs['ray_start'] = 2.35



    print("Reloading Modules!")
    from training.neural_renderer import TriPlaneGenerator
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=False)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    G.set_batch_size(1)

    intrinsics = FOV_to_intrinsics(12.447863, device=device)
    cam_pivot = torch.tensor([0, 0.0649, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    default_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot,
                                                           radius=cam_radius, device=device)
    default_cam_params = torch.cat([default_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    res_dir = data_dir

    update_data_dir = os.path.join(res_dir, 'update_data')
    if not os.path.exists(update_data_dir):
        print('update data not found for ', res_dir)
        return

    print('optimize for ', res_dir)

    log_dir = os.path.join(res_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_img_dir = os.path.join(log_dir, 'img')
    os.makedirs(log_img_dir, exist_ok=True)

    log_ckpt_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_ckpt_dir, exist_ok=True)


    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    ckpt_path = os.path.join(res_dir, 'checkpoints/df.pth')
    if not os.path.exists(ckpt_path):
        print('checkpoints not found for ', res_dir)
        return

    print('Loading checkpoints from "%s"...' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)['model']
    trigrid = {
        8: ckpt['trigrids_8'].to(device).requires_grad_(True),
        16: ckpt['trigrids_16'].to(device).requires_grad_(True),
        32: ckpt['trigrids_32'].to(device).requires_grad_(True),
        64: ckpt['trigrids_64'].to(device).requires_grad_(True),
        128: ckpt['trigrids_128'].to(device).requires_grad_(True),
        256: ckpt['trigrids_256'].to(device).requires_grad_(True),
        512: ckpt['trigrids_512'].to(device).requires_grad_(True),
    }
    ws = ckpt['ws'].to(device)

    epoch_num = 19
    patch_resolution = 256
    lr = 1.0
    params = [
        {'params': trigrid[8], 'lr': lr},
        {'params': trigrid[16], 'lr': lr},
        {'params': trigrid[32], 'lr': lr},
        {'params': trigrid[64], 'lr': lr},
        {'params': trigrid[128], 'lr': lr},
        {'params': trigrid[256], 'lr': lr},
        {'params': trigrid[512], 'lr': lr},
    ]
    # optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))

    from optimizer import Adan

    # Adan usually requires a larger LR
    optimizer = Adan(params, eps=1e-8,  weight_decay=2e-5, max_grad_norm=5.0, foreach=False)


    dataset = Dataset(res_dir)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(epoch_num):
        print('epoch: ', epoch)

        for i, data in enumerate(data_loader):
            print('iter: ', i)
            image, cam = data

            gt_img = image.clone().detach().to(device).permute(0, 3, 1, 2) # 1, 3, 512, 512 [-1,1]
            cam = cam.clone().detach().to(device)
            #print('fetch data done')
            # render
            output = G.render_planes(ws=ws, planes=trigrid, c=cam, noise_mode='const',
                                  neural_rendering_resolution=512, chunk=4096,render_bg = False, patch_resolution = patch_resolution)

            img = output['image_raw'] # 1, 3, 512, 512 [-1,1]
            mask = output['image_mask'] # 1, 1, 512, 512 [0,1]
            patch_info = output['patch_info']


            # L2 loss

            top, left = patch_info[0]
            gt_img = gt_img[:, :, top:top + patch_resolution, left:left + patch_resolution]

            loss = torch.mean((img - gt_img) ** 2 * mask)*1e3
            print('loss: ', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



         # save checkpoint
        if epoch == epoch_num - 1:
            ckpt = {
                'trigrids_8': trigrid[8].clone().detach(),
                'trigrids_16': trigrid[16].clone().detach(),
                'trigrids_32': trigrid[32].clone().detach(),
                'trigrids_64': trigrid[64].clone().detach(),
                'trigrids_128': trigrid[128].clone().detach(),
                'trigrids_256': trigrid[256].clone().detach(),
                'trigrids_512': trigrid[512].clone().detach(),
                'ws': ws,
            }

            torch.save({'model': ckpt}, f'{log_ckpt_dir}/epoch_{epoch:05d}.pth')

        with torch.no_grad():
            output = G.render_planes(ws=ws, planes=trigrid, c=default_cam_params, noise_mode='const',
                                     neural_rendering_resolution=512, chunk=4096, render_bg=False,
                                     patch_resolution=None)

        img = output['image_raw']  # 1, 3, 512, 512 [-1,1]
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        print('save image to ', f'{log_img_dir}/epoch_{epoch}.png')
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{log_img_dir}/epoch_{epoch}.png')





#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------