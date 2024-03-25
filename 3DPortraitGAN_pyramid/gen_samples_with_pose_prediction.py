
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.smpl_triplane import TriPlaneGenerator
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
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

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

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
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

#         return pose
def run_D_pose_prediction(img, c, blur_sigma=0,D = None):
    blur_size = np.floor(blur_sigma * 3)
    if blur_size > 0:
        with torch.autograd.profiler.record_function('blur'):
            f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
            img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
    pose,_ = D.predict_pose( img, c)
    return pose

def get_pose_params(real_img,real_c,D = None,neural_rendering_resolution = None,blur_sigma = None,resample_filter = None, filter_mode = None):



    real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=resample_filter,
                                         filter_mode=filter_mode)

    if True:
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(
                blur_sigma).square().neg().exp2()
            real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

    real_img = {'image': real_img, 'image_raw': real_img_raw}

        # get pose_params from real image
    real_img_tmp_image = real_img['image'].detach().requires_grad_(True)
    real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(True)
    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

    predicted_real_pose = run_D_pose_prediction(real_img_tmp, real_c, blur_sigma=blur_sigma, D = D)
    return predicted_real_pose


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--test_data',  help='Real data dir', required=True)
@click.option('--outdir',  help='output dir', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=12.447863, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    test_data: str,
    outdir: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
        print('resume_data',resume_data.keys())
        G = resume_data['G_ema'].to(device) # type: ignore
        D = resume_data['D_ema'].to(device) # type: ignore

    G.set_batch_size(1)
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * 2)

    G.rendering_kwargs['ray_start'] = 2.35


    # get pose prediction kwargs
    import json
    pose_prediction_kwargs_path = network_pkl.replace('.pkl','-pose_predict_kwargs.json')          # network-snapshot-001400.pkl
    print('Loading pose_prediction_kwargs from "%s"...' % pose_prediction_kwargs_path)
    with open(pose_prediction_kwargs_path, 'r') as f:
        pose_predict_kwargs = json.load(f)


    # read images
    import glob

    real_image_paths = glob.glob(os.path.join(test_data, 'aligned_images/*'))


    path = os.path.join(test_data, 'result.json')
    with open(path, 'r') as f:
        labels = json.load(f)

    poses = []
    cameras = []
    names = []
    from PIL import Image
    intrinsics = np.reshape(
        np.array([6.510416666666667,
                  0.0,
                  0.5,
                  0.0,
                  6.510416666666667,
                  0.5,
                  0.0,
                  0.0,
                  1.0]), (1, 9)
    )

    with torch.no_grad():
        for real_image_path in real_image_paths:
            image = Image.open(real_image_path).convert('RGB')
            image = image.resize((G.img_resolution, G.img_resolution), Image.BILINEAR)
            image = np.array(image)
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image, device=device)
            image = image.to(device).to(torch.float32) / 127.5 - 1
            image = image.unsqueeze(0)
            image_id = os.path.basename(real_image_path).split('.')[0]

            c = labels[os.path.basename(real_image_path)]['camera_pose']
            c = np.reshape(np.array(c),(1,16))
            c = np.concatenate((c, intrinsics), axis=1)

            c = torch.tensor(c, device=device).to(torch.float32)
            resample_filter = pose_predict_kwargs['resample_filter']
            resample_filter = torch.tensor(resample_filter, device=device).to(torch.float32)

            p = get_pose_params(real_img=image,
                                real_c=c,
                                D=D,
                                neural_rendering_resolution=pose_predict_kwargs['neural_rendering_resolution'],
                                blur_sigma=pose_predict_kwargs['blur_sigma'],
                                resample_filter=resample_filter,
                                filter_mode=pose_predict_kwargs['filter_mode'])

            poses.append(p)
            cameras.append(c)
            names.append(image_id)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if True:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=False)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        G.set_batch_size(1)



    camera_lookat_point = torch.tensor([0, 0.0649, 0], device=device)
    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, camera_lookat_point, radius=2.7, device=device)
    focal_length = 6.5104166  # if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    cond_c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    cond_p = torch.zeros([1, 6], device=device)



    for seed_idx, seed in enumerate(seeds):

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        ws = G.mapping(z, cond_c, cond_p, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        for pose_idx in range(len(poses)):
            p = poses[pose_idx]
            c = cameras[pose_idx]
            name = names[pose_idx]

            img = G.synthesis(ws, c=c, noise_mode='const', apply_def=True, pose_params=p)['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            real_image_path = real_image_paths[pose_idx]
            image = Image.open(real_image_path).convert('RGB')
            image = image.resize((G.img_resolution, G.img_resolution), Image.BILINEAR)
            img = np.concatenate((np.array(image), img), axis=1)

            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{seed}_{name}.png')




#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
