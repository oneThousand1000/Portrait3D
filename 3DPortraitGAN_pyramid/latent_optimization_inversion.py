import glob

import numpy as np
import dnnlib
import legacy
from proj.projector import w_projector, w_plus_projector
from proj.configs import global_config, hyperparameters
from PIL import Image
import torch
import json
import os
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing


# ----------------------------------------------------------------------------
class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = hyperparameters.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = hyperparameters.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, use_wandb=False):
        loss = 0.0

        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(global_config.device), None,
                                            truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            if hyperparameters.regulizer_l2_lambda > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)

                loss += l2_loss_val * hyperparameters.regulizer_l2_lambda

            if hyperparameters.regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))

                loss += loss_lpips * hyperparameters.regulizer_lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch, use_wandb):
        ret_val = self.ball_holder_loss_lazy(new_G, hyperparameters.latent_ball_num_of_samples, w_batch, use_wandb)
        return ret_val





def l2_loss(real_images, generated_images):
    l2_criterion = torch.nn.MSELoss(reduction='mean')
    loss = l2_criterion(real_images, generated_images)
    return loss


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def run_D_pose_prediction(img, c, blur_sigma=0, D=None):
    blur_size = np.floor(blur_sigma * 3)
    if blur_size > 0:
        with torch.autograd.profiler.record_function('blur'):
            f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(
                blur_sigma).square().neg().exp2()
            img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
    pose, _ = D.predict_pose(img, c)
    return pose


def get_pose_params(real_img, real_seg, real_c, D=None, neural_rendering_resolution=None, blur_sigma=None,
                    resample_filter=None, filter_mode=None):
    real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=resample_filter,
                                     filter_mode=filter_mode)

    real_seg_raw = filtered_resizing(real_seg, size=neural_rendering_resolution, f=resample_filter,
                                     filter_mode=filter_mode)

    if True:
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(
                blur_sigma).square().neg().exp2()
            real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

    real_img = {'image': real_img, 'image_raw': real_img_raw, 'image_mask': real_seg_raw}

    # get pose_params from real image
    real_img_tmp_image = real_img['image'].detach().requires_grad_(True)
    real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(True)
    real_img_tmp_image_mask = real_img['image_mask'].detach().requires_grad_(True)
    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw,
                    'image_mask': real_img_tmp_image_mask}

    predicted_real_pose = run_D_pose_prediction(real_img_tmp, real_c, blur_sigma=blur_sigma, D=D)
    return predicted_real_pose


if __name__ == '__main__':
    # input_dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input')
    parser.add_argument('--model_pkl', type=str, default='input')
    parser.add_argument('--pose_prediction_kwargs_path', type=str, default='input')
    input_dir = parser.parse_args().input_dir
    model_pkl = parser.parse_args().model_pkl
    pose_prediction_kwargs_path = parser.parse_args().pose_prediction_kwargs_path
    # ----------------------------------------------------------------------------
    sampling_multiplier = 2.0
    
    print('Loading networks from "%s"...' % model_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(model_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
        print('resume_data', resume_data.keys())
        G = resume_data['G_ema'].to(device)  # type: ignore
        D = resume_data['D_ema'].to(device)  # type: ignore
    
    G.set_batch_size(1)
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    
    G.rendering_kwargs['ray_start'] = 2.35
    
    print('Loading pose_prediction_kwargs from "%s"...' % pose_prediction_kwargs_path)
    with open(pose_prediction_kwargs_path, 'r') as f:
        pose_predict_kwargs = json.load(f)
    
    
    
    
    
    camera_path = os.path.join(input_dir, 'result.json')
    print('Loading camera pose from "%s"...' % camera_path)
    with open(camera_path, 'r') as f:
        camera_poses = json.load(f)
    
    print('Loading images from "%s"...' % input_dir)
    image_base_dir = os.path.join(input_dir, 'aligned_images')
    mask_base_path = os.path.join(input_dir, 'mask')
    
    images = glob.glob(os.path.join(image_base_dir, '*'))
    
    print('images', images)
    for image_path in images:
        image_name = os.path.basename(image_path)
        mask_path = os.path.join(mask_base_path, image_name)
        print('projecting image: "%s"' % image_path)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        # image_name = os.path.basename(paths_config.input_data_path)
        camera_pose = camera_poses[image_name]
        cam2world_pose = torch.tensor(camera_pose['camera_pose'], device=device)
        focal_length = 6.5104166
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
        with torch.no_grad():
            image_p = image.resize((G.img_resolution, G.img_resolution), Image.BILINEAR)
            image_p = np.array(image_p)
            image_p = image_p.transpose(2, 0, 1)
            image_p = torch.tensor(image_p, device=device)
            image_p = image_p.to(device).to(torch.float32) / 127.5 - 1
            image_p = image_p.unsqueeze(0)
    
            mask_p = np.array(mask)[:, :, None]
            mask_p = mask_p.transpose(2, 0, 1)
            mask_p = torch.tensor(mask_p, device=device)
            mask_p = mask_p.to(device).to(torch.float32) / 255.0
            mask_p = mask_p.unsqueeze(0)
    
            resample_filter = pose_predict_kwargs['resample_filter']
            resample_filter = torch.tensor(resample_filter, device=device).to(torch.float32)
    
            p = get_pose_params(real_img=image_p,
                                real_seg=mask_p,
                                real_c=c,
                                D=D,
                                neural_rendering_resolution=pose_predict_kwargs['neural_rendering_resolution'],
                                blur_sigma=pose_predict_kwargs['blur_sigma'],
                                resample_filter=resample_filter,
                                filter_mode=pose_predict_kwargs['filter_mode'])
    
        # ----------------------------------------------------------------------------
        image_name = image_name[:-4]
        # coach = SingleIDCoach(None, False, c, p)
        # coach.train(image=image, image_name=image_name[:-4])
        w_path_dir = f'{input_dir}/inversion'
        os.makedirs(w_path_dir, exist_ok=True)
        use_ball_holder = True
        # for fname, image in tqdm(self.data_loader):
        # image_name = fname[0]
    
        embedding_dir = f'{w_path_dir}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)
        image.save(f'{embedding_dir}/original.png')
        w_pivot = None
        # if hyperparameters.use_last_w_pivots:
        #     w_pivot = self.load_inversions(w_path_dir, image_name)
        # elif not hyperparameters.use_last_w_pivots or w_pivot is None:
        #     w_pivot = self.calc_inversions(image, image_name)
        # image = torch.tensor(image, device=device)
        if os.path.exists(f'{embedding_dir}/0.pt'):
            w_pivot = torch.load(f'{embedding_dir}/0.pt').to(global_config.device)
        else:
            image = image.resize((G.img_resolution, G.img_resolution), Image.BILINEAR)
            image = np.array(image)
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image, device=device)
            image = image.to(device).to(torch.float32) / 127.5 - 1
            image = image.unsqueeze(0)
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            # id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            w_pivot = w_projector.project(G, c, p, embedding_dir, id_image, device=torch.device('cuda'), w_avg_samples=600,
                                          num_steps=500,
                                          w_name=image_name)
            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)
            torch.save(w_pivot, f'{embedding_dir}/inversion.pt')

