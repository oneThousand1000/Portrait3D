from .trigrid_rendering.neural_render import NeuralRender as TrigridNeRFRenderer
from .renderer import NeRFRenderer

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import get_encoder

from .utils import safe_normalize

import numpy as np
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 device,
                 trigrid_shapes,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        super().__init__(opt)


        self.triplane_names = {}
        for k in trigrid_shapes:
            self.register_parameter(k, torch.nn.Parameter(torch.randn(trigrid_shapes[k])))

            if k.startswith('trigrid'):
                res = int(k.split('_')[1])
                self.triplane_names[res] = k

        # sort the triplane names by resolution
        self.triplane_names = {k: self.triplane_names[k] for k in sorted(self.triplane_names.keys())}

        params = {'z_dim': 512, 'c_dim': 25, 'w_dim': 512, 'img_resolution': 512, 'img_channels': 3,
                  'rendering_kwargs': {'image_resolution': 512, 'disparity_space_sampling': False,
                                       'clamp_mode': 'softplus',
                                       'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                                       'c_gen_conditioning_zero': False, 'gpc_reg_prob': 0.7,
                                       'decoder_activation': 'none', 'use_torgb_raw': True,
                                       'use_background': True, 'triplane_depth': 3, 'c_scale': 1.0,
                                       'superresolution_noise_mode': 'none', 'density_reg': 0.0,
                                       'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                                       'sr_antialias': True, 'radius_scale': 0.7,
                                       'depth_resolution': 48, 'depth_resolution_importance': 48,
                                       'ray_start': 2.3850000000000002, 'ray_end': 3.12,
                                       'box_warp': 0.7, 'density_noise': 0.0},
                  'batch_size': 1, 'thickness': 0.25,"apply_deformation": self.opt.use_body_pose,
                  }
        self.model = TrigridNeRFRenderer(**params).to(device)
        # self.trigrid_4 = torch.nn.Parameter(torch.randn([1, 3, 16 * 3 , 4, 4]))
        # self.trigrid_8 = torch.nn.Parameter(torch.randn([1, 3, 16 * 3 , 8, 8]))
        # self.trigrid_16=torch.nn.Parameter(torch.randn([1, 3, 16 * 3 , 16, 16]))
        # self.trigrid_32 =torch.nn.Parameter(torch.randn([1, 3, 16* 3, 32, 32]))
        # self.trigrid_64 =torch.nn.Parameter(torch.randn([1, 3, 16* 3, 64, 64]))
        # self.trigrid_128 =torch.nn.Parameter(torch.randn([1,3,  16* 3, 128, 128]))
        # self.trigrid_256 = torch.nn.Parameter(torch.randn([1, 3, 32*3, 256, 256]))

        # self.ws = torch.nn.Parameter(torch.randn([1, 14, 512]))

        # background network
        if self.opt.bg_radius > 0:

            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg

            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)

        else:
            assert self.opt.learnable_bg == False
            self.bg_net = None



        self.train_decoder = opt.train_decoder

    def common_forward(self, x):

        # # sigma
        # enc = self.encoder(x, bound=self.bound, max_level=self.max_level)
        #
        # h = self.sigma_net(enc)
        #
        # sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        # albedo = torch.sigmoid(h[..., 1:])


        # return sigma, albedo

        # TODO
        raise NotImplementedError

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        # dx_pos, _ = self.common_forward(
        #     (x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        # dx_neg, _ = self.common_forward(
        #     (x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        # dy_pos, _ = self.common_forward(
        #     (x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        # dy_neg, _ = self.common_forward(
        #     (x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        # dz_pos, _ = self.common_forward(
        #     (x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        # dz_neg, _ = self.common_forward(
        #     (x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        #
        # normal = torch.stack([
        #     0.5 * (dx_pos - dx_neg) / epsilon,
        #     0.5 * (dy_pos - dy_neg) / epsilon,
        #     0.5 * (dz_pos - dz_neg) / epsilon
        # ], dim=-1)

        # return -normal

        # TODO
        raise NotImplementedError

    def normal(self, x):
        # normal = self.finite_difference_normal(x)
        # normal = safe_normalize(normal)
        # normal = torch.nan_to_num(normal)

        # return normal

        # TODO
        raise NotImplementedError

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        '''
        x: [N, 3], in [-bound, bound]
        d: [N, 3], view direction, nomalized in [-1, 1]
        l: [3], plane light direction, nomalized in [-1, 1]
        ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        '''

        # sigma, albedo = self.common_forward(x)
        #
        # if shading == 'albedo':
        #     normal = None
        #     color = albedo
        #
        # else:  # lambertian shading
        #
        #     # normal = self.normal_net(enc)
        #     normal = self.normal(x)
        #
        #     lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0)  # [N,]
        #
        #     if shading == 'textureless':
        #         color = lambertian.unsqueeze(-1).repeat(1, 3)
        #     elif shading == 'normal':
        #         color = (normal + 1) / 2
        #     else:  # 'lambertian'
        #         color = albedo * lambertian.unsqueeze(-1)

        # return sigma, color, normal

        # TODO
        raise NotImplementedError

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma, albedo = self.common_forward(x)



        # return {
        #     'sigma': sigma,
        #     'albedo': albedo,
        # }

        # TODO
        raise NotImplementedError

    def background(self, d):

        h = self.encoder_bg(d)  # [N, C]

        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr,trigrid_lr_ratio):
        params =[]
        assert len(trigrid_lr_ratio) == len(self.triplane_names)
        resolutions = list(self.triplane_names.keys())
        for i in range(len(trigrid_lr_ratio)):
            print(f'{self.triplane_names[resolutions[i]]} lr: {lr*trigrid_lr_ratio[i]}')

            params.append({'params': getattr(self, self.triplane_names[resolutions[i]]), 'lr': lr*trigrid_lr_ratio[i]})

        # params.append({'params': self.ws, 'lr': lr*0.1})

        if self.train_decoder:
            params.append({'params': self.model.parameters(lr), 'lr': lr})
 

        return params

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):
        raise NotImplementedError




    def render(self, rays_o, rays_d, poses, h, w, staged=False, max_ray_batch=4096, bg_color = None,bg_rays_o=None,bg_rays_d=None, **kwargs):
        cam2world_pose = poses.clone()
        cam2world_pose[:, :3, :3] = cam2world_pose[:, :3, :3] * -1
        cam2world_pose[:, 0, 1] *= -1
        cam2world_pose[:, 0, 2] *= -1
        cam2world_pose[:, 1, 0] *= -1
        cam2world_pose[:, 2, 0] *= -1
        cam2world_pose[:, 0, 3] *= -1

        intrinsics = [6.510416666666667,
                    0.0,
                    0.5,
                    0.0,
                    6.510416666666667,
                    0.5,
                    0.0,
                    0.0,
                    1.0]
        intrinsics = torch.tensor(intrinsics).to(cam2world_pose.device)
        camera_params = torch.cat([cam2world_pose.reshape(1, 16), intrinsics.reshape(1, 9)], 1)

        # rays_o, rays_d: [B, N, 3]
        # return: pred_rgb: [B, N, 3]
        #B, N = rays_o.shape[:2]
        H = h
        W = w

        if self.opt.learnable_bg:
            assert bg_color is None, 'bg_color should be None when learnable_bg is True'
            bg_color = self.background(rays_d.contiguous().view(-1, 3))  # [BHW, 3]
            # from [BHW, 3] to [B, H, W, 3]
            bg_color = bg_color.view(-1, H, W, 3).clamp(0, 1.0)


        device = rays_o.device
        N, M, _ = rays_o.shape

        planes = {}
        for res in self.triplane_names:
            planes[res] = getattr(self, self.triplane_names[res])

        if self.opt.use_body_pose:
            # apply_def=apply_def, ws=None, pose_params=pose_params
            pose_params = self.model.sample_pose_params(camera_params)
            apply_def = True
        else:
            pose_params = None
            apply_def = False



        if staged:


            depth = torch.empty((N,M,1), device=device)
            image = torch.empty((N,M, 32), device=device)
            weights_sum = torch.empty((N,M,1), device=device)

            for b in range(N):
                head = 0
                while head < M:
                    tail = min(head + max_ray_batch, M)

                    render_output = self.model.renderer(planes, self.model.decoder, rays_o[b:b + 1, head:tail],
                                                        rays_d[b:b + 1, head:tail], self.model.rendering_kwargs, apply_def=apply_def, ws=None, pose_params=pose_params)  # channels last
                    # {'rgb_final': rgb_final, 'depth_final': depth_final, 'weights': weights.sum(2)}
                    feature_samples = render_output['rgb_final'] # max_ray_batch,32
                    depth_samples = render_output['depth_final'] # 1, max_ray_batch
                    weights_samples = render_output['weights'] # 1, max_ray_batch, depth

                    weights_sum_samples =  weights_samples.sum(2) # 1, max_ray_batch,1



                    depth[b:b + 1, head:tail] = depth_samples
                    weights_sum[b:b + 1, head:tail] = weights_sum_samples
                    image[b:b + 1, head:tail] = feature_samples
                    head += max_ray_batch

            feature_samples = image
            depth_samples = depth
            weights_sum_samples = weights_sum

            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
            weights_sum_samples = weights_sum_samples.permute(0, 2, 1).reshape(N, 1, H, W)

            # Run superresolution to get final image
            if self.model.decoder.activation == "sigmoid":
                assert self.model.decoder.out_channels == 3
                feature_image = feature_image * 2 - 1  # Scale to (-1, 1), taken from ray marcher

            # Generate Raw image
            if self.model.torgb:
                rgb_image = self.model.torgb(feature_image, self.ws[:, -1], fused_modconv=False)
                rgb_image = rgb_image.to(dtype=torch.float32, memory_format=torch.contiguous_format)

            weights_sum_samples = weights_sum_samples * (1 + 2 * 0.001) - 0.001

            # from [B,C,H,W] to [B, H, W, C]
            rgb_image = (rgb_image.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1.0).contiguous()
            depth_image = depth_image.permute(0, 2, 3, 1).contiguous().squeeze(-1) # [B, H, W]
            weights_sum_samples = weights_sum_samples.permute(0, 2, 3, 1).contiguous().squeeze(-1) # [B, H, W]



            if bg_color is not None and self.opt.learnable_bg:
                assert bg_color.shape == rgb_image.shape, f'bg_color.shape {bg_color.shape} should be equal to rgb_image.shape {rgb_image.shape}'
                rgb_image = rgb_image + (1 - weights_sum_samples).unsqueeze(-1) * bg_color


            return {'image': rgb_image, 'depth': depth_image,
                    "weights_sum": weights_sum_samples}


        else:


            # Create triplanes by running StyleGAN backbone


            # Reshape output into three D*32-channel planes, where D=self.rendering_kwargs['triplane_depth'], defines the depth of the tri-grid

            #self.trigrid.register_hook(lambda grad: print(grad,grad.abs().sum(), grad.abs().max(),grad.abs().min()))
            # Perform volume rendering
            render_output = self.model.renderer(planes, self.model.decoder, rays_o,
                                          rays_d, self.model.rendering_kwargs, apply_def=apply_def, ws=None, pose_params=pose_params)  # channels last


            # {'rgb_final': rgb_final, 'depth_final': depth_final, 'weights': weights.sum(2)}
            feature_samples = render_output['rgb_final']
            depth_samples = render_output['depth_final']
            weights_samples = render_output['weights'] # 1, max_ray_batch, depth,1
            weights_sum_samples = weights_samples.sum(2) # 1, max_ray_batch,1

            # Reshape into 'raw' neural-rendered image

            feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
            depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
            weights_sum_samples = weights_sum_samples.permute(0, 2, 1).reshape(N, 1, H, W)
            depth =  weights_samples.shape[-2]
            weights_samples = weights_samples.squeeze(-1).permute(0, 2, 1).reshape(N,depth, H, W)

            # Run superresolution to get final image
            if self.model.decoder.activation == "sigmoid":
                assert self.model.decoder.out_channels == 3
                feature_image = feature_image * 2 - 1  # Scale to (-1, 1), taken from ray marcher
                feature_image.register_hook(lambda x:print(f'in sigmoid, feature_image.grad = {x}'))

            # Generate Raw image
            if self.model.torgb:
                rgb_image = self.model.torgb(feature_image, self.ws[:, -1], fused_modconv=False)

                rgb_image = rgb_image.to(dtype=torch.float32, memory_format=torch.contiguous_format)

            weights_sum_samples = weights_sum_samples * (1 + 2 * 0.001) - 0.001


            # from [B,C,H,W] to [B, H, W, C]
            rgb_image = (rgb_image.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1.0).contiguous()
            depth_image = depth_image.permute(0, 2, 3, 1).contiguous().squeeze(-1)
            weights_sum_samples = weights_sum_samples.permute(0, 2, 3, 1).contiguous().squeeze(-1)
            weights_samples = weights_samples.permute(0, 2, 3, 1).contiguous() # B, H, W, D


            if bg_color is not None and self.opt.learnable_bg:
                assert bg_color.shape == rgb_image.shape, f'bg_color.shape {bg_color.shape} should be equal to rgb_image.shape {rgb_image.shape}'
                rgb_image = rgb_image + (1 - weights_sum_samples).unsqueeze(-1) * bg_color

            return {'image': rgb_image, 'depth': depth_image,"weights":weights_samples, "weights_sum": weights_sum_samples}


            #results = self.run(rays_o, rays_d, **kwargs)


        #return results