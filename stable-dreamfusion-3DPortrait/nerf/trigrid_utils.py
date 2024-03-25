import os
import gc
import glob
import tqdm
import math
import imageio
import psutil
from pathlib import Path
import random
import shutil
import warnings
import tensorboardX

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torchmetrics import PearsonCorrCoef

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver


def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)  # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)  # [B * K]
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1 - r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = - torch.ones_like(i)
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class TrigridTrainer(object):
    def __init__(self,
                 argv,  # command line args
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 teacher_model,
                 guidance,  # guidance network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.as_latent = True
        self.vgg16 = None
        model.to(self.device)
        teacher_model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

            teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[local_rank])
        self.model = model
        self.teacher_model = teacher_model

        # guide model
        self.guidance = guidance
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if self.opt.images is not None:
            self.pearson = PearsonCorrCoef().to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'latent_trigrid_fit_checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            # Save a copy of image_config in the experiment workspace
            if opt.image_config is not None:
                shutil.copyfile(opt.image_config, os.path.join(self.workspace, os.path.basename(opt.image_config)))

            # Save a copy of images in the experiment workspace
            if opt.images is not None:
                for image_file in opt.images:
                    shutil.copyfile(image_file, os.path.join(self.workspace, os.path.basename(image_file)))

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] opt: {self.opt}')
        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        # text embeddings (stable-diffusion)
        if self.opt.text is not None:

            if 'SD' in self.guidance:
                self.embeddings['SD']['default'] = self.guidance['SD'].get_text_embeds([self.opt.text])
                self.embeddings['SD']['uncond'] = self.guidance['SD'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['SD'][d] = self.guidance['SD'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'IF' in self.guidance:
                self.embeddings['IF']['default'] = self.guidance['IF'].get_text_embeds([self.opt.text])
                self.embeddings['IF']['uncond'] = self.guidance['IF'].get_text_embeds([self.opt.negative])

                for d in ['front', 'side', 'back']:
                    self.embeddings['IF'][d] = self.guidance['IF'].get_text_embeds([f"{self.opt.text}, {d} view"])

            if 'clip' in self.guidance:
                self.embeddings['clip']['text'] = self.guidance['clip'].get_text_embeds(self.opt.text)

        if self.opt.images is not None:

            h = int(self.opt.known_view_scale * self.opt.h)
            w = int(self.opt.known_view_scale * self.opt.w)

            # load processed image
            for image in self.opt.images:
                assert image.endswith(
                    '_rgba.png')  # the rest of this code assumes that the _rgba image has been passed.
            rgbas = [cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image in
                     self.opt.images]
            rgba_hw = np.stack(
                [cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
            rgb_hw = rgba_hw[..., :3] * rgba_hw[..., 3:] + (1 - rgba_hw[..., 3:])
            self.rgb = torch.from_numpy(rgb_hw).permute(0, 3, 1, 2).contiguous().to(self.device)
            self.mask = torch.from_numpy(rgba_hw[..., 3] > 0.5).to(self.device)
            print(f'[INFO] dataset: load image prompt {self.opt.images} {self.rgb.shape}')

            # load depth
            depth_paths = [image.replace('_rgba.png', '_depth.png') for image in self.opt.images]
            depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
            depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
            self.depth = torch.from_numpy(depth.astype(np.float32) / 255).to(
                self.device)  # TODO: this should be mapped to FP16
            print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')

            # load normal   # TODO: don't load if normal loss is 0
            normal_paths = [image.replace('_rgba.png', '_normal.png') for image in self.opt.images]
            normals = [cv2.imread(normal_path, cv2.IMREAD_UNCHANGED) for normal_path in normal_paths]
            normal = np.stack([cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA) for normal in normals])
            self.normal = torch.from_numpy(normal.astype(np.float32) / 255).to(self.device)
            print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')

            # encode embeddings for zero123
            if 'zero123' in self.guidance:
                rgba_256 = np.stack(
                    [cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in
                     rgbas])
                rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                rgb_256 = torch.from_numpy(rgbs_256).permute(0, 3, 1, 2).contiguous().to(self.device)
                guidance_embeds = self.guidance['zero123'].get_img_embeds(rgb_256)
                self.embeddings['zero123']['default'] = {
                    'zero123_ws': self.opt.zero123_ws,
                    'c_crossattn': guidance_embeds[0],
                    'c_concat': guidance_embeds[1],
                    'ref_polars': self.opt.ref_polars,
                    'ref_azimuths': self.opt.ref_azimuths,
                    'ref_radii': self.opt.ref_radii,
                }

            if 'clip' in self.guidance:
                self.embeddings['clip']['image'] = self.guidance['clip'].get_img_embeds(self.rgb)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data, save_guidance_path: Path = None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """

        # perform RGBD loss instead of SDS if is image-conditioned
        do_rgbd_loss = self.opt.images is not None and \
                       (self.global_step % self.opt.known_view_interval == 0)

        # override random camera with fixed known camera
        if do_rgbd_loss:
            data = self.default_view_data

        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = (self.global_step - self.opt.exp_start_iter) / (
                self.opt.exp_end_iter - self.opt.exp_start_iter)

        # progressively relaxing view range
        if self.opt.progressive_view:
            r = min(1.0, self.opt.progressive_view_init_ratio + 2.0 * exp_iter_ratio)
            self.opt.phi_range = [self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[0] * r,
                                  self.opt.default_azimuth * (1 - r) + self.opt.full_phi_range[1] * r]
            self.opt.theta_range = [self.opt.default_polar * (1 - r) + self.opt.full_theta_range[0] * r,
                                    self.opt.default_polar * (1 - r) + self.opt.full_theta_range[1] * r]
            self.opt.radius_range = [self.opt.default_radius * (1 - r) + self.opt.full_radius_range[0] * r,
                                     self.opt.default_radius * (1 - r) + self.opt.full_radius_range[1] * r]
            self.opt.fovy_range = [self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[0] * r,
                                   self.opt.default_fovy * (1 - r) + self.opt.full_fovy_range[1] * r]

        # progressively increase max_level
        if self.opt.progressive_level:
            self.model.max_level = min(1.0, 0.25 + 2.0 * exp_iter_ratio)

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']  # [B, 4, 4]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        teacher_rays_o = data['teacher_rays_o']  # [B, N, 3]
        teacher_rays_d = data['teacher_rays_d']  # [B, N, 3]
        teacher_H = data['teacher_H']
        teacher_W = data['teacher_W']

        # When ref_data has B images > opt.batch_size
        if B > self.opt.batch_size:
            # choose batch_size images out of those B images
            choice = torch.randperm(B)[:self.opt.batch_size]
            B = self.opt.batch_size
            rays_o = rays_o[choice]
            rays_d = rays_d[choice]
            mvp = mvp[choice]

        if do_rgbd_loss:
            ambient_ratio = 1.0
            shading = 'lambertian'  # use lambertian instead of albedo to get normal
            binarize = False
            bg_color = torch.rand((B * N, 3), device=rays_o.device)

            # add camera noise to avoid grid-like artifact
            if self.opt.known_view_noise_scale > 0:
                noise_scale = self.opt.known_view_noise_scale  # * (1 - self.global_step / self.opt.iters)
                rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
                rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        elif exp_iter_ratio <= self.opt.latent_iter_ratio:
            ambient_ratio = 1.0
            shading = 'normal'
            binarize = False
            bg_color = None

        else:
            if exp_iter_ratio <= self.opt.albedo_iter_ratio:
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = self.opt.min_ambient_ratio + (1.0 - self.opt.min_ambient_ratio) * random.random()
                rand = random.random()
                if rand >= (1.0 - self.opt.textureless_ratio):
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
            # binarize_thresh = min(0.5, -0.5 + self.global_step / self.opt.iters)
            # binarize = random.random() < binarize_thresh
            binarize = False

            # random background
            rand = random.random()
            # if self.opt.bg_radius > 0 and rand > 0.5:
            if self.opt.learnable_bg:
                bg_color = None  # use bg_net
            elif self.opt.noise_bg:
                # B, 3, H, W
                # bg_color =  torch.randn(B, 3, H, W).to(self.device)
                # bg_color = bg_color *
                # self.guidance['SD'].
                raise NotImplementedError
            else:
                bg_color = torch.rand(3).to(self.device)  # single color random bg

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, bg_color=bg_color,
                                    ambient_ratio=ambient_ratio, shading=shading, binarize=binarize, as_latent=True)

        if self.as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_latent = outputs['image'].reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous()  # [B, 4, H, W]
            pred_rgb = self.guidance['SD'].decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous() # [B, H, W, 3]
        else:
            raise NotImplementedError

        pred_depth = outputs['depth'].squeeze(-1)  # .reshape(B, H, W)

        with torch.no_grad():
            teacher_output = self.teacher_model.render(teacher_rays_o, teacher_rays_d, mvp, teacher_H, teacher_W,
                                                       staged=True, perturb=True, bg_color=bg_color,
                                                       ambient_ratio=ambient_ratio, shading=shading, binarize=binarize,
                                                       as_latent=False)

            teacher_rgb = teacher_output['image']
            teacher_rgb = teacher_rgb  # .reshape(B, H, W, 3)

            teacher_latent = self.guidance['SD'].encode_imgs(
                teacher_rgb.permute(0, 3, 1, 2).contiguous())  # [B, 4, H, W]

            teacher_depth = teacher_output['depth'].squeeze(-1)  # [B, 1, H, W]
            teacher_depth = F.interpolate(teacher_depth.unsqueeze(1), size=pred_depth.shape[1:3],
                                          mode='nearest').squeeze(1)  # [B, H, W]

        assert teacher_latent.shape == pred_latent.shape, f"teacher_latent.shape {teacher_latent.shape} != pred_rgb.shape {pred_latent.shape}"
        assert teacher_depth.shape == pred_depth.shape, f"teacher_depth.shape {teacher_depth.shape} != pred_depth.shape {pred_depth.shape}"

        loss = 0
        losses = {}
        #print(pred_latent.min(), pred_latent.max(), teacher_latent.min(), teacher_latent.max())
        latent_mse_loss = F.mse_loss(pred_latent, teacher_latent)
        loss = loss + latent_mse_loss
        losses['latent_mse_loss'] = latent_mse_loss

        rgb_mse_loss = F.mse_loss(pred_rgb, teacher_rgb)*20
        loss = loss + rgb_mse_loss
        losses['rgb_mse_loss'] = rgb_mse_loss

        # rgb_perceptual_loss = self.perceptual_loss(pred_rgb, teacher_rgb)
        # loss = loss + rgb_perceptual_loss
        # losses['rgb_perceptual_loss'] = rgb_perceptual_loss

        # depth_mse_loss = F.mse_loss(pred_depth, teacher_depth)
        # loss = loss + depth_mse_loss
        # losses['depth_mse_loss'] = depth_mse_loss

        return pred_latent, pred_depth, teacher_rgb, teacher_depth, loss, losses

    def perceptual_loss(self, synth_images, target):
        '''

        :param synth_images: [0, 1] , [B, 3, H, W]
        :param target: [0, 1] , [B, 3, H, W]
        :return:
        '''
        synth_images = synth_images.permute(0, 3, 1, 2).contiguous()
        target = target.permute(0, 3, 1, 2).contiguous()

        if self.vgg16 is None:
            url = './pretrained/vgg16.pt'
            with open(url, 'rb') as f:
                self.vgg16 = torch.jit.load(f).eval().to(self.device)

        target_images = target * 255  # [-1, 1] -> [0, 255]
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = self.vgg16(target_images, resize_images=False, return_lpips=True)

        synth_images = synth_images * 255  # [-1, 1] -> [0, 255]
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum() * 0.1

        return dist

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

        if not self.opt.dmtet and self.opt.backbone == 'grid':

            if self.opt.lambda_tv > 0:
                lambda_tv = min(1.0, self.global_step / (0.5 * self.opt.iters)) * self.opt.lambda_tv
                self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
            if self.opt.lambda_wd > 0:
                self.model.encoder.grad_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        teacher_rays_o = data['teacher_rays_o']  # [B, N, 3]
        teacher_rays_d = data['teacher_rays_d']  # [B, N, 3]
        teacher_H = data['teacher_H']
        teacher_W = data['teacher_W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=False, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, bg_color=None, as_latent=True)

        if self.as_latent:  # always True
            # from B, H, W, C to B, C, H, W
            pred_rgb = self.guidance['SD'].decode_latents(outputs['image'].permute(0, 3, 1, 2).contiguous()).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     1).contiguous()
        else:
            pred_rgb = outputs['image']

        pred_rgb = pred_rgb  # .reshape(B, H, W, 3)
        pred_depth = outputs['depth'].squeeze(-1)  # .reshape(B, H, W)

        with torch.no_grad():
            teacher_output = self.teacher_model.render(teacher_rays_o, teacher_rays_d, mvp, teacher_H, teacher_W,
                                                       staged=True, perturb=False, light_d=light_d,
                                                       ambient_ratio=ambient_ratio, shading=shading, bg_color=None,
                                                       as_latent=False)

            teacher_rgb = teacher_output['image']
            teacher_rgb = teacher_rgb  # .reshape(B, H, W, 3)

            teacher_depth = teacher_output['depth'].squeeze(-1)  # [B, 1, H, W]
            teacher_depth = F.interpolate(teacher_depth.unsqueeze(1), size=pred_depth.shape[1:3], mode='nearest').squeeze(
                1)  # [B, H, W]

        assert teacher_rgb.shape == pred_rgb.shape, f"teacher_rgb.shape {teacher_rgb.shape} != pred_rgb.shape {pred_rgb.shape}"
        assert teacher_depth.shape == pred_depth.shape, f"teacher_depth.shape {teacher_depth.shape} != pred_depth.shape {pred_depth.shape}"

        # dummy
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, teacher_rgb, teacher_depth, loss

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        mvp = data['mvp']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        teacher_rays_o = data['teacher_rays_o']  # [B, N, 3]
        teacher_rays_d = data['teacher_rays_d']  # [B, N, 3]
        teacher_H = data['teacher_H']
        teacher_W = data['teacher_W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=True, perturb=perturb, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color, as_latent=True)

        if self.as_latent:  # always True
            # from B, H, W, C to B, C, H, W
            pred_rgb = self.guidance['SD'].decode_latents(outputs['image'].permute(0, 3, 1, 2).contiguous()).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     1).contiguous()
        else:
            pred_rgb = outputs['image']

        pred_rgb = pred_rgb  # .reshape(B, H, W, 3)
        pred_depth = outputs['depth'].squeeze(-1)  # .reshape(B, H, W)

        with torch.no_grad():
            teacher_output = self.teacher_model.render(teacher_rays_o, teacher_rays_d, mvp, teacher_H, teacher_W,
                                                       staged=True, perturb=perturb, light_d=light_d,
                                                       ambient_ratio=ambient_ratio, shading=shading, bg_color=bg_color,
                                                       as_latent=False)

            teacher_rgb = teacher_output['image']
            teacher_rgb = teacher_rgb  # .reshape(B, H, W, 3)

            teacher_depth = teacher_output['depth'].squeeze(-1)
            teacher_depth = F.interpolate(teacher_depth.unsqueeze(1), size=pred_depth.shape[1:3], mode='nearest').squeeze(
                1)  # [B, H, W]

        assert teacher_rgb.shape == pred_rgb.shape, f"teacher_rgb.shape {teacher_rgb.shape} != pred_rgb.shape {pred_rgb.shape}"
        assert teacher_depth.shape == pred_depth.shape, f"teacher_depth.shape {teacher_depth.shape} != pred_depth.shape {pred_depth.shape}"

        return pred_rgb, pred_depth, teacher_rgb, teacher_depth

    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution,
                               decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "latent_trigrid_fit_run", self.name))

        start_t = time.time()
        self.evaluate_one_epoch(valid_loader)
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, max_epochs)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.opt.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

            if self.epoch % self.opt.test_interval == 0 or self.epoch == max_epochs:
                self.test(test_loader)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'latent_trigrid_fit_results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        self.teacher_model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, teacher_rgb, teacher_depth = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                teacher_rgb = teacher_rgb[0].detach().cpu().numpy()
                teacher_rgb = (teacher_rgb * 255).astype(np.uint8)

                teacher_depth = teacher_depth[0].detach().cpu().numpy()
                teacher_depth = (teacher_depth - teacher_depth.min()) / (
                            teacher_depth.max() - teacher_depth.min() + 1e-6)
                teacher_depth = (teacher_depth * 255).astype(np.uint8)

                pred = np.concatenate([pred, teacher_rgb], axis=1)
                pred_depth = np.concatenate([pred_depth, teacher_depth], axis=1)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            print('save video...', os.path.join(save_path, f'{name}_rgb.mp4'),
                  os.path.join(save_path, f'{name}_depth.mp4'))
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader, max_epochs):
        self.log(
            f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        total_latent_mse_loss = 0
        total_rgb_mse_loss = 0
        # total_rgb_perceptual_loss = 0
        # total_depth_mse_loss = 0

        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:

            # update grid every 16 steps
            if (
                    self.model.cuda_ray or self.model.taichi_ray) and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.save_guidance and (self.global_step % self.opt.save_guidance_interval == 0):
                    save_guidance_path = save_guidance_folder / f'step_{self.global_step:07d}.png'
                else:
                    save_guidance_path = None
                pred_rgbs, pred_depths, teacher_rgbs, teacher_depths, loss, losses = self.train_step(data,
                                                                                                     save_guidance_path=save_guidance_path)

            # hooked grad clipping for RGB space
            if self.opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        return grad.clamp(grad_scale * -self.opt.grad_clip_rgb, grad_scale * self.opt.grad_clip_rgb)
                    else:
                        return grad.clamp(-self.opt.grad_clip_rgb, self.opt.grad_clip_rgb)

                pred_rgbs.register_hook(_hook)
                # pred_rgbs.retain_grad()

            self.scaler.scale(loss).backward()

            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            total_latent_mse_loss += losses['latent_mse_loss'].item()
            total_rgb_mse_loss += losses['rgb_mse_loss'].item()
            # total_rgb_perceptual_loss += losses['rgb_perceptual_loss'].item()
            # total_depth_mse_loss += losses['depth_mse_loss'].item()

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                        f"latent_mse_loss={losses['latent_mse_loss'].item():.4f} ({total_latent_mse_loss / self.local_step:.4f}), "
                        f"rgb_mse_loss={losses['rgb_mse_loss'].item():.4f} ({total_rgb_mse_loss / self.local_step:.4f}), "
                        # f"rgb_perceptual_loss={losses['rgb_perceptual_loss'].item():.4f} ({total_rgb_perceptual_loss / self.local_step:.4f}), "
                        # f"depth_mse_loss={losses['depth_mse_loss'].item():.4f} ({total_depth_mse_loss / self.local_step:.4f}), "
                        f"lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(
            f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, teacher_rgb, teacher_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save image
                    save_path = os.path.join(self.workspace, 'latent_trigrid_fit_validation',
                                             f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'latent_trigrid_fit_validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    teacher_rgb = teacher_rgb[0].detach().cpu().numpy()
                    teacher_rgb = (teacher_rgb * 255).astype(np.uint8)

                    teacher_depth = teacher_depth[0].detach().cpu().numpy()
                    teacher_depth = (teacher_depth - teacher_depth.min()) / (
                                teacher_depth.max() - teacher_depth.min() + 1e-6)
                    teacher_depth = (teacher_depth * 255).astype(np.uint8)

                    pred = np.concatenate((pred, teacher_rgb), axis=1)
                    pred_depth = np.concatenate((pred_depth, teacher_depth), axis=1)

                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density

        if self.opt.dmtet:
            state['tet_scale'] = self.model.tet_scale.cpu().numpy()

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if self.opt.dmtet:
            if 'tet_scale' in checkpoint_dict:
                new_scale = torch.from_numpy(checkpoint_dict['tet_scale']).to(self.device)
                self.model.verts *= new_scale / self.model.tet_scale
                self.model.tet_scale = new_scale

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")


def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free) / 1024 ** 3) * 1000) / 1000)
        mem += mems[-1]
    return mem, mems