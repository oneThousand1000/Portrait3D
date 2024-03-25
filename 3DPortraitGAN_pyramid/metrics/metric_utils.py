# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, identical_c_p = True,
                 cache=True, metric_pose_sample_mode = None,D = None,pose_predict_kwargs = None):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

        self.metric_pose_sample_mode = metric_pose_sample_mode
        self.D = D
        self.pose_predict_kwargs = pose_predict_kwargs

        self.identical_c_p = identical_c_p

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            random_idx = [np.random.randint(len(dataset)) for _i in range(batch_size) ]


            c = [dataset.get_label(idx) for idx in random_idx]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)

            p = [dataset.get_coarse_pose(idx) for idx in random_idx]
            p = torch.from_numpy(np.stack(p)).pin_memory().to(opts.device)
            yield c,p


from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

def run_D_pose_prediction(img, c, blur_sigma=0,D = None):
    blur_size = np.floor(blur_sigma * 3)
    if blur_size > 0:
        with torch.autograd.profiler.record_function('blur'):
            f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
            img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
    pose,_ = D.predict_pose( img, c)
    return pose

def get_pose_params(real_img,real_seg, real_c,D = None,neural_rendering_resolution = None,blur_sigma = None,resample_filter = None, filter_mode = None):



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
    real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_mask': real_img_tmp_image_mask}

    predicted_real_pose = run_D_pose_prediction(real_img_tmp, real_c, blur_sigma=blur_sigma, D = D)
    return predicted_real_pose

def iterate_random_labels_predicted_pose(opts, batch_size,G):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            random_idx = [np.random.randint(len(dataset)) for _i in range(batch_size) ]


            c = [dataset.get_label(idx) for idx in random_idx]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)

            z = torch.randn([batch_size, opts.G.z_dim], device=opts.device)

            p = G.get_pose_params(z,c)


            yield c,p

def iterate_random_labels_predicted_pose_D(opts, batch_size,D):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            random_idx = [np.random.randint(len(dataset)) for _i in range(batch_size) ]


            c = [dataset.get_label(idx) for idx in random_idx]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)

            # p = [dataset.get_coarse_pose(idx) for idx in random_idx]
            # p = torch.from_numpy(np.stack(p)).pin_memory().to(opts.device)

            image = [dataset.get_image(idx) for idx in random_idx]
            image = torch.from_numpy(np.stack(image)).pin_memory().to(opts.device)
            image = image.to(torch.float32) / 127.5 - 1


            mask = [dataset._seg_dataset.get_image(idx) for idx in random_idx]
            mask = torch.from_numpy(np.stack(mask)).pin_memory().to(opts.device)
            mask = mask.to(torch.float32) / 255.0



            p = get_pose_params(
                real_img = image,
                real_seg = mask,
                real_c = c,
                D = D,
                blur_sigma = opts.pose_predict_kwargs['blur_sigma'],
                neural_rendering_resolution= opts.pose_predict_kwargs['neural_rendering_resolution'],
                resample_filter= opts.pose_predict_kwargs['resample_filter'],
                filter_mode= opts.pose_predict_kwargs['filter_mode'],
            )
            yield c,p

# def iterate_random_poses(opts, batch_size):
#     if opts.G.c_dim == 0:
#         p = torch.zeros([batch_size, 6], device=opts.device)
#         while True:
#             yield p
#     else:
#         dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
#         while True:
#             p = [dataset.get_coarse_pose(np.random.randint(len(dataset))) for _i in range(batch_size)]
#             p = torch.from_numpy(np.stack(p)).pin_memory().to(opts.device)
#             yield p
#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, masks, _labels,_poses in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 8)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    G.set_batch_size(batch_gen)

    if opts.metric_pose_sample_mode == 'G_predict':
        label_iter = iterate_random_labels_predicted_pose(opts=opts, batch_size=batch_gen, G = G)
    else:
        D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
        label_iter = iterate_random_labels_predicted_pose_D(opts=opts, batch_size=batch_gen,D = D)

    if not opts.identical_c_p:
        if opts.metric_pose_sample_mode == 'G_predict':
            cond_label_iter = iterate_random_labels_predicted_pose(opts=opts, batch_size=batch_gen, G = G)
        else:
            D = copy.deepcopy(opts.D).eval().requires_grad_(False).to(opts.device)
            cond_label_iter = iterate_random_labels_predicted_pose_D(opts=opts, batch_size=batch_gen,D = D)


    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)

            if  opts.identical_c_p:
                c,p = next(label_iter)

                img = G(z=z, c=c, pose_params = p,apply_def = True,**opts.G_kwargs)['image']
            else:
                c,p = next(label_iter)
                cond_c,cond_p =  next(cond_label_iter)
                ws = G.mapping(z, cond_c, cond_p)
                img = G.synthesis(ws, c=c,apply_def = True, pose_params = p,**opts.G_kwargs )['image']

            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------
