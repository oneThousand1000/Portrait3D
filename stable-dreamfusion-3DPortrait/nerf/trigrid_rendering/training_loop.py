# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import random
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main,metric_utils
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    h = int(7680 * (training_set.image_shape[2]/512))
    w = int(4320 * (training_set.image_shape[2] / 512))
    gh = np.clip(h // training_set.image_shape[2], 7, 8)
    gw = np.clip(w // training_set.image_shape[1], 4, 4)

    # No labels => show random subset of training samples.
    # if not training_set.has_labels:
    #     all_indices = list(range(len(training_set)))
    #     rnd.shuffle(all_indices)
    #     grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # else:
    #     # Group training samples by label.
    #     label_groups = dict() # label => [idx, ...]
    #     for idx in range(len(training_set)):
    #         label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
    #         if label not in label_groups:
    #             label_groups[label] = []
    #         label_groups[label].append(idx)

    #     # Reorder.
    #     label_order = list(label_groups.keys())
    #     rnd.shuffle(label_order)
    #     for label in label_order:
    #         rnd.shuffle(label_groups[label])

    #     # Organize into grid.
    #     grid_indices = []
    #     for y in range(gh):
    #         label = label_order[y % len(label_order)]
    #         indices = label_groups[label]
    #         grid_indices += [indices[x % len(indices)] for x in range(gw)]
    #         label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]
    label_groups = dict() # label => [idx, ...]
    for idx in range(len(training_set)):
        label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(idx)

        # Reorder.
    label_order = list(label_groups.keys())
    rnd.shuffle(label_order)
    for label in label_order:
        rnd.shuffle(label_groups[label])

        # Organize into grid.
    grid_indices = []
    for y in range(gh):
        for x in range(gw//2):
            label = label_order[(y + x*gh) % len(label_order)]
            indices = list(set(label_groups[label]))
            #grid_indices += [indices[x % len(indices)] for x in range(2)]
            grid_indices += [indices[0],  (indices[0]+ len(training_set)//2)%len(training_set) ]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]


    # Load data.
    images, segs, labels, poses = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images),np.stack(segs), np.stack(labels), np.stack(poses)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    train_g_pose_branch     = None,
    metric_pose_sample_mode = None,
):
    print('Random seed: %d' % random_seed)
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.cuda.set_device(device)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print('Pose shape:', training_set.pose_shape)
        print()
        print('>>>>>>>>>>>>>>> image_snapshot_ticks:', image_snapshot_ticks)
        print('>>>>>>>>>>>>>>> network_snapshot_ticks:', network_snapshot_ticks)

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    D_ema = copy.deepcopy(D).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        if 'D_ema' in resume_data:
            print(f'copy params of D_ema of "{resume_pkl} to D_ema')
            misc.copy_params_and_buffers(resume_data['D_ema'], D_ema, require_all=False)
        else:
            print(f'copy params of D of "{resume_pkl} to D_ema')
            misc.copy_params_and_buffers(resume_data['D'], D_ema, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        p = torch.empty([batch_gpu, 6], device=device)
        img = misc.print_module_summary(G, [z, c, ])
        misc.print_module_summary(D, [img, c ])

        print('plane_shapes:')
        for res_k in G.plane_shapes:
            print(res_k, G.plane_shapes[res_k])
    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema,D_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe,rank = rank,**loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        params_list = []
        params_name_list = []
        for p_name, p in module.named_parameters():
            if name == 'G':
                if 'aligned_SMPL' not in p_name:
                    if not train_g_pose_branch:
                        if 'pose_branch' not in p_name:
                            params_list.append(p)
                            params_name_list.append(p_name)
                    else:
                        params_list.append(p)
                        params_name_list.append(p_name)
            else:
                params_list.append(p)
                params_name_list.append(p_name)



        if rank ==0:
            print(f'params_name_list of {name}:',params_name_list)

        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=params_list, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]


        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(params=params_list, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]



    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
            print('phase: ',phase.name)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images,segs, labels,poses = setup_snapshot_image_grid(training_set=training_set,random_seed=random.randint(0, 1000000))
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        save_image_grid(segs, os.path.join(run_dir, 'segs.jpg'), drange=[0, 255], grid_size=grid_size)
        grid_images = (torch.from_numpy(images).to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
        grid_segs = (torch.from_numpy(segs).to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

        #grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)

        if G.rendering_kwargs['c_gen_conditioning_zero']:
            raise NotImplementedError
            grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        else:
            #raise NotImplementedError
            grid_z = []
            for i in range(labels.shape[0]//2):
                sample_z = torch.randn([1, G.z_dim], device=device)
                grid_z.append(sample_z)
                grid_z.append(sample_z)
            grid_z = torch.cat(grid_z,dim=0).split(batch_gpu)


        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_poses = torch.from_numpy(poses).to(device).split(batch_gpu)

        real_shape_real_pose = []
        for real_pose, c in zip(grid_poses, grid_c):
            real_shape_pose_param = {'pose': real_pose}
            real_shape_real_pose.append(
                G_ema.render_meshes(real_shape_pose_param, resolution=training_set.image_shape[2], cameras=c)
            )
        real_shape_real_pose = np.concatenate(real_shape_real_pose, axis=0)
        save_image_grid(real_shape_real_pose,
                        os.path.join(run_dir, f'mesh_coarse_real_pose.png'),
                        drange=[0, 255], grid_size=grid_size)
    #exit()

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)



    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):

            phase_real_img, phase_real_seg, phase_real_c, phase_real_pose = next(training_set_iterator)


            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_seg = (phase_real_seg.to(device).to(torch.float32) / 255.0).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            phase_real_pose = phase_real_pose.to(device).split(batch_gpu)

            all_gen_z = torch.randn([len(phases) * (batch_size // num_gpus), G.z_dim], device=device) # 4 * 8
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split((batch_size // num_gpus))]

            random_idx = [np.random.randint(len(training_set))  for _ in range(len(phases) * (batch_size // num_gpus))]


            all_gen_c = [training_set.get_label(gen_idx) for gen_idx in random_idx]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split((batch_size // num_gpus))]


            all_gen_pose = [training_set.get_coarse_pose(gen_idx) for gen_idx in random_idx]
            all_gen_pose = torch.from_numpy(np.stack(all_gen_pose)).pin_memory().to(device)
            all_gen_pose = [phase_gen_pose.split(batch_gpu) for phase_gen_pose in all_gen_pose.split((batch_size // num_gpus))]

        assert len(phases) == len(all_gen_z) == len(all_gen_c) ==len(all_gen_pose)
        # Execute training phases.
        for phase, phase_gen_z,phase_gen_c,phase_gen_pose in zip(phases, all_gen_z,all_gen_c,all_gen_pose): # 4
            if batch_idx % phase.interval != 0:
                continue


            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_seg, real_c,real_pose, gen_z,gen_c,gen_pose in \
                    zip(phase_real_img, phase_real_seg, phase_real_c, phase_real_pose, phase_gen_z,phase_gen_c,phase_gen_pose):

                loss.accumulate_gradients(phase=phase.name, real_img=real_img,real_seg = real_seg, real_c=real_c,real_pose = real_pose,
                                          gen_z=gen_z,gen_c = gen_c, gen_pose = gen_pose,

                                          gain=phase.interval, cur_nimg=cur_nimg,cur_nimg_start = resume_kimg * 1000)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):

                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()



            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        with torch.autograd.profiler.record_function('Dema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(D_ema.parameters(), D.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(D_ema.buffers(), D.buffers()):
                b_ema.copy_(b)


        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]

        if loss.swapping_prob is not None:
            fields += [f"swap prob {training_stats.report0('Progress/swap_prob', float(loss.swapping_prob)):.3f}"]
        if loss.neural_rendering_resolution is not None:
            fields += [f"render_res {training_stats.report0('Progress/rendering_res', float(loss.neural_rendering_resolution)):.3f}"]
        # if loss.noise_alpha is not None:
        #     fields += [f"noise_alpha {training_stats.report0('Progress/noise_alpha', float(loss.noise_alpha)):.3f}"]
        # if loss.noise_scale is not None:
        #     fields += [f"noise_scale {training_stats.report0('Progress/noise_scale', float(loss.noise_scale)):.3f}"]

        # if loss.predict_label_alpha is not None:
        #     fields += [f"predict_label_alpha {training_stats.report0('Progress/predict_label_alpha', float(loss.predict_label_alpha)):.3f}"]

        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')




        if  (rank == 0) and ((image_snapshot_ticks is not None) and (done or (cur_tick % image_snapshot_ticks == 0) ) ): # or (cur_tick<50  and cur_tick % 5 == 0 ) )  # (cur_tick!=0) and
            print('gen images...')
            with torch.no_grad():
                predicted_real_pose_params_D = []
                for vis_real_img,vis_real_seg, vis_c in zip(grid_images,grid_segs, grid_c):
                    pose_param = loss.get_pose_params_D(vis_real_img,vis_real_seg, vis_c, cur_nimg)
                    predicted_real_pose_params_D.append(pose_param)

                predicted_fake_pose_params_G = []
                for vis_z, vis_c in zip(grid_z, grid_c):
                    pose_param = loss.get_pose_params_G(vis_z, vis_c)
                    predicted_fake_pose_params_G.append(pose_param)


            real_pose_mesh = []
            for predicted_real_pose, c in zip(predicted_real_pose_params_D, grid_c):
                real_pose_param = {'pose': predicted_real_pose}
                real_pose_mesh.append(
                    G_ema.render_meshes(real_pose_param, resolution=training_set.image_shape[2], cameras=c)
                )
            real_pose_mesh = np.concatenate(real_pose_mesh, axis=0)
            save_image_grid(real_pose_mesh,
                            os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_mesh_real_pose_D.png'),
                            drange=[0, 255], grid_size=grid_size)


            snap_pose = predicted_fake_pose_params_G
            cond_c = torch.tensor([[ 1.0000e+00,  1.0505e-09,  4.3685e-08, -1.1805e-07,  0.0000e+00,
                                    -9.9951e-01,  2.4033e-02, -1.1805e-07,  4.3714e-08, -2.4033e-02,
                                    -9.9951e-01,  2.6992e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                                    1.0000e+00,  6.5104e+00,  0.0000e+00,  5.0000e-01,  0.0000e+00,
                                    6.5104e+00,  5.0000e-01,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).float().to(device)


            #out = [G_ema(z=z, c=c, noise_mode='const',apply_def = True, pose_params = pose) for z, c, pose in zip(grid_z, grid_c, snap_pose)]
            grid_ws = [G_ema.mapping(z, cond_c.expand(z.shape[0], -1),None) for z in grid_z]
            out =[G_ema.synthesis(ws, c=c, noise_mode='const',apply_def = True, pose_params = pose) for ws, c,pose in zip(grid_ws, grid_c,snap_pose)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            #print('images range: ',np.max(images),np.min(images))
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            images_alpha = torch.cat([o['image_mask'].cpu() for o in out]).numpy()
            #background_raw = torch.cat([o['image_background'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_0.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_2_raw.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_4_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            save_image_grid(images_alpha, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_4_alpha.jpg'), drange=[0, 1], grid_size=grid_size)
            #save_image_grid(background_raw, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_4_background.jpg'), drange=[-1, 1], grid_size=grid_size)
            with torch.no_grad():
                predicted_fake_pose_params_D = []
                for o,vis_c,vis_pose in zip(out,grid_c,snap_pose):
                    pose_param = loss.get_pose_params_D(o['image'],o['image_mask'],vis_c, cur_nimg)
                    predicted_fake_pose_params_D.append(pose_param)

            fake_pose_mesh = []
            for predicted_fake_pose, c in zip(predicted_fake_pose_params_D, grid_c):
                fake_pose_param = {'pose': predicted_fake_pose}
                fake_pose_mesh.append(
                    G_ema.render_meshes(fake_pose_param, resolution=training_set.image_shape[2], cameras=c)
                )
            fake_pose_mesh = np.concatenate(fake_pose_mesh, axis=0)
            save_image_grid(fake_pose_mesh,
                            os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_mesh_fake_pose_D.png'),
                            drange=[0, 255], grid_size=grid_size)

            input_pose_mesh = []
            for input_pose, c in zip(predicted_fake_pose_params_G, grid_c):
                input_pose_param = {'pose': input_pose}
                input_pose_mesh.append(
                    G_ema.render_meshes(input_pose_param, resolution=training_set.image_shape[2], cameras=c)
                )
            input_pose_mesh = np.concatenate(input_pose_mesh, axis=0)
            save_image_grid(input_pose_mesh,
                            os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_mesh_input_pose_G.png'),
                            drange=[0, 255], grid_size=grid_size)




            # no_pose_out = [G_ema(z=z, c=c, noise_mode='const', apply_def=False, pose_params=None) for z, c in zip(grid_z, grid_c)]
            no_pose_out =[G_ema.synthesis(ws, c=c, noise_mode='const',apply_def = False, pose_params = None) for ws, c in zip(grid_ws, grid_c)]
            images = torch.cat([o['image'].cpu() for o in no_pose_out]).numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in no_pose_out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in no_pose_out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_1_no_pose.png'), drange=[-1, 1],
                            grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_3_no_pose_raw.png'), drange=[-1, 1],
                            grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_5_no_pose_depth.png'),
                            drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)



        # if (loss.fronzen_D is not None) and ((network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0)):
        #     if rank ==0 :
        #         print('update loss.fronzen_D...')
        #     misc.copy_params_and_buffers(D, loss.fronzen_D, require_all=True)
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('D_ema', D_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

                pose_predict_kwargs = {
                                                          'blur_sigma' : loss.blur_sigma,
                                                         'neural_rendering_resolution': loss.neural_rendering_resolution,
                                                         'resample_filter': loss.resample_filter.cpu().numpy().tolist(),
                                                         'filter_mode': loss.filter_mode
                                                      }
                with open(os.path.join(run_dir, f'pose_predict_kwargs-{cur_nimg//1000:06d}.json'), 'wt') as f:
                    json.dump(pose_predict_kwargs, f, indent=2)


        # Evaluate metrics.
        if  (cur_tick!=0) and (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
            for metric in metrics:
                progress = metric_utils.ProgressMonitor(verbose=True)
                # result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                #                                       dataset_kwargs=training_set_kwargs, num_gpus=num_gpus,
                #                                       rank=rank, device=device, progress=progress
                #                                       )
                result_dict = metric_main.calc_metric(metric=metric,
                                                      G=snapshot_data['G_ema'],
                                                      dataset_kwargs=training_set_kwargs,
                                                      num_gpus=num_gpus,
                                                      rank=rank,
                                                      device=device,
                                                      metric_pose_sample_mode = metric_pose_sample_mode,
                                                      progress=progress,
                                                      D = snapshot_data['D'] if metric_pose_sample_mode == 'D_predict' else None,
                                                      pose_predict_kwargs = {
                                                          'blur_sigma' : loss.blur_sigma,
                                                         'neural_rendering_resolution': loss.neural_rendering_resolution,
                                                         'resample_filter': loss.resample_filter,
                                                         'filter_mode': loss.filter_mode
                                                      }  if metric_pose_sample_mode == 'D_predict' else None
                                                      )

                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
