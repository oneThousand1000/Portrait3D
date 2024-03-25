# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import copy
import traceback
import numpy as np
import torch
import torch.fft
import torch.nn
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error

from camera_utils import LookAtPoseSampler
import os

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

#----------------------------------------------------------------------------

def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

#----------------------------------------------------------------------------

def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self):
        self._device        = torch.device('cuda')
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        self._start_event   = torch.cuda.Event(enable_timing=True)
        self._end_event     = torch.cuda.Event(enable_timing=True)
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

        self.input_data = dict()

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = dnnlib.EasyDict()
        try:
            self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).numpy()
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).numpy()
        if 'error' in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_pyramid_tri_grid_ws(self, pyramid_tri_grid_ckpt, device):

        data = self.input_data.get(pyramid_tri_grid_ckpt, None)
        if data is None:

            print(f'Loading "{pyramid_tri_grid_ckpt}"... ', end='', flush=True)
            ckpt = torch.load(pyramid_tri_grid_ckpt, map_location=lambda storage, loc: storage)['model']
            trigrid = {
                8: ckpt['trigrids_8'].to(device).detach(),
                16: ckpt['trigrids_16'].to(device).detach(),
                32: ckpt['trigrids_32'].to(device).detach(),
                64: ckpt['trigrids_64'].to(device).detach(),
                128: ckpt['trigrids_128'].to(device).detach(),
                256: ckpt['trigrids_256'].to(device).detach(),
                512: ckpt['trigrids_512'].to(device).detach(),
            }
            ws = ckpt['ws'].to(device)
            print('Done.')
            self.input_data[pyramid_tri_grid_ckpt] = {'trigrid': trigrid, 'ws': ws}

        else:
            trigrid = data['trigrid']
            ws = data['ws']

        return trigrid, ws


    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                net = copy.deepcopy(orig_net)
                net = self._tweak_network(net, **tweak_kwargs)
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net


        return net

    def _tweak_network(self, net):
        # Print diagnostics.

        # RELOAD_MODULES = False
        # if RELOAD_MODULES:
        #     from training.triplane import TriPlaneGenerator
        #     from torch_utils import misc
        #     print("Reloading Modules!")
        #     net_new = TriPlaneGenerator(*net.init_args, **net.init_kwargs).eval().requires_grad_(False).to(self._device)
        #     misc.copy_params_and_buffers(net, net_new, require_all=True)
        #     net_new.neural_rendering_resolution = net.neural_rendering_resolution
        #     net_new.rendering_kwargs = net.rendering_kwargs
        #     net = net_new
        #     # net.rendering_kwargs['ray_start'] = 'auto'
        #     # net.rendering_kwargs['ray_end'] = 'auto'
        #     # net.rendering_kwargs['avg_camera_pivot'] = [0, 0, 0]

        if True:
            print("Reloading Modules!")
            from training.smpl_triplane import TriPlaneGenerator
            from torch_utils import misc
            print("Reloading Modules!")
            init_kwargs = net.init_kwargs
            print('G.init_args: ', net.init_args)
            print('G.init_kwargs: ', init_kwargs)
            G_new = TriPlaneGenerator(*net.init_args, **init_kwargs).eval().requires_grad_(False).to(self._device)
            misc.copy_params_and_buffers(net, G_new, require_all=False)
            G_new.neural_rendering_resolution = net.neural_rendering_resolution
            G_new.rendering_kwargs = net.rendering_kwargs
            G_new.batch_size = 1
            G_new.set_batch_size(1)
            net = G_new
        print('>>>> G batch: ', net.batch_size)


        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x


    def _render_impl(self, res,
        pkl             = None,
        pyramid_tri_grid_ckpt = None,
        w0_seeds        = [[0, 1]],
        stylemix_idx    = [],
        stylemix_seed   = 0,
        trunc_psi       = 1,
        trunc_cutoff    = 0,
        random_seed     = 0,
        noise_mode      = 'const',
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        fft_show        = False,
        fft_all         = True,
        fft_range_db    = 50,
        fft_beta        = 8,
        input_transform = None,
        untransform     = False,

        yaw             = 0,
        pitch           = 0,
        lookat_point    = (0, 0.0649, 0),
        conditioning_yaw    = 0,
        conditioning_pitch  = 0,
        conditioning_body_pose = None,
        body_pose            = None,
        focal_length    = 4.2647,
        render_type     = 'image',

        do_backbone_caching = False,

        depth_mult            = 1,
        depth_importance_mult = 1,
    ):
        if not os.path.exists(pyramid_tri_grid_ckpt) or not os.path.exists(pkl):
            res.error = 'Pyramid Tri-Grid or pkl file does not exist'
            return
        if body_pose is None:
            body_pose = np.zeros((1, 6), dtype=np.float32)
        else:
            body_pose = np.array(body_pose, dtype=np.float32)
            body_pose = np.reshape(body_pose, (1, -1))




        # Dig up network details.
        G = self.get_network(pkl, 'G_ema').eval().requires_grad_(False).to('cuda')
        res.img_resolution = G.img_resolution
        res.num_ws = G.backbone.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.backbone.named_buffers())
        res.has_input_transform = (hasattr(G.backbone, 'input') and hasattr(G.backbone.input, 'transform'))

        # set G rendering kwargs
        if 'depth_resolution_default' not in G.rendering_kwargs:
            G.rendering_kwargs['depth_resolution_default'] = G.rendering_kwargs['depth_resolution']
            G.rendering_kwargs['depth_resolution_importance_default'] = G.rendering_kwargs['depth_resolution_importance']

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution_default'] * depth_mult)
        G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance_default'] * depth_importance_mult)

        # G.init_kwargs.batch_size = 1

        pyramid_tri_grid,ws = self.get_pyramid_tri_grid_ws(pyramid_tri_grid_ckpt,self._device)


        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.

        if lookat_point is None:
            #camera_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', (0, 0, 0)))
            camera_pivot = torch.tensor([0, 0.0649, 0])
        else:
            # override lookat point provided
            camera_pivot = torch.tensor(lookat_point)
        camera_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)


        # Run mapping network.
        # w_avg = G.mapping.w_avg
        # Run synthesis network.
        synthesis_kwargs = dnnlib.EasyDict(noise_mode=noise_mode, force_fp32=force_fp32, cache_backbone=do_backbone_caching)
        torch.manual_seed(random_seed)

        # Set camera params
        pose = LookAtPoseSampler.sample(3.14/2 + yaw, 3.14/2 + pitch, camera_pivot, radius=camera_radius)
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
        c = torch.cat([pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(ws.device)



        # if body pose is not 0:
        if not (body_pose == 0).all():
            apply_def = True
            body_pose = torch.tensor(body_pose).to(ws.device)
        else:
            apply_def = False
            body_pose = None


        out = self.run_tri_grid_render(G, ws,pyramid_tri_grid, c)


        # Untransform.
        if untransform and res.has_input_transform:
            out, _mask = _apply_affine_transformation(out.to(torch.float32), G.synthesis.input.transform, amax=6) # Override amax to hit the fast path in upfirdn2d.

        # Select channels and compute statistics.
        if type(out) == dict:
            # is model output. query render type
            out = out[render_type][0].to(torch.float32)
        else:
            out = out[0].to(torch.float32)

        if sel_channels > out.shape[0]:
            sel_channels = 1
        base_channel = max(min(base_channel, out.shape[0] - sel_channels), 0)
        sel = out[base_channel : base_channel + sel_channels]
        res.stats = torch.stack([
            out.mean(), sel.mean(),
            out.std(), sel.std(),
            out.norm(float('inf')), sel.norm(float('inf')),
        ])

        # normalize if type is 'image_depth'
        if render_type == 'image_depth':
            out -= out.min()
            out /= out.max()

            out -= .5
            out *= -2

        # Scale and convert to uint8.
        img = sel
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img

        # FFT.
        if fft_show:
            sig = out if fft_all else sel
            sig = sig.to(torch.float32)
            sig = sig - sig.mean(dim=[1,2], keepdim=True)
            sig = sig * torch.kaiser_window(sig.shape[1], periodic=False, beta=fft_beta, device=self._device)[None, :, None]
            sig = sig * torch.kaiser_window(sig.shape[2], periodic=False, beta=fft_beta, device=self._device)[None, None, :]
            fft = torch.fft.fftn(sig, dim=[1,2]).abs().square().sum(dim=0)
            fft = fft.roll(shifts=[fft.shape[0] // 2, fft.shape[1] // 2], dims=[0,1])
            fft = (fft / fft.mean()).log10() * 10 # dB
            fft = self._apply_cmap((fft / fft_range_db + 1) / 2)
            res.image = torch.cat([img.expand_as(fft), fft], dim=1)



    def run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5: # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == '':
                    name = 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split('.')[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            out = net.synthesis(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

    @staticmethod
    def run_tri_grid_render(net, w, trigrid,c): # => out, layers
        out = net.render_planes(ws=w, planes=trigrid, c=c[0:1], noise_mode='const',
                              neural_rendering_resolution=256, chunk=4096)
        return out

#----------------------------------------------------------------------------
