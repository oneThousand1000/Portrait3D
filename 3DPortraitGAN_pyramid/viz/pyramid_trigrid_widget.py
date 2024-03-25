# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import os
import re

import dnnlib
import imgui
import numpy as np
from gui_utils import imgui_utils

from . import renderer

#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

class PyramidTrigridWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.search_dirs    = []
        self.cur_pyramid_trigrid        = None
        self.cur_ws = None
        self.user_pth       = ''
        self.recent_pths    = []
        self.browse_cache   = dict() # {tuple(path, ...): [dnnlib.EasyDict(), ...], ...}
        self.browse_refocus = False
        self.load('', ignore_errors=True)

    def add_recent(self, pth, ignore_errors=False):
        try:
            resolved = self.resolve_pth(pth)
            if resolved not in self.recent_pths:
                self.recent_pths.append(resolved)
        except:
            if not ignore_errors:
                raise

    def load(self, pth, ignore_errors=False):
        viz = self.viz
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        try:
            resolved = pth
            name = resolved.replace('\\', '/').split('/')[-1]
            self.cur_pth = resolved
            self.user_pth = resolved
            viz.result.message = f'Loading {name}...'
            viz.defer_rendering()
            if resolved in self.recent_pths:
                self.recent_pths.remove(resolved)
            self.recent_pths.insert(0, resolved)
        except:
            self.cur_pth = None
            self.user_pth = pth
            if pth == '':
                viz.result = dnnlib.EasyDict(message='No pyramid tri-grid ckpt loaded')
            else:
                viz.result = dnnlib.EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        recent_pths = [pth for pth in self.recent_pths if pth != self.user_pth]
        if show:
            imgui.text('Pyramid Tri-Grid Ckpt:')
            imgui.same_line(round(viz.font_size * 8.5))
            changed, self.user_pth = imgui_utils.input_text('##pth', self.user_pth, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text='<PATH>.pth')
            if changed:
                self.load(self.user_pth, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_pth != '':
                imgui.set_tooltip(self.user_pth)

            imgui.same_line()
            if imgui_utils.button('Browse...', enabled=len(self.search_dirs) > 0, width=-1):
                imgui.open_popup('browse_pths_popup')
                self.browse_cache.clear()
                self.browse_refocus = True

        if imgui.begin_popup('recent_pths_popup'):
            for pth in recent_pths:
                clicked, _state = imgui.menu_item(pth)
                if clicked:
                    self.load(pth, ignore_errors=True)
            imgui.end_popup()

        if imgui.begin_popup('browse_pths_popup'):
            def recurse(parents):
                key = tuple(parents)
                items = self.browse_cache.get(key, None)
                if items is None:
                    items = self.list_runs_and_pths(parents)
                    self.browse_cache[key] = items
                for item in items:
                    if item.type == 'run' and imgui.begin_menu(item.name):
                        recurse([item.path])
                        imgui.end_menu()
                    if item.type == 'pth':
                        clicked, _state = imgui.menu_item(item.name)
                        if clicked:
                            self.load(item.path, ignore_errors=True)
                if len(items) == 0:
                    with imgui_utils.grayed_out():
                        imgui.menu_item('No results found')
            recurse(self.search_dirs)
            if self.browse_refocus:
                imgui.set_scroll_here()
                viz.skip_frame() # Focus will change on next frame.
                self.browse_refocus = False
            imgui.end_popup()

        paths = viz.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.load(paths[0], ignore_errors=True)

        viz.args.pyramid_tri_grid_ckpt = self.cur_pth

    def list_runs_and_pths(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        pth_regex = re.compile(r'network-snapshot-\d+\.pth')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))
                    if entry.is_file() and pth_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(type='pth', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items


#----------------------------------------------------------------------------
