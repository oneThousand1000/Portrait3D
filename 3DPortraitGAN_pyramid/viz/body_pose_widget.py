# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils

#----------------------------------------------------------------------------

class BodyPoseWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.neck_pose       = dnnlib.EasyDict(x=0, y=0, z=0)
        self.head_pose = dnnlib.EasyDict(x=0, y=0, z=0)
        # self.pose_def   = dnnlib.EasyDict(self.pose)

        # self.lookat_point_choice = 0
        # self.lookat_point_option = ['auto',         'ffhq',            'shapenet',         'afhq',         'manual']
        # self.lookat_point_labels = ['Auto Detect',  'FFHQ Default',    'Shapenet Default', 'AFHQ Default', 'Manual']
        # self.lookat_point = (0.0, 0.0, 0.2)

    # def drag(self, dx, dy):
    #     viz = self.viz
    #     self.pose.yaw   += -dx / viz.font_size * 3e-2
    #     self.pose.pitch += -dy / viz.font_size * 3e-2

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('NeckPose')
            imgui.same_line(viz.label_w)
            neck_pose_x = self.neck_pose.x
            neck_pose_y = self.neck_pose.y
            neck_pose_z = self.neck_pose.z
            with imgui_utils.item_width(viz.font_size * 10):
                changed, (new_neck_pose_x, new_neck_pose_y,new_neck_pose_z) = \
                    imgui.input_float3('##neck_pose', neck_pose_x, neck_pose_y,neck_pose_z,  format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.neck_pose.x = new_neck_pose_x
                    self.neck_pose.y = new_neck_pose_y
                    self.neck_pose.z = new_neck_pose_z

            imgui.text('HeadPose')
            imgui.same_line(viz.label_w)
            head_pose_x = self.head_pose.x
            head_pose_y = self.head_pose.y
            head_pose_z = self.head_pose.z
            with imgui_utils.item_width(viz.font_size * 10):
                changed, (new_head_pose_x, new_head_pose_y, new_head_pose_z) = \
                    imgui.input_float3('##head_pose', head_pose_x, head_pose_y,
                                       head_pose_z, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.head_pose.x = new_head_pose_x
                    self.head_pose.y = new_head_pose_y
                    self.head_pose.z = new_head_pose_z
            # imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            # _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            # if dragging:
            #     self.drag(dx, dy)

            # imgui.same_line()
            # snapped = dnnlib.EasyDict(self.pose, yaw=round(self.pose.yaw, 1), pitch=round(self.pose.pitch, 1))
            # if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.pose != snapped)):
            #     self.pose = snapped
            # imgui.same_line()
            # if imgui_utils.button('Reset', width=-1, enabled=(self.pose != self.pose_def)):
            #     self.pose = dnnlib.EasyDict(self.pose_def)
            #
            # # New line starts here
            # imgui.text('LookAt Point')
            # imgui.same_line(viz.label_w)
            # with imgui_utils.item_width(viz.font_size * 8):
            #     _clicked, self.lookat_point_choice = imgui.combo('', self.lookat_point_choice, self.lookat_point_labels)
            # lookat_point = self.lookat_point_option[self.lookat_point_choice]
            # if lookat_point == 'auto':
            #     self.lookat_point = None
            # if lookat_point == 'ffhq':
            #     self.lookat_point = (0.0, 0.0, 0.2)
            #     changes_enabled=False
            # if lookat_point == 'shapenet':
            #     self.lookat_point = (0.0, 0.0, 0.0)
            #     changes_enabled=False
            # if lookat_point == 'afhq':
            #     self.lookat_point = (0.0, 0.0, 0.0)
            #     changes_enabled=False
            # if lookat_point == 'manual':
            #     if self.lookat_point is None:
            #         self.lookat_point = (0.0, 0.0, 0.0)
            #     changes_enabled=True
            # if lookat_point != 'auto':
            #     imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            #     with imgui_utils.item_width(viz.font_size * 16):
            #         with imgui_utils.grayed_out(not changes_enabled):
            #             _changed, self.lookat_point = imgui.input_float3('##lookat', *self.lookat_point, format='%.2f', flags=(imgui.INPUT_TEXT_READ_ONLY if not changes_enabled else 0))


        viz.args.body_pose   = [self.neck_pose.x, self.neck_pose.y, self.neck_pose.z, self.head_pose.x, self.head_pose.y, self.head_pose.z]

#----------------------------------------------------------------------------
