import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import Pose2Feat, PositionNet, RotationNet, Vposer
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math
# visualization
import colorsys
from utils.vis import vis_mesh, save_obj, render_mesh, vis_keypoints
import numpy as np
from utils.transforms import rot6d_to_axis_angle
import cv2
from utils.preprocessing import generate_patch_image


class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net, vposer):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer.cuda()
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

        # The joint that we want to align to the origin
        self.align_joint_name = 'Neck'
        # 0.0649 is the height between the neck joint and head joint of the template
        self.init_camera_location = torch.tensor([0, 0.0649, 2.7]).float().cuda()

        # get template mesh
        root_pose = torch.zeros((1, 3)).cuda()
        pose_param = torch.zeros((1, 69)).cuda()
        cam_trans = torch.zeros((1, 3)).cuda()
        shape_param = torch.zeros((1, 10)).cuda()
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        coord_output = self.get_coord(pose_param, shape_param, cam_trans)
        self.template_mesh_cam_render = coord_output['mesh_cam_render']

        # align neck joint to origin
        template_align_joint_coorinate = coord_output['align_joint_coorinate']  # 1 x 1 x 3
        # print('template_align_joint_coorinate:',template_align_joint_coorinate)
        # exit()
        self.template_mesh_cam_render -= template_align_joint_coorinate  # 1 x 6890 x 3
        self.template_align_joint_coorinate = template_align_joint_coorinate

        # used for real world rendering, should not rotate
        self.template_mesh_cam_render_no_flip = self.template_mesh_cam_render.clone()

        self.template_mesh_cam_render_no_flip_joint = torch.bmm(
            torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(1, 1, 1),
            self.template_mesh_cam_render_no_flip)

        # in pyrender, should rotate 180 degree around x axis (since y and z axis are flipped)
        R = torch.eye(4).cuda()
        angle = torch.FloatTensor([np.pi]).cuda()
        R[1, 1] = torch.cos(angle)
        R[1, 2] = -torch.sin(angle)
        R[2, 1] = torch.sin(angle)
        R[2, 2] = torch.cos(angle)

        self.template_mesh_R = R

        self.template_mesh_cam_render = torch.matmul(self.template_mesh_cam_render, R[:3, :3])

        x_axis_ = np.array([1, 0, 0])
        y_axis_ = np.array([0, 1, 0])
        z_axis_ = np.array([0, 0, -1])
        self.Axis_original = np.concatenate([x_axis_[:, None], y_axis_[:, None], z_axis_[:, None]], axis=1)

        self.min_box_stride = None

    def get_neck_head_rotated_template_mesh(self, pose_params_input):
        root_pose = torch.zeros((1, 3)).cuda()
        pose_param = torch.zeros((1, 69)).cuda()
        cam_trans = torch.zeros((1, 3)).cuda()
        shape_param = torch.zeros((1, 10)).cuda()
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)

        pose_param[:, [11, 14], :] = pose_params_input[:, [11, 14], :]

        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        coord_output = self.get_coord(pose_param, shape_param, cam_trans)
        mesh_cam_render = coord_output['mesh_cam_render']
        mesh_cam_render -= self.template_align_joint_coorinate
        mesh_cam_render = torch.matmul(mesh_cam_render, self.template_mesh_R[:3, :3])

        return mesh_cam_render

    def get_neck_head_rotated_template_mesh_joint(self, pose_params_input):
        root_pose = torch.zeros((1, 3)).cuda()
        pose_param = torch.zeros((1, 69)).cuda()
        cam_trans = torch.zeros((1, 3)).cuda()
        shape_param = torch.zeros((1, 10)).cuda()
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)

        pose_param[:, [11, 14], :] = pose_params_input[:, [11, 14], :]

        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        coord_output = self.get_coord(pose_param, shape_param, cam_trans)
        joints_3d = coord_output['joints_3d']
        joints_3d -= self.template_align_joint_coorinate

        return joints_3d

    def set_min_box_stride(self, min_box_stride):
        self.min_box_stride = min_box_stride

    def compute_shoulder_points_R(self, mesh_a, mesh_b):
        '''
        :param mesh_a: 1 x 6890 x 3
        :param mesh_b: 1 x 6890 x 3

        shoulder_vertex_index: 55,
        '''

        joints_a = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(mesh_a.shape[0], 1, 1),
                             mesh_a)
        joints_b = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(mesh_b.shape[0], 1, 1),
                             mesh_b)

        selected_joints = [
            'L_Shoulder', 'R_Shoulder',
            # 'L_Thorax', 'R_Thorax',
            'Neck',
            # 'Chest',
            'Pelvis'
        ]
        selected_joints_index = [self.human_model.joints_name.index(joints_name) for joints_name in selected_joints]

        points_a = joints_a[:, selected_joints_index, :]
        points_b = joints_b[:, selected_joints_index, :]

        A = points_a[0, :, :].cpu().numpy()  # 55 x 3
        B = points_b[0, :, :].cpu().numpy()  # 55 x 3
        mean_A = np.mean(A, axis=0, keepdims=True)
        mean_B = np.mean(B, axis=0, keepdims=True)

        A = A - mean_A
        B = B - mean_B

        H = np.transpose(A) @ B

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        return torch.from_numpy(R).cuda().float()  # 3 x 3

    def get_camera_trans(self, cam_param, bbox, is_render):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                    cfg.input_img_shape[0] * cfg.input_img_shape[1]))]).cuda().view(-1)
        if is_render:
            k_value = k_value * math.sqrt(cfg.input_img_shape[0] * cfg.input_img_shape[1]) / (
                        bbox[:, 2] * bbox[:, 3]).sqrt()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float();
        yy = yy[None, None, :, :].cuda().float();

        x = joint_coord_img[:, :, 0, None, None];
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        batch_size = smpl_pose.shape[0]
        mesh_cam, mesh_joints = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(batch_size, 1, 1),
                              mesh_cam)
        joints_3d = joint_cam.clone()
        # head
        align_joint_coorinate = joint_cam[:, self.human_model.joints_name.index(self.align_joint_name), None, :]

        root_joint_idx = self.human_model.root_joint_idx

        # project 3D coordinates to 2D space
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        mesh_cam_render = mesh_cam.clone()
        # root-relative 3D coordinates
        root_cam = joint_cam[:, root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return {
            'joint_proj': joint_proj,
            'joint_cam': joint_cam,
            'mesh_cam': mesh_cam,
            'mesh_cam_render': mesh_cam_render,
            'align_joint_coorinate': align_joint_coorinate,
            'root_cam': root_cam,
            'joints_3d': joints_3d
        }

    def generate_visualization(self, image, mesh_cam_render, joint):

        # princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        # generate random color
        color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        # bbox = out['bbox'][0].cpu().numpy()

        mesh_image = render_mesh(image.copy() * 255, mesh_cam_render, self.human_model.face,
                                 {'focal': cfg.focal, 'princpt': cfg.princpt}, color=color)

        joint_image = vis_keypoints(image.copy() * 255, joint)

        viz = np.concatenate([image.copy() * 255,
                              joint_image.astype(np.uint8),
                              mesh_image.astype(np.uint8)],
                             axis=1)[:, :, ::-1]
        return viz

    def get_visualization(self, inputs, targets, meta_info):
        inputs = inputs
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        output = self.forward(inputs, targets, meta_info, mode='test')
        viz_predicts = []
        for i in range(inputs['img'].shape[0]):
            viz_predict = self.generate_visualization(image=inputs['img'][i].cpu().numpy().transpose(1, 2, 0),
                                                      mesh_cam_render=output['mesh_cam_render'][
                                                          i].detach().cpu().numpy(),
                                                      joint=inputs['joints'][i].detach().cpu().numpy() * (
                                                              cfg.input_img_shape[1] / cfg.output_hm_shape[2])
                                                      )
            viz_predicts.append(viz_predict)

        return viz_predicts

    def forward(self, inputs, targets, meta_info, mode):
        early_img_feat = self.backbone(inputs['img'])  # pose_guided_img_feat

        # get pose gauided image feature
        joint_coord_img = inputs['joints']
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            # remove blob centered at (0,0) == invalid ones
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:, :, :, None]
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
        pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)  # 2048 x 8 x 8

        joint_img, joint_score = self.position_net(pose_guided_img_feat)  # refined 2D pose or 3D pose

        # estimate model parameters
        root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(),
                                                                    joint_score.detach())
        # change root pose 6d + latent code -> axis angles
        root_pose = rot6d_to_axis_angle(root_pose_6d)
        pose_param = self.vposer(z)
        cam_trans = self.get_camera_trans(cam_param, meta_info['bbox'], is_render=(cfg.render and (mode == 'test')))
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)

        body_pose_param = pose_param.clone()

        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        coord_output = self.get_coord(pose_param, shape_param, cam_trans)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = coord_output['joint_proj'], coord_output['joint_cam'], \
        coord_output['mesh_cam'], coord_output['mesh_cam_render']

        if mode == 'train':
            # loss functions
            loss = {}
            # joint_img: 0~8, joint_proj: 0~64, target: 0~64
            loss['body_joint_img'] = (1 / 8) * self.coord_loss(joint_img * 8, self.human_model.reduce_joint_set(
                targets['orig_joint_img']), self.human_model.reduce_joint_set(meta_info['orig_joint_trunc']),
                                                               meta_info['is_3D'])
            loss['smpl_joint_img'] = (1 / 8) * self.coord_loss(joint_img * 8, self.human_model.reduce_joint_set(
                targets['fit_joint_img']),
                                                               self.human_model.reduce_joint_set(
                                                                   meta_info['fit_joint_trunc']) * meta_info[
                                                                                                       'is_valid_fit'][
                                                                                                   :, None, None])
            loss['smpl_pose'] = self.param_loss(pose_param, targets['pose_param'],
                                                meta_info['fit_param_valid'] * meta_info['is_valid_fit'][:, None])
            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'],
                                                 meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = (1 / 8) * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2],
                                                                meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'],
                                                     meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'],
                                                     meta_info['is_valid_fit'][:, None, None])

            return loss

        else:
            # test output
            out = {'cam_param': cam_param}
            # out['input_joints'] = joint_coord_img
            out['joint_img'] = joint_img * 8
            out['joint_proj'] = joint_proj
            out['joint_score'] = joint_score
            out['smpl_mesh_cam'] = mesh_cam
            out['smpl_pose'] = pose_param.clone()
            out['smpl_shape'] = shape_param.clone()
            out['cam_trans'] = cam_trans.clone()

            out['mesh_cam_render'] = mesh_cam_render
            out['mesh_cam_render_joints_3d'] = coord_output['joints_3d']

            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'img2bb_trans' in meta_info:
                out['img2bb_trans'] = meta_info['img2bb_trans']
            if 'bbox' in meta_info:
                out['bbox'] = meta_info['bbox']
            if 'tight_bbox' in meta_info:
                out['tight_bbox'] = meta_info['tight_bbox']
            if 'aid' in meta_info:
                out['aid'] = meta_info['aid']

            out['neck_head_rotated_template_mesh'] = self.get_neck_head_rotated_template_mesh(body_pose_param)

            cam_trans_crop = self.get_camera_trans(cam_param, meta_info['bbox'], is_render=False)
            coord_output_crop = self.get_coord(pose_param, shape_param, cam_trans_crop)
            mesh_cam_render_crop = coord_output_crop['mesh_cam_render']
            out['mesh_cam_render_crop'] = mesh_cam_render_crop
            out['align_joint_coorinate_crop'] = coord_output_crop['align_joint_coorinate']
            # align neck joint to origin
            align_joint_coorinate = coord_output['align_joint_coorinate']  # 1 x 1 x 3
            mesh_cam_render_aligned = mesh_cam_render.clone()  # 1 x 6890 x 3
            # align neck joint to origin
            mesh_cam_render_aligned -= align_joint_coorinate
            out['mesh_cam_render_aligned'] = mesh_cam_render_aligned
            out['align_joint_coorinate'] = align_joint_coorinate

            # ======================translation ===================
            translation_in_pyrender = torch.eye(4, device=mesh_cam_render_aligned.device)
            translation_in_pyrender[:3, 3:4] = -align_joint_coorinate.squeeze(1).T

            # flip y axis and z axis to render in pyrender correctly
            translation_in_pyrender[[1, 2], 3] *= -1
            # =====================================================

            # ======================rotaion =======================
            rotation_in_pyrender = torch.eye(4, device=mesh_cam_render_aligned.device)
            # compute the rotation matrix that rotate template to the aligned mesh
            R = self.compute_shoulder_points_R(self.template_mesh_cam_render, mesh_cam_render_aligned)
            # flip y axis and z axis to render in pyrender correctly
            angles = cv2.Rodrigues(torch.inverse(R).cpu().numpy())[0]
            angles[[1, 2], :] *= -1
            R_in_pyrender = cv2.Rodrigues(angles)[0]
            rotation_in_pyrender[:3, :3] = torch.from_numpy(R_in_pyrender).to(mesh_cam_render_aligned.device)
            # ========================================================

            # ========================== remder template on original image ==================================
            out['camera_pose_in_pyrender'] = rotation_in_pyrender @ translation_in_pyrender
            out['camera_to_render_template_in_pyrender'] = translation_in_pyrender
            out['no_rotation_world2camera_transformation_in_real_world'] = torch.inverse(translation_in_pyrender)  #
            out['no_rotation_world2camera_transformation_in_real_world'][[1, 2]] *= -1

            # ========================== Normalized camera ==========================
            normalized_camerapose_in_pyrender = out['camera_pose_in_pyrender'].cpu().numpy()

            camera_position = normalized_camerapose_in_pyrender[:3, 3]
            camera_position = camera_position / np.linalg.norm(camera_position) * 2.7

            camera_up = normalized_camerapose_in_pyrender[:3, :3] @ np.reshape(np.array([0, 1, 0]), (3, 1))[:, 0]  # 3,

            # we suppose the camera is always looking at the [0, 0.0649, 0]
            Lookat = np.array([0, 0.0649, 0])

            z_axis = Lookat - camera_position
            z_axis = z_axis / np.linalg.norm(z_axis)
            x_axis = -np.cross(camera_up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = -np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            Axis_new = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)

            R = Axis_new @ np.linalg.inv(self.Axis_original)

            normalized_camerapose_in_pyrender[:3, :3] = R
            normalized_camerapose_in_pyrender[:3, 3] = camera_position

            normalized_transformation_in_realworld = np.linalg.inv(normalized_camerapose_in_pyrender)
            normalized_transformation_in_realworld[[1, 2]] *= -1

            out['normalized_camerapose_in_pyrender'] = normalized_camerapose_in_pyrender
            out['normalized_transformation_in_realworld'] = normalized_transformation_in_realworld

            return out

    def get_projected_joints(self, joint_cam, world_2_camera_matrix, princpt):
        # joint in 3d world coordinate
        joint_cam = joint_cam.squeeze(0)  # 30 x 3
        joint_cam = torch.cat([joint_cam, torch.ones(joint_cam.shape[0], 1).cuda()], dim=1)  # 30 x 4
        joint_on_input_image = world_2_camera_matrix @ joint_cam.T  # 4,30
        joint_on_input_image = joint_on_input_image[:3, :].cpu().numpy()

        intrinsic_matrix = np.eye(3)
        intrinsic_matrix[0, 0] = cfg.focal[0]
        intrinsic_matrix[1, 1] = cfg.focal[1]
        intrinsic_matrix[0, 2] = princpt[0]
        intrinsic_matrix[1, 2] = princpt[1]

        joint_on_input_image = intrinsic_matrix @ joint_on_input_image
        joint_on_input_image = joint_on_input_image / joint_on_input_image[2, :]
        joint_on_input_image = joint_on_input_image[:2, :].T  # 30,2

        return joint_on_input_image

    def get_projected_vertex(self, mesh, world2screen_matrix):

        mesh = mesh.squeeze(0)
        mesh = torch.cat([mesh, torch.ones(mesh.shape[0], 1).cuda()], dim=1).cpu().numpy()  # 6890 x 4
        points_image = world2screen_matrix @ mesh.T  # 4,6890
        points_image = points_image[:3, :]  # 3,6890

        points_on_input_image = points_image / points_image[2, :]
        points_on_input_image = points_on_input_image[:2, :].T  # 30,2

        return points_on_input_image

    def __crop_and_process_camera_matrix__(self, model_output, input_image, joint_2d, crop_image_size, model_input_bbox,
                                           head_bbox, use_head_bbox):
        # project template mesh to input full size image
        template_joint_on_input_image = self.get_projected_joints(self.template_mesh_cam_render_no_flip_joint,
                                                                  model_output[
                                                                      'no_rotation_world2camera_transformation_in_real_world'],
                                                                  (model_input_bbox[0] + model_input_bbox[2] / 2,
                                                                   model_input_bbox[1] + model_input_bbox[3] / 2))

        L_Shoulder_2d = template_joint_on_input_image[self.human_model.joints_name.index('L_Shoulder'), :]
        R_Shoulder_2d = template_joint_on_input_image[self.human_model.joints_name.index('R_Shoulder'), :]

        #  project template mesh using the nomalized camera (1024)
        template_joint_on_crop_image = self.get_projected_joints(
            self.template_mesh_cam_render_no_flip_joint,
            torch.from_numpy(model_output['normalized_transformation_in_realworld']).float().cuda(),
            (crop_image_size / 2, crop_image_size / 2))
        L_Shoulder_2d_on_crop_image = template_joint_on_crop_image[self.human_model.joints_name.index('L_Shoulder'), :]
        R_Shoulder_2d_on_crop_image = template_joint_on_crop_image[self.human_model.joints_name.index('R_Shoulder'), :]
        Shoulder_center_on_crop_image = (L_Shoulder_2d_on_crop_image + R_Shoulder_2d_on_crop_image) / 2.0

        # vis = crop_output['cropped_image'].copy()
        # for i in range(template_joint_on_crop_image.shape[0]):
        #     cv2.circle(vis, (int(template_joint_on_crop_image[i, 0]), int(template_joint_on_crop_image[i, 1])), 5,
        #                (0, 255, 255), -1)
        #     cv2.putText(vis, str(i), (int(template_joint_on_crop_image[i, 0]), int(template_joint_on_crop_image[i, 1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        #
        # vis = cv2.resize(vis, (vis.shape[1] // 4, vis.shape[0] // 4))
        # cv2.imshow('input_image', vis.astype(np.uint8))
        # cv2.waitKey(0)
        # exit()

        L_Shoulder_coco = joint_2d[5, :2]
        R_Shoulder_coco = joint_2d[6, :2]
        shoulder_center_coco = (L_Shoulder_coco + R_Shoulder_coco) / 2.0

        '''
        cv2.circle(input_image, (int(L_ear_from_coco[ 0]), int(L_ear_from_coco[ 1])), 10, (0, 0, 255), -1)
        cv2.putText(input_image, "L_ear_from_coco", (int(L_ear_from_coco[0]), int(L_ear_from_coco[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)

        cv2.circle(input_image, (int(R_ear_from_coco[0]), int(R_ear_from_coco[1])), 10, (0, 0, 255), -1)
        cv2.putText(input_image, "R_ear_from_coco", (int(R_ear_from_coco[0]), int(R_ear_from_coco[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)

        cv2.circle(input_image, (int(rotated_L_ear_2d[0]), int(rotated_L_ear_2d[1])), 10, (0, 255, 255), -1)
        cv2.putText(input_image, "rotated_L_ear_2d", (int(rotated_L_ear_2d[0]), int(rotated_L_ear_2d[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 1)

        cv2.circle(input_image, (int(rotated_R_ear_2d[0]), int(rotated_R_ear_2d[1])), 10, (0, 255, 255), -1)
        cv2.putText(input_image, "rotated_R_ear_2d", (int(rotated_R_ear_2d[0]), int(rotated_R_ear_2d[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 1)


        input_image = cv2.resize(input_image, (input_image.shape[1] // 4, input_image.shape[0] // 4))
        cv2.imshow('input_image', input_image)
        cv2.waitKey(0)
        #'''

        # compute the stride of the bbox, using the shoulder distance of head stride
        if use_head_bbox:
            assert len(head_bbox) == 4
            head_stride = min(head_bbox[2], head_bbox[3])
            bbox_stride = head_stride * 2.6
        else:
            shoulder_stride = np.linalg.norm(L_Shoulder_2d - R_Shoulder_2d)
            bbox_stride = shoulder_stride * 1.6

        # shoulder_center_coco should be aligned with Shoulder_center_on_crop_image
        # bbox_center = shoulder_center_coco - (Shoulder_center_on_crop_image - crop_image_size / 2.0)/crop_image_size*bbox_stride   #aligned_joint_2d  + align_translation_on_input_image
        # (shoulder_center_coco - bbox_center)/bbox_stride*crop_image_size = (Shoulder_center_on_crop_image - crop_image_size / 2.0)
        # (shoulder_center_coco - bbox_center) = (Shoulder_center_on_crop_image - crop_image_size / 2.0)*bbox_stride/crop_image_size

        bbox_center = shoulder_center_coco - (
                    Shoulder_center_on_crop_image - crop_image_size / 2.0) * bbox_stride / crop_image_size

        bbox = np.array([bbox_center[0] - bbox_stride / 2, bbox_center[1] - bbox_stride / 2, bbox_stride, bbox_stride])
        # print('bbox',bbox)

        if bbox[2] < self.min_box_stride or bbox[3] < self.min_box_stride:
            return None

        # crop_image
        try:
            img, img2bb_trans, bb2img_trans = generate_patch_image(input_image, bbox, 1.0, 0.0, False,
                                                                   (crop_image_size, crop_image_size),
                                                                   enable_padding=True)
        except:
            return None

        # the cam_param is corresponding to the original bbox
        original_cam_param = {'focal': cfg.focal, 'princpt': (
        model_input_bbox[0] + model_input_bbox[2] / 2, model_input_bbox[1] + model_input_bbox[3] / 2)}

        # rescale to the original image size

        # crop to new bbox
        w_crop = bbox[0]
        h_crop = bbox[1]

        cx_new = original_cam_param['princpt'][0] - w_crop
        cy_new = original_cam_param['princpt'][1] - h_crop

        translated_princpt = (cx_new, cy_new)

        # rescale to the crop image
        new_focal = (cfg.focal[0] / bbox[2] * crop_image_size, cfg.focal[1] / bbox[3] * crop_image_size)
        new_princpt = (
            translated_princpt[0] / bbox[2] * crop_image_size, translated_princpt[1] / bbox[3] * crop_image_size)

        cam_param = {'focal': new_focal, 'princpt': new_princpt}

        out = {}
        out['intrisics'] = cam_param
        out['cropped_image'] = img
        out['bbox'] = bbox
        out['bbox_stride'] = bbox_stride

        return out

    def crop_and_process_camera_matrix(self, model_output, input_image, joint_2d, crop_image_size, model_input_bbox,
                                       head_bbox):

        out = []

        head_bbox_score = head_bbox['score']
        head_bbox_ = head_bbox['bbox']

        if len(head_bbox_) == 4:
            out_ = self.__crop_and_process_camera_matrix__(model_output, input_image, joint_2d, crop_image_size,
                                                           model_input_bbox,
                                                           head_bbox_, use_head_bbox=True)
            if out_ is not None:
                out.append(out_)
            # out_ = self.__crop_and_process_camera_matrix__(model_output, input_image, joint_2d, crop_image_size,
            #                                                model_input_bbox,
            #                                                head_bbox_, use_head_bbox=False)
            # if out_ is not None and len(out) > 0:
            #     if abs(out_['bbox_stride'] - out[0]['bbox_stride']) > out[0]['bbox_stride'] * 0.05:
            #         out.append(out_)

        else:
            # no bbox, use the shoulder as stride
            out_ = self.__crop_and_process_camera_matrix__(model_output, input_image, joint_2d, crop_image_size,
                                                           model_input_bbox,
                                                           head_bbox_, use_head_bbox=False)
            if out_ is not None:
                out.append(out_)

        return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(vertex_num, joint_num, mode):
    backbone = ResNetBackbone(cfg.resnet_type)
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()

    if mode == 'train':
        backbone.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)

    model = Model(backbone, pose2feat, position_net, rotation_net, vposer)
    return model

