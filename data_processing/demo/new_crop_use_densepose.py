import glob
import shutil
import sys
import os
import os.path as osp
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2
import colorsys
import json
import random
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import pyrender

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
# from model import get_model
# from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
# from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton

sys.path.insert(0, cfg.smpl_path)
from utils.smpl import SMPL

import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# check if on a Linux machine
if os.name == 'posix': # Linux
    os.environ["PYOPENGL_PLATFORM"] = "egl"
def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord


def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_path', type=str, default='demo_checkpoint.pth.tar')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--data_dir', type=str, default='101570')

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def get_projected_vertex(mesh, world2screen_matrix):
    mesh = mesh[0, ...]
    mesh = np.concatenate([mesh, np.ones((mesh.shape[0], 1))], axis=1)  # 6890 x 4
    # mesh = torch.cat([mesh, torch.ones((mesh.shape[0], 1))], dim=1)
    points_image = world2screen_matrix @ mesh.T  # 4,6890
    points_image = points_image[:3, :]  # 3,6890

    points_on_input_image = points_image / points_image[2, :]
    points_on_input_image = points_on_input_image[:2, :].T  # 30,2

    return points_on_input_image


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, is_test=True)
cfg.set_data_dir(args.data_dir)
cfg.render = True
cudnn.benchmark = True

# SMPL joint set
joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top
joints_name = (
    'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
    'Neck', 'L_Thorax', 'R_Thorax',
    'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye',
    'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
flip_pairs = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
skeleton = (
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
    (17, 19),
    (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
    (24, 26),
    (25, 27), (26, 28), (24, 29))

# SMPl mesh
vertex_num = 6890
smpl = SMPL()
face = smpl.face
alpha = 0.8
# other joint set
coco_joints_name = (
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
coco_skeleton = (
    (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15),
    (5, 6),
    (11, 17), (12, 17), (17, 18))

vis_joints_name = (
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
vis_skeleton = (
    (0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18),
    (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

human_model_layer = smpl.layer['neutral'].cuda()

# prepare input image
transform = transforms.ToTensor()
# pose2d_result_path = os.path.join(args.input_dir, '2d_pose_result_hrnet.json')
# with open(pose2d_result_path) as f:
#     pose2d_result = json.load(f)

img_dir = os.path.join(args.input_dir, 'images')

debug = True

input_aligned_images_dir = os.path.join(args.input_dir, 'aligned_images')

output_dir = args.output_dir
print('>>>>>>> output_dir', output_dir)
os.makedirs(output_dir, exist_ok=True)

result_json_path = os.path.join(args.input_dir, 'result.json')
with open(result_json_path, 'r') as f:
    result_json = json.load(f)

template_mesh_in_pyrender = np.load('./template_mesh_in_pyrender.npy')
print('template_mesh_in_pyrender.shape', template_mesh_in_pyrender.shape)
template_mesh = np.load('./template_mesh.npy')
print('template_mesh.shape', template_mesh.shape)

from model import get_model

model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

from utils.preprocessing import generate_patch_image

output_aligned_images_dir = os.path.join(output_dir, 'aligned_images')
output_visualization_dir = os.path.join(output_dir, 'visualization')
os.makedirs(output_aligned_images_dir, exist_ok=True)
os.makedirs(output_visualization_dir, exist_ok=True)

output_result_json_path = os.path.join(output_dir, 'result.json')
if os.path.exists(output_result_json_path):
    with open(output_result_json_path, 'r') as f:
        output_result_json = json.load(f)
else:
    output_result_json = {}


def exit_function():
    global output_result_json
    with open(output_result_json_path, 'w') as f:
        json.dump(output_result_json, f, indent=4)
    print('结束')


import atexit
from tqdm import tqdm

atexit.register(exit_function)

color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
template_mesh_cam_render = model.module.template_mesh_cam_render.cpu().numpy()  # 6890, 3

head_vertexes_index = np.where(template_mesh_cam_render[0, :, 1] < -0.0649)[0]

#
# align_joint_name = 'Neck'
# align_joint_index =  model.module.human_model.joints_name.index(align_joint_name)
# print(align_joint_indexmodel.module.joint_regressor.max(), model.module.joint_regressor.shape)
# neck_vertexes_index = np.where(model.module.joint_regressor[align_joint_index,:] == 1 )[0]
# print('neck_vertexes_index', neck_vertexes_index , neck_vertexes_index.shape)

for input_aligned_image_path in tqdm(glob.glob(os.path.join(input_aligned_images_dir, '*'))):
    aligned_image_name = osp.basename(input_aligned_image_path)

    dense_pose_img_paths = glob.glob(osp.join(args.input_dir, 'seg', aligned_image_name.replace('.png', '*.png')))
    original_aligned_image_path = osp.join(args.input_dir, 'aligned_images', aligned_image_name)
    original_visualization_path = osp.join(args.input_dir, 'visualization', aligned_image_name)

    meta_info = result_json[aligned_image_name]
    intrisics_dict = meta_info['intrisics_dict']
    intrisics_dict['focal'][0] = intrisics_dict['focal'][0] / 0.75
    intrisics_dict['focal'][1] = intrisics_dict['focal'][1] / 0.75

    camera_pose = np.array(meta_info['camera_pose'])

    raw_image_name = meta_info['raw_image_name']
    raw_image_path = osp.join(img_dir, raw_image_name)
    raw_image = cv2.imread(raw_image_path)

    if len(dense_pose_img_paths) == 0:
        res_meta_info = {}
        save_key = os.path.basename(input_aligned_image_path)
        original_bbox = meta_info['bbox']

        stride = original_bbox[2]
        center = np.array([original_bbox[0] + stride / 2, original_bbox[1] + stride / 2])

        stride = 0.75 * stride

        new_bbox = [center[0] - stride / 2, center[1] - stride / 2, stride, stride]

        new_aligned_image, img2bb_trans, bb2img_trans = generate_patch_image(raw_image, new_bbox, 1.0, 0.0, False,
                                                                             (1024, 1024),
                                                                             enable_padding=True)
        viz = [new_aligned_image]
        body_pose_param = torch.from_numpy(np.array(meta_info['smpl_pose'])).float().cuda()
        body_pose_param = body_pose_param.reshape(-1, 24, 3)
        body_pose_param = body_pose_param[:, 1:, :]
        mesh_cam_render = model.module.get_neck_head_rotated_template_mesh(body_pose_param)
        image_camera_rotate = render_mesh(new_aligned_image.copy(),
                                          mesh_cam_render[0].cpu().numpy(), face,
                                          intrisics_dict,
                                          color=color, cam_pose=meta_info['normalized_camerapose_in_pyrender'])
        viz.append((image_camera_rotate * alpha + new_aligned_image.copy() * (1.0 - alpha)).astype(np.uint8))

        viz = np.concatenate(viz, axis=0).astype(np.uint8)
        viz = cv2.resize(viz, (viz.shape[1] // 4, viz.shape[0] // 4))

        # output_aligned_images_dir = os.path.join(output_dir, 'aligned_images')
        # output_visualization_dir = os.path.join(output_dir, 'visualization')
        new_aligned_image_path = os.path.join(output_aligned_images_dir, save_key)
        cv2.imwrite(new_aligned_image_path, new_aligned_image)

        new_visualization_path = os.path.join(output_visualization_dir, save_key)
        cv2.imwrite(new_visualization_path, viz)

        res_meta_info['bbox'] = new_bbox
        res_meta_info['camera_pose'] = camera_pose.tolist()
        res_meta_info['smpl_pose'] = meta_info['smpl_pose']
        res_meta_info['raw_image_name'] = raw_image_name

        output_result_json[save_key] = res_meta_info

    for dense_pose_img_path in dense_pose_img_paths:
        res_meta_info = {}
        save_key = os.path.basename(dense_pose_img_path)
        if save_key in output_result_json:
            continue
        dense_pose_img_ = cv2.imread(dense_pose_img_path).astype(np.int32)

        dense_pose_index_ = dense_pose_img_[:, :, 0] * 255 + dense_pose_img_[:, :, 1]
        dense_pose_index_[dense_pose_img_[:, :, 2] == 255] = -1

        # print('dense_pose_index', dense_pose_index.shape, dense_pose_index.min(), dense_pose_index.max()    )
        # mask out dense_pose_index that not in head_vertexes_index
        # for i in range(dense_pose_index.shape[0]):
        #     for j in range(dense_pose_index.shape[1]):
        #         if dense_pose_index[i,j] not in head_vertexes_index:
        #             dense_pose_index[i,j] = -1
        #             dense_pose_img[i,j,:] = 255
        # dense_pose_index = np.ones_like(dense_pose_index_)*-1
        # dense_pose_img = np.ones_like(dense_pose_img_)*255

        dense_pose_2d_points = np.ones((head_vertexes_index.shape[0], 2)) * -1

        for i, selected_vertex in enumerate(head_vertexes_index):
            mask = dense_pose_index_ == selected_vertex
            # dense_pose_index[mask] = selected_vertex
            # dense_pose_img[mask, :] = dense_pose_img_[mask, :]
            if mask.sum() == 0:
                continue
            dense_pose_2d_points[i, :] = np.array([np.mean(np.where(mask)[1]), np.mean(np.where(mask)[0])])
            # cv2.circle(dense_pose_img, (int(dense_pose_2d_points[i,0]), int(dense_pose_2d_points[i,1])), 6, (0,255,0), -1)

        # dense_pose_img = dense_pose_img.astype(np.uint8)
        valid_head_vertexes_index = np.where(dense_pose_2d_points[:, 0] != -1)[0]
        dense_pose_2d_points = dense_pose_2d_points[valid_head_vertexes_index, :]

        if dense_pose_2d_points.shape[0] == 0:
            continue

        # project smpl mesh to img:

        # mesh_cam_render, _ = human_model_layer(torch.from_numpy(np.array(meta_info['smpl_pose'])).float().cuda(),
        #                                        torch.from_numpy(np.array(meta_info['smpl_shape'])).float().cuda(),
        #                                        torch.from_numpy(np.array(meta_info['cam_trans'])).float().cuda())
        #
        # mesh_cam_render = mesh_cam_render.cpu().numpy()
        body_pose_param = torch.from_numpy(np.array(meta_info['smpl_pose'])).float().cuda()
        body_pose_param = body_pose_param.reshape(-1, 24, 3)
        body_pose_param = body_pose_param[:, 1:, :]
        mesh_cam_render = model.module.get_neck_head_rotated_template_mesh(body_pose_param)

        mesh_proj = torch.matmul(mesh_cam_render, model.module.template_mesh_R[:3, :3]).cpu().numpy()

        intrisics = np.array(meta_info['intrisics'])
        # optimize trans and scale
        transl = np.array([0, 0]).reshape(1, 2)
        scale = np.array([1]).reshape(1, 1)

        proj_matrix = np.array(intrisics) @ np.array(meta_info['world2camera_matrix'])
        projected_vertexes = get_projected_vertex(mesh_proj, proj_matrix)
        moved_projected_vertexes = projected_vertexes * scale + transl

        projected_vertexes = moved_projected_vertexes[head_vertexes_index, :][valid_head_vertexes_index, :]

        # vertex_vis = dense_pose_img.copy()
        # for i in range(projected_vertexes.shape[0]):
        #     if projected_vertexes[i, 0] < 0 or projected_vertexes[i, 0] >= vertex_vis.shape[1] or \
        #             projected_vertexes[i, 1] < 0 or projected_vertexes[i, 1] >= vertex_vis.shape[0]:
        #         continue
        #     cv2.circle(vertex_vis, (int(projected_vertexes[i, 0]), int(projected_vertexes[i, 1])), 6, (255, 0, 0), -1)
        # cv2.imshow('vertex_vis', vertex_vis)
        # cv2.waitKey(0)

        # print('dense_pose_2d_points', dense_pose_2d_points.shape)
        # print('projected_vertexes', projected_vertexes.shape)
        # try to align projected_vertexes to dense_pose_index
        height_dense_pose = dense_pose_2d_points[:, 1].max() - dense_pose_2d_points[:, 1].min()
        width_dense_pose = dense_pose_2d_points[:, 0].max() - dense_pose_2d_points[:, 0].min()
        new_center = np.array([1024 / 2, 1024 / 2]).reshape(1, 2)

        height_projected_vertexes = projected_vertexes[:, 1].max() - projected_vertexes[:, 1].min()
        width_projected_vertexes = projected_vertexes[:, 0].max() - projected_vertexes[:, 0].min()

        scale = max(height_projected_vertexes / height_dense_pose, width_projected_vertexes / width_dense_pose)

        scale = max(scale, 0.85)
        scale = min(scale, 2)

        dense_pose_2d_points = dense_pose_2d_points * scale
        new_center = new_center * scale

        center_dense_pose = np.array([dense_pose_2d_points[:, 0].mean(), dense_pose_2d_points[:, 1].mean()]).reshape(1,
                                                                                                                     2)
        center_projected_vertexes = np.array(
            [projected_vertexes[:, 0].mean(), projected_vertexes[:, 1].mean()]).reshape(1, 2)

        transl = center_projected_vertexes - center_dense_pose

        dense_pose_2d_points = dense_pose_2d_points + transl
        new_center = new_center + transl

        # vertex_vis = np.ones_like(dense_pose_img)*255
        # for i in range(projected_vertexes.shape[0]):
        #     if projected_vertexes[i, 0] < 0 or projected_vertexes[i, 0] >= vertex_vis.shape[1] or \
        #             projected_vertexes[i, 1] < 0 or projected_vertexes[i, 1] >= vertex_vis.shape[0]:
        #         continue
        #     cv2.circle(vertex_vis, (int(projected_vertexes[i, 0]), int(projected_vertexes[i, 1])), 6, (255, 0, 0), -1)
        #     cv2.circle(vertex_vis, (int(dense_pose_2d_points[i, 0]), int(dense_pose_2d_points[i, 1])), 6, (0, 255, 0), -1)
        #
        # cv2.imshow('vertex_vis', vertex_vis)
        # cv2.waitKey(0)

        original_bbox = meta_info['bbox']
        stride = original_bbox[2]
        center = np.array([original_bbox[0] + stride / 2, original_bbox[1] + stride / 2])

        new_stride = stride / scale

        # transl_ = transl/1024

        # new_center_on_crop_image = np.array([1024/2, 1024/2]) + transl
        new_center = center - (new_center - np.array([1024 / 2, 1024 / 2]).reshape(1, 2)) / 1024 * stride
        new_center = new_center.reshape(-1)
        # print('new_center', new_center)
        # print('new_stride', new_stride)
        new_stride = new_stride * 0.75
        new_bbox = [new_center[0] - new_stride / 2, new_center[1] - new_stride / 2, new_stride, new_stride]
        meta_info['bbox'] = new_bbox

        try:
            new_aligned_image, img2bb_trans, bb2img_trans = generate_patch_image(raw_image, new_bbox, 1.0, 0.0, False,
                                                                                 (1024, 1024),
                                                                                 enable_padding=True)
        except:
            continue
        viz = [new_aligned_image]
        image_camera_rotate = render_mesh(new_aligned_image.copy(),
                                          mesh_cam_render[0].cpu().numpy(), face,
                                          intrisics_dict,
                                          color=color, cam_pose=meta_info['normalized_camerapose_in_pyrender'])
        viz.append((image_camera_rotate * alpha + new_aligned_image.copy() * (1.0 - alpha)).astype(np.uint8))

        viz = np.concatenate(viz, axis=0).astype(np.uint8)
        viz = cv2.resize(viz, (viz.shape[1] // 4, viz.shape[0] // 4))

        # output_aligned_images_dir = os.path.join(output_dir, 'aligned_images')
        # output_visualization_dir = os.path.join(output_dir, 'visualization')
        new_aligned_image_path = os.path.join(output_aligned_images_dir, save_key)
        cv2.imwrite(new_aligned_image_path, new_aligned_image)

        new_visualization_path = os.path.join(output_visualization_dir, save_key)
        cv2.imwrite(new_visualization_path, viz)

        res_meta_info['bbox'] = new_bbox
        res_meta_info['camera_pose'] = camera_pose.tolist()
        res_meta_info['smpl_pose'] = meta_info['smpl_pose']
        res_meta_info['raw_image_name'] = raw_image_name

        output_result_json[save_key] = res_meta_info

        # print(save_key,scale)

with open(output_result_json_path, 'w') as f:
    json.dump(output_result_json, f, indent=4)































