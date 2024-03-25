import glob
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
    mesh = mesh[0,...]
    mesh = np.concatenate([mesh, np.ones((mesh.shape[0], 1))], axis=1)  # 6890 x 4
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
pose2d_result_path = os.path.join(args.input_dir, '2d_pose_result_hrnet.json')
with open(pose2d_result_path) as f:
    pose2d_result = json.load(f)

img_dir = os.path.join(args.input_dir, 'images')

debug = True

output_dir = args.output_dir
print('>>>>>>> output_dir', output_dir)
os.makedirs(output_dir, exist_ok=True)
aligned_images_dir = os.path.join(output_dir, 'aligned_images')
visualization_dir = os.path.join(output_dir, 'visualization_debug')
os.makedirs(visualization_dir, exist_ok=True)

result_json_path = os.path.join(output_dir, 'result.json')
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


for aligned_image_name in result_json.keys():
    color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
    meta_info = result_json[aligned_image_name]

    visualization_path = os.path.join(visualization_dir, aligned_image_name)
    aligned_image_path = os.path.join(aligned_images_dir, aligned_image_name)
    aligned_image = cv2.imread(aligned_image_path)



    # crop image
    viz = [aligned_image]
    mesh_cam_render, _ = human_model_layer(torch.from_numpy(np.array(meta_info['smpl_pose'])).float().cuda(),
                                           torch.from_numpy(np.array(meta_info['smpl_shape'])).float().cuda(),
                                           torch.from_numpy(np.array(meta_info['cam_trans'])).float().cuda())
    image_mesh = render_mesh(aligned_image.copy(),
                             mesh_cam_render[0].cpu().numpy(), face,
                             meta_info['intrisics_crop_dict'],
                             color=color)
    # image_mesh,_,_ = generate_patch_image(image_mesh, crop_output['bbox'], 1.0, 0.0, False, (crop_image_size,crop_image_size))
    viz.append((image_mesh * alpha + aligned_image.copy() * (1.0 - alpha)).astype(np.uint8))


    image_camera_rotate = render_mesh(aligned_image.copy(),
                                      template_mesh_in_pyrender[0], face,
                                      meta_info['intrisics_dict'],
                                      color=color, cam_pose=meta_info['normalized_camerapose_in_pyrender'])
    viz.append((image_camera_rotate * alpha + aligned_image.copy() * (1.0-alpha)).astype(np.uint8))





    #
    projected_vertexes = get_projected_vertex(template_mesh, np.array(meta_info['intrisics']) @ np.array(meta_info['world2camera_matrix']))
    vertex_vis = aligned_image.copy()
    camera_pose = np.array(meta_info['camera_pose'])
    camera_forward_direction = (camera_pose[:3, :3] @ np.reshape(np.array([0, 0, 1]),(3,1)))[:,0]  # 3,1
    camera_position = camera_pose[:, 3:4][:3, 0]  # 34,1
    not_pass_check = 0
    in_screen = 0
    for i in range(projected_vertexes.shape[0]):
        if projected_vertexes[i, 0] < 0 or projected_vertexes[i, 0] >= vertex_vis.shape[1] or \
                projected_vertexes[i, 1] < 0 or projected_vertexes[i, 1] >= vertex_vis.shape[0]:
            continue
        check = np.sum((template_mesh[0, i, :3] - camera_position) * camera_forward_direction)
        in_screen += 1
        if check < 0:
            not_pass_check += 1
        cv2.circle(vertex_vis, (int(projected_vertexes[i, 0]), int(projected_vertexes[i, 1])), 6, (255, 0, 0), -1)
    print('check', not_pass_check, in_screen)
    if not_pass_check == in_screen:
        raise Exception('all vertexes are before camera')
    viz.append(vertex_vis)

    # flip image
    flip_camerapose_in_pyrender = np.array(meta_info['normalized_camerapose_in_pyrender'])
    flip_camerapose_in_pyrender = flip_yaw(flip_camerapose_in_pyrender)

    image_camera_rotate_flip = render_mesh(cv2.flip(aligned_image.copy(), 1),
                                           template_mesh_in_pyrender[0], face,
                                           meta_info['intrisics_dict'],
                                           color=color, cam_pose=flip_camerapose_in_pyrender)
    viz.append((image_camera_rotate_flip * alpha + cv2.flip(aligned_image.copy(), 1) * (1.0 - alpha)).astype(np.uint8))


    # flip
    camera_pose = np.array(meta_info['camera_pose'])
    flip_camera_pose = flip_yaw(camera_pose)


    flip_world2camera_matrix = np.linalg.inv(flip_camera_pose)

    projected_vertexes = get_projected_vertex(template_mesh, np.array(meta_info['intrisics']) @ flip_world2camera_matrix) #
    # select head & neck vertexes
    template_align_joint_coorinate = model.module.template_align_joint_coorinate.cpu().numpy() # 30, 6890
    template_mesh_cam_render = model.module.template_mesh_cam_render.cpu().numpy() # 6890, 3
    # template_mesh_cam_render -template_align_joint_coorinate > 0
    print(template_mesh_cam_render.shape)
    selected_vertexes = np.where( template_mesh_cam_render[0,:,1]<0 )[0]
    print(selected_vertexes.shape)
    projected_vertexes = projected_vertexes[selected_vertexes, :]
    print(projected_vertexes.shape)


    vertex_vis = cv2.flip(aligned_image.copy(), 1)
    camera_pose = flip_camera_pose
    camera_forward_direction = (camera_pose[:3, :3] @ np.reshape(np.array([0, 0, 1]), (3, 1)))[:, 0]  # 3,1
    camera_position = camera_pose[:, 3:4][:3, 0]  # 34,1
    not_pass_check = 0
    in_screen = 0
    for i in range(projected_vertexes.shape[0]):
        if projected_vertexes[i, 0] < 0 or projected_vertexes[i, 0] >= vertex_vis.shape[1] or \
                projected_vertexes[i, 1] < 0 or projected_vertexes[i, 1] >= vertex_vis.shape[0]:
            continue
        check = np.sum((template_mesh[0, i, :3] - camera_position) * camera_forward_direction)
        in_screen += 1
        if check < 0:
            not_pass_check += 1
        cv2.circle(vertex_vis, (int(projected_vertexes[i, 0]), int(projected_vertexes[i, 1])), 6, (255, 0, 0), -1)
    print('check', not_pass_check, in_screen)
    if not_pass_check == in_screen:
        raise Exception('all vertexes are before camera')



    viz.append(vertex_vis)
    viz = np.concatenate(viz, axis=0)
    cv2.imwrite(visualization_path, cv2.resize(viz, (viz.shape[1] //4, viz.shape[0] //4)))
