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
from tqdm import tqdm
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton

sys.path.insert(0, cfg.smpl_path)
from utils.smpl import SMPL
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--model_path', type=str, default='demo_checkpoint.pth.tar')
    parser.add_argument('--input_dir', type=str, default='')

    parser.add_argument('--data_dir', type=str,
                        default='E:\project/3DCrowdNet_upper_body-main\data')

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
alpha = 0.9
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

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()


def get_projected_vertex(mesh, world2screen_matrix):
    mesh = mesh[0, ...]
    mesh = np.concatenate([mesh, np.ones((mesh.shape[0], 1))], axis=1)  # 6890 x 4
    points_image = world2screen_matrix @ mesh.T  # 4,6890
    points_image = points_image[:3, :]  # 3,6890

    points_on_input_image = points_image / points_image[2, :]
    points_on_input_image = points_on_input_image[:2, :].T  # 30,2

    return points_on_input_image


import shutil

path = args.input_dir
bad_path = os.path.join(args.input_dir,'bad_aligned_images')
os.makedirs(bad_path, exist_ok=True)
image_list = glob.glob(os.path.join(args.input_dir,'aligned_images/*')) #

result_json_path = os.path.join(path, 'result.json')
with open(result_json_path, 'r') as f:
    result_json = json.load(f)

for image_path in tqdm(image_list):
    aligned_image_name = os.path.basename(image_path)

    vis1_path = os.path.join(path, 'aligned_images', aligned_image_name)


    meta_info = result_json[aligned_image_name]
    bbox1 = meta_info['bbox']

    coco_joint = np.array(meta_info['coco_joint'])
    coco_joint1 = coco_joint.copy()
    coco_joint1[:, 0] = coco_joint1[:, 0] - bbox1[0]
    coco_joint1[:, 1] = coco_joint1[:, 1] - bbox1[1]
    coco_joint1 *= 1024 / bbox1[2]

    # vis1 = cv2.imread(image_path)

    pose_params_input = torch.from_numpy(np.array(meta_info['smpl_pose'])).float().cuda().view(1, 24, 3)
    pose_params_input = pose_params_input[:, 1:, :]

    joints_3d = model.module.get_neck_head_rotated_template_mesh_joint(pose_params_input).cpu().numpy()
    # print(joints_3d.shape)
    projected_joints = get_projected_vertex(joints_3d, np.array(meta_info['intrisics']) @ np.array(
        meta_info['world2camera_matrix']))
    # print(projected_joints.shape)

    vis1 = cv2.imread(image_path)
    # joint_names = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear']
    # for j in range(5):
    #     cv2.circle(vis1, (int(coco_joint1[j, 0]), int(coco_joint1[j, 1])), 3, (0, 0, 255), -1)
    #     #cv2.circle(vis2, (int(coco_joint2[j, 0]), int(coco_joint2[j, 1])), 3, (0, 0, 255), -1)
    #     cv2.putText(vis1, joint_names[j], (int(coco_joint1[j, 0]), int(coco_joint1[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #     #cv2.putText(vis2, joint_names[j], (int(coco_joint2[j, 0]), int(coco_joint2[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #
    # for j in range(24,29):
    #     cv2.circle(vis1, (int(projected_joints[j, 0]), int(projected_joints[j, 1])), 3, (0, 255, 0), -1)
    #     #cv2.circle(vis2, (int(projected_joints[j, 0]), int(projected_joints[j, 1])), 3, (0, 255, 0), -1)
    #     cv2.putText(vis1, joint_names[j-24], (int(projected_joints[j, 0]), int(projected_joints[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #     #cv2.putText(vis2, joint_names[j-24], (int(projected_joints[j, 0]), int(projected_joints[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    coco_joints_name = (
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')

    smpl_joints_name = (
    'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
    'Neck', 'L_Thorax', 'R_Thorax',
    'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye',
    'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')

    selected_joint_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder')

    selected_coco_idx = [coco_joints_name.index(joint_name) for joint_name in selected_joint_name]
    selected_smpl_idx = [smpl_joints_name.index(joint_name) for joint_name in selected_joint_name]

    distance = 0
    count = 0
    for i in range(len(selected_joint_name)):
        if coco_joint1[selected_coco_idx[i], 2] > 0.1:
            distance += np.linalg.norm(coco_joint1[selected_coco_idx[i], :2] - projected_joints[selected_smpl_idx[i], :])
            count += 1

        cv2.circle(vis1, (int(coco_joint1[selected_coco_idx[i], 0]), int(coco_joint1[selected_coco_idx[i], 1])), 3, (0, 0, 255), -1)
        cv2.putText(vis1, selected_joint_name[i], (int(coco_joint1[selected_coco_idx[i], 0]), int(coco_joint1[selected_coco_idx[i], 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(vis1, (int(projected_joints[selected_smpl_idx[i], 0]), int(projected_joints[selected_smpl_idx[i], 1])), 3, (0, 255, 0), -1)
        cv2.putText(vis1, selected_joint_name[i], (int(projected_joints[selected_smpl_idx[i], 0]), int(projected_joints[selected_smpl_idx[i], 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if count > 0:
        distance /= count
    else:
        shutil.move(vis1_path, os.path.join(bad_path, os.path.basename(vis1_path)))

    if distance > 150:
        shutil.move(vis1_path, os.path.join(bad_path, os.path.basename(vis1_path)))







    # cv2.imshow('vis1', vis1)
    # cv2.waitKey(0)
    # cv2.imshow('vis2', vis2)
    # cv2.waitKey(0)
    # exit()

    # distance1 = np.linalg.norm(coco_joint1[:5, :2] - projected_joints[24:29, :])
    #
    # # if distance1 > distance2:
    # #     print('move', vis1_path, 'to', os.path.join(bad_path, os.path.basename(vis1_path)))
    # #     shutil.move(vis1_path, os.path.join(bad_path, os.path.basename(vis1_path)))
    # # else:
    # #     print('move', vis2_path, 'to', os.path.join(bad_path, os.path.basename(vis2_path)))
    # #     shutil.move(vis2_path, os.path.join(bad_path, os.path.basename(vis2_path)))
    #
    # if distance1 > 50:
    #     cv2.imshow('vis1', vis1)
    #     cv2.waitKey(0)

    # exit()
