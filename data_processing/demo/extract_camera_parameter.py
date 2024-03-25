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
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton
import atexit
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
    parser.add_argument('--crop_image_size', type=int, default=1024)
    parser.add_argument('--debug', type=int, default=0)

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


def bad_image_vis(image, joint,vis_skeleton):
    joint[-1], joint[-2] = joint[-2].copy(), joint[-1].copy()
    image = vis_coco_skeleton(image, joint.T, vis_skeleton)
    image = cv2.resize(image, (512, int(image.shape[0]/image.shape[1] *512)))
    return image




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

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
pose2d_result_path = os.path.join(args.input_dir, '2d_pose_result_hrnet.json')
with open(pose2d_result_path) as f:
    pose2d_result = json.load(f)


head_bbox_path = os.path.join(args.input_dir, 'head_bbox_yolov5_crowdhuman.json')
with open(head_bbox_path) as f:
    head_bbox_result = json.load(f)

img_dir = os.path.join(args.input_dir, 'images')

output_dir = args.output_dir
print('>>>>>>> output_dir', output_dir)
os.makedirs(output_dir, exist_ok=True)

aligned_images_dir = os.path.join(output_dir, 'aligned_images')
os.makedirs(aligned_images_dir, exist_ok=True)

bad_images_dir = os.path.join(output_dir, 'bad_images')
os.makedirs(bad_images_dir, exist_ok=True)

visualization_dir = os.path.join(output_dir, 'visualization')
os.makedirs(visualization_dir, exist_ok=True)


result_json_path = os.path.join(output_dir, 'result.json')
if os.path.exists(result_json_path):
    with open(result_json_path, 'r') as f:
        result_json = json.load(f)
else:
    result_json = {}

def exit_function():
    global result_json
    with open(result_json_path, 'w') as f:
        json.dump(result_json, f)
    print('结束')

atexit.register(exit_function)



if not os.path.exists('./template_mesh.npy'):
    print(
        f'save template mesh (shape {model.module.template_mesh_cam_render_no_flip.cpu().numpy().shape}) to ./template_mesh.npy')
    np.save('./template_mesh.npy', model.module.template_mesh_cam_render_no_flip.cpu().numpy())
    template_mesh = model.module.template_mesh_cam_render_no_flip.cpu().numpy()
else:
    print('load template_mesh from ', './template_mesh.npy')
    template_mesh = np.load('./template_mesh.npy')


if not os.path.exists('./template_mesh_in_pyrender.npy'):
    print(
        f'save template mesh (shape {model.module.template_mesh_cam_render.cpu().numpy().shape}) to ./template_mesh_in_pyrender.npy')
    np.save('./template_mesh_in_pyrender.npy', model.module.template_mesh_cam_render.cpu().numpy())


min_box_stride = 50

model.module.set_min_box_stride(min_box_stride)

image_list = glob.glob(os.path.join(img_dir, "*"))
for img_idx,img_path in enumerate(image_list):

    print(f'{img_idx}/{len(image_list)}',img_path)
    original_img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    if img_name not in pose2d_result or img_name not in head_bbox_result:
        raise ValueError('please generate 2d pose result and head bbox result for all images first!')
    # print(img_name)
    # debug
    # if img_name.split('.')[0] not in ['pexels-photo-15829424']:
    #     continue

    original_img_height, original_img_width = original_img.shape[:2]
    coco_joint_list = pose2d_result[img_name]
    head_bbox_list = head_bbox_result[img_name]
    if len(coco_joint_list) > 50:
        coco_joint_list = coco_joint_list[:50]
        head_bbox_list = head_bbox_list[:50]

    assert len(coco_joint_list) == len(head_bbox_list), 'len(coco_joint_list) != len(head_bbox_list)'

    drawn_joints = []
    c = coco_joint_list

    result_count = 0

    used_joints = []

    for idx in range(len(coco_joint_list)):

        image_name = os.path.basename(img_path).split('.')[0]
        file_name = f'{image_name}_{idx}.jpg'


        if f'{image_name}_{idx}.png' in result_json or f'{image_name}_{idx}_h.png' in result_json or f'{image_name}_{idx}_s.png' in result_json:
            result_count += 1
            continue


        image = original_img.copy()
        input = original_img.copy()
        input2 = original_img.copy()
        """ 2D pose input setting & hard-coding for filtering """
        pose_thr = 0.05
        coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]

        # if there is a similar joint in used_joints, skip this joint
        if len(used_joints) > 0:
            for joint in used_joints:
                #print(np.linalg.norm(joint - coco_joint_img)/ np.linalg.norm(coco_joint_img))
                distance = max(
                    max(coco_joint_img[:, 0])-min(coco_joint_img[:, 0]),
                    max(coco_joint_img[:, 1])-min(coco_joint_img[:, 1])
                )
                #print( np.linalg.norm(joint - coco_joint_img)/ distance)
                if np.linalg.norm(joint - coco_joint_img)/ distance < 0.15:
                    print('skip similar', np.linalg.norm(joint - coco_joint_img) / np.linalg.norm(coco_joint_img))
                    continue
        used_joints.append(coco_joint_img)

        coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
        coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
        coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)

        """head bbox"""
        head_bbox = head_bbox_list[idx]

        # if len(head_bbox)<4:
        # #     bad_vis = bad_image_vis(image, coco_joint_img.copy(), vis_skeleton)
        # #     cv2.imwrite(os.path.join(bad_images_dir, file_name), bad_vis)
        #     continue
        # filter inaccurate inputs
        det_score = sum(coco_joint_img[:, 2])
        if det_score < 0.3:
            print('skip low det score', det_score)
            continue
        if len(coco_joint_img[:, 2:].nonzero()[0]) < 1:
            print('skip no det score', det_score)
            continue
        # filter the same targets
        tmp_joint_img = coco_joint_img.copy()
        continue_check = False
        for ddx in range(len(drawn_joints)):
            drawn_joint_img = drawn_joints[ddx]
            drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
            diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
            diff = diff[diff != 0]
            if diff.size == 0:
                continue_check = True
            elif diff.mean() < 20:
                continue_check = True
        if continue_check:
            print('skip continue_check')
            # bad_vis = bad_image_vis(image, coco_joint_img.copy(), vis_skeleton)
            # cv2.imwrite(os.path.join(bad_images_dir, file_name), bad_vis)
            continue


        drawn_joints.append(tmp_joint_img)

        tmp_joint_img[-1], tmp_joint_img[-2] = tmp_joint_img[-2].copy(), tmp_joint_img[-1].copy()




        """ Prepare model input """
        # prepare bbox
        # bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0])  # xmin, ymin, width, height
        bbox = get_bbox(coco_joint_img, np.ones_like(coco_joint_valid[:, 0]))
        if bbox[2] < min_box_stride or bbox[3] < min_box_stride:
            print('skip too small bbox', bbox[2], bbox[3])
            continue
        orig_bbox = bbox.copy()
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        if bbox is None:
            print('skip invalid bbox')
            continue
        img, img2bb_trans, bb2img_trans = generate_patch_image(input2[:, :, ::-1], bbox, 1.0, 0.0, False,
                                                               cfg.input_img_shape)
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]

        coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)
        coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
        coco_joint_img[:, 0] = coco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        coco_joint_img[:, 1] = coco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

        coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)
        coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
        coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

        # check truncation
        coco_joint_trunc = coco_joint_valid * (
                (coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < cfg.output_hm_shape[2]) * (
                coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
            -1, 1).astype(np.float32)
        coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :, :], torch.from_numpy(
            coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]

        """ Model forward """
        inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}
        targets = {}
        meta_info = {'bbox': bbox}
        with torch.no_grad():
            out = model(inputs, targets, meta_info, 'test')


        #print("file name: ", file_name)

        bbox = out['bbox'][0].cpu().numpy()
        princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)



        color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

        intrisics_full_image_dict = {'focal': cfg.focal, 'princpt': princpt}
        camera_to_render_template_in_pyrender = out['camera_to_render_template_in_pyrender'].cpu().numpy()
        camera_pose_in_pyrender = out['camera_pose_in_pyrender'].cpu().numpy()



        if args.debug:
            viz = [original_img]

            image_mesh = render_mesh(image, out['mesh_cam_render'][0].cpu().numpy(), face,
                                     intrisics_full_image_dict,
                                     color=color)

            viz.append((image_mesh * alpha + original_img * (1.0 - alpha)).astype(np.uint8))

            image_template = render_mesh(image.copy(), model.module.template_mesh_cam_render[0].cpu().numpy(), face,
                                         intrisics_full_image_dict,
                                         color=color, cam_pose=camera_to_render_template_in_pyrender)
            viz.append((image_template * alpha + original_img * (1.0 - alpha)).astype(np.uint8))

            image_camera_rotate = render_mesh(image.copy(),model.module.template_mesh_cam_render[0].cpu().numpy(), face,
                                              intrisics_full_image_dict,
                                              color=color, cam_pose=camera_pose_in_pyrender)
            viz.append((image_camera_rotate * alpha + original_img * (1.0 - alpha)).astype(np.uint8))


            viz_full_image = np.concatenate(viz, axis=0 if original_img.shape[1] > original_img.shape[0] else 1)
            viz_full_image = cv2.resize(viz_full_image, (int( viz_full_image.shape[1] / viz_full_image.shape[0] * 853),853 ))







        crop_image_size = args.crop_image_size
        crop_outputs = model.module.crop_and_process_camera_matrix(out,
                                                                  original_img.copy(),
                                                                  joint_2d= tmp_joint_img, # used to realign
                                                                  crop_image_size=crop_image_size,
                                                                  model_input_bbox=bbox,
                                                                  head_bbox = head_bbox)
        model_input_bbox = bbox.copy()
        if len(crop_outputs) == 0:
            continue
        if len(crop_outputs) == 1:
            save_keys = [f'{image_name}_{idx}.png']
        else:
            save_keys = [f'{image_name}_{idx}_h.png' ,f'{image_name}_{idx}_s.png']


        for crop_idx in range(len(crop_outputs)):
            crop_output = crop_outputs[crop_idx]
            if crop_output is None:
                continue
            save_key = save_keys[crop_idx]
            intrisics_crop = np.eye(4)
            intrisics_crop[0, 0] = crop_output['intrisics']['focal'][0]
            intrisics_crop[1, 1] = crop_output['intrisics']['focal'][1]
            intrisics_crop[0, 2] =  crop_output['intrisics']['princpt'][0]
            intrisics_crop[1, 2] =  crop_output['intrisics']['princpt'][1]
            intrisics_crop_dict = {'focal': (intrisics_crop[0, 0], intrisics_crop[1, 1]),
                                   'princpt': [intrisics_crop[0, 2], intrisics_crop[1, 2]]}

            intrisics_standard = np.eye(4)
            intrisics_standard[0, 0] = cfg.focal[0]
            intrisics_standard[1, 1] = cfg.focal[1]
            intrisics_standard[0, 2] = crop_image_size / 2
            intrisics_standard[1, 2] = crop_image_size / 2
            intrisics_standard_dict = {'focal': cfg.focal, 'princpt': [crop_image_size / 2, crop_image_size / 2]}

            normalized_camerapose_in_pyrender = out['normalized_camerapose_in_pyrender']
            normalized_transformation_in_realworld = out['normalized_transformation_in_realworld']
            camerapose_in_realworld = np.linalg.inv(normalized_transformation_in_realworld)


            # realign image


            viz = [crop_output['cropped_image']]

            # image_mesh = render_mesh(crop_output['cropped_image'],
            #                          out['mesh_cam_render'][0].cpu().numpy(), face,
            #                         intrisics_crop_dict,
            #                         color=color)
            image_mesh = render_mesh(crop_output['cropped_image'].copy(),out['neck_head_rotated_template_mesh'][0].cpu().numpy(), face,
                                     intrisics_standard_dict,
                                     color=color,
                                     cam_pose=normalized_camerapose_in_pyrender)

            #image_mesh,_,_ = generate_patch_image(image_mesh, crop_output['bbox'], 1.0, 0.0, False, (crop_image_size,crop_image_size))
            viz.append((image_mesh * alpha + crop_output['cropped_image'] * (1.0-alpha)).astype(np.uint8))

            # image_template = render_mesh(crop_output['cropped_image'].copy(),
            #                              model.module.template_mesh_cam_render[0].cpu().numpy(), face,
            #                              intrisics_crop_dict,
            #                              color=color,
            #                              cam_pose=camera_to_render_template_in_pyrender)
            # viz.append((image_template * alpha + crop_output['cropped_image'] * (1.0-alpha)).astype(np.uint8))

            # image_camera_rotate = render_mesh(crop_output['cropped_image'].copy(),
            #                                   model.module.template_mesh_cam_render[0].cpu().numpy(), face,
            #                                   intrisics_standard_dict,
            #                                   color=color,
            #                                   cam_pose=normalized_camerapose_in_pyrender)
            # viz.append((image_camera_rotate * alpha + crop_output['cropped_image'] * (1.0-alpha)).astype(np.uint8))


            if args.debug:
                projected_vertexes = model.module.get_projected_vertex(torch.from_numpy(template_mesh).float().cuda(),
                                                                       intrisics_standard @ normalized_transformation_in_realworld)

                vertex_vis = crop_output['cropped_image'].copy()


                camera_forward_direction = (camerapose_in_realworld[:3, :3] @ np.reshape(np.array([0, 0, 1]), (3, 1)))[:, 0]  # 3,1
                camera_position = camerapose_in_realworld[:3, 3]  # 3,1

                not_pass_check = 0
                in_screen = 0
                for i in range(projected_vertexes.shape[0]):
                    if projected_vertexes[i, 0] < 0 or projected_vertexes[i, 0] >= vertex_vis.shape[1] or \
                            projected_vertexes[i, 1] < 0 or projected_vertexes[i, 1] >= vertex_vis.shape[0]:
                        continue
                    # print(template_mesh[0, i, :].shape, camera_position.shape, camera_forward_direction.shape)
                    check = np.sum((template_mesh[0, i, :] - camera_position) * camera_forward_direction)
                    in_screen += 1
                    if check < 0:
                        not_pass_check += 1
                    cv2.circle(vertex_vis, (int(projected_vertexes[i, 0]), int(projected_vertexes[i, 1])), 5, (255, 255, 255), -1)

                viz.append(vertex_vis)

                if not_pass_check == in_screen:
                    raise Exception('all vertexes are before camera')



            # tmp_joint_img 19 x 2
            # rescale tmp_joint_img accroding to bbox
            tmp_joint_img_on_croppped_image =  tmp_joint_img.copy()
            tmp_joint_img_on_croppped_image[:, 0] = tmp_joint_img[:, 0] - crop_output['bbox'][0]
            tmp_joint_img_on_croppped_image[:, 1] = tmp_joint_img[:, 1] - crop_output['bbox'][1]
            tmp_joint_img_on_croppped_image*=  crop_image_size/crop_output['bbox'][2]

            skeleton_vis = vis_coco_skeleton(crop_output['cropped_image'].copy(), tmp_joint_img_on_croppped_image.T, vis_skeleton)
            if len(head_bbox['bbox']) ==4 and crop_idx == 0:
                tmp_head_bbox = np.array(head_bbox['bbox'].copy())
                tmp_head_bbox[0] = head_bbox['bbox'][0] - crop_output['bbox'][0]
                tmp_head_bbox[1] = head_bbox['bbox'][1] - crop_output['bbox'][1]
                tmp_head_bbox *= crop_image_size / crop_output['bbox'][2]
                cv2.rectangle(skeleton_vis, (int(tmp_head_bbox[0]), int(tmp_head_bbox[1])),
                              (int(tmp_head_bbox[0] + tmp_head_bbox[2]), int(tmp_head_bbox[1] + tmp_head_bbox[3])),
                              (0, 255, 0), 4)

            viz.append(skeleton_vis)

            viz = np.concatenate(viz,  axis=0 )
            if args.debug:
                viz =  cv2.resize(viz, (int(viz.shape[1]/viz.shape[0] * viz_full_image.shape[0]), viz_full_image.shape[0]))
                viz = np.concatenate([viz_full_image, viz], axis=1)
            else:
                viz = cv2.resize(viz, (viz.shape[1]//6,viz.shape[0]//6))

            cv2.imwrite(os.path.join(visualization_dir, save_key),viz)

            #'''

            cv2.imwrite(os.path.join(aligned_images_dir,save_key), crop_output['cropped_image'])

            # final =========================================
            res = {
                'bbox':crop_output['bbox'].tolist(),

                'coco_joint': tmp_joint_img.tolist(),
                'model_input_bbox': model_input_bbox.tolist(),
                'raw_image_name': img_name,

                # real world
                'intrisics': intrisics_standard.tolist(),
                'intrisics_dict': intrisics_standard_dict,
                'world2camera_matrix': normalized_transformation_in_realworld.tolist(),
                'camera_pose': camerapose_in_realworld.tolist(),


                # pyrender
                    # original
                    'intrisics_full_image_dict': intrisics_full_image_dict,
                    'camera_to_render_template_in_pyrender':camera_to_render_template_in_pyrender.tolist(),
                    'camera_pose_in_pyrender':camera_pose_in_pyrender.tolist(),

                    #crop
                    'intrisics_crop_dict': intrisics_crop_dict,
                    'normalized_camerapose_in_pyrender': normalized_camerapose_in_pyrender.tolist(),



                # smpl
                'smpl_pose': out['smpl_pose'].cpu().numpy().tolist(),
                'smpl_shape': out['smpl_shape'].cpu().numpy().tolist(),
                'cam_trans': out['cam_trans'].cpu().numpy().tolist(),


            }

            result_json[save_key] = res

            result_count += 1

    if result_count == 0:
        print(f">>>>>>> No result in {img_path}!")
        shutil.move(img_path, os.path.join(bad_images_dir, os.path.basename(img_path)))
        # ==============================================


with open(result_json_path, 'w') as f:
    json.dump(result_json, f)
