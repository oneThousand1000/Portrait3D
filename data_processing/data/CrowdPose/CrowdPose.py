import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import scipy.io as sio
import cv2
import random
import math
import torch
import transforms3d
from pycocotools.coco import COCO

from utils.posefix import replace_joint_img
from utils.smpl import SMPL
from utils.preprocessing import load_img, process_bbox, augmentation, compute_iou, get_bbox
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton
from utils.transforms import world2cam, cam2pixel, pixel2cam, transform_joint_to_other_db


class CrowdPose(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join(cfg.data_dir, 'CrowdPose', 'images')
        self.annot_path = osp.join(cfg.data_dir, 'CrowdPose', 'annotations')
        self.target_data_split = 'val'
        self.fitting_thr = 5.0  # pixel in cfg.output_hm_shape space

        # mscoco skeleton
        self.coco_joint_num = 18  # original: 17, manually added pelvis
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis')
        self.coco_skeleton = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_joint_regressor = np.load(osp.join(cfg.data_dir, 'MSCOCO', 'J_regressor_coco_hip_smpl.npy'))

        # crowdpose skeleton
        self.crowdpose_jonit_num = 14+1  # manually added pelvis
        self.crowdpose_joints_name = ('L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Head_top', 'Neck', 'Pelvis')
        self.crowdpose_skeleton = ((0,2), (0,13),  (1,3), (1,13), (2,4), (3,5), (6,14), (7,14), (6,8), (7,9), (8,10), (9,11), (12,13), (13,14) )
        self.crowdpose_flip_pairs = ((0, 1), (1, 2), (3, 4), (5, 6), (6, 7), (8, 9), (10, 11))
        self.crowdpose_coco_common_jidx = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14)  # for posefix, exclude pelvis

        # smpl skeleton
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.flip_pairs = self.smpl.flip_pairs
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        self.datalist = self.load_data()
        print("crowdpose data len: ", len(self.datalist))

    def add_pelvis(self, joint_coord):
        lhip_idx = self.crowdpose_joints_name.index('L_Hip')
        rhip_idx = self.crowdpose_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # joint_valid
        pelvis = pelvis.reshape(1, 3)
        joint_coord = np.concatenate((joint_coord, pelvis))
        return joint_coord

    def load_data(self):
        datalist = []
        if self.data_split == 'train':
            split_list = ['train'] if self.data_split == 'train' else [self.target_data_split]

            datalist = []
            for split in split_list:
                db = COCO(osp.join(self.annot_path, f'crowdpose_{split}.json'))
                # smpl parameter load
                with open(osp.join(self.annot_path, f'CrowdPose_{split}_SMPL_NeuralAnnot.json'), 'r') as f:
                    smpl_params = json.load(f)

                for iid in db.imgs.keys():
                    aids = db.getAnnIds([iid])

                    tmplist = []
                    for aid in aids:
                        ann = db.anns[aid]
                        img = db.loadImgs(ann['image_id'])[0]
                        img_path = osp.join(self.img_path, img['file_name'])
                        # bbox
                        if split != 'val':  # correct reversed img width,height info
                            width, height = img['height'], img['width']
                        else:
                            width, height = img['width'], img['height']

                        if sum(ann['keypoints']) == 0:
                            continue

                        # bbox
                        # tight_bbox = np.array(ann['bbox'])
                        # bbox = process_bbox(tight_bbox, width, height)
                        # if bbox is None: continue

                        # joint coordinates
                        joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                        joint_img = self.add_pelvis(joint_img)
                        joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                        joint_img[:, 2] = joint_valid[:, 0]  # for posefix, only good for 2d datasets

                        # bbox
                        if cfg.use_bbox_in_ann:
                            tight_bbox = np.array(ann['bbox'])
                        else:
                            tight_bbox = get_bbox(joint_img, np.ones_like(joint_img[:, 0]), crop_bottom_body=True)
                        # bbox = process_bbox(tight_bbox, width, height)
                        # if bbox is None: continue

                        if str(aid) in smpl_params:
                            smpl_param = smpl_params[str(aid)]
                            if smpl_param['fit_err'] < self.fitting_thr:
                                smpl_param = None
                        else:
                            smpl_param = None

                        tmplist.append({'img_path': img_path,
                                        'img_shape': (height, width),
                                        #'bbox': bbox,
                                        'tight_bbox': tight_bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'neural_annot_result': smpl_param})

                    for i, person in enumerate(tmplist):
                        tight_bbox = person['tight_bbox']

                        # for swap
                        num_overlap = 0
                        near_joints = []
                        other_persons = tmplist[:i] + tmplist[i + 1:]
                        for other in other_persons:
                            other_bbox = other['tight_bbox']
                            iou = compute_iou(tight_bbox[None, :], other_bbox[None, :])
                            if iou < 0.1:
                                continue
                            num_overlap += 1
                            other_joint = transform_joint_to_other_db(other['joint_img'], self.crowdpose_joints_name, self.coco_joints_name)
                            near_joints.append(other_joint)

                        person['num_overlap'] = num_overlap
                        person['near_joints'] = near_joints

                    datalist.extend(tmplist)

        return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(1, -1);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # flip smpl pose parameter (axis-angle)
        if do_flip:
            smpl_pose = smpl_pose.view(-1, 3)
            for pair in self.flip_pairs:
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose):  # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                    smpl_pose[pair[0], :], smpl_pose[pair[1], :] = smpl_pose[pair[1], :].clone(), smpl_pose[pair[0], :].clone()
            smpl_pose[:, 1:3] *= -1;  # multiply -1 to y and z axis of axis-angle
            smpl_pose = smpl_pose.view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer['neutral'](smpl_pose, smpl_shape, smpl_trans)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        # smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
        # smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
        # smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))
        smpl_joint_coord = np.dot(self.joint_regressor, smpl_mesh_coord)

        # flip translation
        if do_flip:  # avg of old and new root joint should be image center.
            focal, princpt = cam_param['focal'], cam_param['princpt']
            flip_trans_x = 2 * (((img_shape[1] - 1) / 2. - princpt[0]) / focal[0] * (smpl_joint_coord[self.root_joint_idx, 2])) - 2 * smpl_joint_coord[self.root_joint_idx][0]
            smpl_mesh_coord[:, 0] += flip_trans_x
            smpl_joint_coord[:, 0] += flip_trans_x

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, tight_bbox = data['img_path'], data['img_shape'], data['tight_bbox']
        # check if image is full body
        # self.crowdpose_jonit_num = 14+1  # manually added pelvis
        # self.crowdpose_joints_name = ('L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Head_top', 'Neck', 'Pelvis')
        # data['joint_valid'].shape = (15, 1)
        is_full_body = np.sum(
            data['joint_valid'][6:12, :]) > 3  # 6:12 is the index of L_Hip, R_Hip, L_Knee, R_Knee, L_Ankle, R_Ankle

        # image load and affine transform
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip,bbox = augmentation(img, tight_bbox, self.data_split,is_full_body = is_full_body)
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == 'train':
            # coco gt
            crowdpose_joint_img = data['joint_img']
            crowdpose_joint_valid = data['joint_valid']
            if do_flip:
                crowdpose_joint_img[:, 0] = img_shape[1] - 1 - crowdpose_joint_img[:, 0]
                for pair in self.crowdpose_flip_pairs:
                    crowdpose_joint_img[pair[0], :], crowdpose_joint_img[pair[1], :] = crowdpose_joint_img[pair[1], :].copy(), crowdpose_joint_img[pair[0], :].copy()
                    crowdpose_joint_valid[pair[0], :], crowdpose_joint_valid[pair[1], :] = crowdpose_joint_valid[pair[1], :].copy(), crowdpose_joint_valid[pair[0], :].copy()

            crowdpose_joint_img_xy1 = np.concatenate((crowdpose_joint_img[:, :2], np.ones_like(crowdpose_joint_img[:, :1])), 1)
            crowdpose_joint_img[:, :2] = np.dot(img2bb_trans, crowdpose_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
            # for swap
            if len(data['near_joints']) > 0:
                near_joint_list = []
                for nj in data['near_joints']:
                    near_joint = np.ones((self.coco_joint_num, 3), dtype=np.float32)
                    nj_xy1 = np.concatenate((nj[:, :2], np.ones_like(nj[:, :1])), axis=1)
                    near_joint[:, :2] = np.dot(img2bb_trans, nj_xy1.transpose(1, 0)).transpose(1, 0)
                    near_joint_list.append(near_joint)
                near_joints = np.asarray(near_joint_list, dtype=np.float32)
            else:
                near_joints = np.zeros((1, self.coco_joint_num, 3), dtype=np.float32)

            input_crowdpose_joint_img = crowdpose_joint_img.copy()
            crowdpose_joint_img[:, 0] = crowdpose_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            crowdpose_joint_img[:, 1] = crowdpose_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # check truncation
            crowdpose_joint_trunc = crowdpose_joint_valid * ((crowdpose_joint_img[:, 0] >= 0) * (crowdpose_joint_img[:, 0] < cfg.output_hm_shape[2]) * (crowdpose_joint_img[:, 1] >= 0) * (crowdpose_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
                -1, 1).astype(np.float32)

            # transform coco joints to target db joints
            crowdpose_joint_img = transform_joint_to_other_db(crowdpose_joint_img, self.crowdpose_joints_name, self.joints_name)
            crowdpose_joint_cam = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
            crowdpose_joint_valid = transform_joint_to_other_db(crowdpose_joint_valid, self.crowdpose_joints_name, self.joints_name)
            crowdpose_joint_trunc = transform_joint_to_other_db(crowdpose_joint_trunc, self.crowdpose_joints_name, self.joints_name)

            # apply PoseFix
            tmp_joint_img = transform_joint_to_other_db(input_crowdpose_joint_img, self.crowdpose_joints_name, self.coco_joints_name)
            tmp_joint_img = replace_joint_img(tmp_joint_img, data['tight_bbox'], near_joints, data['num_overlap'], img2bb_trans)
            tmp_joint_img = transform_joint_to_other_db(tmp_joint_img, self.coco_joints_name, self.crowdpose_joints_name)
            input_crowdpose_joint_img[self.crowdpose_coco_common_jidx, :2] = tmp_joint_img[self.crowdpose_coco_common_jidx, :2]
            """
            # debug PoseFix result
            newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), input_crowdpose_joint_img.T, self.crowdpose_skeleton)
            cv2.imshow(f'{img_path}', newimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            # import pdb; pdb.set_trace()
            """
            input_crowdpose_joint_img[:, 0] = input_crowdpose_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            input_crowdpose_joint_img[:, 1] = input_crowdpose_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            input_crowdpose_joint_img = transform_joint_to_other_db(input_crowdpose_joint_img, self.crowdpose_joints_name, self.joints_name)

            neural_annot_result = data['neural_annot_result']
            if neural_annot_result is not None:
                # use fitted mesh
                smpl_param, cam_param = neural_annot_result['smpl_param'], neural_annot_result['cam_param']
                smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
                smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
                smpl_coord_img = cam2pixel(smpl_coord_cam, cam_param['focal'], cam_param['princpt'])

                # x,y affine transform, root-relative depth
                smpl_coord_img_xy1 = np.concatenate((smpl_coord_img[:, :2], np.ones_like(smpl_coord_img[:, 0:1])), 1)
                smpl_coord_img[:, :2] = np.dot(img2bb_trans, smpl_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                """
                # vis smpl
                newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), smpl_coord_img[6890:].T, self.skeleton)
                cv2.imshow(f'{img_path}', newimg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                """
                smpl_coord_img[:, 2] = smpl_coord_img[:, 2] - smpl_coord_cam[self.vertex_num + self.root_joint_idx][2]
                smpl_coord_img[:, 0] = smpl_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
                smpl_coord_img[:, 1] = smpl_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
                smpl_coord_img[:, 2] = (smpl_coord_img[:, 2] / (cfg.bbox_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]

                # check truncation
                smpl_trunc = (
                            (smpl_coord_img[:, 0] >= 0) * (smpl_coord_img[:, 0] < cfg.output_hm_shape[2]) * (smpl_coord_img[:, 1] >= 0) * (smpl_coord_img[:, 1] < cfg.output_hm_shape[1]) * (smpl_coord_img[:, 2] >= 0) * (
                                smpl_coord_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1, 1).astype(np.float32)

                # split mesh and joint coordinates
                smpl_mesh_img = smpl_coord_img[:self.vertex_num];
                smpl_joint_img = smpl_coord_img[self.vertex_num:];
                smpl_mesh_trunc = smpl_trunc[:self.vertex_num];
                smpl_joint_trunc = smpl_trunc[self.vertex_num:];

                # already checked in load_data()
                is_valid_fit = True

            else:
                smpl_joint_img = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
                smpl_joint_cam = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
                smpl_mesh_img = np.zeros((self.vertex_num, 3), dtype=np.float32)  # dummy
                smpl_pose = np.zeros((72), dtype=np.float32)  # dummy
                smpl_shape = np.zeros((10), dtype=np.float32)  # dummy
                smpl_joint_trunc = np.zeros((self.joint_num, 1), dtype=np.float32)
                smpl_mesh_trunc = np.zeros((self.vertex_num, 1), dtype=np.float32)
                is_valid_fit = False

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32)
            # parameter
            smpl_pose = smpl_pose.reshape(-1, 3)
            root_pose = smpl_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            smpl_pose[self.root_joint_idx] = root_pose.reshape(3)
            smpl_pose = smpl_pose.reshape(-1)
            # smpl coordinate
            smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[self.root_joint_idx, None]  # root-relative
            smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1, 0)).transpose(1, 0)

            # SMPL pose parameter validity
            smpl_param_valid = np.ones((self.smpl.orig_joint_num, 3), dtype=np.float32)
            for name in ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'):
                smpl_param_valid[self.joints_name.index(name)] = 0
            smpl_param_valid = smpl_param_valid.reshape(-1)

            inputs = {'img': img, 'joints': input_crowdpose_joint_img[:, :2], 'joints_mask': crowdpose_joint_trunc}
            targets = {'orig_joint_img': crowdpose_joint_img, 'fit_joint_img': smpl_joint_img, 'orig_joint_cam': crowdpose_joint_cam, 'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
            meta_info = {'orig_joint_valid': crowdpose_joint_valid, 'orig_joint_trunc': crowdpose_joint_trunc, 'fit_param_valid': smpl_param_valid, 'fit_joint_trunc': smpl_joint_trunc, 'is_valid_fit': float(is_valid_fit),'bbox': bbox,
                         'is_3D': float(False)}
            return inputs, targets, meta_info

