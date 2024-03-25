import torch
import copy
import os
import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
from config import cfg
import json
import cv2
import random
import math

from utils.smpl import SMPL
from utils.transforms import pixel2cam, transform_joint_to_other_db, cam2pixel
from utils.preprocessing import load_img, augmentation, process_bbox, get_bbox
from utils.vis import vis_keypoints, vis_3d_skeleton, vis_keypoints_with_skeleton


class MuPoTs(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'MuPoTs', 'data', 'MultiPersonTestSet')
        self.test_annot_path = osp.join(cfg.data_dir, 'MuPoTs', 'data', 'MuPoTS-3D.json')
        self.hhrnet_result_path = osp.join(cfg.data_dir, 'MuPoTs', 'data', 'MuPoTs_test_hhrnet_result.json')
        self.hhrnet_thr = 0.1
        self.openpose_result_path = osp.join(cfg.data_dir, 'MuPoTs', 'data', 'MuPoTs_test_openpose_result.json')
        self.openpose_thr = 0.05

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        # MuCo-3DHP
        self.muco_joint_num = 21
        self.muco_joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe',
        'L_Toe')

        # MuPoTS
        self.mupots_joint_num = 17
        self.mupots_joints_name = (
        'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head')  #
        self.mupots_flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
        self.mupots_skeleton = (
        (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3),
        (3, 4), (1, 5), (5, 6), (6, 7))
        self.mupots_eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.mupots_root_idx = self.mupots_joints_name.index('Pelvis')

        # H36M joint set
        # Spine Thorax, Head
        self.h36m_joint_num = 17
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Spine', 'Thorax', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.h36m_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        # self.h36m_joint_regressor = np.load(osp.join('..', 'data', 'Human36M', 'J_regressor_h36m_from_pav.npy')) #'J_regressor_h36m_correct.npy'))
        # self.h36m_pav_joints_name = ('Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Hip', 'R_Knee', 'R_Ankle', 'Spine', 'Thorax', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        # self.h36m_joint_regressor = transform_joint_to_other_db(self.h36m_joint_regressor, self.h36m_pav_joints_name, self.h36m_joints_name)

        # MPI-INF-3DHP joint set
        self.mpii3d_joint_num = 17
        self.mpii3d_joints_name = (
                               'Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow',
                               'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine',
                               'Head')
        self.mpii3d_flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
        self.mpii3d_smpl_regressor = np.load(osp.join(cfg.data_dir, 'MPI_INF_3DHP', 'J_regressor_mi_smpl.npy'))[:17]
        self.mpii3d_root_idx = self.mpii3d_joints_name.index('Pelvis')

        # MSCOCO joint set
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        # OpenPose joint set
        self.openpose_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear', 'Pelvis')

        self.datalist = self.load_data()
        print('mupots data len: ', len(self.datalist))

    def add_pelvis(self, joint_coord, joints_name):
        lhip_idx = joints_name.index('L_Hip')
        rhip_idx = joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2]  # confidence for openpose
        pelvis = pelvis.reshape(1, 3)

        joint_coord = np.concatenate((joint_coord, pelvis))#, neck))

        return joint_coord

    def add_neck(self, joint_coord, joints_name):
        lshoulder_idx = joints_name.index('L_Shoulder')
        rshoulder_idx = joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
        neck = neck.reshape(1,3)

        joint_coord = np.concatenate((joint_coord, neck))

        return joint_coord

    def load_data(self):
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0

        with open(self.hhrnet_result_path) as f:
            hhrnet_result = json.load(f)
        with open(self.openpose_result_path) as f:
            openpose_result = json.load(f)

        data = []
        db = COCO(self.test_annot_path)

        count_dummy = 0
        # use gt bbox and root
        print("Get bounding box and root from groundtruth")
        for aid in db.anns.keys():
            ann = db.anns[aid]
            if ann['is_valid'] == 0:
                continue

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            fx, fy, cx, cy = img['intrinsic']
            f = np.array([fx, fy]);
            c = np.array([cx, cy]);

            joint_cam = np.array(ann['keypoints_cam'])
            root_cam = joint_cam[self.mupots_root_idx]

            joint_img = np.array(ann['keypoints_img'])
            joint_img = np.concatenate([joint_img, joint_cam[:, 2:]], 1)
            joint_img[:, 2] = joint_img[:, 2] - root_cam[2]
            joint_valid = np.ones((self.mupots_joint_num, 1))

            hhrnetpose = np.array(hhrnet_result[str(aid)]['coco_joints'])
            hhrnetpose = self.add_pelvis(hhrnetpose, self.coco_joints_name)
            hhrnetpose = self.add_neck(hhrnetpose, self.coco_joints_name)

            openpose = np.array(openpose_result[str(aid)]['coco_joints'])
            openpose = self.add_pelvis(openpose, self.openpose_joints_name)

            if openpose.sum() == 0:
                count_dummy += 1
            bbox = np.array(ann['bbox'])
            img_width, img_height = img['width'], img['height']
            # bbox = process_bbox(bbox, img_width, img_height)
            # if bbox is None: continue

            data.append({
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'tight_bbox': np.array(ann['bbox']),
                'joint_img': joint_img,  # [org_img_x, org_img_y, depth - root_depth]
                'joint_cam': joint_cam,  # [X, Y, Z] in camera coordinate
                'joint_valid': joint_valid,
                'root_cam': root_cam,  # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c,
                'hhrnetpose': hhrnetpose,
                'openpose': openpose
            })

        print("dummy predictions: ", count_dummy)
        return data

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path = data['img_path']

        input_joint_name = 'openpose'
        if input_joint_name == 'gt':
            joint_coord_img = data['joint_img']
            joint_coord_img[:, 2] = 1
            joint_valid = data['joint_valid']
            joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.mupots_joints_name, self.joints_name)
            joint_valid = transform_joint_to_other_db(joint_valid, self.mupots_joints_name, self.joints_name)
        elif input_joint_name == 'hhrnet':
            joint_coord_img = data['hhrnetpose']
            joint_valid = (joint_coord_img[:, 2:] > self.hhrnet_thr)
            joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.coco_joints_name, self.joints_name)
            joint_valid = transform_joint_to_other_db(joint_valid, self.coco_joints_name, self.joints_name)
        elif input_joint_name == 'openpose':
            joint_coord_img = data['openpose']
            joint_valid = (joint_coord_img[:, 2:] > self.openpose_thr)
            joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.openpose_joints_name, self.joints_name)
            joint_valid = transform_joint_to_other_db(joint_valid, self.openpose_joints_name, self.joints_name)

        # get bbox from joints
        try:
            bbox = get_bbox(joint_coord_img, joint_valid[:, 0])
        except:  # in case of perfect occlusion
            bbox = data['bbox']
        img_height, img_width = data['img_shape']
        bbox = process_bbox(bbox.copy(), img_width, img_height, is_3dpw_test=True)

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, _, _, _ = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        """
        # debug
        img = cv2.imread(img_path)
        input_img = vis_keypoints_with_skeleton(img, joint_coord_img.T, self.skeleton, kp_thresh=0.1, alpha=1, kps_scores=joint_coord_img[:, 2:].round(3))
        cv2.imshow('mupots', input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # import pdb;
        # pdb.set_trace()
        """

        # x,y affine transform, root-relative depth
        joint_coord_img_xy1 = np.concatenate((joint_coord_img[:, :2], np.ones_like(joint_coord_img[:, 0:1])), 1)
        joint_coord_img[:, :2] = np.dot(img2bb_trans, joint_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        joint_coord_img[:, 0] = joint_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        joint_coord_img[:, 1] = joint_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

        # check truncation
        joints_mask = joint_valid * (
                (joint_coord_img[:, 0] >= 0) * (joint_coord_img[:, 0] < cfg.output_hm_shape[2]) * (joint_coord_img[:, 1] >= 0) * (joint_coord_img[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(np.float32)

        inputs = {'img': img, 'joints': joint_coord_img, 'joints_mask': joints_mask}
        targets = {}
        meta_info = {'bbox': bbox,
                     'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        gts = self.datalist
        sample_num = len(outs)
        joint_num = self.mupots_joint_num

        pred_2d_save = {}
        pred_3d_save = {}
        for n in range(sample_num):
            gt = gts[cur_sample_idx+n]
            f = gt['f']
            c = gt['c']
            gt_3d_root = gt['root_cam']
            img_name = gt['img_path'].split('/')
            img_name = img_name[-2] + '_' + img_name[-1].split('.')[0]  # e.g., TS1_img_0001

            # h36m joint from output mesh
            out = outs[n]
            mesh_out_cam = out['smpl_mesh_cam'] * 1000
            pred = np.dot(self.mpii3d_smpl_regressor, mesh_out_cam)
            pred = pred - pred[self.mpii3d_root_idx, None]  # root-relative
            pred_3d_kpt = transform_joint_to_other_db(pred, self.mpii3d_joints_name, self.mupots_joints_name)
            pred_3d_kpt += gt_3d_root

            pred_3d_save.setdefault(img_name + '_3d', []).append(pred_3d_kpt)

            pred_2d_kpt = cam2pixel(pred_3d_kpt, f, c)
            pred_2d_save.setdefault(img_name + '_2d', []).append(pred_2d_kpt[:, :2])

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1, 500))

                pred_2d_kpt[:, 2] = 1
                # tmpimg = vis_keypoints(cvimg, pred_2d_kpt, alpha=1)
                tmpimg = vis_keypoints_with_skeleton(cvimg, pred_2d_kpt.T, self.mupots_skeleton, kp_thresh=0.1, alpha=1)
                # cv2.imwrite(filename + '_output.jpg', tmpimg)
                cv2.imshow('mupots', tmpimg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                import pdb; pdb.set_trace()

        eval_result = {**pred_2d_save, **pred_3d_save}
        return eval_result

    def print_eval_result(self, eval_result):
        pred_2d_save = {}
        pred_3d_save = {}

        for k, v in eval_result.items():
            if '2d' in k:
                key = k.split('_2d')[0]
                pred_2d_save[key] = v
            elif '3d' in k:
                key = k.split('_3d')[0]
                pred_3d_save[key] = v

        result_dir = osp.join(cfg.result_dir, 'MuPoTs')
        output_path = osp.join(result_dir, 'preds_2d_kpt_mupots.mat')
        sio.savemat(output_path, pred_2d_save)
        print("Testing result is saved at " + output_path)
        output_path = osp.join(result_dir, 'preds_3d_kpt_mupots.mat')
        sio.savemat(output_path, pred_3d_save)
        print("Testing result is saved at " + output_path)