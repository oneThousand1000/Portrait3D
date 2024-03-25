import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import pickle
import transforms3d
from pycocotools.coco import COCO
from config import cfg
from utils.posefix import replace_joint_img
from utils.smpl import SMPL
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, compute_iou
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, vis_bbox
import transforms3d


class MuCo(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'MuCo', 'data')
        self.annot_path = osp.join(cfg.data_dir, 'MuCo', 'data', 'MuCo-3DHP.json')
        self.smpl_param_path = osp.join(cfg.data_dir, 'MuCo', 'data', 'smpl_param.json')
        self.fitting_thr = 25 # milimeter

        # COCO joint set
        self.coco_joint_num = 17  # original: 17
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')

        # MuCo joint set
        self.muco_joint_num = 21
        self.muco_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.muco_flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        self.muco_skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        self.muco_root_joint_idx = self.muco_joints_name.index('Pelvis')
        self.muco_coco_common_jidx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

        # H36M joint set
        self.h36m_joint_regressor = np.load(osp.join(cfg.data_dir, 'Human36M', 'J_regressor_h36m_correct.npy')) # use h36m joint regrssor (only use subset from original muco joint set)
        self.h36m_flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')

        # SMPL joint set
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
        print("muco data len: ", len(self.datalist))

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(self.annot_path)
            with open(self.smpl_param_path) as f:
                smpl_params = json.load(f)
        else:
            print('Unknown data subset')
            assert 0

        datalist = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            img_path = osp.join(self.img_dir, imgname)
            focal = img["f"]
            princpt = img["c"]
            cam_param = {'focal': focal, 'princpt': princpt}

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)

            root_depths = [ann['keypoints_cam'][self.muco_root_joint_idx][2] for ann in anns]
            closest_pid = root_depths.index(min(root_depths))
            pid_list = [closest_pid]
            for pid in pid_list:
                joint_cam = np.array(anns[pid]['keypoints_cam'])
                joint_img = np.array(anns[pid]['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_valid = np.ones((self.muco_joint_num,1))

                if cfg.use_bbox_in_ann:
                    tight_bbox = np.array(anns[pid]['bbox'])
                else:
                    tight_bbox = get_bbox(joint_img, np.ones_like(joint_img[:, 0]), crop_bottom_body=True)

                # for swap
                num_overlap = 0
                near_joints = []
                other_persons = anns[:pid] + anns[pid+1:]
                for other in other_persons:
                    other_tight_bbox = np.array(other['bbox'])
                    iou = compute_iou(tight_bbox[None, :], other_tight_bbox[None, :])
                    if iou < 0.1:
                        continue
                    num_overlap += 1
                    other_joint = np.array(other['keypoints_img'])
                    other_joint = np.concatenate((other_joint, np.ones_like(other_joint[:, :1])), axis=1)
                    other_joint = transform_joint_to_other_db(other_joint, self.muco_joints_name, self.coco_joints_name)
                    near_joints.append(other_joint)
                if num_overlap == 0:
                    near_joints = []

                # bbox = process_bbox(tight_bbox, img_width, img_height)
                # if bbox is None: continue
                
                # check smpl parameter exist
                try:
                    smpl_param = smpl_params[str(ann_ids[pid])]
                except KeyError:
                    smpl_param = None

                datalist.append({
                    'img_path': img_path,
                    'img_shape': (img_height, img_width),
                    #'bbox': bbox,
                    'tight_bbox': tight_bbox,
                    'joint_img': joint_img,
                    'joint_cam': joint_cam, 
                    'joint_valid': joint_valid,
                    'cam_param': cam_param,
                    'smpl_param': smpl_param,
                    'near_joints': near_joints,
                    'num_overlap': num_overlap
                })

        return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(1,-1); smpl_shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1,3) # translation vector from smpl coordinate to muco world coordinate
       
        # flip smpl pose parameter (axis-angle)
        if do_flip:
            smpl_pose = smpl_pose.view(-1,3)
            for pair in self.flip_pairs:
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose): # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                    smpl_pose[pair[0], :], smpl_pose[pair[1], :] = smpl_pose[pair[1], :].clone(), smpl_pose[pair[0], :].clone()
            smpl_pose[:,1:3] *= -1; # multiply -1 to y and z axis of axis-angle
            smpl_pose = smpl_pose.view(1,-1)
        
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer['neutral'](smpl_pose, smpl_shape, smpl_trans)
       
        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3); 
        # smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
        # smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
        # smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))
        smpl_joint_coord = np.dot(self.joint_regressor, smpl_mesh_coord)

        # flip translation
        if do_flip: # avg of old and new root joint should be image center.
            focal, princpt = cam_param['focal'], cam_param['princpt']
            flip_trans_x = 2 * (((img_shape[1] - 1)/2. - princpt[0]) / focal[0] * (smpl_joint_coord[self.root_joint_idx,2] * 1000)) / 1000 - 2 * smpl_joint_coord[self.root_joint_idx][0]
            smpl_mesh_coord[:,0] += flip_trans_x
            smpl_joint_coord[:,0] += flip_trans_x

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # meter -> milimeter
        smpl_mesh_coord *= 1000; smpl_joint_coord *= 1000;
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy()

    def get_fitting_error(self, muco_joint, smpl_mesh, do_flip):
        muco_joint = muco_joint.copy()
        muco_joint = muco_joint - muco_joint[self.muco_root_joint_idx,None,:] # root-relative
        if do_flip:
            muco_joint[:,0] = -muco_joint[:,0]
            for pair in self.muco_flip_pairs:
                muco_joint[pair[0],:] , muco_joint[pair[1],:] = muco_joint[pair[1],:].copy(), muco_joint[pair[0],:].copy()
        muco_joint_valid = np.ones((self.muco_joint_num,3), dtype=np.float32)
      
        # transform to h36m joint set
        h36m_joint = transform_joint_to_other_db(muco_joint, self.muco_joints_name, self.h36m_joints_name)
        h36m_joint_valid = transform_joint_to_other_db(muco_joint_valid, self.muco_joints_name, self.h36m_joints_name)
        h36m_joint = h36m_joint[h36m_joint_valid==1].reshape(-1,3)

        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl[h36m_joint_valid==1].reshape(-1,3)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:] # translation alignment
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, tight_bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['tight_bbox'], data['smpl_param'], data['cam_param']
         
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip,bbox = augmentation(img, tight_bbox, self.data_split,is_full_body = True) # always full body
        img = self.transform(img.astype(np.float32))/255.
        
        # muco gt
        muco_joint_img = data['joint_img']
        muco_joint_cam = data['joint_cam']
        muco_joint_cam = muco_joint_cam - muco_joint_cam[self.muco_root_joint_idx,None,:] # root-relative
        muco_joint_valid = data['joint_valid']
        if do_flip:
            muco_joint_img[:,0] = img_shape[1] - 1 - muco_joint_img[:,0]
            muco_joint_cam[:,0] = -muco_joint_cam[:,0]
            for pair in self.muco_flip_pairs:
                muco_joint_img[pair[0],:], muco_joint_img[pair[1],:] = muco_joint_img[pair[1],:].copy(), muco_joint_img[pair[0],:].copy()
                muco_joint_cam[pair[0],:], muco_joint_cam[pair[1],:] = muco_joint_cam[pair[1],:].copy(), muco_joint_cam[pair[0],:].copy()
                muco_joint_valid[pair[0],:], muco_joint_valid[pair[1],:] = muco_joint_valid[pair[1],:].copy(), muco_joint_valid[pair[0],:].copy()

        muco_joint_img_xy1 = np.concatenate((muco_joint_img[:,:2], np.ones_like(muco_joint_img[:,:1])),1)
        muco_joint_img[:,:2] = np.dot(img2bb_trans, muco_joint_img_xy1.transpose(1,0)).transpose(1,0)
        # for swap
        if len(data['near_joints']) > 0:
            near_joint_list = []
            for nj in data['near_joints']:
                near_joint = np.ones((self.coco_joint_num, 3), dtype=np.float32)
                nj_xy1 = np.concatenate((nj[:, :2], np.ones_like(nj[:, :1])), axis=1)
                near_joint[:, :2] = np.dot(img2bb_trans, nj_xy1.transpose(1,0)).transpose(1,0)
                near_joint_list.append(near_joint)
            near_joints = np.asarray(near_joint_list, dtype=np.float32)
        else:
            near_joints = np.zeros((1, self.coco_joint_num, 3), dtype=np.float32)

        input_muco_joint_img = muco_joint_img.copy()
        muco_joint_img[:,0] = muco_joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        muco_joint_img[:,1] = muco_joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        muco_joint_img[:,2] = muco_joint_img[:,2] - muco_joint_img[self.muco_root_joint_idx][2] # root-relative
        muco_joint_img[:,2] = (muco_joint_img[:,2] / (cfg.bbox_3d_size * 1000 / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter

        # check truncation
        muco_joint_trunc = muco_joint_valid * ((muco_joint_img[:,0] >= 0) * (muco_joint_img[:,0] < cfg.output_hm_shape[2]) * \
                    (muco_joint_img[:,1] >= 0) * (muco_joint_img[:,1] < cfg.output_hm_shape[1]) * \
                    (muco_joint_img[:,2] >= 0) * (muco_joint_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)

        # transform muco joints to target db joints
        muco_joint_img = transform_joint_to_other_db(muco_joint_img, self.muco_joints_name, self.joints_name)
        muco_joint_cam = transform_joint_to_other_db(muco_joint_cam, self.muco_joints_name, self.joints_name)
        muco_joint_valid = transform_joint_to_other_db(muco_joint_valid, self.muco_joints_name, self.joints_name)
        muco_joint_trunc = transform_joint_to_other_db(muco_joint_trunc, self.muco_joints_name, self.joints_name)

        # apply PoseFix
        input_muco_joint_img[:, 2] = 1 # joint valid
        tmp_joint_img = transform_joint_to_other_db(input_muco_joint_img, self.muco_joints_name, self.coco_joints_name)
        tmp_joint_img = replace_joint_img(tmp_joint_img, data['tight_bbox'], near_joints, data['num_overlap'], img2bb_trans)
        tmp_joint_img = transform_joint_to_other_db(tmp_joint_img, self.coco_joints_name, self.muco_joints_name)
        input_muco_joint_img[self.muco_coco_common_jidx, :2] = tmp_joint_img[self.muco_coco_common_jidx, :2]
        """
        # debug PoseFix result
        newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), input_muco_joint_img.T, self.muco_skeleton)
        cv2.imshow(f'{img_path}', newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        """
        input_muco_joint_img[:, 0] = input_muco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        input_muco_joint_img[:, 1] = input_muco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        input_muco_joint_img = transform_joint_to_other_db(input_muco_joint_img, self.muco_joints_name, self.joints_name)

        if smpl_param is not None:
            # smpl coordinates
            smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
            smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
            focal, princpt = cam_param['focal'], cam_param['princpt']
            smpl_coord_img = cam2pixel(smpl_coord_cam, focal, princpt)

            # affine transform x,y coordinates. root-relative depth
            smpl_coord_img_xy1 = np.concatenate((smpl_coord_img[:,:2], np.ones_like(smpl_coord_img[:,:1])),1)
            smpl_coord_img[:,:2] = np.dot(img2bb_trans, smpl_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            """
            # vis smpl
            newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), smpl_coord_img[6890:].T, self.skeleton)
            cv2.imshow(f'{img_path}', newimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            """
            smpl_coord_img[:,2] = smpl_coord_img[:,2] - smpl_coord_cam[self.vertex_num + self.root_joint_idx][2]
            smpl_coord_img[:,0] = smpl_coord_img[:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            smpl_coord_img[:,1] = smpl_coord_img[:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            smpl_coord_img[:,2] = (smpl_coord_img[:,2] / (cfg.bbox_3d_size * 1000  / 2) + 1)/2. * cfg.output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter

            # check truncation
            smpl_trunc = ((smpl_coord_img[:,0] >= 0) * (smpl_coord_img[:,0] < cfg.output_hm_shape[2]) * \
                        (smpl_coord_img[:,1] >= 0) * (smpl_coord_img[:,1] < cfg.output_hm_shape[1]) * \
                        (smpl_coord_img[:,2] >= 0) * (smpl_coord_img[:,2] < cfg.output_hm_shape[0])).reshape(-1,1).astype(np.float32)
            
            # split mesh and joint coordinates
            smpl_mesh_img = smpl_coord_img[:self.vertex_num]; smpl_joint_img = smpl_coord_img[self.vertex_num:];
            smpl_mesh_trunc = smpl_trunc[:self.vertex_num]; smpl_joint_trunc = smpl_trunc[self.vertex_num:];

            # if fitted mesh is too far from muco gt, discard it
            is_valid_fit = True
            error = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam, do_flip)
            if error > self.fitting_thr:
                is_valid_fit = False
            
        else:
            smpl_joint_img = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            smpl_joint_cam = np.zeros((self.joint_num,3), dtype=np.float32) # dummy
            smpl_mesh_img = np.zeros((self.vertex_num,3), dtype=np.float32) # dummy
            smpl_pose = np.zeros((72), dtype=np.float32) # dummy
            smpl_shape = np.zeros((10), dtype=np.float32) # dummy
            smpl_joint_trunc = np.zeros((self.joint_num,1), dtype=np.float32) # dummy
            smpl_mesh_trunc = np.zeros((self.vertex_num,1), dtype=np.float32) # dummy
            is_valid_fit = False
        
        # 3D data rotation augmentation
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
        [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
        [0, 0, 1]], dtype=np.float32)
        # muco coordinate
        muco_joint_cam = np.dot(rot_aug_mat, muco_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter
        # parameter
        smpl_pose = smpl_pose.reshape(-1,3)
        root_pose = smpl_pose[self.root_joint_idx,:]
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        smpl_pose[self.root_joint_idx] = root_pose.reshape(3)
        smpl_pose = smpl_pose.reshape(-1)
        # smpl coordinate
        smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[self.root_joint_idx,None] # root-relative
        smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter

        # SMPL pose parameter validity
        smpl_param_valid = np.ones((self.smpl.orig_joint_num, 3), dtype=np.float32)
        for name in ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'):
            smpl_param_valid[self.joints_name.index(name)] = 0
        smpl_param_valid = smpl_param_valid.reshape(-1)

        inputs = {'img': img, 'joints': input_muco_joint_img[:, :2], 'joints_mask': muco_joint_trunc}
        targets = {'orig_joint_img': muco_joint_img, 'fit_joint_img': smpl_joint_img, 'orig_joint_cam': muco_joint_cam, 'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
        meta_info = {'orig_joint_valid': muco_joint_valid, 'orig_joint_trunc': muco_joint_trunc, 'fit_param_valid': smpl_param_valid, 'fit_joint_trunc': smpl_joint_trunc, 'is_valid_fit': float(is_valid_fit), 'bbox': bbox,
                     'is_3D': float(True)}

        return inputs, targets, meta_info


