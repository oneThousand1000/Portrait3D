import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO

from utils.posefix import replace_joint_img
from utils.preprocessing import compute_iou, process_bbox, load_img, augmentation,get_bbox
from utils.smpl import SMPL
from utils.transforms import transform_joint_to_other_db, cam2pixel
from utils.vis import vis_keypoints_with_skeleton


class MPII(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join(cfg.data_dir, 'MPII', 'data')
        self.annot_path = osp.join(cfg.data_dir, 'MPII', 'data', 'annotations')

        # mpii skeleton
        self.mpii_joint_num = 16
        self.mpii_joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')
        self.mpii_flip_pairs = ((0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13))
        self.mpii_skeleton = ((0,1), (1,2), (2,6), (3,6), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12) , (7, 12), (7, 13), (13, 14), (14, 15))

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

        self.coco_joint_num = 18  # original: 17, manually added pelvis
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis')
        self.mpii_coco_common_idx = (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15)

        self.datalist = self.load_data()
        print("mpii data len: ", len(self.datalist))

    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'train.json'))
        with open(osp.join(self.annot_path, 'MPII_train_SMPL_NeuralAnnot.json')) as f:
            smpl_params = json.load(f)

        datalist = []
        for iid in db.imgs.keys():
            aids = db.getAnnIds([iid])

            tmplist = []
            for aid in aids:
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                img_path = osp.join(self.img_path, img['file_name'])
                width, height = img['width'], img['height']

                # bbox
                # tight_bbox = np.array(ann['bbox'])
                # bbox = process_bbox(tight_bbox, width, height)
                # if bbox is None: continue

                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
                joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
                joint_img[:, 2] = joint_valid[:, 0]  # for posefix, only good for 2d datasets

                # bbox
                if cfg.use_bbox_in_ann:
                    tight_bbox = np.array(ann['bbox'])
                else:
                    tight_bbox = get_bbox(joint_img, np.ones_like(joint_img[:, 0]), crop_bottom_body=True)

                # smpl parameter
                if str(aid) in smpl_params:
                    smpl_param = smpl_params[str(aid)]
                else:
                    smpl_param = None

                tmplist.append({'img_path': img_path, 'img_shape': (height, width),
                                #'bbox': bbox,
                                'tight_bbox': tight_bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'smpl_param': smpl_param})

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
                    other_joint = transform_joint_to_other_db(other['joint_img'], self.mpii_joints_name, self.coco_joints_name)
                    near_joints.append(other_joint)

                person['num_overlap'] = num_overlap
                person['near_joints'] = near_joints

            datalist.extend(tmplist)
            """
            if num_overlap > 2:
                tmpimg = cv2.imread(img_path)
                newimg = vis_keypoints_with_skeleton(tmpimg, joint_img.T, self.mpii_skeleton)
                cv2.imshow(f'{img_path}', newimg / 255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                import pdb;
                pdb.set_trace()
            """

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
        # self.mpii_joint_num = 16
        # self.mpii_joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')
        # data['joint_valid'].shape = (15, 1)
        is_full_body = np.sum(
            data['joint_valid'][0:6, :]) > 3  # 0:6 is the index of R_Ankle, R_Knee, R_Hip, L_Hip, L_Knee, L_Ankle

        # image load and affine transform
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip,bbox = augmentation(img, tight_bbox, self.data_split, is_full_body= is_full_body)
        img = self.transform(img.astype(np.float32)) / 255.

        # mpii gt
        mpii_joint_img = data['joint_img']
        mpii_joint_valid = data['joint_valid']
        if do_flip:
            mpii_joint_img[:, 0] = img_shape[1] - 1 - mpii_joint_img[:, 0]
            for pair in self.mpii_flip_pairs:
                mpii_joint_img[pair[0], :], mpii_joint_img[pair[1], :] = mpii_joint_img[pair[1], :].copy(), mpii_joint_img[pair[0], :].copy()
                mpii_joint_valid[pair[0], :], mpii_joint_valid[pair[1], :] = mpii_joint_valid[pair[1], :].copy(), mpii_joint_valid[pair[0], :].copy()

        mpii_joint_img_xy1 = np.concatenate((mpii_joint_img[:, :2], np.ones_like(mpii_joint_img[:, :1])), 1)
        mpii_joint_img[:, :2] = np.dot(img2bb_trans, mpii_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
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

        input_mpii_joint_img = mpii_joint_img.copy()
        mpii_joint_img[:, 0] = mpii_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        mpii_joint_img[:, 1] = mpii_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

        # check truncation
        mpii_joint_trunc = mpii_joint_valid * (
                    (mpii_joint_img[:, 0] >= 0) * (mpii_joint_img[:, 0] < cfg.output_hm_shape[2]) * (mpii_joint_img[:, 1] >= 0) * (mpii_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(-1,
            1).astype(np.float32)

        # transform coco joints to target db joints
        mpii_joint_img = transform_joint_to_other_db(mpii_joint_img, self.mpii_joints_name, self.joints_name)
        mpii_joint_cam = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
        mpii_joint_valid = transform_joint_to_other_db(mpii_joint_valid, self.mpii_joints_name, self.joints_name)
        mpii_joint_trunc = transform_joint_to_other_db(mpii_joint_trunc, self.mpii_joints_name, self.joints_name)

        # apply PoseFix
        tmp_joint_img = transform_joint_to_other_db(input_mpii_joint_img, self.mpii_joints_name, self.coco_joints_name)
        tmp_joint_img = replace_joint_img(tmp_joint_img, data['tight_bbox'], near_joints, data['num_overlap'], img2bb_trans)
        tmp_joint_img = transform_joint_to_other_db(tmp_joint_img, self.coco_joints_name, self.mpii_joints_name)
        input_mpii_joint_img[self.mpii_coco_common_idx, :2] = tmp_joint_img[self.mpii_coco_common_idx, :2]
        """
        # debug PoseFix result
        newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), input_mpii_joint_img.T, self.mpii_skeleton)
        cv2.imshow(f'{img_path}', newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        import pdb; pdb.set_trace()
        """
        input_mpii_joint_img[:, 0] = input_mpii_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        input_mpii_joint_img[:, 1] = input_mpii_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        input_mpii_joint_img = transform_joint_to_other_db(input_mpii_joint_img, self.mpii_joints_name, self.joints_name)

        smpl_param = data['smpl_param']
        if smpl_param is not None:
            # use fitted mesh
            smpl_param, cam_param = smpl_param['smpl_param'], smpl_param['cam_param']
            smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
            smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
            smpl_coord_img = cam2pixel(smpl_coord_cam, cam_param['focal'], cam_param['princpt'])

            # x,y affine transform, root-relative depth
            smpl_coord_img_xy1 = np.concatenate((smpl_coord_img[:, :2], np.ones_like(smpl_coord_img[:, 0:1])), 1)
            smpl_coord_img[:, :2] = np.dot(img2bb_trans, smpl_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            smpl_coord_img[:, 2] = smpl_coord_img[:, 2] - smpl_coord_cam[self.vertex_num + self.root_joint_idx][2]
            smpl_coord_img[:, 0] = smpl_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            smpl_coord_img[:, 1] = smpl_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            smpl_coord_img[:, 2] = (smpl_coord_img[:, 2] / (cfg.bbox_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]

            # check truncation
            smpl_trunc = ((smpl_coord_img[:, 0] >= 0) * (smpl_coord_img[:, 0] < cfg.output_hm_shape[2]) * (smpl_coord_img[:, 1] >= 0) * (smpl_coord_img[:, 1] < cfg.output_hm_shape[1]) * (smpl_coord_img[:, 2] >= 0) * (
                        smpl_coord_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1, 1).astype(np.float32)

            # split mesh and joint coordinates
            smpl_joint_img = smpl_coord_img[self.vertex_num:];
            smpl_joint_trunc = smpl_trunc[self.vertex_num:];

            """
            # vis smpl joint coord
            # tmpimg = cv2.imread(img_path)
            newimg = vis_keypoints_with_skeleton(img.numpy().transpose(1, 2, 0), (smpl_joint_img.T)*4, self.skeleton)
            cv2.imshow(f'{img_path}', newimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            import pdb; pdb.set_trace()
            """

            # if fitted mesh is too far from h36m gt, discard it
            is_valid_fit = True
        else:
            smpl_joint_img = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
            smpl_joint_cam = np.zeros((self.joint_num, 3), dtype=np.float32)  # dummy
            smpl_pose = np.zeros((72), dtype=np.float32)  # dummy
            smpl_shape = np.zeros((10), dtype=np.float32)  # dummy
            smpl_joint_trunc = np.zeros((self.joint_num, 1), dtype=np.float32)
            is_valid_fit = False

        # SMPL pose parameter validity
        smpl_param_valid = np.ones((self.smpl.orig_joint_num, 3), dtype=np.float32)
        for name in ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'):
            smpl_param_valid[self.joints_name.index(name)] = 0
        smpl_param_valid = smpl_param_valid.reshape(-1)

        inputs = {'img': img, 'joints': input_mpii_joint_img[:, :2], 'joints_mask': mpii_joint_trunc}
        targets = {'orig_joint_img': mpii_joint_img, 'fit_joint_img': smpl_joint_img, 'orig_joint_cam': mpii_joint_cam, 'fit_joint_cam': smpl_joint_cam, 'pose_param': smpl_pose, 'shape_param': smpl_shape}
        meta_info = {'orig_joint_valid': mpii_joint_valid, 'orig_joint_trunc': mpii_joint_trunc, 'fit_param_valid': smpl_param_valid, 'fit_joint_trunc': smpl_joint_trunc,'bbox': bbox,
                     'is_valid_fit': float(is_valid_fit), 'is_3D': float(False)}

        return inputs, targets, meta_info


