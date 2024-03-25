import os.path as osp
import torch
import numpy as np
import copy
import cv2
import json
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob


class MuPoTs(torch.utils.data.Dataset):
    def __init__(self):
        self.data_split = 'test'
        self.img_dir = osp.join('..', 'data', 'MuPoTs', 'data', 'MultiPersonTestSet')
        self.test_annot_path = osp.join('..', 'data', 'MuPoTs', 'data', 'MuPoTS-3D.json')

        # MuPoTS
        self.mupots_joint_num = 17
        self.mupots_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head')  #
        self.mupots_flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13))
        self.mupots_skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7))

        # MSCOCO
        self.coco_joint_num = 17
        self.coco_joints_name = ('Head', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
        self.coco_skeleton = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

        self.openpose_joints_name = ('Nose', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear')

        self.datalist = self.load_data()
        print('mupots data len: ', len(self.datalist))

    def load_data(self):
        data = []
        db = COCO(self.test_annot_path)

        print("Get bounding box and root from groundtruth")
        for aid in db.anns.keys():
            ann = db.anns[aid]
            if ann['is_valid'] == 0:
                continue

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            file_name = img['file_name']
            img_path = osp.join(self.img_dir, file_name)
            img_name = '_'.join(file_name.split('/'))

            joint_img = np.array(ann['keypoints_img'])
            joint_img = np.concatenate((joint_img, np.ones_like(joint_img[:, :1])), axis=1)
            joint_img = transform_joint_to_other_db(joint_img, self.mupots_joints_name, self.openpose_joints_name)  # self.coco_joints_name

            data.append({
                'img_name': img_name,
                'img_path': img_path,
                'img_id': image_id,
                'ann_id': aid,
                'joint_img': joint_img
            })

        return data

    def __len__(self):
        return len(self.datalist)

    def getitem(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_name, img_id, aid = data['img_name'], data['img_id'], data['ann_id']

        # for prediction matching
        gtpose = data['joint_img']

        # img_path = data['img_path']
        # tmpimg = cv2.imread(img_path)
        # newimg = vis_keypoints_with_skeleton(tmpimg, gtpose.T, self.coco_skeleton)
        # cv2.imshow('img w gt pose', newimg/255)
        # cv2.waitKey(0)
        # import pdb; pdb.set_trace()

        return data['img_path'], img_name, img_id, aid, gtpose


class PoseMatcher:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        result_path = '/home/hongsukchoi/projects/pytorch_Realtime_Multi-Person_Pose_Estimation'  # '/home/redarknight/projects/HHRNet'
        self.candidates = self.load_2dpose_results(result_path)
        
    def run(self):
        output_list = []
        for idx in range(len(self.dataloader)):
            img_path, img_name, img_id, aid, gtpose = self.dataloader.getitem(idx)
            candidates = self.candidates[img_name]

            output = {}
            output['candidates'] = candidates
            output['target'] = {
                'gtpose': gtpose
            }
            output['meta'] = {
                'aid': aid,
                'img_id': img_id,
                'img_path': img_path
            }

            output_list.append(output)

        output_list = filter_bbox(output_list)

        save_output(output_list)

    def load_2dpose_results(self, result_path):
        result_jsons = glob.glob(f'{result_path}/*.json')

        detector_results = {}
        for rj in result_jsons:
            with open(rj) as f:
                pose_outputs = json.load(f)

            seq_name = rj.split('openpose_result_')[-1][:-5]
            for img_name in sorted(pose_outputs.keys()):
                pose_candidates = pose_outputs[img_name]
                try:
                    pose_candidates = np.asarray(pose_candidates, dtype=np.float32)[:,:18,:3]
                except IndexError:  # when the result is empty
                    pose_candidates = []
                img_name = seq_name + '_' + img_name

                detector_results[img_name] = pose_candidates

        return detector_results


# open pose valid joint compare
def filter_bbox(output_list):
    result = {}
    d_count = 0
    for out in output_list:
        candidates = out['candidates']
        gtpose_from_dataset = out['target']['gtpose']
        aid = out['meta']['aid']
        img_id = out['meta']['img_id']
        img_path = out['meta']['img_path']

        if len(candidates) == 0:
            continue

        valid_gtpose_joints = (gtpose_from_dataset[:, 2] > 0.0)

        match_idx = 0
        err = 60  # pixel
        for idx in range(len(candidates)):
            pred_pose = candidates[idx]
            valid_pred_joints = (pred_pose[:, 2] > 0.1)
            valid_idx = (valid_gtpose_joints * valid_pred_joints).nonzero()[0]
            l1_err = np.abs(pred_pose[valid_idx, :2] - gtpose_from_dataset[valid_idx, :2])
            # import pdb; pdb.set_trace()

            euc_dst = np.sqrt((l1_err**2).sum(axis=1)).mean()

            if euc_dst < err:
                match_idx = idx
                err = euc_dst

        if err == 60:
            # continue

            candidates[match_idx] = np.zeros_like(candidates[match_idx])
            d_count += 1
            """
            coco_skeleton = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
            tmpimg = cv2.imread(img_path)
            gimg = vis_keypoints(tmpimg, gtpose_from_dataset) #vis_keypoints_with_skeleton(tmpimg, openpose_from_dataset.T, coco_skeleton, kp_thresh=0.0)
            cv2.imshow('gtpose', gimg/255)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
            # simg = vis_keypoints(tmpimg, smplpose_from_dataset) #vis_keypoints_with_skeleton(tmpimg, smplpose_from_dataset.T, coco_skeleton, kp_thresh=0.0)
            # cv2.imshow('smplpose', simg / 255)
            # cv2.waitKey(0)
            print("Match idx: ", match_idx)
            for idx in range(len(candidates)):
                pimg = vis_keypoints(tmpimg, candidates[idx]) #vis_keypoints_with_skeleton(tmpimg, candidates[idx].T, coco_skeleton, kp_thresh=0.0)
                cv2.imshow(f'hhrnetpose {idx}', pimg)
                cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            import pdb; pdb.set_trace()
            """

        res = {}
        res['coco_joints'] = candidates[match_idx].tolist()  # 17 x2
        res['img_id'] = img_id
        result[aid] = res

    print(f"{d_count} dummy out of {len(output_list)}")

    return result


def save_output(output):
    save_file_name = f'MuPoTs_test_openpose_result.json'
    print("Saving result to ", save_file_name)
    with open(save_file_name, 'w') as f:
        json.dump(output, f)


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    device = box1.device
    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).to(device)) * torch.max(
        inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).to(device))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    # x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    x_img = x_img[joint_valid > 0.01];
    y_img = y_img[joint_valid > 0.01];

    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1, kps_scores=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

            if kps_scores is not None:
                cv2.putText(kp_mask, str(kps_scores[i2, 0]), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


if __name__ == '__main__':
    testset_loader = MuPoTs()
    pose_matcher = PoseMatcher(testset_loader)
    pose_matcher.run()