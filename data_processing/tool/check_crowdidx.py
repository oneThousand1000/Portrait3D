import pickle

import numpy as np
import os.path as osp
from pycocotools.coco import COCO


def compute_CrowdIndex(ref_bbox, ref_kps, intf_kps):

    na = 0
    for ref_kp in ref_kps:
        count = get_inclusion(ref_bbox, ref_kp)
        na += count

    nb = 0
    for intf_kp in intf_kps:
        count = get_inclusion(ref_bbox, intf_kp)
        nb += count

    if na < 4:  # invalid ones, e.g. truncated images
        return 0
    else:
        return nb / na


def get_inclusion(bbox, kp):
    if bbox[0] > kp[0] or (bbox[0] + bbox[2]) < kp[0]:
        return 0

    if bbox[1] > kp[1] or (bbox[1] + bbox[3]) < kp[1]:
        return 0

    return 1


def compute_iou(src_roi, dst_roi):
    # IoU calculate with GTs
    xmin = np.maximum(dst_roi[:, 0], src_roi[:, 0])
    ymin = np.maximum(dst_roi[:, 1], src_roi[:, 1])
    xmax = np.minimum(dst_roi[:, 0] + dst_roi[:, 2], src_roi[:, 0] + src_roi[:, 2])
    ymax = np.minimum(dst_roi[:, 1] + dst_roi[:, 3], src_roi[:, 1] + src_roi[:, 3])

    interArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    boxAArea = dst_roi[:, 2] * dst_roi[:, 3]
    boxBArea = np.tile(src_roi[:, 2] * src_roi[:, 3], (len(dst_roi), 1))
    sumArea = boxAArea + boxBArea

    iou = interArea / (sumArea - interArea + 1e-5)

    return iou


def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1];
    y_img = y_img[joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

class PW3D():
    def __init__(self, data_split):
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'PW3D', 'data')
        self.seq_iou_list, self.seq_crowd_idx_list = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.data_path, '3DPW_latest_' + self.data_split + '.json'))

        seq_iou_list = {}
        seq_crowd_idx_list = {}
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            img_width, img_height = img['height'], img['width']

            aids = db.getAnnIds(iid)
            if len(aids) < 2:
                continue

            data_dict = {}
            data_dict['img_id'] = iid
            data_dict['img_path'] = img_path

            # compute iou
            ann1 = db.anns[aids[0]]
            ann2 = db.anns[aids[1]]

            bbox1 = np.array(ann1['bbox'])
            bbox2 = np.array(ann2['bbox'])
            iou = compute_iou(bbox1[None, :], bbox2[None, :])[0,0]

            seq_iou_list.setdefault(sequence_name, []).append(iou)

            # compute crowd index
            joint_img1 = np.array(ann1['joint_img'], dtype=np.float32).reshape(-1, 2)
            joint_img2 = np.array(ann2['joint_img'], dtype=np.float32).reshape(-1, 2)

            ci1 = compute_CrowdIndex(bbox1, joint_img1, joint_img2)
            ci2 = compute_CrowdIndex(bbox2, joint_img2, joint_img1)

            crowd_idx = (ci1+ci2) / 2

            seq_crowd_idx_list.setdefault(sequence_name, []).append(crowd_idx)

        for seq in seq_iou_list.keys():
            seq_iou_list[seq] = sum(seq_iou_list[seq]) / len(seq_iou_list[seq])
        for seq in seq_crowd_idx_list.keys():
            seq_crowd_idx_list[seq] = sum(seq_crowd_idx_list[seq]) / len(seq_crowd_idx_list[seq])

        return seq_iou_list, seq_crowd_idx_list

    def print_statistics(self):
        total_mean_iou, total_mean_crowd_idx = 0, 0
        for seq in self.seq_iou_list:
            print(f"Average iou / crowd index of {seq}: {self.seq_iou_list[seq]}, {self.seq_crowd_idx_list[seq]}")
            total_mean_iou += self.seq_iou_list[seq]
            total_mean_crowd_idx += self.seq_crowd_idx_list[seq]
        print(f"All iou / crowd index: {total_mean_iou/len(self.seq_iou_list)}, {total_mean_crowd_idx/len(self.seq_iou_list)}")


class MuPoTs():
    def __init__(self):
        self.test_annot_path = osp.join('..', 'data', 'MuPoTs', 'data', 'MuPoTS-3D.json')
        self.seq_iou_list, self.seq_crowd_idx_list = self.load_data()

    def load_data(self):
        db = COCO(self.test_annot_path)

        seq_iou_list = {}
        seq_crowd_idx_list = {}
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_name = img['file_name']
            sequence_name = img_name.split('/')[0]

            aids = db.getAnnIds(iid)
            if len(aids) < 2:
                continue

            for aid_idx in range(len(aids)):
                ref_ann = db.anns[aids[aid_idx]]
                other_aids = aids[:aid_idx] + aids[aid_idx+1:]
                ref_bbox = np.array(ref_ann['bbox'])
                ref_joint = np.array(ref_ann['keypoints_img'])
                for oaid in other_aids:
                    other_ann = db.anns[oaid]
                    other_bbox = np.array(other_ann['bbox'])
                    other_joint = np.array(other_ann['keypoints_img'])

                    iou = compute_iou(ref_bbox[None, :], other_bbox[None, :])[0, 0] / 2.0  # compensate twice count
                    crowd_idx = compute_CrowdIndex(ref_bbox, ref_joint, other_joint)

                    seq_iou_list.setdefault(sequence_name, []).append(iou)
                    seq_crowd_idx_list.setdefault(sequence_name, []).append(crowd_idx)

        for seq in seq_iou_list.keys():
            seq_iou_list[seq] = sum(seq_iou_list[seq]) / len(seq_iou_list[seq])
        for seq in seq_crowd_idx_list.keys():
            seq_crowd_idx_list[seq] = sum(seq_crowd_idx_list[seq]) / len(seq_crowd_idx_list[seq])

        return seq_iou_list, seq_crowd_idx_list

    def print_statistics(self):
        total_mean_iou, total_mean_crowd_idx = 0, 0
        for seq in self.seq_iou_list:
            print(f"Average iou / crowd index of {seq}: {self.seq_iou_list[seq]}, {self.seq_crowd_idx_list[seq]}")
            total_mean_iou += self.seq_iou_list[seq]
            total_mean_crowd_idx += self.seq_crowd_idx_list[seq]
        print(f"All iou / crowd index: {total_mean_iou/len(self.seq_iou_list)}, {total_mean_crowd_idx/len(self.seq_iou_list)}")



class CMUP():
    def __init__(self):
        self.seq_list = ['160906_pizza1', '160422_ultimatum1', '160422_haggling1', '160422_mafia2']

        self.seq_iou_list, self.seq_crowd_idx_list = {}, {}

        for seq_name in self.seq_list:
            self.annot_path = osp.join('..', 'data', 'CMU-Panoptic', 'data', f'{seq_name}.pkl')
            mean_iou, mean_crowd_idx = self.load_data()
            self.seq_iou_list[seq_name], self.seq_crowd_idx_list[seq_name] = mean_iou, mean_crowd_idx

    def load_data(self):
        with open(self.annot_path,'rb') as f:
            db = pickle.load(f)

        seq_iou_list = []
        seq_crowd_idx_list = []
        for img_idx in range(len(db)):

            for i in range(db[img_idx]['kpts2d'].shape[0]):
                ref_joint = db[img_idx]['kpts2d'][i] #24 3
                ref_bbox = get_bbox(ref_joint[:, :2], ref_joint[:, 2])

                other_joints = np.concatenate((db[img_idx]['kpts2d'][:i],db[img_idx]['kpts2d'][i+1:]), axis=0)
                for other_joint in other_joints:
                    other_bbox = get_bbox(other_joint[:, :2], other_joint[:, 2])
                    iou = compute_iou(ref_bbox[None, :], other_bbox[None, :])[0, 0] / 2.0  # compensate twice count
                    crowd_idx = compute_CrowdIndex(ref_bbox, ref_joint, other_joint)

                    seq_iou_list.append(iou)
                    seq_crowd_idx_list.append(crowd_idx)

        mean_iou, mean_crowd_idx = sum(seq_iou_list) / len(seq_iou_list), sum(seq_crowd_idx_list) / len(seq_crowd_idx_list)
        return mean_iou, mean_crowd_idx

    def print_statistics(self):
        total_mean_iou, total_mean_crowd_idx = 0, 0
        for seq in self.seq_list:
            print(f"Average iou / crowd index of {seq}: {self.seq_iou_list[seq]}, {self.seq_crowd_idx_list[seq]}")
            total_mean_iou += self.seq_iou_list[seq]
            total_mean_crowd_idx += self.seq_crowd_idx_list[seq]
        print(f"All iou / crowd index: {total_mean_iou/len(self.seq_list)}, {total_mean_crowd_idx/len(self.seq_list)}")




if __name__ == '__main__':
    dataset = PW3D('validation')
    # dataset = MuPoTs()
    # dataset = CMUP()
    dataset.print_statistics()
