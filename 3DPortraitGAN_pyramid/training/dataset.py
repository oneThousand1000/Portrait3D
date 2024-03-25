# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

def matrix2angle(R):
    """ 
    https://github.com/sizhean/panohead
    compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    
    if abs(y) > np.pi/2:
        if x > 0:
            x = np.pi - x
        else:
            x = -np.pi - x
    y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
    z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    return x, y, z


def get_poseangle(eg3dparams):
    '''
    https://github.com/sizhean/panohead
    '''
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    entry_cam = np.array([float(p) for p in eg3dparams][:16]).reshape((4,4))

    world2cam = np.linalg.inv(entry_cam@convert)
    pose = matrix2angle(world2cam[:3,:3])
    angle  = [p * 180 / np.pi for p in pose]

    return angle



class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        rebal_raw_idx = None, # Rebalance the dataset by sampling from the raw_idx list
        data_rebalance=False, # Rebalance the dataset by sampling from the raw_idx list
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._raw_poses = None
        self._label_shape = None
        self._pose_shape = None


        if data_rebalance:
            raise NotImplementedError
            assert rebal_raw_idx is not None, "rebal_raw_idx must be provided if data_rebalance is True"
            self._raw_idx = rebal_raw_idx
        else:
            self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)


        self._raw_idx = self._filter_samples()

        # Apply max_size.
        if (max_size is not None) and (self._raw_idx.size > max_size):
            raise NotImplementedError
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _filter_samples(self): # to be overridden by subclass
        raise NotImplementedError


    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels,self._raw_poses = self._load_raw_labels() if self._use_labels else None

            if self._raw_labels is None:
                raise Exception("_raw_labels is None")
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)

            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)


            if self._raw_poses is None:
                raise Exception("_raw_poses is None")
                self._raw_poses = np.zeros([self._raw_poses[0], 0], dtype=np.float32)

            assert isinstance(self._raw_poses, np.ndarray)
            assert self._raw_poses.shape[0] == self._raw_shape[0]
            assert self._raw_poses.dtype in [np.float32, np.int64]
            if self._raw_poses.dtype == np.int64:
                assert self._raw_poses.ndim == 1
                assert np.all(self._raw_poses >= 0)
            self._raw_poses_std = self._raw_poses.std(0)

        return self._raw_labels

    def _get_raw_poses(self):
        if self._raw_poses is None:
            _ = self._get_raw_labels()
            #raise Exception("please run _get_raw_labels first")

        return self._raw_poses


    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError


    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None, _raw_poses=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size




    def __getitem__(self, idx):


        label = self.get_label(idx)
        pose = self.get_coarse_pose(idx)

        # image = self._load_raw_image(self._raw_idx[idx])
        # assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        # if self._xflip[idx]:
        #     assert image.ndim == 3 # CHW
        #     image = image[:, :, ::-1]
        #     # # flip label
        #     # label = self.flip_yaw(label)
        #     # # flip pose
        #     # pose[[1, 2, 4, 5]] *= -1

        image = self.get_image(idx)


        return image, label,pose

    def flip_yaw(self, c):
        pose_matrix = c.copy()
        flipped = pose_matrix[:16].reshape(4,4)
        flipped[0, 1] *= -1
        flipped[0, 2] *= -1
        flipped[1, 0] *= -1
        flipped[2, 0] *= -1
        flipped[0, 3] *= -1

        flipped = flipped.reshape(16)
        pose_matrix[:16] = flipped

        return pose_matrix

    def get_image(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        return image.copy()


    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]].copy()
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot

        if self._xflip[idx]:
            assert label.shape == (25,)
            label[[1, 2, 3, 4, 8]] *= -1

        return label

    def get_coarse_pose(self, idx):
        pose = self._get_raw_poses()[self._raw_idx[idx]].copy()
        if pose.dtype == np.int64:
            raise TypeError("pose should be float32")
            onehot = np.zeros(self.pose_shape, dtype=np.float32)
            onehot[pose] = 1
            pose = onehot

        if self._xflip[idx]:
            pose_flip = pose.copy()
            pose_flip[[1, 2, 4, 5]] *= -1

            return pose_flip

        else:
            return pose



    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        # d.pose = self.get_coarse_pose(idx).copy()

        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def pose_shape(self):
        if self._pose_shape is None:
            self._get_raw_labels()
            if self._raw_poses.dtype == np.int64:
                self._pose_shape = [int(np.max(self._raw_poses)) + 1]
            else:
                self._pose_shape = self._raw_poses.shape[1:]
        return list(self._pose_shape)


    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        back_repeat = None,
        resolution      = None, # Ensure specific resolution, None = highest available.
        data_rebalance_idx_file = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.min_yaw = 0  
        self.max_yaw = 180 
        self.max_pitch = 90 
        self.back_repeat = 1 if back_repeat is None else back_repeat
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            raise NotImplementedError('Does not support directories yet')
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        if data_rebalance_idx_file is not None:
            raise NotImplementedError('data_rebalance is not implemented yet')
            rebal_idx_list_path =data_rebalance_idx_file
            #print('load rebal_idx_list from ',rebal_idx_list_path)
            with open(rebal_idx_list_path, 'r') as f:
                rebal_raw_idx = json.load(f)
                rebal_raw_idx = np.array(rebal_raw_idx)
        else:
            rebal_raw_idx = None


        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, rebal_raw_idx = rebal_raw_idx,**super_kwargs)


    def _filter_samples(self):
        if self.back_repeat>1:
            raw_labels = self._get_raw_labels()[self._raw_idx]
            label_list = []
            for entry in raw_labels:
                label_list.append(get_poseangle(entry))
            poses = np.array(label_list)
            # find [min_yaw, max_yaw] boolean
            valid = (np.abs(poses[:,0])>=self.min_yaw) & (np.abs(poses[:,0])<=self.max_yaw) & (np.abs(poses[:,1])<=self.max_pitch)
            # find back boolean: [max(90, self.min_yaw), max_yaw]
            back_valid = (np.abs(poses[:,0])>= max(90, self.min_yaw)) & (np.abs(poses[:,0])<=self.max_yaw) & (np.abs(poses[:,1])<=self.max_pitch)
            if not np.all(valid):
                print(f"filtering samples by pose: ratio = {valid.sum()}/{len(self._raw_idx)}")
            # boolean to index
            valid_idx = self._raw_idx[valid]
            back_idx = self._raw_idx[back_valid]
            front_idx = np.array(list(set(valid_idx) - set(back_idx)))
            
            front_num = valid.sum()-len(back_idx)
            front_back_ratio_min = front_num/2/len(back_idx)
            print(f"if back num be the half of front num, at least repeat ({int(front_back_ratio_min)}) times.")
            back_repeat = max(int(front_num/2/len(back_idx)), self.back_repeat)




            # TODO: support the repeat times < 1
            # repeat [max(90, self.min_yaw), max_yaw] for multiple times
            back_repeat_idx = np.tile(back_idx, back_repeat)
            # merge front index and repeated back index
            new_idx = np.concatenate((front_idx, back_repeat_idx))
            print(f"Repeat {len(back_idx)} back images till abs({self.max_yaw}) degree {back_repeat} times")
            return new_idx
        else:
            return self._raw_idx
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = np.squeeze(labels)
        #print('labels shape', labels.shape) # N, 31
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        poses = labels[:,25:]
        labels = labels[:,:25]

        # print('labels shape', labels.shape) # N, 25
        # print('poses shape', poses.shape) # N, 6

        return labels, poses


#----------------------------------------------------------------------------


class MaskLabeledDataset(ImageFolderDataset):

    def __init__(self,
                 img_path,  # Path to directory or zip.
                 seg_path,  # Path to directory or zip.
                 back_repeat = None,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self.min_yaw = 0  
        self.max_yaw = 180 
        self.max_pitch = 90 
        self.back_repeat = 1 if back_repeat is None else back_repeat
        super().__init__(path=img_path,  back_repeat = None,**super_kwargs)
        
        self._seg_dataset = ImageFolderDataset(seg_path, **super_kwargs)

        # Build the mapping from seg fname to seg raw index
        seg_dict = {os.path.splitext(fname)[0]: idx for idx, fname in enumerate(self._seg_dataset._image_fnames)}

        # Build the mapping from index to seg raw index
        self._seg_raw_idx = []
        for raw_idx in self._raw_idx:
            fname = self._image_fnames[raw_idx]
            key = os.path.splitext(fname)[0]
            self._seg_raw_idx.append(seg_dict[key])
        self._seg_raw_idx = np.array(self._seg_raw_idx)

    def _filter_samples(self):
        if self.back_repeat>1:
            raw_labels = self._get_raw_labels()[self._raw_idx]
            label_list = []
            for entry in raw_labels:
                label_list.append(get_poseangle(entry))
            poses = np.array(label_list)
            # find [min_yaw, max_yaw] boolean
            valid = (np.abs(poses[:,0])>=self.min_yaw) & (np.abs(poses[:,0])<=self.max_yaw) & (np.abs(poses[:,1])<=self.max_pitch)
            # find back boolean: [max(90, self.min_yaw), max_yaw]
            back_valid = (np.abs(poses[:,0])>= max(90, self.min_yaw)) & (np.abs(poses[:,0])<=self.max_yaw) & (np.abs(poses[:,1])<=self.max_pitch)
            if not np.all(valid):
                print(f"filtering samples by pose: ratio = {valid.sum()}/{len(self._raw_idx)}")
            # boolean to index
            valid_idx = self._raw_idx[valid]
            back_idx = self._raw_idx[back_valid]
            front_idx = np.array(list(set(valid_idx) - set(back_idx)))
            
            front_num = valid.sum()-len(back_idx)
            front_back_ratio_min = front_num/2/len(back_idx)
            print(f"if back num be the half of front num, at least repeat ({int(front_back_ratio_min)}) times.")
            back_repeat = max(int(front_num/2/len(back_idx)), self.back_repeat)




            # TODO: support the repeat times < 1
            # repeat [max(90, self.min_yaw), max_yaw] for multiple times
            back_repeat_idx = np.tile(back_idx, back_repeat)
            # merge front index and repeated back index
            new_idx = np.concatenate((front_idx, back_repeat_idx))
            print(f"Repeat {len(back_idx)} back images till abs({self.max_yaw}) degree {back_repeat} times")
            return new_idx
        else:
            return self._raw_idx



    def __getitem__(self, idx):
        # already flipped in the ImageFolderDataset
        image = self.get_image(idx)
        mask = self._seg_dataset.get_image(idx)
        label = self.get_label(idx)
        pose = self.get_coarse_pose(idx)


        return image.copy(), mask.copy(), label,pose

