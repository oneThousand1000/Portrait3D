import glob
import json
import os.path

import cv2
import sys
sys.path.append('../common')
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import Pose2Feat, PositionNet, RotationNet, Vposer
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math
# visualization
import colorsys
from utils.vis import vis_mesh, save_obj, render_mesh, vis_keypoints
import numpy as np
from utils.transforms import rot6d_to_axis_angle
import cv2
from utils.preprocessing import generate_patch_image
with open('G:/full-head-dataset/pexels/00000000/result.json')as f:
    result = json.load(f)

for image_name in result:
    bbox = result[image_name]['bbox']
    if (bbox[2] < 400 or bbox[3] < 400):
        # os.remove(f'G:/full-head-dataset/pexels/00000000/visualization/{image_name}')
        # if os.path.exists(f'G:/full-head-dataset/pexels/00000000/aligned_images/{image_name}'):
        #     os.remove(f'G:/full-head-dataset/pexels/00000000/aligned_images/{image_name}')
        continue
    if not os.path.exists(f'G:/full-head-dataset/pexels/00000000/visualization/{image_name}'):
        continue
    if os.path.exists(f'G:/full-head-dataset/pexels/00000000/aligned_images/{image_name}'):
        continue
    raw_image_name = image_name.split('_')[0]
    image_path = glob.glob(f'G:/full-head-dataset/pexels/00000000/images/{raw_image_name}' + '*')[0]
    print(image_path)

    img, _, _ = generate_patch_image(cv2.imread(image_path), bbox, 1.0, 0.0, False, (1024,1024),enable_padding=True)
    cv2.imwrite(f'G:/full-head-dataset/pexels/00000000/aligned_images/{image_name}', img)