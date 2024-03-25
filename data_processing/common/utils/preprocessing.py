import numpy as np
import cv2
import random
from config import cfg
import math



def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

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

# def trunc_bbox(bbox):
#     if False and random.random() >= 0.3:
#         return bbox
#     else:
#         x, y, w, h = bbox
#         x_aug_range, y_aug_range = w/2, h/2
#         x_aug, y_aug = random.random() * x_aug_range, random.random() * y_aug_range
#
#         if random.random() <= 0.5:
#             x, y = x+x_aug, y+y_aug
#         else: # good
#             w, h = w-x_aug, h-y_aug
#
#     return [x,y,w,h]

def trunc_tight_bbox(tight_bbox, img, is_full_body):
    xmin, ymin, width, height = tight_bbox
    xmax = xmin + width
    ymax = ymin + height

    height = height * 1.2
    y_center = (ymin + ymax) / 2

    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    if is_full_body:

        crop_half_bottom = random.random()<0.8
    else:
        crop_half_bottom = False

    ymin = ymin + height * 0.1 * random.random()  # 0.0 ~ 0.1
    if crop_half_bottom: # for is_full_body, we only preserve its upper body (or crop the bottom body)
        cropped_height = height * 0.25 + height * 0.25 * random.random()  # 0.25 ~ 0.5
        ymax = ymin + cropped_height  # 0.25 ~ 0.6
    # lower
    else:  # prob_preserve_more_than_half
        cropped_height = height * 0.5 + height * 0.3 * random.random()  # 0.5 ~ 0.8
        ymax = ymin + cropped_height  # 0.5 ~ 0.9

    tight_bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)

    # Since we crop the tight bbox to simulate upper body, we need to set the lower part to 0
    img[int(ymax):, :, :] = 0
    img[:int(ymin), :, :] = 0

    return tight_bbox, img

def process_bbox(bbox, img_width, img_height, is_3dpw_test=False):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if is_3dpw_test:
        bbox = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
    else:
        if w*h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
        else:
            return None

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox
def get_aug_config(exclude_flip):
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.2 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    do_crop_bbox = random.random() <= 0.7

    return scale, rot, color_scale, do_flip , do_crop_bbox


def augmentation(img, tight_bbox, data_split, exclude_flip=False,is_full_body=False):
    if data_split == 'train':
        scale, rot, color_scale, do_flip, do_crop_bbox = get_aug_config(exclude_flip, )
    else:
        scale, rot, color_scale, do_flip, do_crop_bbox = 1.0, 0.0, np.array([1, 1, 1]), False, False

    orig_tight_bbox = tight_bbox.copy()
    if do_crop_bbox:
        tight_bbox, img = trunc_tight_bbox(tight_bbox, img, is_full_body=is_full_body)

    bbox = process_bbox(tight_bbox, img.shape[1], img.shape[0])



    '''
    bbox_viz = cv2.rectangle(img.copy(), (int(orig_tight_bbox[0]), int(orig_tight_bbox[1])), (int(orig_tight_bbox[0]+orig_tight_bbox[2]), int(orig_tight_bbox[1]+orig_tight_bbox[3])), (0,255,0), 2)
    bbox_viz = cv2.rectangle(bbox_viz, (int(tight_bbox[0]), int(tight_bbox[1])), (int(tight_bbox[0]+tight_bbox[2]), int(tight_bbox[1]+tight_bbox[3])), (0,0,255), 2)
    bbox_viz = cv2.rectangle(bbox_viz, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255,0,0), 2)
    cv2.imshow('bbox', bbox_viz/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    #'''
    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)

    '''
    cv2.imshow('aug', img/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    #'''

    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, do_flip, bbox


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape, enable_padding=False):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape



    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    if enable_padding and (bbox[0]<0 or bbox[1]<0 or bbox[0]+bbox[2]>img_width or bbox[1]+bbox[3]>img_height):
        assert do_flip == False
        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
        # print('trans:',trans.shape,trans)
        # img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
        # img_patch = img_patch.astype(np.float32)
        inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                            inv=True)
        # reflection padding
        # top, bottom, left, right
        padding_top = max(int(-bbox[1]),0)
        padding_bottom = max(int(bbox[1]+bbox[3]-img_height),0)
        padding_left = max(int(-bbox[0]),0)
        padding_right = max(int(bbox[0]+bbox[2]-img_width),0)
        img_padding = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_REFLECT)
        #print(img_padding.shape,np.pad(img.astype(np.float32), ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)), 'reflect').shape)
        blur_size = int(img.shape[0]//512*5)//2*2 +1

        img_padding = img_padding.astype(np.float32)
        h, w, _ = img_padding.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        pad = [padding_left+1e-6, padding_top+1e-6, padding_right+1e-6, padding_bottom+1e-6]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))

        low_res = cv2.resize(img_padding, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        # blur = qsize * 0.02 * 0.1
        low_res = cv2.GaussianBlur(low_res, (blur_size, blur_size), 0)
        low_res = cv2.resize(low_res, (img_padding.shape[1], img_padding.shape[0]), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        # cv2.imshow('low_res', cv2.resize(low_res, (0, 0), fx=0.5, fy=0.5).astype(np.uint8))
        # cv2.waitKey(0)
        img_padding += (low_res - img_padding) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        median = cv2.resize(img_padding.astype(np.uint8), (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        median = np.median(median, axis=(0, 1))
        img_padding += (median - img_padding) * np.clip(mask, 0.0, 1.0)
        img_padding = np.uint8(np.clip(np.rint(img_padding), 0, 255))

        # cv2.imshow('img_padding', cv2.resize(img_padding, (0, 0), fx=0.5, fy=0.5).astype(np.uint8))
        # cv2.waitKey(0)

        temp_bbox = np.array([padding_left+bbox[0], padding_top+bbox[1], bbox[2], bbox[3]])
        temp_bb_c_x = float(temp_bbox[0] + 0.5 * temp_bbox[2])
        temp_bb_c_y = float(temp_bbox[1] + 0.5 * temp_bbox[3])
        temp_bb_width = float(temp_bbox[2])
        temp_bb_height = float(temp_bbox[3])
        temp_trans = gen_trans_from_patch_cv(temp_bb_c_x, temp_bb_c_y, temp_bb_width,temp_bb_height, out_shape[1], out_shape[0], scale, rot)
        img_patch = cv2.warpAffine(img_padding, temp_trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)

    else:
        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
        # print('trans:',trans.shape,trans)
        img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
        img_patch = img_patch.astype(np.float32)
        inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                            inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

