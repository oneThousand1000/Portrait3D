#!/usr/bin/env python

import functools
import os.path
import random
import sys
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image
import pickle




def load_pascal_occluder(pascal_voc_root_path):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_person = (obj.find('name').text == 'person')
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

            if object_with_mask.size == 0:
                continue

            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

    print("total # of occluders: ", len(occluders))
    return occluders

def load_coco_person_occluder(data_path, data_split):
    img_dir_path = os.path.join(data_path, f'{data_split}2017')
    part_seg_path = os.path.join(data_path, 'densepose_output', 'DensePose_maskRCNN_output')

    dp_dict = load_dp_result(part_seg_path, data_split)
    print("loaded dp result..., total imgs: ", len(dp_dict.keys()))
    from densepose.data.structures import DensePoseResult
    from timer import Timer
    load_timer = Timer()

    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    for img_name in dp_dict.keys():
        img_path = os.path.join(img_dir_path, img_name)
        load_timer.tic()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = img[:, :, ::-1].copy()
        # img = np.asarray(PIL.Image.open(img_path))
        load_timer.toc()

        dp_outputs = dp_dict[img_name]

        for output in dp_outputs:
            encoded_dp = output['dp']
            iuv_arr = DensePoseResult.decode_png_data(*encoded_dp)
            _, h, w = iuv_arr.shape
            dp_bbox = output['bbox']
            xmin, ymin = int(dp_bbox[0] + 0.5), int(dp_bbox[1] + 0.5)
            xmax, ymax = xmin+w, ymin+h

            object_mask = (iuv_arr[0] != 0).astype(np.uint8) * 255
            object_image = img[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 5000:
                # Ignore small objects or low resolution objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

            if object_with_mask.size == 0:
                continue

            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

        if len(occluders) > 5000:
            break

    print("img load time: ", load_timer.total_time)
    print("total # of occluders: ", len(occluders))
    return occluders

def load_dp_result(part_seg_path, data_split):
    print(f'Load DensePose Result of COCO {data_split} set')
    data_path = os.path.join(part_seg_path, f'coco_{data_split}.pkl')
    with open(data_path, 'rb') as f:
        raw_data_list = pickle.load(f)

    data_dict = {}
    for rd in raw_data_list:
        key = rd['file_name'].split('/')[-1]
        scores = rd['scores']
        pred_data_list = []
        for idx in range(len(scores)):
            if scores[idx] > 0.5:
                pred_data = {}
                pred_data['bbox'] = rd['pred_boxes_XYXY'][idx]
                pred_data['dp'] = rd['pred_densepose'].results[idx]
                pred_data_list.append(pred_data)
        data_dict[key] = pred_data_list
    return data_dict

def occlude_with_objects(im, occluders):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    count = np.random.randint(1, 5)

    for _ in range(count):
        occluder = random.choice(occluders)
        im_scale_factor = min(width_height) / max(occluder.shape[:2])

        random_scale_factor = np.random.uniform(0.2, 0.5)
        scale_factor = random_scale_factor * im_scale_factor

        try:
            occluder = resize_by_factor(occluder, scale_factor)
        except Exception as e:
            print("error")
            continue

        # center = np.random.uniform([0, 0], width_height)
        center = np.random.uniform(width_height/8, width_height/8*7)
        paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).
    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)

    return im_dst


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))



def main():
    """Demo of how to use the code"""

    # path = 'something/something/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    path = sys.argv[1]

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_pascal_occluder(pascal_voc_root_path=path)
    print('Found {} suitable objects'.format(len(occluders)))

    original_im = cv2.resize(skimage.data.astronaut(), (256, 256))
    fig, axarr = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axarr.ravel():
        occluded_im = occlude_with_objects(original_im, occluders)
        ax.imshow(occluded_im, interpolation="none")
        ax.axis('off')

    fig.tight_layout(h_pad=0)
    # plt.savefig('examples.jpg', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dp_path = '/home/redarknight/projects/detectron2/projects/DensePose/'
    sys.path.insert(0, dp_path)
    occluder = load_coco_person_occluder('/media/disk2/hongsuk/data/COCO/2017/', data_split='train')
    # img = occlude_with_objects(dummy, occluder)
