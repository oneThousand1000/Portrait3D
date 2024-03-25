import argparse
import os.path
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import json
import numpy as np

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    skeleton_path = os.path.join(opt.source,'2d_pose_result_hrnet.json')
    source = os.path.join(opt.source,'images')

    with open(skeleton_path) as f:
        pose2d_result = json.load(f)

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    bbox_results = {}
    result_json_path = os.path.join(opt.source, 'head_bbox_yolov5_crowdhuman.json')
    print('result_json_path', result_json_path)

    if os.path.exists(result_json_path):
        with open(result_json_path) as f:
            bbox_results = json.load(f)


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source,bbox_results, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()





    for path, img, im0s, vid_cap in dataset:
        img_name = os.path.basename(path)
        if img_name in bbox_results:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        coco_joint_list = pose2d_result[img_name]
        bbox_list_wo_sort = []


        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    label = f'{names[int(cls)]} {conf:.2f}'
                    if 'head' in label:
                        bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]-xyxy[0]), float(xyxy[3]-xyxy[1])] # x, y, w, h
                        #print(im0.shape)
                        bbox_list_wo_sort.append(bbox)





                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if opt.heads or opt.person:
                            if 'head' in label and opt.heads:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            if 'person' in label and opt.person:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                print('resize to',(512, int(im0.shape[0]/im0.shape[1]*512)))
                cv2.imshow(str(p), cv2.resize(im0, (512, int(im0.shape[0]/im0.shape[1]*512))))
                cv2.waitKey(0)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #
            #             fourcc = 'mp4v'  # output video codec
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)

        # sort bbox
        bbox_list_sort = []

        for idx in range(len(coco_joint_list)):
            coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]

            face_points = coco_joint_img[:5, :3]
            face_center = np.mean(face_points[:,:2], axis=0, keepdims=True)
            #print('face_points', face_points)
            #
            clip_tresh = 0.5
            face_points_valid = face_points[face_points[:,2] > clip_tresh]
            face_center_valid = np.mean(face_points_valid[:,:2], axis=0, keepdims=True)
            #print('face_points_valid', face_points_valid)
            # if valid face num >=1, match bbox to coco joint
            if face_points_valid.shape[0] >= 1:
                for bbox in bbox_list_wo_sort:
                    relax = 0.1
                    relaxed_bbox = [bbox[0] - bbox[2] * relax, bbox[1] - bbox[3] * relax, bbox[2] * (1 + 2 * relax),
                                    bbox[3] * (1 + 2 * relax)]
                    check = True
                    for points_idx in range(face_points.shape[0]):
                        if not (relaxed_bbox[0] <= face_points[points_idx][0] <= relaxed_bbox[0] + relaxed_bbox[2] and
                                relaxed_bbox[1] <= face_points[points_idx][1] <= relaxed_bbox[1] + relaxed_bbox[3]):
                            check = False
                            break
                    if check:
                        bbox_list_sort.append({'bbox':bbox,'score':1.0})
                        break
            else:
                # if no valid face, use face center to match bbox (nearest )
                min_dist = 1e8
                min_bbox = None
                for bbox in bbox_list_wo_sort:
                    bbox_c = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    if np.linalg.norm(bbox_c - face_center) < min_dist:
                        min_dist = np.linalg.norm(bbox_c - face_center)
                        min_bbox = bbox

                if min_bbox is not None:
                    bbox_list_sort.append({'bbox':min_bbox,'score':1.0})




            # no bbox detec, use coco joint to generate bbox
            if len(bbox_list_sort) != idx+1:
                # face_points_valid = face_points[face_points[:, 2] > clip_tresh]
                # face_center_valid = np.mean(face_points_valid, axis=0, keepdims=True)

                if face_points_valid.shape[0] < 2:
                    bbox_list_sort.append({'bbox':[],'score':0.0})
                    continue

                head_stride = max(np.max(face_points[:, 0]) - np.min(face_points[:, 0]),
                                  np.max(face_points[:, 1]) - np.min(face_points[:, 1])) * 1.3
                temp_bbox = [face_center[0][0]-head_stride/2, face_center[0][1]-head_stride/2, head_stride, head_stride]
                bbox_list_sort.append({'bbox':temp_bbox,'score':0.0})

        if len(bbox_list_sort) != len(coco_joint_list):
            raise ValueError('bbox_list_sort and coco_joint_list have different length')

        bbox_results[img_name] = bbox_list_sort
        # save bbox
    with open(result_json_path, 'w') as f:
        json.dump(bbox_results, f)


    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--person', action='store_true', help='displays only person')
    parser.add_argument('--heads', action='store_true', help='displays only person')
    opt = parser.parse_args()
    print(opt)
    #check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
