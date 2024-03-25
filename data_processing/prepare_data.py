import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='')
args = parser.parse_args()

input_dir = args.input_dir

root = os.path.dirname(os.path.abspath(__file__))
print(root)


os.chdir(os.path.join(root,'HigherHRNet-Human-Pose-Estimation'))
command = f'python tools/get_keypoints.py  --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml  --input_dir {input_dir} TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth '
print(command)
os.system(command)
# head-detection

os.chdir(os.path.join(root,'yolov5_crowdhuman'))
command = f'python detect_head_bbox.py --weights crowdhuman_yolov5m.pt --source {input_dir}   --heads'
print(command)
os.system(command)
