import os
import argparse

# python runmy.py
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='')
args = parser.parse_args()

path = args.input_dir

root = os.path.dirname(os.path.abspath(__file__))
print(root)




head_box_path = f'{path}/head_bbox_yolov5_crowdhuman.json'
if not os.path.exists(path) or not os.path.exists(head_box_path):
    raise Exception('path or head_box_path not exists')


data_dir = os.path.join(root,'data')
os.chdir(os.path.join(root,'demo'))
command = f'python extract_camera_parameter.py --gpu 0  --input_dir {path} --output_dir {path}  --data_dir {data_dir}'
print(command)
os.system(command)

# os.chdir(os.path.join(root,'MANIQA'))
# command = f'python imagedups.py -r -d -N -p {path}/aligned_images'
# print(command)
# os.system(command)

# command = f'python remove_blurr_images.py  --input_dir {path}'
# print(command)
# os.system(command)
#
# command = f'python delete_images.py  --input_dir {path}'
# print(command)
# os.system(command)
