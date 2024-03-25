import os
import argparse

# python runmy.py
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='')
args = parser.parse_args()

path = args.input_dir

root = os.path.dirname(os.path.abspath(__file__))
print(root)


os.chdir(os.path.join(root,'demo'))

data_dir = os.path.join(root,'data')
command = f'python new_crop_use_densepose.py --gpu 0  --input_dir  {path}  --output_dir  {path}_new_crop  --data_dir {data_dir}'
print(command)
os.system(command)
