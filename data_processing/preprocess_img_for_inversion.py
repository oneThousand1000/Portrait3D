import glob

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test_data_dir', type=str,default='../test_data')


opt = parser.parse_args()
test_data_dir = opt.test_data_dir


for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):
    samples_dir = os.path.join(sub_dir, 'samples')

    if os.path.exists(samples_dir):
        new_crop_samples_dir = os.path.join(sub_dir, 'samples_new_crop')
        if os.path.exists(new_crop_samples_dir):
            continue
        images_dir = os.path.join(samples_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        for image in glob.glob(os.path.join(samples_dir, '*.png')):
            os.rename(image, os.path.join(images_dir, os.path.basename(image)))

root = os.path.dirname(os.path.abspath(__file__))
print(root)





os.chdir(root)
# os.system(cmd)
for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):
    samples_dir = os.path.join(sub_dir, 'samples')

    if os.path.exists(samples_dir):
        new_crop_samples_dir = os.path.join(sub_dir, 'samples_new_crop')
        if os.path.exists(new_crop_samples_dir):
            continue
        cmd = f'python prepare_data.py --input_dir {samples_dir}'
        os.system(cmd)
        # os.system(cmd)


# os.system(cmd)
for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):
    samples_dir = os.path.join(sub_dir, 'samples')

    if os.path.exists(samples_dir):
        new_crop_samples_dir = os.path.join(sub_dir, 'samples_new_crop')
        if os.path.exists(new_crop_samples_dir):
            continue
        cmd = f'python runmy.py --input_dir {samples_dir}'
        os.system(cmd)
        # os.system(cmd)


# os.system(cmd)
os.chdir(os.path.join(root,'detectron2/projects/DensePose'))
for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):
    samples_dir = os.path.join(sub_dir, 'samples')

    if os.path.exists(samples_dir):
        new_crop_samples_dir = os.path.join(sub_dir, 'samples_new_crop')
        if os.path.exists(new_crop_samples_dir):
            continue
        cmd = f'python apply_net.py show configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml R_101_FPN_DL_soft_s1x.pkl {samples_dir}/aligned_images  dp_vertex  --output {samples_dir}/seg --min_score 0.8'
        os.system(cmd)
        # os.system(cmd)


os.chdir(root)
for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):

    samples_dir = os.path.join(sub_dir, 'samples')
    if os.path.exists(samples_dir):
        new_crop_samples_dir = os.path.join(sub_dir, 'samples_new_crop')
        if os.path.exists(new_crop_samples_dir):
            continue
        cmd = f'python runmy_new_crop.py --input_dir {samples_dir}'
        os.system(cmd)
        # os.system(cmd)


for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):
    samples_dir = os.path.join(sub_dir, 'samples')
    if os.path.exists(samples_dir):
        new_crop_samples_dir = os.path.join(sub_dir, 'samples_new_crop')
        new_crop_mask_samples_dir = os.path.join(sub_dir, 'samples_new_crop/mask')
        if os.path.exists(new_crop_mask_samples_dir):
            continue
        os.makedirs(new_crop_mask_samples_dir, exist_ok=True)
        cmd = f'python segmentation_example.py --base_path {new_crop_samples_dir}'
        os.system(cmd)
        # os.system(cmd)





