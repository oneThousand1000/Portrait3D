import os
import argparse

parser = argparse.ArgumentParser(description=' ')
# general

for i in range(0,1):
    path = f'E:/project/unsplash/{i*1000:08d}'
    head_box_path = f'{path}/head_bbox_yolov5_crowdhuman.json'
    if not os.path.exists(path) or not os.path.exists(head_box_path):
        continue
    
    os.chdir('E:/project/3DCrowdNet_upper_body-main/demo')
    command =f'python extract_camera_parameter.py --gpu 0  --input_dir {path} --output_dir {path}  --data_dir E:/project/3DCrowdNet_upper_body-main/data'
    print(command)
    os.system(command)

    os.chdir('E:/project/3DCrowdNet_upper_body-main/MANIQA')
    command =f'python imagedups.py -r -d -N -p {path}/aligned_images'
    print(command)
    os.system(command)

    command = f'python remove_blurr_images.py  --input_dir {path}'
    print(command)
    os.system(command)

    command = f'python delete_images.py  --input_dir {path}'
    print(command)
    os.system(command)