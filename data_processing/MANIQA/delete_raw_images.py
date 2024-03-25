import os
import glob
import argparse
import json
import numpy as np
parser = argparse.ArgumentParser(description='Test keypoints network')
    # general

parser.add_argument('--input_dir',
                        help='experiment configure file name',
                        required=True,
                        type=str)
args = parser.parse_args()
count = 0
for image_path in glob.glob(os.path.join(args.input_dir, 'images/*')):
    image_name = os.path.basename(image_path).split('.')[0]
    #print(os.path.join(args.input_dir, 'aligned_images',image_name + '*'))
    if len(glob.glob(os.path.join(args.input_dir, 'aligned_images',image_name + '*'))) == 0:
        os.remove(image_path)
        print(image_path)
        count+=1
print(count)

json_save_path = os.path.join(args.input_dir, 'result.json')
with open(json_save_path, 'r') as f:
    results = json.load(f)
# check:
for image_path in glob.glob(os.path.join(args.input_dir, 'aligned_images/*')):
    raw_image_name = results[os.path.basename(image_path)]['raw_image_name']
    if not os.path.exists(os.path.join(args.input_dir, 'images',raw_image_name)):
        raise Exception(image_path)




