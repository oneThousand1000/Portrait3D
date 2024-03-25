import os
import glob
import argparse
parser = argparse.ArgumentParser(description='Test keypoints network')
    # general

parser.add_argument('--input_dir',
                        help='experiment configure file name',
                        required=True,
                        type=str)
args = parser.parse_args()
count = 0
for image_path in glob.glob(os.path.join(args.input_dir, 'aligned_images/*')):
    image_name = os.path.basename(image_path)
    if not os.path.exists(os.path.join(args.input_dir, 'visualization',image_name)):
        os.remove(image_path)
        count+=1
print(count)
count = 0
#for image_path in glob.glob('G:/full-head-dataset/pexels/00000000/visualization/*'):
for image_path in glob.glob(os.path.join(args.input_dir, 'visualization/*')):
    image_name = os.path.basename(image_path)
    #if not os.path.exists('G:/full-head-dataset/pexels/00000000/aligned_images/' + image_name):
    if not os.path.exists(os.path.join(args.input_dir, 'aligned_images',image_name)):
        os.remove(image_path)
        count+=1



print(count)