import os

import glob
import cv2
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general

    parser.add_argument('--input_dir',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()



    image_list = glob.glob(os.path.join(args.input_dir, 'aligned_images/*'))

    for image_path in tqdm(image_list):
        # data load
        # model defination
        image = cv2.imread(image_path)
        img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
        if imageVar < 4:
            os.remove(image_path)

