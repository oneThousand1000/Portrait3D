import math
import json
import os.path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import random

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    # theta = math.atan2(y, x)
    # phi = math.acos(z / r)
    # return r, theta, phi
    theta = math.atan2(z, x) # 0~2pi
    phi = math.acos(y / r) # 0~pi
    return r, theta, phi

thetas = []
phis = []
thetas_imgs = []
phis_imgs = []

stride = 2
stride_rad = stride / 180 * math.pi

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


for i in range(180//stride):
    phis_imgs.append([])
for i in range(360//stride):
    thetas_imgs.append([])
for i in range(0,5):
    path = f'G:/full-head-dataset/pexels/{i * 1000:08d}'
    image_list = glob.glob(f'{path}/aligned_images/*')
    result_json_path = os.path.join(path, 'result.json')
    with open(result_json_path, 'r') as f:
        result = json.load(f)




    for aligned_image_path in image_list:
        aligned_image_name = os.path.basename(aligned_image_path)

        camera_pose = result[aligned_image_name]['camera_pose']
        camera_pose = np.reshape(camera_pose, (4, 4))
        #radius = np.linalg.norm(camera_pose[:3,3])
        _, theta, phi = cartesian_to_spherical(camera_pose[0,3], camera_pose[1,3], camera_pose[2,3])

        thetas.append(theta)
        phis.append(phi)

        flip_camerapose_in_pyrender = np.array(result[aligned_image_name]['normalized_camerapose_in_pyrender'])
        flip_camerapose_in_pyrender = flip_yaw(flip_camerapose_in_pyrender)
        flip_world2camera_matrix = np.linalg.inv(flip_camerapose_in_pyrender)
        flip_world2camera_matrix[[1, 2]] *= -1
        camera_pose = np.linalg.inv(flip_world2camera_matrix)
        _, theta, phi = cartesian_to_spherical(camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3])

        thetas.append(theta)
        phis.append(phi)



plt.scatter(thetas, phis)
plt.show()

#         if abs(theta - np.pi/2) < 0.1:
#             phi_bin = int(phi/stride_rad)
#             phis_imgs[phi_bin].append(aligned_image_path)
# 
# count = 0
# for i in range(len(phis_imgs)):
#     if len(phis_imgs[i]) > 0:
#         cv2.imwrite(f'G:/full-head-dataset/pexels/theta_phi/{count}.png', cv2.imread(random.choice(phis_imgs[i])))
#         count+=1