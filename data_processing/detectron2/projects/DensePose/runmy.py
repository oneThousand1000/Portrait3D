import os
import argparse
# dataset_name = 'pexels'
# for i in range(50):
#     path = f'G:/full-head-dataset/{dataset_name}/{i * 1000:08d}'
#
#     cmd = f'python apply_net.py show configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml R_101_FPN_DL_soft_s1x.pkl {path}/aligned_images dp_vertex  --output {path}/seg --min_score 0.8'
#     print(cmd)
#     os.system(cmd)


dataset_name = 'unsplash'
for i in range(58,64):
    path = f'G:/full-head-dataset/{dataset_name}/{i * 1000:08d}'

    cmd = f'python apply_net.py show configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml R_101_FPN_DL_soft_s1x.pkl {path}/aligned_images dp_vertex  --output {path}/seg --min_score 0.8'
    print(cmd)
    os.system(cmd)
