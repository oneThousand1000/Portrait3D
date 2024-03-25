import glob
import os

import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument('--test_data_dir', type=str,default='../test_data')
parser.add_argument('--df_ckpt', type=str,default='SG161222/Realistic_Vision_V5.1_noVAE')
parser.add_argument('--sample_num', type=int,default=6)
parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

opt = parser.parse_args()
test_data_dir = opt.test_data_dir
df_ckpt = opt.df_ckpt
sample_num = opt.sample_num
scale = opt.scale

for sub_dir in glob.glob(os.path.join(test_data_dir, '*')):
    prompt_path = os.path.join(sub_dir, 'prompt.txt')
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
        # print(prompt)
        samples_dir = os.path.join(sub_dir, 'samples')
        seed = random.randint(0, 100000)
        if not os.path.exists(samples_dir):
            cmd = f'python scripts/txt2realistic_human.py --outdir {sub_dir} --seed {seed} --H 512 --W 512 --n_samples 1 --scale {scale} --n_iter {sample_num} --prompt "{prompt}" --plms  --ckpt {df_ckpt}'
            print(cmd)
            os.system(cmd)
