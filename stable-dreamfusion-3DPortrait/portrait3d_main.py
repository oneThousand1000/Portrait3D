import os 

import glob
import random
import argparse
#
parser = argparse.ArgumentParser()
parser.add_argument('--trigrid_decoder_ckpt', type=str)
parser.add_argument('--inversion_name', type=str)
parser.add_argument('--network_path', type=str)
parser.add_argument('--test_data_dir', type=str,default='../test_data')
parser.add_argument('--df_ckpt', type=str,default='SG161222/Realistic_Vision_V5.1_noVAE')

opt = parser.parse_args()
trigrid_decoder_ckpt = opt.trigrid_decoder_ckpt
inversion_name = opt.inversion_name
network_path = opt.network_path
test_data_dir = opt.test_data_dir
df_ckpt = opt.df_ckpt

exp_name = 'text_to_3dportrait'

# the current file's path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root)

todo = glob.glob(os.path.join(test_data_dir, '*/prompt.txt'))
for prompt_file in todo:

    with open(prompt_file, 'r') as f:
        prompt = f.read()
        
    prompt = prompt.replace('\n', '')
    
    dir_ = os.path.dirname(prompt_file)
    name = dir_.split('/')[-1].split('\\')[-1]

    if os.path.exists(f'output/{exp_name}/{name}/results_final/small_pose_final.mp4'):
        continue
    trigrid_list = glob.glob(f'{dir_}/samples_new_crop/{inversion_name}/*/inversion_trigrid.pkl')
    if len(trigrid_list) == 0:
        continue
    inversion_trigrid = trigrid_list[0]


    # change dir
    os.chdir(os.path.join(root, 'stable-dreamfusion-3DPortrait'))
    cmd = f'python main_3DPortraitGAN_cam.py  --workspace output/{exp_name}/{name} --latent_iter_ratio 0 --trigrid_lr_ratio 200 200 200 200 200 40 20 --t_range 0.02 0.4 --vram_O --w 128 --h 128 --H 512 --W 512 --iters 2000 --text "{prompt}"   --hf_key {df_ckpt}  --trigrid_path  {inversion_trigrid}  --trigrid_decoder_ckpt {trigrid_decoder_ckpt}'
    print(cmd)
    os.system(cmd)

    os.chdir(os.path.join(root, '3DPortraitGAN_pyramid'))
    cmd = f'python gen_quality_improve_data_from_triplane.py --data_dir={root}/stable-dreamfusion-3DPortrait/output/{exp_name}/{name}    --grid=1x1   --network={network_path}'
    print(cmd)
    os.system(cmd)

    os.chdir(os.path.join(root, 'stable-dreamfusion-3DPortrait'))
    cmd = f'python guidance/sdedit.py  --data_dir {root}/stable-dreamfusion-3DPortrait/output/{exp_name}/{name} --hf_key {df_ckpt} -H 512 -W 512 --seed 42 --test_data_dir={test_data_dir}'
    print(cmd)
    os.system(cmd)

    os.chdir(os.path.join(root, '3DPortraitGAN_pyramid'))
    cmd = f'python optimize_trigrid.py --data_dir={root}/stable-dreamfusion-3DPortrait/output/{exp_name}/{name}    --grid=1x1   --network={network_path}'
    print(cmd)
    os.system(cmd)

    os.chdir(os.path.join(root, '3DPortraitGAN_pyramid'))
    cmd = f'python gen_videos_shapes_from_optimized_triplane.py  --data_dir={root}/stable-dreamfusion-3DPortrait/output/{exp_name}/{name}    --grid=1x1   --network={network_path}'
    print(cmd)
    os.system(cmd)


