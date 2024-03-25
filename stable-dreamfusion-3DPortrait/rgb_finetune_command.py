import os 

import glob
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--trigrid_decoder_ckpt', type=str)
# parser.add_argument('--inversion_name', type=str)
# opt = parser.parse_args()
# trigrid_decoder_ckpt = opt.trigrid_decoder_ckpt
# inversion_name = opt.inversion_name

count = 0

inversion_name = 'hierarchy_inversion_4000'
trigrid_decoder_ckpt ='F:\high_quality_3DPortraitGAN\exp/3DPortraitGAN-hierarchy\models/network-snapshot-004000_decoder.ckpt'
for prompt_file in glob.glob(f'F:/high_quality_3DPortraitGAN/exp/test_data/*/prompt.txt'):

    with open(prompt_file, 'r') as f:
        prompt = f.read()
        
    prompt = prompt.replace('/n', '')
    
    dir_ = os.path.dirname(prompt_file)
    name = dir_.split('/')[-1].split('\\')[-1]
    #print(dir_.split('/'),dir_.split('/')[-1].split('\\'))
    count_ = 0
    # if len(glob.glob(f'F:\high_quality_3DPortraitGAN\exp\stable-dreamfusion\output/2023-11-*-with-inversion-initialization-{name}_*')) > 0:
    #     continue
    for inversion_trigrid in glob.glob(f'{dir_}/samples_new_crop/{inversion_name}/*/inversion_trigrid.pkl'):
        name_ =name+ f'_{count_}'
        cmd = f'python main_3DPortraitGAN.py  --workspace output/2023-11-22-{name_}_{inversion_name}   --save_guidance --backbone trigrid_heirarchy_aggregate --latent_iter_ratio 0 --t_range 0.02 0.4 --vram_O --w 128 --h 128 --H 512 --W 512 --iters 3000 --text "{prompt}"   --hf_key F:\high_quality_3DPortraitGAN\exp\stable-dreamfusion\pretrained\SG161222Realistic_Vision_V5.1_noVAE  --trigrid_path  {inversion_trigrid}  --trigrid_decoder_ckpt {trigrid_decoder_ckpt}'
        print(cmd)
        count_ += 1
        break
    count += 1



