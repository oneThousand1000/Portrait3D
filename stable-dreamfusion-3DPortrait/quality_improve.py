# 190,140,22,129,104,113,133,164,15,31,72,135,83,149,85,169
for name in [91,111,96,53,143]:

    print('cd F:/high_quality_3DPortraitGAN/exp/3DPortraitGAN-hierarchy-v2')
    print('activate 3dportraitgan')

    cmd = f'python gen_quality_improve_data_from_triplane.py --data_dir=F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion-hierarchy-v2/output/better_direction_prompt/{name}    --grid=1x1   --network=F:/high_quality_3DPortraitGAN/exp/3DPortraitGAN-hierarchy-v2/models/model.pkl'

    print(cmd)


    print('cd F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion-hierarchy-v2')
    print('activate ldm_3dgan_kaolin')

    cmd = f'python guidance/sdedit.py  --data_dir F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion-hierarchy-v2/output/better_direction_prompt/{name} --hf_key F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion/pretrained/SG161222Realistic_Vision_V5.1_noVAE -H 512 -W 512 --seed 42'

    print(cmd)


    print('cd F:/high_quality_3DPortraitGAN/exp/3DPortraitGAN-hierarchy-v2')
    print('activate 3dportraitgan')
    cmd = f'python optimize_trigrid.py --data_dir=F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion-hierarchy-v2/output/better_direction_prompt/{name}    --grid=1x1   --network=F:/high_quality_3DPortraitGAN/exp/3DPortraitGAN-hierarchy-v2/models/model.pkl'
    print(cmd)


    cmd = f'python gen_videos_shapes_from_optimized_triplane.py  --data_dir=F:/high_quality_3DPortraitGAN/exp/stable-dreamfusion-hierarchy-v2/output/better_direction_prompt/{name}    --grid=1x1   --network=F:/high_quality_3DPortraitGAN/exp/3DPortraitGAN-hierarchy-v2/models/model.pkl'
    print(cmd)