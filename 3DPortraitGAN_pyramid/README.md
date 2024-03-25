# 3DPortraitGAN_pyramid Training

**Note: Upon the acceptance of our [3DPortraitGAN](https://arxiv.org/abs/2307.14770), we plan to release our 360°PHQ dataset to facilitate reproducibility of research. We encourage you to utilize our provided pre-trained models. Stay tuned for updates! **



## Training

```shell
cd 3DPortraitGAN_pyramid

# stage 1
python  train.py \
	--outdir=./training-runs/stage1 --cfg=full-head \
	--data=$DATASET_PATH/360PHQ-512.zip  --seg_data=$DATASET_PATH/360PHQ-512-mask.zip \
    --gpus=8 --batch=32  --gamma=5.0 --cbase=18432 --cmax=144 \
	--gamma_seg=5.0 --use_torgb_raw=1 --decoder_activation="none" \
	--bcg_reg_prob 0.2  --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --back_repeat=1 \
	--gen_pose_cond=True --gpc_reg_prob=0.7  --mirror=True  --data_rebalance=False  --image-snap=25  --kimg=20000 \
	--neural_rendering_resolution_initial=64  \
	--pose_loss_weight=10 --input_pose_params_reg_loss_weight=5 --input_pose_params_reg_loss_kimg=200 \
	--train_g_pose_branch=True \
	--explicitly_symmetry=True \
	--metric_pose_sample_mode=G_predict 


# stage 2
python  train.py \
	--outdir=./training-runs/stage2 --cfg=full-head \
	--data=$DATASET_PATH/360PHQ-512.zip  --seg_data=$DATASET_PATH/360PHQ-512-mask.zip \
	--gpus=8 --batch=32  --gamma=5.0 --cbase=18432 --cmax=144 \
	--gamma_seg=5.0 --use_torgb_raw=1 --decoder_activation="none" \
	--bcg_reg_prob 0.2  --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --back_repeat=1 \
	--gen_pose_cond=True --gpc_reg_prob=0.7  --mirror=True  --data_rebalance=False  --image-snap=25  --kimg=20000 \
	--neural_rendering_resolution_initial=64  \
	--pose_loss_weight=10 --input_pose_params_reg_loss_weight=5 --input_pose_params_reg_loss_kimg=200 \
	--train_g_pose_branch=False \
	--explicitly_symmetry=True \
	--metric_pose_sample_mode=D_predict \
	--resume=stage1.pkl  --resume_kimg=NUM_KIMGS
        
# stage 3
python  train.py \
	--outdir=./training-runs/stage3 --cfg=full-head \
	--data=$DATASET_PATH/360PHQ-512.zip  --seg_data=$DATASET_PATH/360PHQ-512-mask.zip \
	--gpus=8 --batch=32  --gamma=5.0 --cbase=18432 --cmax=144 \
	--gamma_seg=5.0 --use_torgb_raw=1 --decoder_activation="none" \
	--bcg_reg_prob 0.2  --triplane_depth 3 --density_noise_fade_kimg 200 --density_reg 0 --back_repeat=1 \
	--gen_pose_cond=True --gpc_reg_prob=0.7  --mirror=True  --data_rebalance=False  --image-snap=25  --kimg=20000 \
	--neural_rendering_resolution_initial=64  --neural_rendering_resolution_final=128 \
	--neural_rendering_resolution_fade_kimg=1000 \
	--pose_loss_weight=10 --input_pose_params_reg_loss_weight=5 --input_pose_params_reg_loss_kimg=200 \
	--train_g_pose_branch=False \
	--explicitly_symmetry=True \
	--metric_pose_sample_mode=D_predict \
	--resume=stage2.pkl  --resume_kimg=NUM_KIMGS

```

