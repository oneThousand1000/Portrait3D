## Running 3DCrowdNet
In this repository, we provide training and testing codes for 3DPW-Crowd (Table 5) and 3DPW (Table 8).
We use the pre-trained ResNet-50 weights of [xiao2018simple](https://github.com/microsoft/human-pose-estimation.pytorch) to achieve faster convergence, but you can get the same result by training longer.
Download the [file of weights](https://drive.google.com/drive/folders/1UsntO3wdIHOiajcb8oicMhQ82SmFvulp?usp=sharing) and place it under `${ROOT}/tool/`.

### Train  
Use the appropriate config file to reproduce results.
For example, to reproduce 3DPW-Crowd (Table 5), run 
```bash  
python train.py --amp --continue --gpu 0-3 --cfg ../assets/yaml/3dpw_crowd.yml
```  
Remove `--continue` if you don't want to the use pre-trained ResNet-50 weights.  
Add `--exp_dir` argument to resume training.

> Note: CUDA version may matter on the training time. Normally it takes 2hours per epoch when I used cuda-10.1. But when I use cuda-10.2, it takes 4~6hours per epoch. Pytorch version is 1.6.0.

### Test  
Download the experiment directories from [here](https://drive.google.com/drive/folders/19ntGuC0zaXQa3cCN_2Ox_hWYX3nLLP2J?usp=sharing) and place them under `${ROOT}/output/`.  
To evaluate on 3DPW-Crowd (Table 5), run 
```bash  
python test.py --gpu 0-3 --cfg ../assets/yaml/3dpw_crowd.yml --exp_dir ../output/exp_03-28_18:26 --test_epoch 6 
```  
To evaluate on 3DPW (Table 8), run 
```bash  
python test.py --gpu 0-3 --cfg ../assets/yaml/3dpw.yml --exp_dir ../output/exp_04-06_23:43 --test_epoch 10
``` 
You can replace the `--exp_dir` with your own experiments.