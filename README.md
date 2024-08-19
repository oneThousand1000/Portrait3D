# Portrait3D 

This is the official code repository for our SIG'24 paper:  
> **Portrait3D: Text-Guided High-Quality 3D Portrait Generation Using Pyramid Representation and GANs Prior**
>
> **ACM Transactions on Graphics (Proceedings of SIGGRAPH 2024)**
>
> [Yiqian Wu](https://onethousandwu.com/), [Hao Xu](https://xh38.github.io/), [Xiangjun Tang](https://yuyujunjun.github.io/), [Xien Chen](https://vision.cs.yale.edu/members/xien-chen.html), [Siyu Tang](https://inf.ethz.ch/people/person-detail.MjYyNzgw.TGlzdC8zMDQsLTg3NDc3NjI0MQ==.html),  Zhebin Zhang, Chen Li, [Xiaogang Jin*](http://www.cad.zju.edu.cn/home/jin)

<div align="center">

[![Project](https://img.shields.io/badge/Portrait3D-1?label=Project&color=8B93FF&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMTA3NzIxIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjkgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM4NzUiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNMTAwMS40MjMyMzggNDk0LjU5MnEyMS41MDQgMjAuNDggMjIuNTI4IDQ1LjA1NnQtMTYuMzg0IDQwLjk2cS0xOS40NTYgMTcuNDA4LTQ1LjA1NiAxNi4zODR0LTQwLjk2LTE0LjMzNnEtNS4xMi00LjA5Ni0zMS4yMzItMjguNjcydC02Mi40NjQtNTguODgtNzcuODI0LTczLjcyOC03OC4zMzYtNzQuMjQtNjMuNDg4LTYwLjQxNi0zMy43OTItMzEuNzQ0cS0zMi43NjgtMjkuNjk2LTY0LjUxMi0yOC42NzJ0LTYyLjQ2NCAyOC42NzJxLTEwLjI0IDkuMjE2LTM4LjQgMzUuMzI4dC02NS4wMjQgNjAuOTI4LTc3LjgyNCA3Mi43MDQtNzUuNzc2IDcwLjY1Ni01OS45MDQgNTUuODA4LTMwLjIwOCAyNy4xMzZxLTE1LjM2IDEyLjI4OC00MC45NiAxMy4zMTJ0LTQ0LjAzMi0xNS4zNnEtMjAuNDgtMTguNDMyLTE5LjQ1Ni00NC41NDR0MTcuNDA4LTQxLjQ3MnE2LjE0NC02LjE0NCAzNy44ODgtMzUuODR0NzUuNzc2LTcwLjY1NiA5NC43Mi04OC4wNjQgOTQuMjA4LTg4LjA2NCA3NC43NTItNzAuMTQ0IDM2LjM1Mi0zNC4zMDRxMzguOTEyLTM3Ljg4OCA4My45NjgtMzguNHQ3Ni44IDMwLjIwOHE2LjE0NCA1LjEyIDI1LjYgMjQuMDY0dDQ3LjYxNiA0Ni4wOCA2Mi45NzYgNjAuOTI4IDcwLjY1NiA2OC4wOTYgNzAuMTQ0IDY4LjA5NiA2Mi45NzYgNjAuOTI4IDQ4LjEyOCA0Ni41OTJ6TTQ0Ny40MzkyMzggMzQ2LjExMnEyNS42LTIzLjU1MiA2MS40NC0yNS4wODh0NjQuNTEyIDI1LjA4OHEzLjA3MiAzLjA3MiAxOC40MzIgMTcuNDA4bDM4LjkxMiAzNS44NHEyMi41MjggMjEuNTA0IDUwLjY4OCA0OC4xMjh0NTcuODU2IDUzLjI0OHE2OC42MDggNjMuNDg4IDE1My42IDE0Mi4zMzZsMCAxOTQuNTZxMCAyMi41MjgtMTYuODk2IDM5LjkzNnQtNDUuNTY4IDE4LjQzMmwtMTkzLjUzNiAwIDAtMTU4LjcycTAtMzMuNzkyLTMxLjc0NC0zMy43OTJsLTE5NS41ODQgMHEtMTcuNDA4IDAtMjQuMDY0IDEwLjI0dC02LjY1NiAyMy41NTJxMCA2LjE0NC0wLjUxMiAzMS4yMzJ0LTAuNTEyIDUzLjc2bDAgNzMuNzI4LTE4Ny4zOTIgMHEtMjkuNjk2IDAtNDcuMTA0LTEzLjMxMnQtMTcuNDA4LTM3Ljg4OGwwLTIwMy43NzZxODMuOTY4LTc2LjggMTUyLjU3Ni0xMzkuMjY0IDI4LjY3Mi0yNi42MjQgNTcuMzQ0LTUyLjczNnQ1Mi4yMjQtNDcuNjE2IDM5LjQyNC0zNi4zNTIgMTkuOTY4LTE4Ljk0NHoiIHAtaWQ9IjM4NzYiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48L3N2Zz4=)](https://onethousandwu.com/portrait3d.github.io/)
[![Paper](https://img.shields.io/badge/Paper%20(ACM%20Digital%20Library)-1?color=58A399&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMTQyMjM1IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjU5MTQiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNODIzLjI5NiA2MC40MTZxNjUuNTM2IDAgOTkuMzI4IDM4LjR0MzMuNzkyIDkzLjY5NnY1NDMuNzQ0cTAgMjUuNi0yMS41MDQgNDYuMDhsLTE3MS4wMDggMTYzLjg0cS0xMy4zMTIgMTEuMjY0LTIyLjUyOCAxNC4zMzZ0LTIzLjU1MiAzLjA3MkgyNTguMDQ4cS0yMy41NTIgMC00Ny4xMDQtOS43Mjh0LTQxLjk4NC0yNy42NDgtMzAuMjA4LTQzLjAwOC0xMS43NzYtNTUuODA4di02MzQuODhxMC02MC40MTYgMzMuMjgtOTYuMjU2dDk0LjcyLTM1Ljg0aDU2OC4zMnogbS0yMTUuMDQgNjQyLjA0OHExMy4zMTIgMCAyMi41MjgtOS4yMTZUNjQwIDY3MC43MnEwLTE0LjMzNi05LjIxNi0yMy4wNHQtMjIuNTI4LTguNzA0SDI4Ny43NDRxLTEzLjMxMiAwLTIyLjUyOCA4LjcwNFQyNTYgNjcwLjcycTAgMTMuMzEyIDkuMjE2IDIyLjUyOHQyMi41MjggOS4yMTZoMzIwLjUxMnogbTEyOC0xOTIuNTEycTEzLjMxMiAwIDIyLjUyOC05LjIxNlQ3NjggNDc4LjIwOHQtOS4yMTYtMjIuNTI4LTIyLjUyOC05LjIxNkgyODcuNzQ0cS0xMy4zMTIgMC0yMi41MjggOS4yMTZUMjU2IDQ3OC4yMDh0OS4yMTYgMjIuNTI4IDIyLjUyOCA5LjIxNmg0NDguNTEyeiBtNjMuNDg4LTE5MS40ODhxMTMuMzEyIDAgMjIuNTI4LTkuMjE2dDkuMjE2LTIzLjU1MnEwLTEzLjMxMi05LjIxNi0yMi41Mjh0LTIyLjUyOC05LjIxNmgtNTEycS0xMy4zMTIgMC0yMi41MjggOS4yMTZUMjU2IDI4NS42OTZxMCAxNC4zMzYgOS4yMTYgMjMuNTUydDIyLjUyOCA5LjIxNmg1MTJ6IiBwLWlkPSI1OTE1IiBmaWxsPSIjZmZmZmZmIj48L3BhdGg+PC9zdmc+)](https://dl.acm.org/doi/10.1145/3658162)
[![Arxiv](https://img.shields.io/badge/Arxiv-1?color=A34343&logo=Arxiv)](https://arxiv.org/abs/2404.10394)
[![Suppl](https://img.shields.io/badge/Supplementary-1?color=378CE7&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMTgyMzc1IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9Ijg5MDIiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNNDg2LjQgNDA5LjZIMjgyLjE4NTE0M2E0MS4zOTg4NTcgNDEuMzk4ODU3IDAgMCAwLTQxLjc2NDU3MiA0MC45NmMwIDIyLjY3NDI4NiAxOC43MjQ1NzEgNDAuOTYgNDEuNzY0NTcyIDQwLjk2aDk2LjMyOTE0M2EzNjIuNzg4NTcxIDM2Mi43ODg1NzEgMCAwIDAtOTYuMzI5MTQzIDI0NS43NmMwIDk0LjU3MzcxNCAzNi42NDQ1NzEgMTgwLjUxNjU3MSA5Ni4zMjkxNDMgMjQ1Ljc2SDE1Ni43NDUxNDNBODIuNzk3NzE0IDgyLjc5NzcxNCAwIDAgMSA3My4xNDI4NTcgOTAxLjEyVjgxLjkyQzczLjE0Mjg1NyAzNi42NDQ1NzEgMTEwLjU5MiAwIDE1Ni43NDUxNDMgMGg2NjguNzQ1MTQzYzQ2LjA4IDAgODMuNjAyMjg2IDM2LjY0NDU3MSA4My42MDIyODUgODEuOTJ2MzgxLjE0NzQyOWEzODAuMTk2NTcxIDM4MC4xOTY1NzEgMCAwIDAtNDIyLjYxOTQyOC01My4zOTQyODZ6IG0yNTUuNDg4LTE2My44NGMwLTIyLjY3NDI4Ni0xOC43MjQ1NzEtNDAuOTYtNDEuODM3NzE0LTQwLjk2SDI4Mi4xMTJhNDEuMzk4ODU3IDQxLjM5ODg1NyAwIDAgMC00MS43NjQ1NzEgNDAuOTZjMCAyMi42NzQyODYgMTguNzI0NTcxIDQwLjk2IDQxLjc2NDU3MSA0MC45Nmg0MTcuOTM4Mjg2YzIzLjExMzE0MyAwIDQxLjgzNzcxNC0xOC4yODU3MTQgNDEuODM3NzE0LTQwLjk2ek02NTguMjg1NzE0IDQ1MC41NmMxNjEuNjQ1NzE0IDAgMjkyLjU3MTQyOSAxMjguMzY1NzE0IDI5Mi41NzE0MjkgMjg2LjcyUzgxOS45MzE0MjkgMTAyNCA2NTguMjg1NzE0IDEwMjRzLTI5Mi41NzE0MjktMTI4LjM2NTcxNC0yOTIuNTcxNDI4LTI4Ni43MiAxMzAuOTI1NzE0LTI4Ni43MiAyOTIuNTcxNDI4LTI4Ni43MnogbS0xMjUuMzY2ODU3IDMyNy42OGg4My42MDIyODZ2ODEuOTJjMCAyMi42NzQyODYgMTguNjUxNDI5IDQwLjk2IDQxLjc2NDU3MSA0MC45NiAyMy4xMTMxNDMgMCA0MS43NjQ1NzEtMTguMjg1NzE0IDQxLjc2NDU3Mi00MC45NnYtODEuOTJoODMuNjAyMjg1YzIzLjExMzE0MyAwIDQxLjgzNzcxNC0xOC4yODU3MTQgNDEuODM3NzE1LTQwLjk2IDAtMjIuNjc0Mjg2LTE4LjcyNDU3MS00MC45Ni00MS44Mzc3MTUtNDAuOTZINzAwLjA1MDI4NlY2MTQuNGMwLTIyLjY3NDI4Ni0xOC42NTE0MjktNDAuOTYtNDEuNzY0NTcyLTQwLjk2YTQxLjM5ODg1NyA0MS4zOTg4NTcgMCAwIDAtNDEuNzY0NTcxIDQwLjk2djgxLjkySDUzMi45MTg4NTdhNDEuMzk4ODU3IDQxLjM5ODg1NyAwIDAgMC00MS44Mzc3MTQgNDAuOTZjMCAyMi42NzQyODYgMTguNzI0NTcxIDQwLjk2IDQxLjgzNzcxNCA0MC45NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9Ijg5MDMiPjwvcGF0aD48L3N2Zz4=)](https://drive.google.com/file/d/1LasG-urCA7rEoITHofBwEk0CloXM0DwX/view?usp=sharing)
[![Video](https://img.shields.io/badge/Video-1?color=E178C5&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMjU5OTgxIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjEyOTE2IiBpZD0ibXhfbl8xNzEyNDkwMjU5OTgyIiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCI+PHBhdGggZD0iTTYyMy4zMzAyOTE0ODQ0NDQ0IDIwMy44OTk4OTE0ODQ0NDQ0Mkg5Ny43NDc3NTI5NjAwMDAwMXMtODguMDI4NjAyNTk1NTU1NTUgMC04OC4wMjg2MDI1OTU1NTU1NSA4NS40Mzk1MjU1NDY2NjY2NnY0MzIuMzc1NzgyOTY4ODg4ODdjMCA4NS40Mzk1MjU1NDY2NjY2NiA4OC4wMjg2MDI1OTU1NTU1NSA4NS40Mzk1MjU1NDY2NjY2NiA4OC4wMjg2MDI1OTU1NTU1NSA4NS40Mzk1MjU1NDY2NjY2Nmg1MjUuNTgyNTM4NTI0NDQ0NHM4OC4wMjg2MDI1OTU1NTU1NSAwIDg4LjAyODYwMjU5NTU1NTU1LTg1LjQzOTUyNTU0NjY2NjY2VjI5MS45Mjg0OTQwODAwMDAwNmMwLTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1ek05ODMuMjExOTMwNzM3Nzc3OCAyNDcuOTE0MTkyMjEzMzMzM2MtNy43NjcyMzAwMDg4ODg4ODgtMi41ODkwNzcwNDg4ODg4ODg2LTE1LjUzNDQ1ODg3OTk5OTk5OS0yLjU4OTA3NzA0ODg4ODg4ODYtMjAuNzEyNjExODQgMi41ODkwNzcwNDg4ODg4ODg2bC0xNzMuNDY4MTI5MjggMTM0LjYzMTk4MDM3MzMzMzMzYy01LjE3ODE1Mjk2IDUuMTc4MTUyOTYtNy43NjcyMzAwMDg4ODg4ODggMTAuMzU2MzA1OTItNy43NjcyMjg4NzExMTExMTE1IDE1LjUzNDQ1ODg3OTk5OTk5OXYyMTQuODkzMzUyOTYwMDAwMDJjMCA1LjE3ODE1Mjk2IDIuNTg5MDc3MDQ4ODg4ODg4NiAxMi45NDUzODI5Njg4ODg4ODggNy43NjcyMjg4NzExMTExMTE1IDE1LjUzNDQ2MDAxNzc3Nzc3N2wxNzMuNDY4MTI5MjggMTM0LjYzMTk3OTIzNTU1NTU4YzIuNTg5MDc3MDQ4ODg4ODg4NiAyLjU4OTA3NzA0ODg4ODg4ODYgNy43NjcyMzAwMDg4ODg4ODggNS4xNzgxNTI5NiAxMi45NDUzODE4MzExMTExMTIgNS4xNzgxNTQwOTc3Nzc3NzcgMi41ODkwNzcwNDg4ODg4ODg2IDAgNS4xNzgxNTI5NiAwIDEwLjM1NjMwNzA1Nzc3Nzc3OC0yLjU4OTA3NzA0ODg4ODg4ODYgNy43NjcyMzAwMDg4ODg4ODgtMi41ODkwNzcwNDg4ODg4ODg2IDEwLjM1NjMwNTkyLTEwLjM1NjMwNTkyIDEwLjM1NjMwNTkyLTE4LjEyMzUzNTkyODg4ODg4OFYyNjYuMDM3NzI4MTQyMjIyMjVjMC03Ljc2NzIzMDAwODg4ODg4OC01LjE3ODE1Mjk2LTE1LjUzNDQ1ODg3OTk5OTk5OS0xMi45NDUzODI5Njg4ODg4ODgtMTguMTIzNTM1OTI4ODg4ODg4eiIgZmlsbD0iI2ZmZmZmZiIgcC1pZD0iMTI5MTciPjwvcGF0aD48L3N2Zz4=)](https://youtu.be/z7xWiD1p1_4)
[![Github](https://img.shields.io/github/stars/oneThousand1000/Portrait3D)](https://github.com/oneThousand1000/Portrait3D)
[![dataset-on-hf](https://huggingface.co/datasets/huggingface/badges/raw/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/onethousand/Portrait3D_gallery)

</div>

![Representative_Image](./assets/teaser-0.png)

![Representative_Image](./assets/teaser.png)



https://github.com/oneThousand1000/Portrait3D/assets/32099648/bc5721db-6edb-4d6a-a88f-07305476cd67



##  Requirements

1. Tested on Python 3.8
3. At least 12 GB of memory
4. Tested on NVIDIA RTX 3080Ti with 12 GB of memory (Windows, 1.5h per portrait)
5. Tested on NVIDIA RTX 4090 with 24 GB of memory (Linux, 0.5h per portrait)
6. CUDA>=11.6

## Installation

Clone this repo to `$PROJECT_ROOT$`.

**Create environment**

```
cd $PROJECT_ROOT$
conda env create -f environment.yaml
conda activate text_to_3dportrait
```

**Torch and torchvision Installation**

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

**OSMesa Dependencies (For Linux)**

```
sudo apt install  libosmesa6  libosmesa6-dev
```

**Installing Additional Requirements**

```
pip install -r requirements.txt
```

**kaolin Installation** 

```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu116.html
```

**Stable-diffusion Installation**

```
cd stable-diffusion
pip install -e .
cd ..
```



**SMPL Model Setup**

1. Download [SMPL_python_v.1.0.0.zip](https://smpl.is.tue.mpg.de/download.php) (version 1.0.0 for Python 2.7 (female/male. 10 shape PCs) ). Save `basicModel_f_lbs_10_207_0_v1.0.0.pkl` to `3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_FEMALE.pkl`, save `basicModel_m_lbs_10_207_0_v1.0.0.pkl` to `3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_MALE.pkl`.

2. Download [SMPLIFY_CODE_V2.ZIP](http://smplify.is.tue.mpg.de/), and save `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_NEUTRAL.pkl`.

| Download Link                                                | Save Path                                                |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| [basicModel_f_lbs_10_207_0_v1.0.0.pkl](https://smpl.is.tue.mpg.de/download.php) | 3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_FEMALE.pkl  |
| [basicModel_m_lbs_10_207_0_v1.0.0.pkl](https://smpl.is.tue.mpg.de/download.php) | 3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_MALE.pkl    |
| [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](http://smplify.is.tue.mpg.de/) | 3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_NEUTRAL.pkl |



## Inference 

### 3DPortraitGAN_pyramid Model

Our 3DPortraitGAN_pyramid draws inspiration from the 3D-aware StyleGAN2 backbone implemented in [SeanChenxy/Mimic3D](https://github.com/SeanChenxy/Mimic3D), and integrates concepts of mask guidance, background synthesis, and tri-grid representation adapted from [SizheAn/PanoHead](https://github.com/SizheAn/PanoHead). We extend our sincere gratitude for these significant contributions!

#### (Recommended) Pretrained models 

Download the pre-trained model of 3DPortraitGAN_pyramid:

| Download Link                                                | Description                                         | Save Path                      |
| ------------------------------------------------------------ | --------------------------------------------------- | ------------------------------ |
| [model_512.pkl](https://drive.google.com/file/d/1P6k4UwGGNmxa6-rQr2oyIOmAPiLAd_WE/view?usp=sharing) | Pre-trained model of 3DPortraitGAN_pyramid          | ./3DPortraitGAN_pyramid/models |
| [model_512.json](https://drive.google.com/file/d/1R6FoQXi4PyIvXtOVoKRohfOXkEkWXdJb/view?usp=sharing) | Pose prediction parameters of 3DPortraitGAN_pyramid | ./3DPortraitGAN_pyramid/models |
| [decoder_512.ckpt](https://drive.google.com/file/d/1r0Lqu1TMm-1Pjj8K963RVM_y72OglJdu/view?usp=sharing) | Decoder checkpoint extracted from model_512.pkl     | ./3DPortraitGAN_pyramid/models |
| [vgg16.pt](https://drive.google.com/file/d/1av5jH9jzuOobV9s2gyzx0w9a4xqco82H/view?usp=sharing) | vgg16                                               | ./3DPortraitGAN_pyramid/models |

#### (Optional)  Training

<u>Omit this section if utilizing the pre-trained 3DPortraitGAN_pyramid model aforementioned.</u>

For those interested in the training process, we kindly direct you to our training instructions available [here](https://github.com/oneThousand1000/Portrait3D/tree/main/3DPortraitGAN_pyramid).



### Random Image Generation

#### Preparing Prompts

First, prepare your prompts. These should be organized in the following structure:

```
test_data
│
└─── 001  
│   │
│   └─── prompt.txt (should initiate with "upper body photo")
└─── 002
│   │
│   └─── prompt.txt (should initiate with "upper body photo")
└─── ...
```

An example is available in `$PROJECT_ROOT$/test_data`.



#### Image generation

Download the Realistic_Vision_V5.1_noVAE model [here](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE). 

Please download all the folders and the Realistic_Vision_V5.1.ckpt file.

We employ the original stable diffusion in this use case. To convert the diffusers-version model to the original-stable-diffusion-version, follow the steps below:

```
cd stable-diffusion

activate text_to_3dportrait

git clone git@github.com:huggingface/diffusers.git

cd diffusers/scripts

python convert_diffusers_to_original_stable_diffusion.py --model_path $PATH_of_Realistic_Vision_V5.1_noVAE$ --checkpoint_path $PATH_of_Realistic_Vision_V5.1_noVAE$/realisticVisionV51_v51VAE.ckpt

cd ../../../
```

Then randomly generate images:

```
cd stable-diffusion

activate text_to_3dportrait

python get_test_data_df.py --test_data_dir ../test_data --sample_num 6  --scale 5 --df_ckpt $PATH_of_Realistic_Vision_V5.1_noVAE$/realisticVisionV51_v51VAE.ckpt 

cd ..
```

The generated images will be stored at `$PROJECT_ROOT$/test_data/image_id/samples`

**Note:** We discovered that using a smaller scale (for example, ` --scale 3`) tends to generate superior results for specific characters, like ''Tyrion Lannister in the Game of Thrones''. Feel free to experiment with different scales to improve the outcome.



#### Image Processing 

Our image processing code is largely adapted from [hongsukchoi/3DCrowdNet_RELEASE](https://github.com/hongsukchoi/3DCrowdNet_RELEASE).  

**Installation**

```text
conda create -n portrait3d_data python=3.8

activate portrait3d_data

cd data_processing

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

python -m pip install -e detectron2

cd ..
```



For windows:

```
pip install pywin32==306
```



For windows users who experience errors during detectron2 installation, please open a `x64 Native Tools Command Prompt` for Visual Studio and execute `python -m pip install -e detectron2`.



**Pretrained models**

| Download Link                                                | Save Path                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [R_101_FPN_DL_soft_s1x.pkl](https://drive.google.com/file/d/1rgrW9bAVbarft57mogUfawRSu2JCUKIT/view?usp=sharing) | `./data_processing/detectron2/projects/DensePose`            |
| [phi_smpl_27554_256.pkl](https://dl.fbaipublicfiles.com/densepose/data/cse/lbo/phi_smpl_27554_256.pkl) | `./data_processing/detectron2/projects/DensePose`            |
| [pose_higher_hrnet_w32_512.pth](https://drive.google.com/drive/folders/1zJbBbIHVQmHJp89t5CD1VF5TIzldpHXn) | `./data_processing/HigherHRNet-Human-Pose-Estimation/models/pytorch/pose_coco` |
| [crowdhuman_yolov5m.pt](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) | `./data_processing/yolov5_crowdhuman`                        |
| [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](http://smplify.is.tue.mpg.de/) | `./data_processing/common/utils/smplpytorch/smplpytorch/native/models` |
| [VPOSER_CKPT](https://drive.google.com/drive/folders/1KNw99d4-_6DqYXfBp2S3_4OMQ_nMW0uQ?usp=sharing) | `./data_processing/common/utils/human_model_files/smpl/VPOSER_CKPT` |
| [J_regressor_extra.npy](https://drive.google.com/file/d/1B9e65ahe6TRGv7xE45sScREAAznw9H4t/view?usp=sharing) | `./data_processing/data`                                     |
| [demo_checkpoint.pth.tar](https://drive.google.com/drive/folders/1YYQHbtxvdljqZNo8CIyFOmZ5yXuwtEhm?usp=sharing) | `./data_processing/demo`                                     |

If you encounter `RuntimeError: Subtraction, the - operator, with a bool tensor is not supported.`, you may refer to [this issue](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527) for a solution or change L301~L304 of `anaconda3/lib/python3.8/site-packages/torchgeometry/core/conversion.py` to below:

```
mask_c0 = mask_d2.float() * mask_d0_d1.float()
mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
```



Then process the randomly generated images to produce aligned images following the alignment setting of 3DPortraitGAN_pyramid:

```
cd data_processing

activate portrait3d_data
python preprocess_img_for_inversion.py --test_data_dir=$PROJECT_ROOT$/test_data

cd ..
```



**Note:** Manually review and discard any subpar images located in `$PROJECT_ROOT$/test_data/image_id/samples_new_crop/aligned_images`. For optimal inversion results, it is recommended to maintain an aligned image with a frontal view and minor body poses.



### 3D Portrait Inversion

**Inversion**

Before proceeding further, always ensure that you have removed all unsatisfactory images in `test_data/image_id/samples_new_crop/aligned_images`. This step is crucial to prevent suboptimal results.

Notice that we only run projection for the first image in `test_data/image_id/samples_new_crop/aligned_images`.

```
cd 3DPortraitGAN_pyramid

activate text_to_3dportrait

python run_inversion_with_pose_optimization.py \
	--model_pkl=./models/model_512.pkl \
	--pose_prediction_kwargs_path=./models/model_512.json \
	--test_data_dir=../test_data \
	--inversion_name=final_inversion \
	--with_pose_optim
```



**Generate Pyramid Tri-grid from Inversion results**

```
python run_trigrid_gen.py  \
	--network=./models/model_512.pkl \
    --inversion_name=final_inversion
    
cd ..
```



### 3D Portrait Generation and Optimization

Our image generation code is largely adapted from [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion). We express our gratitude for their significant contributions!

```
cd stable-dreamfusion-3DPortrait

python portrait3d_main.py \
	--trigrid_decoder_ckpt=../3DPortraitGAN_pyramid/models/decoder_512.ckpt \
	--inversion_name=final_inversion \
	--network_path=../3DPortraitGAN_pyramid/models/model_512.pkl \
	--test_data_dir=../test_data  \
	--df_ckpt=$PATH_of_Realistic_Vision_V5.1_noVAE$ 
```

The results will be stored and organized as:

```
stable-dreamfusion-3DPortrait/output/text_to_3dportrait/image_id
│
└─── trigrid.pkl [Original pyramid tri-grid generated from inversion results]
│
└─── validation [SDS validation images]
│
└─── checkpoints [SDS checkpoints]
│
└─── run [SDS run file]
│
└─── results [SDS rendering results]
|
└─── data [21 rendered views, refer to Section 3.5 in our paper]
|
└─── update_data [21 refined views, refer to Section 3.5 in our paper]
|
└─── log [Pyramid tri-grid optimization log files, refer to Section 3.5 in our paper]
│    │
│    └─── ckpt
│    │    │
│    │    └─── epoch_00019.pth [Final pyramid tri-grid]
│    └─── img 
│
└─── results_final [Final rendering results]
```



## Results Gallery 

We offer a gallery of 300 3D portraits (with their corresponding prompts) generated by our method, all viewable and accessible on [huggingface](https://huggingface.co/datasets/onethousand/Portrait3D_gallery).

```
Portrait3D_gallery
│
└─── 000  
│   │
│   └─── 000_pyramid_trigrid.pth [the pyramid trigrid file] 
│   │
│   └─── 000_prompt.txt [the prompt]
│   │
│   └─── 000_preview.png [the preview image]
│   │
│   └─── ...
└─── 001
│   │
│   └─── ...
└─── 002
│   │
│   └─── ...
│
└─── ...
```

To visualize these  3D portraits, use the following visualizer:

```
cd 3DPortraitGAN_pyramid

activate text_to_3dportrait

python pyramid_trigrid_visualizer.py
```

Input the path of your `model_512.pkl` into the `Pickle` field, and input the pyramid tri-grid path into the `Pyramid Tri-Grid Ckpt` field.

Please observe that we **maintain the neural rendering resolution at 256** for optimal rendering speed.


https://github.com/oneThousand1000/Portrait3D/assets/32099648/ebc3fcc2-2f7c-400d-8f5e-c41e56d72a1e


Enjoy traversing through these results 😉!



## Contact

[onethousand@zju.edu.cn](mailto:onethousand@zju.edu.cn) / [onethousand1250@gmail.com](mailto:onethousand1250@gmail.com)



## Citation

If you find this project helpful to your research, please consider citing:

```
@article{10.1145/3658162,
author = {Wu, Yiqian and Xu, Hao and Tang, Xiangjun and Chen, Xien and Tang, Siyu and Zhang, Zhebin and Li, Chen and Jin, Xiaogang},
title = {Portrait3D: Text-Guided High-Quality 3D Portrait Generation Using Pyramid Representation and GANs Prior},
year = {2024},
issue_date = {July 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3658162},
doi = {10.1145/3658162},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {45},
numpages = {12},
keywords = {3D portrait generation, 3D-aware GANs, diffusion models}
}
```



## Acknowledgements

The work is supported by the Information Technology Center and State Key Lab of CAD&CG, Zhejiang University. We extend our sincere gratitude for the generous provision of necessary computing resources.

We also want to express our thanks to those in the open-source community for their valuable contributions.



