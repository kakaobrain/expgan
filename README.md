# Exp-GAN: 3D-Aware Facial Image Generation with Expression Control

![image](https://user-images.githubusercontent.com/29425882/194230500-ca3f9337-d540-4194-8d96-008d420fde7a.png)

This repository is the official implementation of the ACCV 2022 paper Exp-GAN: 3D-Aware Facial Image Generation with Expression Control

## Installation

Requirements for using pytorch3d
- Python >= 3.7
- PyTorch >= 1.12.0
```
pip install -r requirements.txt
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout v0.7.0
pip install -e .
cd -
```

## Dataset and model

Download the aligned FFHQ dataset images from [the official repository](https://drive.google.com/open?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL),
and place them under `data/FFHQ/img`.
Annotations for DECA parameters (head pose, shape and expression) can be downloaded here; place the files under `data/FFHQ/annots`.
Please refer `experiments/config/config.yaml` to see how the data is used.

DECA is used to generate facial texture, download required assets by running 
```
cd data
sh download_deca.sh
cd -
```

A pretrained model can be downloaded here; refer the demo notebook file to see how to use the checkpoint.
<!-- - dataset:
    - FFHQ : 
        - dataset_root: `data/FFHQ/`
        - images: `<dataset_root>/img`
        - DECA parameters: `<dataset_root>/ffhq_deca_ear_ortho_1217.pkl`, `<dataset_root>/ffhq_deca_ear_ortho_flipped_0207.json`
        - masking indices : `<dataset_root>/indices_ear_noeye.pkl`

    - DECA : `data/DECA/`

- pre-trained model : `data/pretrained_checkpoint/is_alpha10.ckpt`
- config : `data/pretrained_checkpoint/is_alpha10_config.yaml` -->


## Training

```
cd experiments/config
sh train.sh
```

## evaluation
Run the following script to generate images for the FID evaluation:
```
python eval.py --cfg <cfg> --ckpt <ckpt> --savedir <savedir>
```

Then run the following to measure the FID between generated and real images:
```
python fid.py --root_real <root_real> --root_fake <root_fake> --batch_size 50
```
where `<root_real>` contains downsampled FFHQ images and `<root_fake>` contains images generated by `eval.py`.

## Demo

Please check `demo.ipynb` to see how to generate some examples by using a pretrained model.

## License
Copyright (c) 2022 POSTECH, Kookmin University, Kakao Brain Corp. All Rights Reserved. Licensed under the Apache License, Version 2.0 (see [LICENSE](./LICENSE) for details)
