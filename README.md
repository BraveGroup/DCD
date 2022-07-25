# DCD
Released code for Densely Constrained Depth Estimator for Monocular 3D Object Detection (ECCV22). [arxiv](https://arxiv.org/abs/2207.10047)
Yingyan Li, Yuntao Chen, Jiawei He, Zhaoxiang Zhang

## Code still needs to be cleaned, please wait a few days.

## Environment

This repo is tested with Ubuntu 16.04, python==3.8, pytorch==1.7.0 and cuda==10.1.

```bash
conda create -n dcd python=3.8
conda activate dcd
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

You also need ot build DCNv2 and this project as:
```bash
cd DGDE/models/backbone/DCNv2
python setup.py develop
cd ../../..
python setup.py develop
```

## Directory Structure
We need [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and keypoints annotation [Google Drive](https://drive.google.com/drive/folders/1vO-A0bHktUEvCq5Jp9vkHxWdksaWRXCp?usp=sharing). 

After download them, please organize as:

```
|DGDE
  |dataset
    |kitti
      |training/
        |calib/
        |image_2/
        |label/
        |ImageSets/
      |testing/
        |calib/
        |image_2/
        |ImageSets/
  |kpts_ann
    |kpts_ann_train.json
    |kpts_ann_val.json
```

## Training and evaluation pipeline
The whole pipeline including 3 parts: a) training DGDE first. b) using DGDE to generate needed data for GMW. c) training GMW and evaluate.

a) training DGDE
Training with 2 GPUs. 

```bash
cd DGDE
CUDA_VISIBLE_DEVICES=0,1 \
python tools/plain_train_net.py --batch_size 8 --config runs/DGDE.yaml \
--output output/DGDE --num_gpus 2 \
```

b) using DGDE to generate needed data for GMW.
Finishing training for DGDE, please generate data on 1 GPU as:
```bash
cd DGDE
CUDA_VISIBLE_DEVICES=0 \
python tools/plain_train_net.py --batch_size 8 --config runs/DGDE.yaml \
--output output/DGDE --num_gpus 1 \
--generate_for_GMW \
--ckpt output/DGDE/model_final.pth
```
after this step, you could see gen_data_train.json and gen_data_infer.json in DGDE/gen_data/

c) training GMW and evaluate.
```bash
cd GMW
python -m torch.distributed.launch --master_port 33521 --nproc_per_node=4 \
main.py --log-dir ./logs/GMW \
-b 8 --lr 1e-4 --epoch 100 --val_freq 5 \
--train_data_path ../DGDE/gen_data/gen_data_train.json \
--val_data_path ../DGDE/gen_data/gen_data_infer.json
```
It will be evaluated periodically. You can also run the following command for evaluation:
```bash
python -m torch.distributed.launch --master_port 24281 --nproc_per_node=4 \
main.py --log-dir ./logs/GMW/
-b 36 -e \
--resume logs/GMW/checkpoint_epoch_100.pth.tar
```

You can also use the pre-trained weights of [DGDE,WGM (Google Drive)](https://drive.google.com/drive/folders/1_t3gm5CUzleL_MQH0SaCnSIDb5rIwyTE?usp=sharing).

## Acknowlegment

The code is mainly based on [MonoFlex](https://github.com/zhangyp15/MonoFlex) and [BPnP](https://github.com/dylan-campbell/bpnpnet). Thanks for their great work.
