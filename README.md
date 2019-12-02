# pytorch-deeplab-xception

**It refer to [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)，参考了[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)的项目**

**It provide model trained on Pascal VOC datasets. 实现了在Pascal VOC数据集下的训练测试与验证**  

**It support multi-gpu training. 支持多GPU训练，使用时需要注意arg.gpu-ids参数的设置**  

**It support synchronized-batchnormalization**

**It need to maintain the file structure of VOC dataset, modify the train.txt, val.txt, test.txt in the ImageSets folder to control the pictures participating in training, verification or test. 需要保持VOC数据集的文件结构(tree)形式，通过修改ImageSet内的train.txt,val.txt,test.txt来控制参与训练，验证或是测试的图片。**


# The backbone Pretrained Model url， backbone权重的下载链接

| Backbone  |Pretrained Model|
| :-------- |:--------------:|
| ResNet    | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |
| MobileNet | [google drive](https://drive.google.com/open?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt) |
| DRN       | [google drive](https://drive.google.com/open?id=131gZN_dKEXO79NknIQazPJ-4UmRrZAfI) |


### Introduction
This is a Program supports PyTorch(1.0.0) and later versions implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). 
基于pytorch1.0编写的DeepLabV3.


### Installation
The code was implemented with Python 3.5. :

1. Clone the repo:
    ```Shell
    git clone https://github.com/zyLee1-git/pytorch-DeepLab-V3.git
    cd pytorch-DeepLab-V3
    ```

2. requiresment packages 需要pip的包:

This programm need matplotlib, pillow, numpy, PIL, tqdm
    ```Shell
    pip install matplotlib pillow numpy PIL tensorboardX tqdm
    ```
### Training
Fellow steps below to train your model:

0. Configure your dataset path in train.py(lines:28~30). 在train.py中(28~30行)对应修改你数据集的地址，test时相同
```python
self.baseroot = None
        if args.dataset == 'pascal':
            self.baseroot = '/path/to/your/VOCdevkit/VOC2012/'
```

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}] [--pattern train]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```

2. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone:
    ```Shell
    bash train_voc.sh
    ```
    or run train.py in pycharm

### Testing
0. Configure your dataset path in train.py(lines:23~25) same as training
```python
 self.baseroot = None
        if args.dataset == 'pascal':
            self.baseroot = '/path/to/your/VOCdevkit/VOC2012/'
```

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: test.py [-h] [--backbone {resnet,xception,drn,mobilenet}] [--pattern test]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--test-batch-size N] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--checkname CHECKNAME]
    ```


### Acknowledgement
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)

