# Aligning Bag of Regions for Open-Vocabulary Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aligning-bag-of-regions-for-open-vocabulary/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=aligning-bag-of-regions-for-open-vocabulary)

## Introduction

This is an official release of the paper **Aligning Bag of Regions for Open-Vocabulary Object Detection**.

> [**Aligning Bag of Regions for Open-Vocabulary Object Detection**](https://arxiv.org/abs/2302.13996),            
> Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, Chen Change Loy           
> In: Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023           
> [[arXiv](https://arxiv.org/abs/2302.13996)][[project page(TBD)](https://www.mmlab-ntu.com/)][[Bibetex](https://github.com/wusize/ovdet#citation)]

## Results

The results of BARON and their corresponding configs on each segmentation task are shown as below.
The model checkpoints and logs will be released soon.

### Open Vocabulary COCO

| Backbone | Method |     Supervision     | Novel AP50 | Config | Download |
|:--------:| :---: |:-------------------:|:-----:| :---: | :---: |
| R-50-C4  | BARON  |        CLIP         |       |[config](configs/baron/ov_coco/baron_caption_faster_rcnn_r50_caffe_c4_90k.py) | [model]() &#124;  [log]() |
| R-50-C4  | BARON  | CLIP (share branch) |       |[config](configs/baron/ov_coco/baron_kd_share_batch_faster_rcnn_r50_caffe_c4_90k.py) | [model]() &#124;  [log]() |
| R-50-C4  | BARON  |    COCO Caption     |       |[config](configs/baron/ov_coco/baron_caption_faster_rcnn_r50_caffe_c4_90k.py) | [model]() &#124;  [log]() |



## Installation

This project is based on [MMDetection 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x)

It requires the following OpenMMLab packages:

- MMEngine >= 0.6.0
- MMCV-full >= v2.0.0rc4
- MMDetection >= v3.0.0rc6
- lvisapi

```bash
pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install mmdet>=3.0.0rc6
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Usage

### Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection). 
Obtain the json files for OV-COCO from [GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing) and put them
under `data/coco/wusize`
The data structure looks like below:

```text
checkpoints/
├── clip_vitb32.pth
data/
├── coco
│   ├── annotations
│   │   ├── instances_{train,val}2017.json
│   ├── wusize
│   │   ├── instances_train2017_base.json
│   │   ├── instances_val2017_base.json
│   │   ├── instances_val2017_novel.json
│   │   ├── captions_train2017_tags_allcaps.json
│   ├── train2017
│   ├── val2017
│   ├── test2017

```

### CLIP checkpoints
Obtain the checkpoint of ViT-B-32 from 
[GoogleDrive](https://drive.google.com/file/d/1ilxBhjb3JXNDar8lKRQ9GA4hTmjxADfu/view?usp=sharing) and put it
under `checkpoints`.

### Training and testing

The training and testing support is coming soon.

## Citation

```bibtex
@inproceedings{wu2023baron,
    title={Aligning Bag of Regions for Open-Vocabulary Object Detection},
    author={Size Wu and Wenwei Zhang and Sheng Jin and Wentao Liu and Chen Change Loy},
    year={2023},
    booktitle={CVPR},
}
```

