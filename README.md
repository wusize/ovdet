# Aligning Bag of Regions for Open-Vocabulary Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aligning-bag-of-regions-for-open-vocabulary/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=aligning-bag-of-regions-for-open-vocabulary)

## Introduction

This is an official release of the paper **Aligning Bag of Regions for Open-Vocabulary Object Detection**.

> [**Aligning Bag of Regions for Open-Vocabulary Object Detection**](https://arxiv.org/abs/2302.13996),            
> Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, Chen Change Loy           
> In: Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023           
> [[arXiv](https://arxiv.org/abs/2302.13996)][[project page(TBD)](https://www.mmlab-ntu.com/project/baron/index.html)][[Bibetex](https://github.com/wusize/ovdet#citation)]

## Results

The results of BARON and their corresponding configs on each segmentation task are shown as below.
The model checkpoints and logs will be released soon.

### Open Vocabulary COCO

| Backbone | Method | Supervision | Novel AP50 | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-50-FPN  | BARON| CLIP | 34.0 |[config](configs/baron/) | [model]() &#124;  [log]() |
| R-50-C4  | BARON| COCO Caption | 33.1 |[config](configs/baron/) | [model]() &#124;  [log]() |
| R-50-C4  | BARON| COCO Caption + CLIP | 42.7|[config](configs/baron/) | [model]() &#124;  [log]() |

### Open Vocabulary LVIS

| Backbone | Method | Branch Ensembel | Learned Prompt | Mask AP_novel | Mask AP | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R-50-FPN  | BARON| N | N | 18.0 | 25.1 |[config](configs/baron/) | [model]() &#124;  [log]() |
| R-50-FPN  | BARON| Y | N | 19.2 | 26.5 |[config](configs/baron/) | [model]() &#124;  [log]() |
| R-50-FPN  | BARON| Y | Y | 22.6 | 27.6 |[config](configs/baron/) | [model]() &#124;  [log]() |


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
The preparation of OV-COCO and OV-LVIS is coming soon.
The data structure looks like below:

```text
data/
├── coco
│   ├── annotations
│   │   ├── instance_{train,val}2017.json
│   │   ├── image_info_test-dev2017.json  # for test-dev submissions
│   ├── train2017
│   ├── val2017
│   ├── test2017

```

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

