# Aligning Bag of Regions for Open-Vocabulary Object Detection

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aligning-bag-of-regions-for-open-vocabulary/open-vocabulary-object-detection-on-mscoco&#41;]&#40;https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=aligning-bag-of-regions-for-open-vocabulary&#41;)

## Introduction

This is an official release of the paper **Aligning Bag of Regions for Open-Vocabulary Object Detection**.

> [**Aligning Bag of Regions for Open-Vocabulary Object Detection**](https://arxiv.org/abs/2302.13996),            
> Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, Chen Change Loy           
> In: Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023           
> [[arXiv](https://arxiv.org/abs/2302.13996)][[project page(TBD)](https://www.mmlab-ntu.com/)][[Bibetex](https://github.com/wusize/ovdet#citation)]


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

The training and testing on [OV-COCO](configs/baron/ov_coco/README.md) are supported now.


## Citation

```bibtex
@inproceedings{wu2023baron,
    title={Aligning Bag of Regions for Open-Vocabulary Object Detection},
    author={Size Wu and Wenwei Zhang and Sheng Jin and Wentao Liu and Chen Change Loy},
    year={2023},
    booktitle={CVPR},
}
```

