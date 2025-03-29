
![Full pipeline of our Point2Quad](https://github.com/cognaclee/Point2Quad/main/assets/pipeline.png)


## Introduction

### Update 29/03/2025: New [PyTorch implementation](https://github.com/cognaclee/Point2Quad) available. 

This repository contains the implementation of *Point2Quad: Generating Quad Meshes from Point Clouds via Face Prediction**, a quad mesh generator
presented in our tcsvt paper ([arXiv](https://arxiv.org/abs/1904.08889)). If you find our work useful in your 
research, please consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```

## Installation

A step-by-step installation guide for Ubuntu 16.04 is provided in [INSTALL.md](./INSTALL.md). Windows is currently 
not supported as the code uses tensorflow custom operations.


## Experiments

We provide scripts for many experiments. The instructions to run these experiments are in the [doc](./doc) folder.


## Performances


## Acknowledgment

Our code uses <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a> as the backbone.

## License
Our code is released under MIT License (see LICENSE file for details).

