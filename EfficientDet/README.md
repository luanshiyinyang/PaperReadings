# EfficientDet 解读

## 简介

这篇发表于CVPR2020的检测论文不同于大火的anchor-free，还是基于one-stage的范式做的设计，是ICML2019的EfficientNet的拓展，将分类模型引入到了目标检测任务中。近些年目标检测发展迅猛，精度提升的同时也使得模型越来越大、算力需求越来越高，这制约了算法的落地。近些年出现了很多高效的目标检测思路，如one-stage、anchor-free以及模型压缩策略，它们基本上都是以牺牲精度为代价获得效率的。EfficientDet直指当前目标检测的痛点：有没有可能在大量的资源约束前提下，实现高效且高精度的目标检测框架？**这就是EfficientDet的由来。**


- 论文标题

    EfficientDet: Scalable and Efficient Object Detection

- 论文地址

    http://arxiv.org/abs/1911.09070

- 论文源码

    https://github.com/google/automl/tree/master/efficientdet


## 介绍

之前提到，EfficientDet是EfficientNet的拓展，我们首先来简单聊一聊EfficientNet，感兴趣的请阅读[原文](https://arxiv.org/abs/1905.11946)。在EfficientNet中提到了一个很重要的概念Compound Scaling（符合缩放）