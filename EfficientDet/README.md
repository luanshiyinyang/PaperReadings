# EfficientDet 解读

## 简介

这篇发表于 CVPR2020 的检测论文不同于大火的 anchor-free，还是基于 one-stage 的范式做的设计，是 ICML2019 的 EfficientNet 的拓展，将分类模型引入到了目标检测任务中。近些年目标检测发展迅猛，精度提升的同时也使得模型越来越大、算力需求越来越高，这制约了算法的落地。近些年出现了很多高效的目标检测思路，如 one-stage、anchor-free 以及模型压缩策略，它们基本上都是以牺牲精度为代价获得效率的。EfficientDet 直指当前目标检测的痛点：有没有可能在大量的资源约束前提下，实现高效且高精度的目标检测框架？**这就是 EfficientDet 的由来。**

- 论文标题

  EfficientDet: Scalable and Efficient Object Detection

- 论文地址

  http://arxiv.org/abs/1911.09070

- 论文源码

  https://github.com/google/automl/tree/master/efficientdet

## 介绍

之前提到，EfficientDet 是 EfficientNet 的拓展，我们首先来简单聊一聊 EfficientNet，感兴趣的请阅读[原文](https://arxiv.org/abs/1905.11946)。在 EfficientNet 中提到了一个很重要的概念 Compound Scaling（符合缩放），这是基于一个公认的事实：调整模型的深度、宽度以及输入的分辨率在一定范围内会对模型性能有影响，但是过大的深度、宽度和分辨率对性能改善不大还会严重影响模型前向效率，所以 EfficientNet 提出复合系数$\phi$统一缩放网络的宽度、深度和分辨率，具体如下。

![](./assets/fai.png)

这里的$\alpha, \beta, \gamma$都是由一个很小范围的网络搜索得到的常量，直观上来讲，$\phi$是一个特定的系数，可以用来控制资源的使用量，$\alpha, \beta, \gamma$决定了具体是如何分配资源的。值得注意的是，常规卷积操作的计算量是和$d, w^{2}, r^{2}$成正比的，加倍深度会使得 FLOPS 加倍，但是加倍宽度和分辨率会使得 FLOPS 加 4 倍。由于卷积 ops 经常在 CNN 中占据了大部分计算量，使用等式上式缩放卷积网络将会使得整体计算量近似增加$\left(\alpha \cdot \beta^{2} \cdot \gamma^{2}\right)^{\phi}$倍。由于 EfficientNet 对任意$\phi$增加了约束$\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$，整体的计算量近似增加了$2^{\phi}$倍。

**对比 EfficientNet 从 B0 到 B7 的提升，不难知道，这种复合缩放可以较大的提升模型性能，所以 EfficientDet 也将其引入了进来。**

论文首先分析了目前 OD（object detection，目标检测）的两个挑战：**高效多尺度特征融合**和**模型缩放**。

**多尺度特征融合**：FPN 如今被广泛用于多尺度特征融合，最近 PANet、NAS-FPN 等方法研究了更多跨尺度特征融合的结构。不过，这些方法融合不同尺度特征的方式都是简单加和，这默认了不同尺度特征的贡献是同等的，然而往往不是这样的。为了解决这个问题，论文提出了一种简单但是高效的加权双向特征金字塔网络（**BiFPN**），它对不同的输入特征学习权重。

**模型缩放**：之前的方法依赖于更大的 backbone 或者更大分辨率的输入，论文发现放大特征网络和预测网络对精度和速度的考量是很重要的。基于 EfficientNet 的基础，论文为目标检测设计了一种复合缩放（**Compound Scaling**）方法，联合调整 backbone、特征网络、预测网络的深度、宽度和分辨率。

和 EfficientNet 一样，EfficientDet 指的是一系列网络，如下图包含 D1 到 D7，速度逐渐变慢，精度逐渐提升。在理解 EfficientDet 两个核心工作（**BiFPN**和**Compound Scaling**）之前，可以先看看下图的 SOTA 方法比较，可以看到 EfficientDet-D7 的效果非常惊人，在 FLOPs 仅为 AmoebaNet+NAS-FPN+AA 的十分之一的前提下，COCO2017 验证集上的 AP 到达了 55.1，超越了 SOTA5 个点。

![](./assets/contrast.png)

## BiFPN

