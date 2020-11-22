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

![](https://i.loli.net/2020/11/22/rQsZnAzVWE986He.png)

这里的$\alpha, \beta, \gamma$都是由一个很小范围的网络搜索得到的常量，直观上来讲，$\phi$是一个特定的系数，可以用来控制资源的使用量，$\alpha, \beta, \gamma$决定了具体是如何分配资源的。值得注意的是，常规卷积操作的计算量是和$d, w^{2}, r^{2}$成正比的，加倍深度会使得 FLOPS 加倍，但是加倍宽度和分辨率会使得 FLOPS 加 4 倍。由于卷积 ops 经常在 CNN 中占据了大部分计算量，使用等式上式缩放卷积网络将会使得整体计算量近似增加$\left(\alpha \cdot \beta^{2} \cdot \gamma^{2}\right)^{\phi}$倍。由于 EfficientNet 对任意$\phi$增加了约束$\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$，整体的计算量近似增加了$2^{\phi}$倍。

**对比 EfficientNet 从 B0 到 B7 的提升，不难知道，这种复合缩放可以较大的提升模型性能，所以 EfficientDet 也将其引入了进来。**

论文首先分析了目前 OD（object detection，目标检测）的两个挑战：**高效多尺度特征融合**和**模型缩放**。

**多尺度特征融合**：FPN 如今被广泛用于多尺度特征融合，最近 PANet、NAS-FPN 等方法研究了更多跨尺度特征融合的结构。不过，这些方法融合不同尺度特征的方式都是简单加和，这默认了不同尺度特征的贡献是同等的，然而往往不是这样的。为了解决这个问题，论文提出了一种简单但是高效的加权双向特征金字塔网络（**BiFPN**），它对不同的输入特征学习权重。

**模型缩放**：之前的方法依赖于更大的 backbone 或者更大分辨率的输入，论文发现放大特征网络和预测网络对精度和速度的考量是很重要的。基于 EfficientNet 的基础，论文为目标检测设计了一种复合缩放（**Compound Scaling**）方法，联合调整 backbone、特征网络、预测网络的深度、宽度和分辨率。

和 EfficientNet 一样，EfficientDet 指的是一系列网络，如下图包含 D1 到 D7，速度逐渐变慢，精度逐渐提升。在理解 EfficientDet 两个核心工作（**BiFPN**和**Compound Scaling**）之前，可以先看看下图的 SOTA 方法比较，可以看到 EfficientDet-D7 的效果非常惊人，在 FLOPs 仅为 AmoebaNet+NAS-FPN+AA 的十分之一的前提下，COCO2017 验证集上的 AP 到达了 55.1，超越了 SOTA5 个点。而且单尺度训练的 EfficientD7 现在依然霸榜 PaperWithCode 上。

![](https://i.loli.net/2020/11/22/NU35fMiRe6Fo2BT.png)

![](https://i.loli.net/2020/11/22/LI5h3EBkVtWQzjO.png)

此外，查看官方仓库提供的模型，其参数量其实是不大的（当然，这不绝对意味着计算量小）。

![](https://i.loli.net/2020/11/22/giL6hNfOYMVA2I1.png)

## BiFPN

CVPR2017 的 FPN 指出了不同层之间特征融合的重要性如下图 a，不过它采用的是自上而下的特征图融合，融合方式也是很简单的高层特征加倍后和低层特征相加的方式。此后，下图 b 所示的 PANet 在 FPN 的基础上又添加了自下而上的信息流。再后来出现了不少其他的融合方式，直到 NAS-FPN 采用了 NAS 策略搜索最佳 FPN 结构，得到的是下图 c 的版本，不过 NAS-FPN 虽然简单高效，但是精度和 PANet 还是有所差距的，并且 NAS-FPN 这种结构是很怪异的，难以理解的。所以，EfficientDet 在 PANet 的基础上进行了优化如下图的 d：移除只有一个输入的节点；同一个 level 的输入和输出节点进行连接，类似 skip connection；PANet 这种一次自上而下再自下而上的特征融合可以作为一个单元重复多次从而获得更加丰富的特征，不过重复多少次是速度和精度的权衡选择，这在后面的复合 22 缩放部分讲到。

![](https://i.loli.net/2020/11/22/iuAG38EdxKI9DqF.png)

上述是 FPN 特征流动的结构，如何数学上组合这些特征也是一个重要的方面。此前的方法都是上一层特征 resize 之后和当前层特征相加。这种方式存在诸多不合理之处，因为这样其实默认融合的两层特征是同权重的，事实上不同尺度的特征对输出特征的贡献是不平等的，应当对每个输入特征加权，这个权重需要网络自己学习。当然，学习到的权重需要归一化到和为 1，采用 softmax 是一个选择，但是 softmax 指数运算开销大，所以作者这里简化为快速标准化融合的方式（Fast normalized fusion），它的计算方法如下，其实就是去掉了 softmax 的指数运算，这种方式在 GPU 上快了很多，精度略微下降，可以接受。

$$
O=\sum_{i} \frac{w_{i}}{\epsilon+\sum_{j} w_{j}} \cdot I_{i}
$$

## Compound Scaling

在看复合缩放之前，我们先要知道，有了 BiFPN 又有了 EfficientNet 再加上 head 部分，其实网络框架已经确定了，如下图所示，左边是 backbone（EfficientDet），中间是多层 BiFPN，右边是 prediction head 部分。

![](https://i.loli.net/2020/11/22/P28ZjJ4sgwYSXGh.png)

结合 EfficientNet 的联合调整策略，论文提出目标检测的联合调整策略，用复合系数$\phi$统一调整.调整的内容包括 backbone（EfficientNet 版本，B0 到 B6）、neck 部分的 BiFPN（通道数、layer 数）以及 head 部分（包括层数）还要输入图像分辨率。不过，和 EfficientNet 不同，由于参数太多采用网格搜索计算量很大，论文采用启发式的调整策略。其中 backbone 的选择系数控制，BiFPN 的配置用下面第一个式子计算，head 的层数和输入分辨率是下面 2、3 式的计算方式。

$$
W_{b i f p n}=64 \cdot\left(1.35^{\phi}\right), \quad D_{b i f p n}=3+\phi
$$

$$
D_{b o x}=D_{c l a s s}=3+\lfloor\phi / 3\rfloor
$$

$$
R_{\text {input}}=512+\phi \cdot 128
$$

最后得到的 8 种结构的配置表如下图。

![](https://i.loli.net/2020/11/22/fw7sLOZj8KNtUm2.png)

## 实验结果

在简介里我已经提到很多这个检测框架的过人之处了，这里就简单看一下在 COCO 验证集的效果，可以说无论是速度还是精度都是吊打其他 SOTA 方法的，至今依然在 COCO 验证集榜首的位置。

![](https://i.loli.net/2020/11/22/KUOb2RJcBCQt5fu.png)

此外作者也将其拓展到语义分割，潜力也是比较大的。还做了不少消融实验，感兴趣的可以自行查看论文原文。

## 总结

本文最大的亮点在于提出了目标检测网络联合调整复杂度的策略，从而刷新了 SOTA 结果。这个思路来自 EfficientDet，同样 backbone 的高效也源自该网络。文中另一个突出的成果在于设计了 BiFPN 以及堆叠它，可以看到效果还是很显著的。此外，除了官方的 TF 实现外，这里也推荐一个目前公认最好的 PyTorch 实现（由国内大佬完成），[Github 地址](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)给出，这也是唯一一个达到论文效果的 PyTorch 复现（作者复现时官方还没有开源）。

## 参考文献

[1]Tan M, Pang R, Le Q V. EfficientDet: Scalable and Efficient Object Detection[J]. arXiv:1911.09070 [cs, eess], 2020.
