# CSTrack 解读

## 简介

自从 FairMOT 的公开以来，MOT 似乎进入了一个高速发展阶段，先是 CenterTrack 紧随其后发布并开源 ，然后是后来的 RetinaTrack、MAT、FGAGT 等 SOTA 方法出现，它们不断刷新着 MOT Challenge 的榜单。最近，CSTrack 这篇文章则在 JDE 范式的基础上进行了改进，获得了相当不错的跟踪表现（基本上暂时稳定在榜单前 5），本文就简单解读一下这篇短文（目前在 Arxiv 上开放的是一个 4 页短文的版本）。

- 论文标题

  Rethinking the competition between detection and ReID in Multi-Object Tracking

- 论文地址

  http://arxiv.org/abs/2010.12138

- 论文源码

  https://github.com/JudasDie/SOTS

## 介绍

为了追求速度和精度的平衡，联合训练检测模型和 ReID 模型的 JDE 范式（如下图，具体提出参考 JDE 原论文 Towards Real-Time Multi-Object Tracking）受到了学术界和工业界越来越多的关注。这主要是针对之前的 two-stage 方法先是使用现有的检测器检测出行人然后再根据检测框提取对应行人的外观特征进行关联的思路，这种方法在精度上的表现不错，然而由于检测模型不小且 ReID 模型需要在每个检测框上进行推理，计算量非常之大，所以 JDE 这种 one-shot 的思路的诞生是一种必然。

![](https://i.loli.net/2020/11/20/XSvleZbIjwDo1Yd.png)

然而，就像之前 FairMOT 分析的那样，检测和 ReID 模型是**存在不公平的过度竞争**的，这种竞争制约了两个任务（检测任务和 ReID 任务 ）的表示学习，导致 了学习的混淆。具体而言，检测任务需要的是同类的不同目标拥有相似的语义信息（类间距离最大），而 ReID 要求的是同类目标有不同的语义信息（类内距离最大）。此外，**目标较大的尺度变化**依然是 MOT 的痛点。在 ReID 中图像被调整到统一的固定尺寸来进行查询 ，而在 MOT 中，提供在 ReID 网络的特征需要拥有尺度感知能力，这是因为沿着帧目标可能会有巨大的 size 变化。

为了解决上述的过度竞争问题，论文提出了一种新的互相关网络（CCN）来改进单阶段跟踪框架下 detection 和 ReID 任务之间的协作学习。作者首先将 detection 和 ReID 解耦为两个分支，分别学习。然后两个任务的特征通过自注意力方式获得自注意力权重图和互相关性权重图。自注意力图是促进各自任务的学习，互相关图是为了提高两个任务的协同学习。而且，为了解决上述的尺度问题，设计了尺度感知注意力网络（SAAN）用于 ReID 特征的进一步优化，SAAN 使用了空间和通道注意力，该网络能够获得目标 不同尺度的外观信息，最后 不同尺度外观特征融合输出即可。

## 框架设计

整体的思路还是采用 JDE 的框架，下图的右图是整体 pipeline 的设计，和左侧的 JDE 相比，中间增加了一个 CCN 网络（互相关网络）用于构建 detection 和 ReID 两个分支不同的特征图（这里解释一下为什么要解耦成两个特征图送入后续的两个任务中，其实原始的 JDE 就是一个特征图送入检测和 ReID 分支中，作者这边认为这会造成后续的混淆，所以在这里分开了，FairMOT 也发现了这个问题，只是用另一种思路解决的而已）。构建的两个特征图分别送入 Detection head 和 SAAN（多尺度+注意力+ReID）中，Detection head 将 JDE 的 YOLO3 换为了更快更准的 YOLO5，其他没什么变动，下文检测这边我就不多提了。检测完成的同时，SAAN 也输出了多尺度融合后的 ReID 特征，至此也就完成了 JDE 联合检测和 ReID 的任务，后续就是关联问题了。

![](https://i.loli.net/2020/11/20/KIpS3jV1ezicCk5.png)

所以，从整个框架来看，CSTrack 对 JDE 的改动主要集中在上图的 CCN 和 SAAN，而这两部分都是在特征优化上做了文章，且主要是基于注意力手段的特征优化（不知道是好是坏呢）。

## CCN

CCN（Cross-correlation Network）用于提取更适合 detection 和 ReID 任务的一般特征和特定特征。在特定性学习方面，通过学习反映不同特征通道之间相互关系的自联系，增强了每个任务的特征表示。对于一般性学习，可以通过精心设计的相互关系机制来学习两个任务之间的共享信息。

![](https://i.loli.net/2020/11/20/XoOUnxREIqfCy3K.png)

CCN 的结构如上图，我们对着这个图来理解 CCN 的思路。从检测器的 backbone 得到的特征图为$\mathbf{F} \in R^{C \times H \times W}$，首先，这个特征经过平均池化降维获得统计信息（更精炼的特征图）$\mathbf{F}^{\prime} \in R^{C \times H^{\prime} \times W^{\prime}}$。然后，两个不同的卷积层作用于$\mathbf{F}^{\prime}$生成两个特征图$\mathbf{T_1}$和$\mathbf{T_2}$，这两个特征图被 reshape 为特征$\left\{\mathbf{M}_{\mathbf{1}}, \mathbf{M}_{\mathbf{2}}\right\} \in R^{C \times N^{\prime}}$（$N^{\prime}=H^{\prime} \times W^{\prime}$）。下面的上下两个分支操作是一致的，先用矩阵$\mathbf{M_1}$或者$\mathbf{M_2}$和自己的转置矩阵相乘获得各自的自注意力图$\left\{\mathbf{W}_{\mathrm{T}_{1}}, \mathbf{W}_{\mathrm{T}_{2}}\right\} \in R^{\mathrm{C} \times \mathrm{C}}$，然后$\mathbf{M_1}$和$\mathbf{M_2}$的转置进行矩阵乘法获得互注意力图$\left\{\mathbf{W}_{\mathrm{S}_{1}}, \mathbf{W}_{\mathrm{S}_{2}}\right\} \in R^{\mathrm{C} \times \mathrm{C}}$（这是$\mathbf{M_1}$的，转置之后 softmax 就是$\mathbf{M_2}$的）。然后，对每个分支，自注意力图和互注意力图相加获得通道级别的注意力图，和原始的输入特征图$\mathbf{F}$相乘再和$\mathbf{F}$相加得到输出特征图$\mathrm{F}_{\mathrm{T} 1}$和$\mathrm{F}_{\mathrm{T} 2}$。

上述学到的$\mathrm{F}_{\mathrm{T} 1}$用于 Detection head 的检测处理，后者则用于下面的 SAAN 中 ReID 特征的处理。

![](https://i.loli.net/2020/11/20/KhYU9rfR3g7VnFe.png)

上图就是作者设计的 ReID 分支，用于对 ReID 特征进行多尺度融合，这个设计挺简单的，不同分支采用不同的下采样倍率获得不同尺度的特征图（其中通过空间注意力进行特征优化），然后融合产生的特征通过空间注意力加强，最终输出不同目标的 embedding$\mathbf{E} \in R^{512 \times W \times H}$（特征图每个通道对应不同的 anchor 的 embedding）。

**这样，整个 JDE 框架就完成了。**

## 实验

在 MOT16 和 MOT17 上实验结果如下图，比较的方法都比较新，MOTA 也是刷到了 70 以上，不过速度稍许有点慢了，总的精度还是很不错的。

![](https://i.loli.net/2020/11/20/Tcv3j8x1ruYHsQh.png)

## 总结

CSTrack 在 JDE 的基础上使用了更强的检测器也对 ReID 特征进行了优化，获得了相当不错的表现。不过，从结果上看这种暴力解耦还是会对整个跟踪的速度有影响的。

## 参考文献

[1]Liang C, Zhang Z, Lu Y, et al. Rethinking the competition between detection and ReID in Multi-Object Tracking[J]. arXiv:2010.12138 [cs], 2020.
