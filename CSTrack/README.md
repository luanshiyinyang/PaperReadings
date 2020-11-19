# CSTrack 解读

## 简介

自从 FairMOT 的公开以来，MOT 似乎进入了一个高速发展阶段，先是 CenterTrack 紧随其后发布并开源 ，然后是后来的 RetinaTrack、MAT、FGAGT 等 SOTA 方法出现，它们不断刷新着 MOT Challenge 的榜单。最近，CSTrack 这篇文章则在 JDE 范式的基础上进行了改进，获得了相当不错的跟踪表现（基本上暂时稳定在榜单前 5），本文就简单解读一下这篇短文（目前在 Arxiv 上开放的是一个 4 页短文的版本）。

- 论文地址

  http://arxiv.org/abs/2010.12138

## 介绍

为了追求速度和精度的平衡，联合训练检测模型和 ReID 模型的 JDE 范式（如下图，具体提出参考 JDE 原论文 Towards Real-Time Multi-Object Tracking）受到了学术界和工业界越来越多的关注。这主要是针对之前的two-stage方法先是使用现有的检测器检测出行人然后再根据检测框提取对应行人的外观特征进行关联的思路，这种方法在精度上的表现不错，然而由于检测模型不小且ReID模型需要在每个检测框上进行推理，计算量非常之大，所以JDE这种one-shot的思路的诞生是一种必然。

![](./assets/jde.png)

然而，就像之前 FairMOT 分析的那样，检测和 ReID 模型是存在不公平的过度竞争的，这种竞争制约了两个任务的表示学习

## 总结

文中采用的是 YOLO

## 参考文献

[1]Liang C, Zhang Z, Lu Y, et al. Rethinking the competition between detection and ReID in Multi-Object Tracking[J]. arXiv:2010.12138 [cs], 2020.
