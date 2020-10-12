## 简介

目前密集目标检测器很受欢迎，其速度很快且精度不低，不过这种这种基于点的特征虽然使用方便，但会缺少关键的边界信息。旷视于 ECCV2020 发表的这篇 BorderDet，其中的核心就是设计了 Border Align 操作来从边界极限点提取边界特征用于加强点的特征。以此为基础设计了 BorderDet 框架，该框架依据 FCOS 的 baseline 插入 Border Align 构成，其在多个数据集上涨点明显。Border Align 是适用于几乎所有基于点的密集目标检测算法的即插即用模块。

- 论文地址

  https://arxiv.org/abs/2007.11056

- 论文源码

  https://github.com/Megvii-BaseDetection/BorderDet

## 介绍

目前大多数 point-based 的目标检测算法（如 SSD、RetinaNet、FCOS 等方法）都使用特征图上的 single-point 进行目标的回归和分类，但是，single-point 特征没有足够的信息表示一个目标实例，主要是因为缺乏边界信息。此前有很多方法来补充 single-point 的表示能力，但是这些方法往往带来较大计算量的同时并没有引入太多有用的信息，反而带来一些无用的背景信息。这篇文章设计了新的特征提取操作 BorderAlign 来直接利用边界特征优化 single-point 特征，以 BorderAlign 为拓展配合作为 baseline 的 FCOS，提出新的检测框架 BorderDet，实现 SOTA。

本文的贡献文中列了不少，但在我看来，只有一个核心：**分析密集目标检测器的特征表示，发现边界信息对 single-point 特征的重要性，并设计了一个高效的边界特征提取器 BorderAlign。** 其他的贡献都是顺理成章的附属产物。

## BorderAlign

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104009198.png#pic_center)


BorderAlign 的提出是基于大量的实验对比的，我这边就按照作者的思路来进行阐述。首先，采用如上图不同的特征增强方式在 FCOS 的基础上评估效果，结果如下表，根据效果最好的二四两行，发现，只使用边界上中心点做增强效果媲美 region-based 的方法。因此，得出结论，**point-based 方法做目标检测确实缺乏完整的目标特征，但从完整的边界框中密集提取特征是没必要且冗余的，高效的边界特征提取策略可以获得更好的特征增强效果。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104024405.png#pic_center)


针对上述结论，一种高效显式自适应提取边界特征的方法，BorderAlign 被提出。如下图所示，一个$5C$的 border-sensitive 特征图作为输入，其中$4C$维度对应边界框的四条边，另外$C$维度对应原始 anchor 点的特征。对于一个 anchor 点预测的边界框，对其四个边界在特征图上的特征做池化操作，由于框的位置是小数，所以采用双线性插值取边界特征。

这里具体的实现如下：假设输入的 5 个通道表示(single point, left border, top border, right border, bottom border)，那么对 anchor 点$(i, j)$对应的 bbox 各边均匀采样$N$个点，$N$默认是 5，如下图所示。采样点的值采用上面所说的双线性插值，然后通过逐通道最大池化得到输出，每个边只会输出值最大的采样点，那么每个 anchor 点最后采用 5 个点的特征作为输出，所以输出也是$5C$维度的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104036324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)


输出特征图相对输入特征图，各通道计算式如下，$(x_0, y_0, x_!, y_1)$为 anchor 点预测的 bbox。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104051290.png#pic_center)


显然，BorderAlign 是一种自适应的通过边界极限点得到边界特征的方法。文章中对其进行了一些可视化工作，下图所示的边上的小圆圈是边界极限点，大圆圈是不同 channel 上预测的边界极限点。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104100220.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)


## BAM(Border Alignment Module)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100410411168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)


该模块用于修正粗糙的 detection 结果，因而必须保证输入输出是同维张量，而其中的 BorderAlign 需求的是 5 个通道，所以必然要经历**降维、特征增强、升维**的过程，为了验证 border feature 的效果，BAM 采用 1x1 卷积实现维度变换。

## BorderDet

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104119644.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)


上图的框架采用 FCOS 作为 baseline，上面是分类分支，下面是回归分支，coarse cls score 和 coarse box reg 表示 FCOS 的输出。在四个卷积层后引出一个分支做 BorderAlign 操作，也就是进入 BAM 模块，该模块需要 bbox 位置信息，所以看到 coarse box reg 送入两个 BAM 中。最终这两个 BAM 预测得到 border cls score 和 border box reg，和检测器原始输出组合变为最终输出。

最后补充一点，BorderDet 在推理时对两种分类结果进行直接的相乘输出，而对于 bbox 定位则使用 border 定位预测对初步定位的 bbox 进行原论文中公式(2)的反向转换，对所有的结果进行 NMS 输出（IOU 阈值设置为 0.6）。

## 实验

论文进行了非常丰富的消融实验以对比 BorderAlign 的效果。

### 各分支效果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104128200.png#pic_center)


### 相比其他特征增强效果

和其他经典的特征增强手段相比，BorderAlign 在速度（使用 CUDA 实现了 BorderAlign）和精度上都有突破。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104137194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)

### 集成到检测器涨点效果

有比较明显的改进。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104208270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)


### 和主流检测器对比

可以看到，即使不使用多尺度策略，BorderDet 和当前 SOTA 相比效果也是不遑多让的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201004104218708.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob3VjaGVuMTk5OA==,size_16,color_FFFFFF,t_70#pic_center)


## 总结

边界信息对于 OD 问题十分重要，BorderDet 的核心思想 BorderAlign 高效地将边界特征融入到目标预测中，而且能够 PnP 融入到各种 point-based 目标检测算法中以带来较大的性能提升。

## 参考文献

[1]Qiu H, Ma Y, Li Z, et al. BorderDet: Border Feature for Dense Object Detection[J]. arXiv preprint arXiv:2007.11056, 2020.

[2]https://zhuanlan.zhihu.com/p/163044323

