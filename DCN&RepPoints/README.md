# DCN & RepPoints解读

## 简介

近几年，Anchor-free的目标检测方法受到了很大的关注，究其原因，该类方法不需要像Anchor-base方法那样受限于anchor的配置（anchor的设置要求开发者对数据很了解）就可以获得不错的检测结果，大大减少了数据分析的复杂过程。Anchor-free方法中有一类方法是基于关键点的，它通过检测目标的边界点（如角点）来配对组合成边界框，RepPoints系列就是其代表之作，这包括了RepPoints、Dense RepPoints和RepPoints v2。不过，回顾更久远的历史，从模型的几何形变建模能力的角度来看，RepPoints其实也是对可变形卷积（Deformable Convolutional Networks，DCN）系列的改进，所以本文会从DCN开始讲起，简单回顾这几个工作对几何建模的贡献，其中，DCN系列包括DCN和DCN v2。

- [DCN](http://arxiv.org/abs/1703.06211)（ICCV2017）
- [DCN v2](http://arxiv.org/abs/1811.11168)（CVPR2019）
- [RepPoints](http://arxiv.org/abs/1904.11490)（ICCV2019）
- [Dense RepPoints](http://arxiv.org/abs/1912.11473)（ECCV2020）
- [RepPoints v2](http://arxiv.org/abs/2007.08508)（暂未收录）


## DCN

首先，我们来看DCN v1。在计算机视觉中，同一物体在不同的场景或者视角中未知的几何变化是识别和检测任务的一大挑战。为了解决这类问题，通常可以在数据和算法两个方面做文章。从数据的角度看来，通过充分的**数据增强**来构建各种几何变化的样本来增强模型的尺度变换适应能力；从算法的角度来看，设计一些**几何变换不变的特征**即可，比如SIFT特征。

上述的两种方法都很难做到，前者是因为样本的限制必然无法构建充分数据以保证模型的泛化能力，后者则是因为手工特征设计对于复杂几何变换是几乎不可能实现的。所以作者设计了Deformable Conv（可变形卷积）和Deformable Pooling（可变形池化）来解决这类问题。

### **可变形卷积**

顾名思义，可变形卷积的含义就是进行卷积运算的位置是可变的，不是传统的矩形网格，以原论文里的一个可视化图所示，左边的传统卷积的感受野是固定的，在最上层的特征图上其作用的区域显然不是完整贴合目标，而右边的可变形卷积在顶层特征图上自适应的感受野很好的捕获了目标的信息（这可以直观感受得到）。

![](./assets/conv2dcn.png)

那么可变形卷积是如何实现的呢，其实是通过针对每个卷积采样点的偏移量来实现的。如下图所示，其中淡绿色的表示常规采样点，深蓝色的表示可变卷积的采样点，它其实是在正常的采样坐标的基础上加上了一个偏移量（图中的箭头）。

![](./assets/dcn_sample.png)

我们先来看普通的卷积的实现。使用常规的网格$\mathcal{R}$在输入特征图$x$上进行采样，采样点的值和权重$w$相乘加和得到输出值。举个例子，一个3x3的卷积核定义的网格$\mathcal{R}$表示如下式，中心点为$(0,0)$，其余为相对位置，共9个点。

$$
\mathcal{R}=\{(-1,-1),(-1,0), \ldots,(0,1),(1,1)\}
$$

那么，对输出特征图$y$上的任意一个位置$p_0$都可以以下式进行计算，其中$\mathbf{p}_n$表示就是网格$\mathcal{R}$中的第$n$个点。

$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}\right)
$$

而可变形卷积干了啥呢，它对原本的卷积操作加了一个偏移量$\left\{\Delta \mathbf{p}_{n} \mid n=1, \ldots, N\right\}$，也就是这个偏移量使得卷积可以不规则进行，所以上面的计算式变为了下式。不过要注意的是，这个偏移量可以是小数，所以偏移后的位置特征需要通过双线性插值得到，计算式如下面第二个式子。

$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}+\Delta \mathbf{p}_{n}\right)
$$

$$
\mathbf{x}(\mathbf{p})=\sum_{\mathbf{q}} G(\mathbf{q}, \mathbf{p}) \cdot \mathbf{x}(\mathbf{q})
$$

至此，可变卷积的实现基本上理清楚了，现在的问题就是，这个偏移量如何获得？不妨看一下论文中一个3x3可变卷积的解释图（下图），图中可以发现，上面绿色的分支其实学习了一个和输入特征图同尺寸且通道数为$2N$的特征图（$N$为卷积核数目），这就是偏移量，之所以两倍是因为网格上偏移有x和y两个方向。

![](./assets/dcn3x3.png)

### **可变形RoI池化**

![](./assets/drp.png)

理解了可变形卷积，理解可变形RoI就没有太大的难度了。原始的RoI pooling在操作时将输入RoI划分为$k\times k = K$个区域，这些区域叫做bin，偏移就是针对这些bin做的。针对每个bin学习偏移量，这里通过全连接层进行学习，因此deformable RoI pooling的输出如下式（含义参考上面的可变卷积即可）。

$$
\mathbf{y}(i, j)=\sum_{\mathbf{p} \in \operatorname{bin}(i, j)} \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}+\Delta \mathbf{p}_{i j}\right) / n_{i j}
$$

**至此，关于DCN的解读就完成了，下图是一个来自原论文对的DCN效果的可视化，可以看到绿点标识的目标基本上被可变形卷积感受野覆盖，且这种覆盖能够针对不同尺度的目标。这说明，可变形卷积确实能够提取出感兴趣目标的完整特征，这对目标检测大有好处。**

![](./assets/dcn1_rst.png)