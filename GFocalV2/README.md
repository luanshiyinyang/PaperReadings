# GFocalV2解读

> 本文对作者基于GFocalV1的工作的扩展，取得了不错的效果，想要更加直白理解这篇文章的话可以参考作者的[《大白话 Generalized Focal Loss V2》](https://zhuanlan.zhihu.com/p/313684358)。

## 简介

在[之前的文章](https://zhouchen.blog.csdn.net/article/details/114972223)介绍了GFocalV1这个开创性的工作，它将边界框建模为通用分布表示，这个分布式的表示其实在GFocalV1中对整个模型的贡献并不是非常大（相比于分类-质量联合表示），因此如何充分利用成为GFocalV2的出发点，事实上，既然分布的形状和真实的定位质量高度相关，那么这个边框的分布表示其实可以算出统计量指导定位质量估计，这就是GFocalV2的核心创新点。

![](https://i.loli.net/2021/03/20/WSYk3tfLroGKMIw.png)

- 论文标题

    Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection
- 论文地址

    http://arxiv.org/abs/2011.12885
- 论文源码

    https://github.com/implus/GFocalV2

## 介绍

当前的密集检测器由于其优雅且高效，成为业界比较关注的研究热点之一。这类框架的一个重要组件就是Localization Quality Estimation（LQE，定位质量评估模块），通过LQE，高质量检测框相比于低质量检测框有着更高的得分从而降低被NMS消除的风险。LQE的实现形式多变，早有基于IoU的Objectness，近有基于偏中心距离的Centerness，不过它们都有共同的特点，那就是它们都基于原始的卷积特征进行质量估计，如基于点、边界框或者区域的特征。

不同于之前的工作，GFocalV2从一个新的角度实现LQE，它利用了边界框分布的统计学特征而不使用原始的卷积特征。这里的边界框通用分布源于GFocalV1，它通过学习每个要预测的边的概率分布来描述边界框回归的不确定性，如下图的(a)所示。有趣的是，这个通用分布其实和真实的定位质量高度相关，作者将预测框分布的top1值和真实的IoU定位质量做了一个散点图，如下图的(b)所示，可以发现整个散点图其实是有y=x这一趋势的，这就是说，统计意义上来看，分布的形状其实与定位质量高相关。

![](https://i.loli.net/2021/03/20/ORMhGf3wctZexYX.png)

更具体一点，从上图的(c)和(d)可以发现，边界框分布的尖锐程度可以清晰描述预测框的定位质量，对于那些模型很确信的预测它的分布是很尖锐的（在一个位置非常突出），而对于模型比较容易混淆的（上图的伞柄）预测框，分布是比较平缓甚至呈现双峰分布的。因此，分布的形状可以指导模型获得更好的LQE，那么如何刻画分布的形状呢？其实论文这里采用了一个非常简单的方法，就是topk数值来进行描述，这点后文会详细讲解。

总之，论文作者设计了一个非常轻量的子网络基于这些分布的统计量产生更加可靠的LQE得分，取得了可观的性能提升，文章将这个子网络称为Distribution-Guided Quality Predictor (DGQP)。引入这个DGQP之后的GFocalV1就成为了GFocalV2，这是一个新的密集检测器，在ATSS上获得了2.6AP的收益。

## 方法

在之前的文章中，我已经介绍了GFocalV1了，那篇文章的核心创新点是**将分类得分和IoU定位质量估计联合到了一起**和**将边界框表示建模为通用分布**，这里就不多做回顾了。

### 分解分类和IoU的表示

在尽管在GFocalV1中分类和质量估计的联合表示解决了训练和推理时的不一致问题，然而只是使用分类分支来产生联合表示还是有所局限，所以这篇论文中作者将其显式分解为利用分类（$C$）和回归（$I$）两个分支的结构，联合表示的$J=C\times I$。其中$\mathbf{C}=\left[C_{1}, C_{2}, \ldots, C_{m}\right], C_{i} \in[0,1]$是$m$个类别的分类表示，而$I \in[0,1]$是表示IoU表示的标量。尽管这里将$J$分解为了两部分，但是依然同时在训练和推理阶段使用$J$，所以依然保持了GFocalV1追求的一致性。

那么这个$C$是如何得到的呢，其实是通过设计的DGQP模块从$C$和$I$计算而来，J的监督依然由QFL损失进行，关于这个DGQP模块，下文会讲解。

### DGQP模块

Distribution-Guided Quality Predictor（DGQP）模块是GFocalV2最核心的模块，它如下图中的红色框所示，将预测的边框分布$P$输入到一个非常简单的全连接子网络中，得到一个IoU质量标量$I$，它和分类得分相乘作为最终的联合表示$J$。和GFocalV1类似，它将相对四个边的偏移量作为回归目标，用通用分布表示，将四个边表示为$\{l,r,t,b\}$，定义$w$边的概率分布为$\mathbf{P}^{w}=\left[P^{w}\left(y_{0}\right), P^{w}\left(y_{1}\right), \ldots, P^{w}\left(y_{n}\right)\right]$，其中$w \in\{l, r, t, b\}$。

![](https://i.loli.net/2021/03/20/KsZ23TFoQ4MlWNH.png)

关于如何选择分布的统计量，作者这里采用的是分布向量$P^w$的Top-k个值和整个向量的均值作为统计特征，因此四条边概率向量的基础统计特征$\mathbf{F} \in \mathbb{R}^{4(k+1)}$通过下式计算得到，其中的Topkm表示求topk值加上均值计算，concat表示串联。

$$
\mathbf{F}=\operatorname{Concat}\left(\left\{\operatorname{Topkm}\left(\mathbf{P}^{w}\right) \mid w \in\{l, r, t, b\}\right\}\right)
$$

之所以选择topk和均值，有以下两个好处。

1. 由于是概率分布，因此概率分布的和是固定为1的，因此topk和均值就能基本反映分布的平坦程度，越大越尖锐，越小越平坦。
2. topk和均值可以尽量和对象的尺度无关，也就是如下图所示的，不管是左边的小尺度目标还是右边的大尺度目标，它们的分布都是尖锐的（形状类似），因而统计值也应该差不多。这种表示方法更有鲁棒性。

![](https://i.loli.net/2021/03/20/AoeFYGd3DvHuXwQ.png)

下面来具体看一下DGQP模块的结构实现，如下图所示，上面的统计特征$F$输入DGQP模块（记为$\mathcal{F}$）中，这个模块只有两个全连接层（配合以激活函数），因而最终IoU预测标量如下式。

![](https://i.loli.net/2021/03/20/rkWa9GvENXsHT7o.png)

$$
I=\mathcal{F}(\mathbf{F})=\sigma\left(\mathbf{W}_{2} \delta\left(\mathbf{W}_{1} \mathbf{F}\right)\right)
$$

其中参数$\mathbf{W}_{1} \in \mathbb{R}^{p \times 4(k+1)}$和$\mathbf{W}_{2} \in \mathbb{R}^{1 \times p}$，$k$就是topk的取值，$p$是隐藏层的神经元数目（论文中$k=4$且$p=64$），最终输出的$I$和分类得分$C$相乘得到联合表示。

### 复杂度分析

需要注意的是，DGQP是非常轻量的，它引入的参数是可以忽略不急的。它也几乎不会带来计算开销。

## 实验

实验部分作者设计了不少消融实验，如统计量的选择、DGQP结构复杂度、统计特征指导和各种类型卷积特征的对比、联合表示的分解形式、和主流检测器的兼容性，这里我就不贴太多表格了。

![](https://i.loli.net/2021/03/20/fD1LgoujM6q5xvE.png)

上图是GFocalV2和主流SOTA的对比，性能改善是蛮大的。此外，作者还进行下图的可视化，可以看到其他算法准确的框score往往都在第三第四而GFocalV2质量较高的框得分也较高，这进一步说明GFocalV2可以利用好更好的定位质量估计来保障更精准的结果。

![](https://i.loli.net/2021/03/20/UpdRihj4WZMA1zN.png)

## 总结

这篇文章作者从一个新的角度地思考质量估计，从而提出边框的分布表示的统计特征指导质量估计，这是一个全新的工作，应该算是检测历史上第一个用学习到的分布的统计学特征指导质量估计的，本文也只是对这篇文章进行了比较粗糙的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。

