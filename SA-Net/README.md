# SA-Net解读

## 简介

这篇文章是南京大学Yu-Bin Yang等人于2021年初开放的一篇文章，已经被收录于ICASSP2021，文章提出了一种新的视觉注意力机制，称为Shuffle Attention（置换注意力），它通过置换单元组合空间注意力和通道注意力，相比此前的混合注意力更加高效，是一种非常轻量的注意力结构。实验表明，在ImageNet、COCO等benchmark上，超越了当前的SOTA注意力模型如SE、SGE等，且拥有更低的计算复杂度和参数量。

- 论文标题

    SA-Net: Shuffle Attention for Deep Convolutional Neural Networks
- 论文地址

    http://arxiv.org/abs/2102.00240
- 论文源码

    https://github.com/wofmanaf/SA-Net

![](https://i.loli.net/2021/03/08/BU2gOqlutNsfJvH.png)


## 介绍

注意力机制如今已经被广泛用于卷积神经网络中，大大提升了很多任务上的性能表现。目前视觉中的注意力机制主要有两种，如下图所示，分别是**通道注意力**和**空间注意力**（我在[之前的文章](https://zhouchen.blog.csdn.net/article/details/111302952)介绍了视觉中一些比较有名的注意力方法，可以访问查看）。

![](https://i.loli.net/2021/03/08/qZLEOnafbC4Nxk9.png)

通道注意力着重于捕获通道间的依赖而空间注意力则关于像素间的关系捕获，不过它们都是通过不同的聚合策略、转换方法和强化函数来从所有位置聚合相同的特征来强化原始特征。CBAM和GCNet同时处理空间和通道信息，获得了较好的精度提升，然而它们通常存在收敛困难、计算负担大等问题。也有一些工作关于简化注意力结构，如ECA-Net将SE模块中通道权重的计算改用了1D卷积来简化。SGE沿着通道维度对输入进行分组形成表示不同语义的子特征，在每个子特征上进行空间注意力。遗憾的是，这些方法都没用很好地利用空间注意力和通道注意力之间的相关性，所以效率偏低，所以自然而然引出这篇文章的出发点：能否以一个高效轻量的方式融合不同的注意力模块？

不妨先回顾一下轻量级网络的代表之一的ShuffleNetv2，它构建了一个可并行的多分支结构，如下图所示，在每个单元的入口，含有$c$个通道的输入被分割为$c-c'$和$c'$两个分支，接着几个卷积层用来提取关于输入更高级的信息，然后结果concat到一起保证和输入通道数相同，最后channel shuffle操作用来进行两个分支之间的信息通讯。类似的，SGE将输入沿着通道分组，所有子特征并行增强。

![](https://i.loli.net/2021/03/08/Bvy21fpmVnxuReO.png)

基于这些前人的工作，论文作者提出了一种更加轻量但是更加高效的Shuffle Attention（SA）模块，它也是将输入按照通道进行分组，对每组子特征，使用Shuffle Unit来同时构建通道注意力和空间注意力。对每个注意力模块，论文还设计了针对每个位置的注意力掩码来抑制噪声加强有效的语义信息。论文主要的贡献为设计了一个轻量但是有效的注意力模块，SA，它将输入特征图按照通道分组并对每个分组用Shuffle Unit实现空间和通道注意力.

## Shuffle Attention

![](https://i.loli.net/2021/03/08/U2Pxr5bm1tO7kGe.png)

所谓一图胜千言，这篇文章其实忽略掉一些细节上图就已经展示了Shuffle Attention模块（SA模块）的设计了，不过我这里还是按照论文的思路来逐步理解作者的设计，整个SA模块其实分为三个步骤，分别为**特征分组**、**混合注意力**、**特征聚合**。

### Feature Grouping

首先来看特征分组这个操作，它将输入特征图分为多组，每组为一个子特征（sub-feature）。具体来看，输入特征图$X \in \mathbb{R}^{C \times H \times W}$被沿着通道维度分为$G$组，表示为$X=\left[X_{1}, \cdots, X_{G}\right], X_{k} \in \mathbb{R}^{C / G \times H \times W}$，其中每个子特征$X_k$随着训练会逐渐捕获一种特定的语义信息。这部分对应上图最左边的**Group**标注的部分。

### Channel Attention and Spatial Attention

接着，$X_k$会被分为两个分支，依然是沿着通道维度划分，两个子特征表示为$X_{k 1}, X_{k 2} \in \mathbb{R}^{C / 2 G \times H \times W}$，如上图中间**Split**标注后的部分，两个分支分别是绿色和蓝色表示，上面的绿色分支实现通道注意力开采通道间的依赖，下面的蓝色分支则捕获特征之间的空间依赖生成空间注意力图，这样，模型同时完成了语义和位置信息的注意。

具体来看通道注意力这个分支，这里可以作者没用采用SE模块的设计，主要是考虑到轻量化设计的需求（SE的参数还是比较多的），也没有使用ECA-Net的设计采用一维卷积（ECA-Net要想精度高，对卷积核尺寸要求比较大），而是采用最简单的GAP+Scale+Sigmoid的单层变换，公式如下。

$$
s=\mathcal{F}_{g p}\left(X_{k 1}\right)=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{k 1}(i, j)
$$

$$
X_{k 1}^{\prime}=\sigma\left(\mathcal{F}_{c}(s)\right) \cdot X_{k 1}=\sigma\left(W_{1} s+b_{1}\right) \cdot X_{k 1}
$$

上面的式子只有两个变换参数，即$W_{1} \in \mathbb{R}^{C / 2 G \times 1 \times 1}$和$b_{1} \in \mathbb{R}^{C / 2 G \times 1 \times 1}$。

下面来看空注意力这个分支，思路也很简单，先是对输入特征图进行Group Norm，然后也是通过一个变换$\mathcal{F}_{c}(\cdot)$来增强输入的表示，具体公式如下。这里的参数也只有$W_2$和$b_2$，它们的尺寸都是$\mathbb{R}^{C / 2 G \times 1 \times 1}$。

$$
X_{k 2}^{\prime}=\sigma\left(W_{2} \cdot G N\left(X_{k 2}\right)+b_{2}\right) \cdot X_{k 2}
$$

最后，两种注意力的结果被concat到一起$X_{k}^{\prime}=\left[X_{k 1}^{\prime}, X_{k 2}^{\prime}\right] \in \mathbb{R}^{C / G \times H \times W}$，此时它已经和该组的输入尺寸一致了。

### Aggregation

最后一步的聚合也很简单，通过ShuffleNetv2采用的channel shuffle操作来保证各组子特征之间的交互，最后得到和输入$X$同维的注意力图。这是上图中最右边的部分。

至此，SA模块的构建已经完成，其实$W_1$、$b_1$、$W_2$和$b_2$就是整个SA模块所有的参数，它可以通过PyTorch轻易的实现。SA模块可以取代SE模块，因此替换SENet的网络称为SA-Net。

![](https://i.loli.net/2021/03/08/trHARNiDCfo6kZK.png)

为了验证SA模块对语义信息的提取能力，作者在ImageNet上训练了没用channel-shuffle的SA-Net50B和有channel-shuffle的SA-Net50，给出精度对比如下图。一方面在SA采用之后，top1精度表示出了提升，这也就意味着特征分组可以显著提升特征的语义表达能力；另一方面，不同类别的分布在前面的层中非常相似，这也就意味着在前期特征分组的重要性被不同类别共享，而随着深度加深，不同的特征激活表现出了类别相关性。

![](https://i.loli.net/2021/03/08/GaLpREQmSoKOrBX.png)

为了验证SA有效性，基于GradCAM进行可视化，得到下图的结果，可以发现，SA模块使得分类模型关注于目标信息更相关的区域，进而有效的提升分类精度。

![](https://i.loli.net/2021/03/08/wkSidlBmEcJ4TYj.png)

## 实验

为了验证性能，作者也在ImageNet、COCO上进行了分类、检测和分割任务的实验，分别如下。

![](https://i.loli.net/2021/03/08/jwUe8ErmDzhgqCo.png)

![](https://i.loli.net/2021/03/08/8MAJWjps96YPZRn.png)

![](https://i.loli.net/2021/03/08/jO6mX9BhzsnDIkH.png)

实验表明，SA模块相比SE等注意力方法在参数量更少的情况下，达到了更高的精度，是非常高效的轻量级注意力机制。

## 总结

从ShuffleNet得到启发，作者设计了轻量级网络上适用的高效注意力机制Shuffle Attention，推动了注意力在轻量级网络上的发展，在多个任务上达到SOTA表现。