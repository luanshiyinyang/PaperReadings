# DLA 论文解读

## 简介

沿着卷积神经网络在计算机视觉的发展史，可以发现，丰富的表示（representations）是视觉任务最需要的。随着网络的深度增加，单独一层的信息是远远不够的，只有聚合各层信息才能提高网络对 what（识别）和 where（定位）两个问题的推断能力。现有的很多网络设计工作主要致力于设计更深更宽的网络，但是如何更好地组合不同网络层（layer）、不同结构块（block）其实值得更多的关注。尽管跳跃连接（skip connecions）常用来组合多层，不过这样的连接依然是浅层的（因为其只采用了简单的单步运算）。该论文通过更深层的聚合来增强标准网络的能力。DLA 结构能够迭代式层级组合多层特征以提高精度的同时减参。实验证明，DLA 相比现有的分支和合并结构，效果更好。

- 论文地址

  https://arxiv.org/abs/1707.06484

- 论文源码

  https://github.com/ucbdrive/dla

## 介绍
表示学习和迁移学习的发展推动了计算机视觉的发展，可以简单组合的特性催生了很多深度网络。为了满足各种不同的任务，寻找合适的网络结构至关重要。随着网络尺寸的增加，模块化的设计更为重要，所以现在的网络越来越深的同时，更紧密的连接能否带来提升呢？

更多的非线性、更强的表示能力、更大的感受野一般能够提高网络精度，但是会带来难以优化和计算量大的问题。为了克服这些缺陷，不同的block和modul被集成到一起来平衡和优化这些特点，如使用bottlenecks进行降维、使用residual、gated和concatnative连接特征以及梯度传播算法。这些技术使得网络可以到达100甚至1000层。

然而，如何连接这些layer和module还需要更多的探索。简单地通过序列化堆叠层来构造网络，如LeNet、AlexNet以及ResNet。经过复杂的分析，更深的网络层能提取到更多语义和全局的特征，但是这并不能表明最后一层就是任务需要的表示。实际上“跳跃连接”已经证明了对于分类、回归以及其他结构化问题的有效性。因此，如何聚合，尤其是深度与宽度上的聚合，对于网络结构的优化是一个非常重要的技术。

![](https://i.loli.net/2020/10/22/ELq8voPCiXpZRWk.png)

论文研究了如何聚合各层特征来融合语义和空间信息以进行识别与定位任务。通过扩展现有的“浅跳跃连接”（单层内部进行连接），论文提出的聚合结构实现了更深的信息共享。文中主要引入两种DLA结构：iterative deep aggregation (IDA，迭代式深度聚合)和hierarchal deep aggregation (HDA，层级深度聚合)。为了更好的兼容现有以及以后的网络结构，IDA和HDA结构通过独立于backbone的结构化框架实现。IDA主要进行跨分辨率和尺度的融合，而HDA主要用于融合各个module和channel的特征。**从上图也可以看出来，DLA集成了密集连接和特征金字塔的优势，IDA根据基础网络结构，逐级提炼分辨率和聚合尺度（语义信息的融合（发生在通道和深度上），类似残差模块），HDA通过自身的树状连接结构，将各个层级聚合为不同等级的表征（空间信息的融合（发生在分辨率和尺度上），类似于FPN）。本文的策略可以通过混合使用IDA与HDA来共同提升效果。** 

DLA通过实验在现有的ResNet和ResNeXt网络结构上采用DLA架构进行拓展，来进行图像分类、细粒度的图像识别、语义分割和边界检测任务。实验表明，DLA的可以在现有的ResNet、ResNeXt、DenseNet等网络结构的基础上提升模型的性能、减少参数数量以及减少显存消耗。DLA达到了目前分类任务中compact models的最佳精度，在分割等任务也达到超越SOTA。DLA是一种通用而有效的深度网络拓展技术。

## DLA
文中将“聚合”定义为跨越整个网络的多层组合，文中研究的也是那些深度、分辨率、尺度上能有效聚合的一系列网络。由于网络可以包含许多层和连接，模块化的设计可以通过分组和重复来克服复杂度问题。多个layer组合为一个block，多个block再根据分辨率组合为一个stage，DLA则主要探讨block和stage的组合（stage间网络保持一致分辨率，那么空间融合发生在stage间，语义融合发生在stage内）。

![](https://i.loli.net/2020/10/22/6cypY4rg9Hkt12h.png)

### IDA
IDA沿着迭代堆叠的backbone进行，依据分辨率对整个网络分stage，越深的stage含有更多的语义信息但空间信息很少。“跳跃连接”由浅至深融合了不同尺度以及分辨率信息，但是这样的“跳跃连接”都是线性且都融合了最浅层的信息，如上图b中，每个stage都只融合上一步的信息。

因此，论文提出了IDA结构，从最浅最小的尺度开始，迭代式地融合更深更大尺度地信息，这样可以使得浅层网络信息在后续stage中获得更多地处理从而得到精炼，上图c就是IDA基本结构。

IDA对应的聚合函数$I$对不同层的特征，随着加深的语义信息表示如下（N表示聚合结点，后文会提到）：

$$
I\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right)=\left\{\begin{array}{ll}
\mathbf{x}_{1} & \text { if } n=1 \\
I\left(N\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right), \ldots, \mathbf{x}_{n}\right) & \text { otherwise }
\end{array}\right.
$$

### HDA
HDA以树的形式合并block和stage来保持和组合特征通道，通过HDA，浅层和深层的网络层可以组合到一起，这样的组合信息可以跨越各个层级从而学得更加丰富。尽管IDA可以高效组合stage，但它依然是序列性的，不足以用来融合网络各个block信息。上图d就是HDA的深度分支结构，这是一个明显的树形结构。

在基础的HDA结构上，可以改进其深度和效率，将一个聚合结点的输出重置为主干网络用于下一棵子树的输入。结构如上图e所示，这样，之前所有block的信息都被送入后续处理中。同时，为了效率，作者将同一深度的聚合结点（指的是父亲和左孩子，也就是相同特征图尺寸的）合并，如上图f。

HDA的聚合函数$T_n$计算如下，$n$表示深度，$N$依然是聚合结点。

$$
\begin{array}{r}
T_{n}(\mathbf{x})=N\left(R_{n-1}^{n}(\mathbf{x}), R_{n-2}^{n}(\mathbf{x}), \ldots,\right. 
\left.R_{1}^{n}(\mathbf{x}), L_{1}^{n}(\mathbf{x}), L_{2}^{n}(\mathbf{x})\right),
\end{array}
$$

上面式子的$R$和$L$定义如下，表示左树和右树，下式$B$表示卷积块。

$$
\begin{aligned}
L_{2}^{n}(\mathbf{x}) &=B\left(L_{1}^{n}(\mathbf{x})\right), \quad L_{1}^{n}(\mathbf{x})=B\left(R_{1}^{n}(\mathbf{x})\right) \\
R_{m}^{n}(\mathbf{x}) &=\left\{\begin{array}{ll}
T_{m}(\mathbf{x}) & \text { if } m=n-1 \\
T_{m}\left(R_{m+1}^{n}(\mathbf{x})\right) & \text { otherwise }
\end{array}\right.
\end{aligned}
$$

### 结构元素

**聚合结点（Aggregation Nodes ）**

其主要功能是组合压缩输入，它通过学习如何选择和投射重要的信息，以在它们的输出中保持与单个输入相同的维度。论文中，IDA都是二分的（两个输入），HDA则根据树结构深度不同有不定量的参数。

虽然聚合结点可以采用任意结构，不过为了简单起见，文中采用了conv-BN-激活函数的组合。图像分类中所有聚合结点采用1x1卷积，分割任务中，额外一个IDA用来特征插值，此时采用3x3卷积。

由于残差连接的有效性，本文聚合结点也采用了残差连接，这能保证梯度不会消失和爆炸。基础聚合函数定义如下，其中$\sigma$是非线性激活函数，$w$和$b$是卷积核参数。

$$
N\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right)=\sigma\left(\text { BatchNorm }\left(\sum_{i} W_{i} \mathbf{x}_{i}+\mathbf{b}\right)\right)
$$

包含残差连接的聚合函数变为下式。

$$
N\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right)=\sigma\left(\text { Batch } \operatorname{Norm}\left(\sum_{i} W_{i} \mathbf{x}_{i}+\mathbf{b}\right)+\mathbf{x}_{n}\right)
$$

**块和层（Blocks and Stages）**

DLA是一系列适用于各种backbone的结构，它对block和stage内部结构没有要求。论文主要实验了三种残差块，分别如下。

- Basic blocks：将堆叠的卷积层通过一个跳跃连接连接起来；
- Bottleneck blocks：通过1x1卷积对堆叠的卷积层进行降维来进行正则化；
- Split blocks：将不同的通道分组到不同的路径（称为cardinality split）来对特征图进行分散。


## 实验
作者设计了分类网络和密集预测网络，前者基于ResNet等修改设计了如下的DLA网络，下表是设计的一系列网络的配置。

![](https://i.loli.net/2020/10/22/kJOICgQwhpxYnKD.png)

![](https://i.loli.net/2020/10/22/xNY5ERprfaWTnMm.png)

DLA网络在Imagenet上对比其他的紧凑网络，结果如下，从精度和参数量来看，性能都是卓越的。

![](https://i.loli.net/2020/10/22/U4E2mg5ZM8GITuy.png)

作者还在检测等任务上也做了实验，这里我就不分析了，感兴趣的可以去阅读原论文。

## 补充说明

在很多人还在致力于研究如何设计更深更宽的网络的时候，DLA 想到的问题是如何更好地聚合一个网络中不同层和块的信息，并开创性地提出了 IDA 和 HDA 两种聚合思路。在多类任务上效果都有明显改善，包括分类、分割、细粒度分类等。而且，相比于普通 backbone，DLA 结构的网络参数量更少（并不意味着运算速度快，因为这种跳跃连接结构是很耗内存的），在原始网络上进行 DLA 改进往往能获得更为不错的效果。


