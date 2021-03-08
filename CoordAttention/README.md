# CoordAttention解读

## 简介

在轻量级网络上的研究表明，通道注意力会给模型带来比较显著的性能提升，但是通道注意力通常会忽略对生成空间选择性注意力图非常重要的位置信息。因此，新加坡国立大学的Qibin Hou等人提出了一种为轻量级网络设计的新的注意力机制，该机制将位置信息嵌入到了通道注意力中，称为coordinate attention（简称CoordAttention，下文也称CA），该论文已被CVPR2021收录。不同于通道注意力将输入通过2D全局池化转化为单个特征向量，CoordAttention将通道注意力分解为两个沿着不同方向聚合特征的1D特征编码过程。这样的好处是可以沿着一个空间方向捕获长程依赖，沿着另一个空间方向保留精确的位置信息。然后，将生成的特征图分别编码，形成一对方向感知和位置敏感的特征图，它们可以互补地应用到输入特征图来增强感兴趣的目标的表示。

![](https://i.loli.net/2021/03/08/j5ytfTsdqpRFkh9.png)

CoordAttention简单灵活且高效，可以插入经典的轻量级网络（如MobileNetV2）在几乎不带来额外计算开销的前提下，提升网络的精度。实验表明，CoordAttention不仅仅对于分类任务有不错的提高，对目标检测、实例分割这类密集预测的任务，效果提升更加明显。

- 论文标题

    Coordinate Attention for Efficient Mobile Network Design
- 论文地址

    http://arxiv.org/abs/2103.02907
- 论文源码

    https://github.com/Andrew-Qibin/CoordAttention


## 介绍

[注意力机制](https://zhouchen.blog.csdn.net/article/details/111302952)常用来告诉模型需要更关注哪些内容和哪些位置，已经被广泛使用在深度神经网络中来加强模型的性能。然而，在模型容量被严格限制的轻量级网络中，注意力的应用是非常滞后的，这主要是因为大多数注意力机制的计算开销是轻量级网络负担不起的。

![](https://i.loli.net/2021/03/08/97YAhCH2wjGNiVR.png)

考虑到轻量级网络有限的计算能力，目前最流行的注意力机制仍然是SENet提出的SE Attention。如上图所示，它通过2D全局池化来计算通道注意力，在相当低的计算成本下提供了显著的性能提升。遗憾的是，SE模块只考虑了通道间信息的编码而忽视了位置信息的重要性，而位置信息其实对于很多需要捕获目标结构的视觉任务至关重要。因此，后来CBAM等方法通过减少通道数继而使用大尺寸卷积来利用位置信息，如下图所示。然而，卷积仅仅能够捕获局部相关性，建模对视觉任务非常重要的长程依赖则显得有些有心无力。

![](https://i.loli.net/2021/03/08/pEU62YbflIPk7KD.png)

因此，这篇论文的作者提出了一种新的高效注意力机制，通过将位置信息嵌入到通道注意力中，使得轻量级网络能够在更大的区域上进行注意力，同时避免了产生大量的计算开销。为了缓解2D全局池化造成的位置信息丢失，论文作者将通道注意力分解为两个并行的1D特征编码过程，有效地将空间坐标信息整合到生成的注意图中。更具体来说，作者利用两个一维全局池化操作分别将垂直和水平方向的输入特征聚合为两个独立的方向感知特征图。然后，这两个嵌入特定方向信息的特征图分别被编码为两个注意力图，每个注意力图都捕获了输入特征图沿着一个空间方向的长程依赖。因此，位置信息就被保存在生成的注意力图里了，两个注意力图接着被乘到输入特征图上来增强特征图的表示能力。由于这种注意力操作能够区分空间方向（即坐标）并且生成坐标感知的特征图，因此将提出的方法称为坐标注意力（coordinate attention）。

## Coordinate Attention

相比此前的轻量级网络上的注意力方法，coordinate attention存在以下优势。首先，它不仅仅能捕获跨通道的信息，还能捕获方向感知和位置感知的信息，这能帮助模型更加精准地定位和识别感兴趣的目标；其次，coordinate attention灵活且轻量，可以被容易地插入经典模块，如MobileNetV2提出的inverted residual block和MobileNeXt提出的 sandglass block，来通过强化信息表示的方法增强特征；最后，作为一个预训练模型，coordinate attention可以在轻量级网络的基础上给下游任务带来巨大的增益，特别是那些存在密集预测的任务（如语义分割）。 

![](https://i.loli.net/2021/03/08/5azgNKbvPdkZtBL.png)

一个coordinate attention模块可以看作一个用来增强特征表示能力的计算单元。它可以将任何中间张量$\mathbf{X}=\left[\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{C}\right] \in \mathbb{R}^{C \times H \times W}$作为输入并输出一个有着增强的表示能力的同样尺寸的输出$\mathbf{Y}=\left[\mathbf{y}_{1}, \mathbf{y}_{2}, \ldots, \mathbf{y}_{C}\right]$。

### SE模块

由于CA（coordinate attention）是基于SENet的思考，所以首先来回顾一下SE Attention（详细关于SENet的解读可以参考[我的博文](https://zhouchen.blog.csdn.net/article/details/110826497)）。标准的卷积操作是很难建立通道之间的关系的，但是显式建模通道之间的关系可以增强模型对信息通道的敏感性，从而对最终的决策产生更多的影响。因此，SE模块对通道关系进行显式建模，取得了突破性的进展。

![](https://i.loli.net/2021/03/08/97YAhCH2wjGNiVR.png)

从上图的结构上来看，SE模块可以分为两步：**压缩（squeeze）**和**激励（excitation）**，分别用于全局信息的嵌入和自适应通道关系的加权。给定输入$X$，第$c$个通道的squeeze操作可以表述如下式，$z_c$就是第$c$个通道的输出。输入$X$来自固定核大小的卷积层，因此可以被看作一堆局部描述的集合。squeeze操作使得模型能够收集全局的信息。

$$
z_{c}=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c}(i, j)
$$

SE模块的第二步是excitation操作，旨在完全捕获通道之间的依赖，可以表述如下式，这里的$\cdot$表示逐通道乘法，$\sigma$表示Sigmoid函数，$\hat{\mathbf{z}}$是通过变换函数生成的，变换函数为$\hat{\mathbf{z}}=T_{2}\left(\operatorname{ReLU}\left(T_{1}(\mathbf{z})\right)\right)$，其中$T_1$和$T_2$表示两个可学习的线性变换，用来捕获每个通道的重要性（对应上图的两个Fully Connected及其中间的部分）。

$$
\hat{\mathbf{X}}=\mathbf{X} \cdot \sigma(\hat{\mathbf{z}})
$$

近些年来，SE模块被广泛使用在轻量级网络中，并且成为获得SOTA表现的关键组件。然而，它仅仅考虑了建模通道间的关系来对每个通道加权，忽略了位置信息，而位置信息对生成空间选择性特征图是分外重要的。因此，论文作者设计了一种新的同时考虑通道关系和位置信息的注意力模块，coordinate attention block（CA模块）。

### CA模块

CA模块通过精确的位置信息对通道关系和长程依赖进行编码，类似SE模块，也分为两个步骤：坐标信息嵌入（coordinate information embedding）和坐标注意力生成（coordinate attention generation），它的具体结构如下图。

![](https://i.loli.net/2021/03/08/5azgNKbvPdkZtBL.png)

首先，我们来看坐标信息嵌入这部分。全局池化常用于通道注意力中来全局编码空间信息为通道描述符，因此难以保存位置信息。为了促进注意力模块能够捕获具有精确位置信息的空间长程依赖，作者将全局池化分解为一对一维特征编码操作。具体而言，对输入$X$，先使用尺寸$(H,1)$和$(1,W)$的池化核沿着水平坐标方向和竖直坐标方向对每个通道进行编码，因此，高度为$h$的第$c$个通道的输出表述如下。

$$
z_{c}^{h}(h)=\frac{1}{W} \sum_{0 \leq i<W} x_{c}(h, i)
$$

类似，宽度为$w$的第$c$个通道的输出表述如下。

$$
z_{c}^{w}(w)=\frac{1}{H} \sum_{0 \leq j<H} x_{c}(j, w)
$$

上面这两个变换沿着两个空间方向进行特征聚合，返回一对方向感知注意力图。这和SE模块产生一个特征向量的方法截然不同，这两种变换也允许注意力模块捕捉到沿着一个空间方向的长程依赖，并保存沿着另一个空间方向的精确位置信息，这有助于网络更准确地定位感兴趣的目标。这个coordinate information embedding操作对应上图的X Avg Pool和Y Avg Pool这个部分。

接着，为了更好地利用上面coordinate information embedding模块产生的具有全局感受野并拥有精确位置信息的表示，设计了coordinate attention generation操作，它生成注意力图，遵循如下三个标准。

- 首先，对于移动环境中的应用来说，这种转换应该尽可能简单高效；
- 其次，它可以充分利用捕获到的位置信息，精确定位感兴趣区域；
- 最后，它还应该能够有效地捕捉通道之间的关系，这是根本。

作者设计的coordinate attention generation操作具体来看，首先级联之前模块生成的两个特征图，然后使用一个共享的1x1卷积进行变换$F_1$，表述如下式，生成的$\mathbf{f} \in \mathbb{R}^{C / r \times(H+W)}$是对空间信息在水平方向和竖直方向的中间特征图，这里的$r$表示下采样比例，和SE模块一样用来控制模块的大小。

$$
\mathbf{f}=\delta\left(F_{1}\left(\left[\mathbf{z}^{h}, \mathbf{z}^{w}\right]\right)\right)
$$

接着，沿着空间维度将$\mathbf{f}$切分为两个单独的张量$\mathbf{f}^{h} \in \mathbb{R}^{C / r \times H}$和$\mathbf{f}^{w} \in \mathbb{R}^{C / r \times W}$，再利用两个1x1卷积$F_{h}$和$F_{w}$将特征图$\mathbf{f}^{h}$ and $\mathbf{f}^{w}$变换到和输入$X$同样的通道数，得到下式的结果。

$$
\begin{aligned}
\mathbf{g}^{h} &=\sigma\left(F_{h}\left(\mathbf{f}^{h}\right)\right) \\
\mathbf{g}^{w} &=\sigma\left(F_{w}\left(\mathbf{f}^{w}\right)\right)
\end{aligned}
$$

然后对$g_h$和$g^w$进行拓展，作为注意力权重，CA模块的最终输出可以表述如下式。

$$
y_{c}(i, j)=x_{c}(i, j) \times g_{c}^{h}(i) \times g_{c}^{w}(j)
$$

这部分coordinate attention generation对应上图剩余的部分，至此CA模块同时完成了水平方向和竖直方向的注意力，同时它也是一种通道注意力。

## 实验

作者采用下图所示的结构进行实验，验证设计的注意力机制的效果，分别是MobileNetV和MobileNeXt设计的两种残差模块。

![](https://i.loli.net/2021/03/08/x2P3WGH8TVKUFIb.png)

作者首先对两个方向的必必要性进行验证，结果如下图，显然，两个方向都是必要的，CA模块可以在保证参数量的前提下，提高精度。

![](https://i.loli.net/2021/03/08/aY7UefAJSsZGVg1.png)

接着，进行权重因子的消融实验，下图先后是MobileNetV2和MobileNeXt基础上的结果，CA模块均取得了最好的效果，无论以哪个模型为baseline或者选择怎样的权重因子，CA模块均靠设计上的优越性取得了最好效果。

![](https://i.loli.net/2021/03/08/O8EsLI7RbKvcoiY.png)

![](https://i.loli.net/2021/03/08/8YCHhWIB3KqvS5a.png)

关于下采样比例也做了实验，CA模块随着r的下调精度上升但是模型变大，依旧表现最佳，鲁棒性很强。

![](https://i.loli.net/2021/03/08/fgxWbSXw9raqBvp.png)

之后，还对SE、CBAM和CA模块注意力结果可视化，大致能看出来CA更能精确关注感兴趣目标。

![](https://i.loli.net/2021/03/08/AxY3uVqmnzileOb.png)

为了检验所提CA模块的性能，采用EfficientNet-b0作为baseline，作者简单地用CA模块代替SE模块。并和其他同样强大的网络对比，CA模块依旧有着强大的表现。

![](https://i.loli.net/2021/03/08/Sc4GqBudWCsnrgF.png)

此外，作者还做了目标检测和语义分割任务上的实验，性能提升更大，由于位置信息的加入，这种依赖位置信息的密集预测效果明显更好，我这里就不贴了。

## 总结

为了将空间信息加入通道注意力，论文作者设计了Coordinate Attention，在轻量级网络上取得了比较大的成功，它既能捕获通道之间的依赖也能很好地建模位置信息和长程依赖，实验表明其在图像识别、目标检测和语义分割任务上都有不错的改进。