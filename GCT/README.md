# GCT解读

> 浙江大学等机构发布的一篇收录于CVPR2021的文章，提出了一种新的通道注意力结构，在几乎**不引入参数**的前提下优于大多SOTA通道注意力模型，如SE、ECA等。这篇文章虽然叫Gaussian Context Transformer，但是和Transformer并无太多联系，这里可以理解为高斯上下文变换器。

## 简介

此前，大量的通道注意力模块被提出用于增强卷积神经网络（CNN）的表示能力，这些方法通常使用全连接层或者线性变换来学习全局上下文和注意力激活值之间的关系（relationship）。但是，实验结果表明，这些注意力模块虽然引入了不少参数，但也许并没有很好地学习到这种关系。这篇论文中，作者假定这种关系是可以预先确定的，基于这个假设，作者提出了一种简单但是极其高效的通道注意力模块，名为 Gaussian Context Transformer(GCT)，它通过满足预设关系的高斯函数来实现上下文特征激励。GCT对大多数视觉网络都有比较明显的性能提升，并且性能优于现有的SOTA通道注意力模型，如SENet、ECANet等，且效率更高。

- 论文标题

    Gaussian Context Transformer
- 论文地址

    https://openaccess.thecvf.com/content/CVPR2021/html/Ruan_Gaussian_Context_Transformer_CVPR_2021_paper.html
- 论文源码

    暂未开源

## 介绍

卷积神经网络（CNN）已经在计算机视觉中取得了巨大的成功，推动了图像分类、分割和目标检测等任务的发展。然而，卷积核局部上下文感知的特性使得CNN难以有效捕获图像上的全局上下文信息，而这种全局感知对于语义任务是至关重要的。为了解决这个问题，注意力机制被引入卷积神经网络中，他们的核心思想通常是在原有网络结构的基础上添加一个轻量的模块来捕获全局的长程依赖（global long-range dependencies）。而在这其中，通道注意力往往受到了更多的关注，因为相比之下这类注意力更加简洁有效。此前，最著名的通道注意力为SENet，它通过捕获逐像素的依赖自适应地增强重要的通道并抑制不重要的通道，从而给CNN带来了巨大的提升。通道注意力的大体范式如下，输入一个$C\times H \times W$的特征图，注意力模块学习一个$C \times 1 \times 1$的注意力向量并和原始特征图相乘，得到注意力后的结果，不论是哪种通道注意力都大体在这个范式下，关于比较经典的视觉注意力，我也做过几篇文章的解读，欢迎[访问](https://zhouchen.blog.csdn.net/article/details/111302952)。

![](https://i.loli.net/2021/06/19/GxlY1dUpgwA4rMX.png)

SENet提出之后，不少方法对其进行改进，有的简化了特征变换模块、有的改变了融合模式，也有的集成了空间注意力机制。不过，这些方法或多或少都引入了不少参数来学习全局上下文和注意力激活值之间的关系，然而，这样学得的关系也许并不是足够好的。

LCT（linear context transform）观察所得，如下图所示，SE倾向于学习一种负相关，即全局上下文偏离均值越多，得到的注意力激活值就越小。为了更加精准地学习这种相关性，LCT使用一个逐通道地变换来替代SE中的两个全连接层。然而，实验表明，LCT学得的这种负相关质量并不是很高，下图中右侧可以看出，LCT的注意力激活值波动是很大的。

![](https://i.loli.net/2021/06/19/cKS1Z3QwXRrev6Y.png)

为了缓解上述问题，论文作者假设全局上下文和注意力激活值之间确实是一种负相关的关系。基于这个假设，作者提出了一个新的通道注意力模块，名为Gaussian Context Transformer（GCT），该模块通过一个表示预设负相关的高斯函数直接将全局注意力映射为注意力图。GCT的基础结构如下图所示，具体而言，GAP之后，GCT首先对通道向量进行标准化以稳定全局上下文的分布，然后一个高斯函数对标准化后的全局上下文进行激励操作（包括转换和激活两步）来获得注意力激活值（attention map，注意力图）。若高斯函数是固定的，这个模型称为GCT-B0，需要注意，GCT-B0是一个无参数的注意力块，它建模全局上下文不需要特征变换的学习。

![](https://i.loli.net/2021/06/19/cxTlDbG1wm4Rg7N.png)

![](https://i.loli.net/2021/06/19/xTVEXp3I8Gds61K.png)

上表是常见的几个通道注意力模块引入的参数量和性能比较，可以看到，GCT-B0在不引入任何参数的前提下就已经有相当不错的性能提升，这也证明了**全局特征变换并非必要的**。作者进一步设计了一个可学习的GCT版本，为GCT-B1，它能自适应学习高斯函数中的标准差，**因此它的参数量是一个数值**。实验表明，GCT-B1在分类任务上通常优于GCT-B0，但是在检测和分割任务上性能类似。

## GCT

受到LCT的启发，作者认为通道注意力机制学习的其实是一种负相关，基于这种假设GCT被设计用于建模全局上下文，这一整节将详细介绍GCT的细节，它的整个结构图如下图所示。

![](https://i.loli.net/2021/06/19/cxTlDbG1wm4Rg7N.png)

### Gaussian Context Transformer

GCT由三个操作构成，包括全局上下文聚合（global context aggregation，GCA）、标准化（normalization）和高斯上下文激励（gaussian context excitation，GCE），在上图中它是先后进行的。

首先来看GCA操作，它是为了捕获逐通道统计信息，手段是在特征图的空间上进行全局信息的聚合，GCA有助于网络捕获长程依赖。和SE类似，这篇论文也采用GAP作为聚合的方式。具体而言，一个$\mathbf{X} \in \mathbb{R}^{C \times H \times W}$的输入特征图，全局上下文信息可以表述为$\mathbf{z}=\operatorname{avg}(\mathbf{X})=\left\{z_{k}= \frac{1}{H \times W} \sum_{i=1}^{W} \sum_{j=1}^{H} \mathbf{X}_{k}(i, j): k \in\{1, \ldots, C\}\right\}$，其中$C$为通道数，$H$和$W$为特征图的宽和高。

此前的工作接下来会通过转换和激活来实现激励操作，即先通过线性层对全局上下文进行变换然后使用sigmoid激活函数将上下文激活为注意力图。不过，和这些工作不太一样，GCT设计了一种新的激励方式，它通过基于负相关假设的一个函数$f(\cdot)$实现转换和激活操作。具体而言，定义均值偏移为$\mathbf{z}-\mu$，这里的$\mu=\frac{1}{C} \sum_{k=1}^{C} z_{k}$表示全局上下文$\mathbf{z}$的均值，均值偏移就很好地度量了$\mathbf{z}$和$\mu$之间的偏差。但是，直接设置均值偏差作为输入会使得$f(\cdot)$不稳定，这是因为不同输入样本的均值偏差分布不一致。为了缓解这个问题，作者引入了一个特定实例的因子$\sigma$来稳定分布，保证0均值和1方差，这个过程描述为下面的式子，其中的$\sigma$为全局上下文的标准差，通过$\sqrt{\frac{1}{C} \sum_{k=1}^{C}\left(z_{k}-\mu\right)^{2}}+\epsilon$计算获得（$\epsilon$为一个很小的常数）。

$$
\hat{\mathbf{z}}=\frac{1}{\sigma}(\mathbf{z}-\mu)
$$

可以发现，这种获得$\hat{\mathbf{z}}$的方式与对$\mathbf{z}$进行归一化的结果一致，因此为了简洁，记其为标准化操作$\hat{\mathbf{z}}=\operatorname{norm}(\mathbf{z})$。

接着，为了满足负相关的假设，需要找到一个连续的函数$f(\hat{\mathbf{z}})$满足下面的条件：
1. $f(\hat{\mathbf{z}})$的取值范围应在$(0,1]$内才能作为注意力值，即$f(\hat{\mathbf{z}}) \in(0,1]$；
2. 当$\hat{\mathbf{z}}$取值为0的时候，$f(\hat{\mathbf{z}})$取得唯一最大值1；
3. 当$\hat{\mathbf{z}}$小于0时$f(\hat{\mathbf{z}})$单调增加，当$\hat{\mathbf{z}}$大于0时$f(\hat{\mathbf{z}})$单调减少；
4. 有$\lim _{\hat{\mathbf{z}} \rightarrow \pm \infty} f(\hat{\mathbf{z}})=0$。

在已知的函数中，高斯函数满足上面的所有条件，因此论文中采用该函数，因此有下面的GCA操作的定义，其中$a$为高斯函数的幅值，设为1以满足条件1；$b$则表示高斯函数的均值，设为0来满足条件2；$c$则表示高斯函数的标准差，控制通道注意力图的多样性，标准差越大，通道间激活值多样性越小。

$$
G(\hat{\mathbf{z}})=a e^{-\frac{(\hat{\mathbf{z}}-b)^{2}}{2 c^{2}}}
$$

通过设置，$G$可以简化为下面的式子，这里的$c$可以是常数，也可以是可学习的参数，$\mathbf{g}$则为注意力激活值，它和原始特征图相乘即可得到注意力后的特征图。

$$
\mathbf{g}=G(\hat{\mathbf{z}})=e^{-\frac{\hat{\mathbf{z}}^{2}}{2 c^{2}}}
$$

将上述所有的操作组合到一个式子中构建GCT模块，如下所示。

$$
\mathbf{Y}=e^{-\frac{norm(\operatorname{avg}(\mathbf{X}))^{2}}{2 c^{2}}} \mathbf{X}
$$

### Parameter-free GCT

当标准差$c$为常数的时候，GCT就是一个parameter-free（无参）注意力模块，称为GCT-B0，实验表明$c=2$可以获得一个较好的结果。

### Parameterized GCT

当然，$c$也可以是可学习的参数，这个版本的GCT为GCT-B1。为了约束$c \in[\beta, \alpha+\beta]$，通过下面的式子对其进行上下界约束，其中$\alpha$和$\beta$是常数，$\theta$为可学习参数，初始化为0。

$$
c=\alpha \cdot \operatorname{sigmoid}(\theta)+\beta
$$

**值得关注的是，GCT-B1只有一个可学习参数$\theta$，并没有引入其他的参数。**

## 实验

关于参数的消融实验这里不多赘述了，只列举几个SOTA对比的结果，分别是ImageNet上图像分类的结果、COCO上目标检测和实例分割的结果，可以发现，相比此前的通道注意力SOTA均有显著改进。

![](https://i.loli.net/2021/06/19/sNFpEZ5f9jgKJeA.png)

![](https://i.loli.net/2021/06/19/awsQ7rRTH4Lbdxe.png)

![](https://i.loli.net/2021/06/19/t5sXhvf79MDPQbk.png)

作者还可视化了均值偏移和激活值在不同stage的变化，GCT是很稳定的。

![](https://i.loli.net/2021/06/19/B5TKloUC2kNR4rH.png)

## 总结

这篇文章提出了一种新的通道注意力机制，GCT，仅仅通过高斯函数即可完成通道注意力的过程，几乎没有引入任何的参数并获得了SOTA效果，是非常有趣且实用的工作。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。