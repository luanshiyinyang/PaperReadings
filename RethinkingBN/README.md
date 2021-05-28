# RethinkingBN解读

> 最近针对BN做的工作还是蛮多的，本文介绍的这篇文章是FAIR提出的明确针对BN的，《Rethinking Skip Connection with Layer Normalization in Transformers and ResNets》也是最近的工作，它是对跳跃连接提出的改进，也care了一下BN。

## 简介

BatchNorm已经成为如今卷积神经网络的核心模块，Conv-BN-ReLU已经是很多网络的核心block。和深度学习中其他的操作相比，它并非直接作用于单个样本而是作用于一批样本。因此，它也有很多需要注意的地方，设置不当可能会对模型性能有着负面的影响。本文彻底回顾了视觉识别任务中的此类问题，并指出解决这些问题的关键是重新思考BatchNorm中“batch”概念的不同选择。通过提出这些警告及其缓解措施，作者希望这篇文章可以帮助研究人员更有效地使用BatchNorm。

- 论文标题
- 论文地址
- 论文源码

## 介绍

BatchNorm如今已经广泛应用于CNN中，实践证明，它可以使模型对学习率和初始化不那么敏感，并可以针对各种网络体系结构进行训练。BN提高了模型收敛速度，并且还提供了消除过拟合的正则化效果。由于这些特性，自BN出现以来，它已经被广泛应用到几乎所有的SOTA CNN结构。

不过，尽管有着这些缺点，但是在不同的场景下如何精准使用BN仍需要多加考究。有时候，不那么完美的选择可能会损害模型的性能，也就是说模型仍然能够训练但未必能收敛到较高的精度。因此，BN在设计CNN时常被视作“necessary evil”。这篇论文的目的是总结应用BatchNorm时研究者可能会遇到的问题，并提供克服这些问题的建议。

BatchNorm和其他深度学习操作最大的不同在于它是作用在一批数据上而不是单个样本上的，BatchNorm在一批数据中进行统计量的计算，而其他算子即使是批处理也是单独处理每个样本的。因此，BatchNorm的输出不仅仅依赖于单个样本的性质，也依赖于batch的采样方式。正是因为BatchNorm的norm发生在batch这个维度，后来的很多方法通过修改作用的维度来改进BatchNorm，诞生了如下图所示的LayerNorm、InstanceNorm、GroupNorm等方法，**但是这些方法的标准化操作的含义都是确定的，而BN的batch却可以有各种各样的采样方法，这篇论文的初衷就是探索不同的batch采样方式会有什么影响。**

![](./assets/norms.png)

如下图所示，左图为三种数据集的采样方式，分别是entire dataset, mini-batches和subset of mini-batches，右图则是多域数据集的采样方式，分别为combined domain、each domain和a mixture of each domain。

![](./assets/batch_methods.png)

这篇论文探索了batch的不同获取方式对于BatchNorm而言非常重要，并通过丰富实验证明使用BN不考虑batch的采样方式会有许多负面影响，合理选择batch采样方式会改善模型性能。

## 回顾BN

首先，简单回归一下Batch Normalization，下文简称BN且均以CNN中的使用展开叙述。假定BN层的输入$x$的shape为$(C,H,W)$，其中$C$为通道数，$H$和$W$表示特征图的高和宽。如果使用逐通道统计量$\mu, \sigma^{2} \in \mathbb{R}^{C}$对$x$进行标准化，那么输出$y$则通过下面的式子计算。

$$
y=\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}
$$

具体计算$\mu$和$\sigma^2$的方式可能是不同的。最常见的计算方法是设置$\mu$和$\sigma^2$为mini-batch的统计量$\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^{2}$，也就是训练时一小批$N$个样本的经验均值和方差。这一批的$N$个样本被打包为一个四维张量$(N, C, H, W)$，然后就可以通过下面的式子计算均值和方差了。

$$
\begin{aligned}
\mu_{\mathcal{B}} &=\operatorname{mean}(X, \operatorname{axis}=[N, H, W]) \\
\sigma_{\mathcal{B}}^{2} &=\operatorname{var}(X, \text { axis }=[N, H, W])
\end{aligned}
$$

推理阶段，不再采用mini-batch的均值和方差了，而是采用整个训练集上得到的均值和方差$\mu_{\text {pop }}, \sigma_{\text {pop }}^{2}$来进行标准化。

事实上，关于如何获取这个batch，方法有很多，也就是说在哪些数据上计算均值和方差选择诸多。批处理的大小，批处理的数据源或用于计算统计量的算法在不同的情况下可能会有所不同，从而导致不一致，最终会影响模型的泛化能力。

不过，这篇论文关于BatchNorm的定义和此前略有不同，并不将通常紧跟标准化操作的逐通道仿射变换作为BN的一部分。尽管作者的实验都包含这个仿射变换，但是将其视为一个独立的、常规的层，因为它是对每个样本独立进行操作的，它实际上等价于一个depth-wise的1x1卷积层。这样区分开可以将关注的重点放在BN中batch这一独特的使用上。

通过排除这个仿射变换，总体统计信息$\mu_{pop}$和$\sigma_{pop}$成为BN的唯一的可学习参数，但是和神经网络中常见的可学习参数不一样，它既不被使用也不通过梯度进行更新，相反它通过其他算法（如EMA）进行训练。

## Whole Population as a Batch

训练时，BN在一个小批量的样本中计算用于标准化的统计量，但是，在推理时。通常没有mini-batch这个概念。在BatchNorm的原始论文中，作者提出在测试时特征应该使用在整个训练集上的整体统计量$\mu_{pop}$和$\sigma_{pop}$来进行标准化。这里的$\mu_{pop}, \sigma_{pop}$的定义和$\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}$是一样的，只不过是把整个训练集当作一个batch罢了。

那么，如何计算这个整体的$\mu_{pop}$和$\sigma_{pop}$呢？既然它来自所有数据，那么就应该在给定的数据集上进行“训练”，这个训练的方式可以有不同的算法。而且，这个训练的过程通常是无监督且代价不大的，但是其对于模型的泛化能力又是至关重要。在这一节，沿着论文的思路来了解并不总是那么准确的EMA算法以及提出的可替代算法。

### Inaccuracy of EMA

BN原始论文提出了指数移动平均（exponential moving average，EMA）来高效计算整体统计量，这种方案已经成为现在很多深度学习库的标准实现。然而，尽管EMA很流行，但是它也许并不能很好的估算整体统计量。

为了估算$\mu_{pop}$,$\sigma_{pop}$，每次训练迭代时，EMA通过下式进行更新，这里的$\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}$就是当前这一次迭代所用的这批数据的均值和方差，动量参数$\lambda$的范围是$\lambda \in[0,1]$。

$$
\begin{array}{l}
\mu_{E M A} \leftarrow \lambda \mu_{E M A}+(1-\lambda) \mu_{\mathcal{B}} \\
\sigma_{E M A}^{2} \leftarrow \lambda \sigma_{E M A}^{2}+(1-\lambda) \sigma_{\mathcal{B}}^{2}
\end{array}
$$

EMA造成整体统计量的次优估计可能有下面的两个原因：
1. 若$\lambda$过大，则统计量收敛过慢；
2. 若$\lambda$过小，统计量主要受最近的几个mini-batch影响，无法表示整个数据集的情况。

论文中通过分类任务做了个EMA的实验，结果如下图，可以看到，在训练的早期，EMA并不能准确地表示mini-batch的统计量或者整体统计量。因此，使用EMA来获得整体统计量可能对模型的效果是有害的。

![](./assets/inaccuracy.png)

### Towards Precise Population Statistics

为了计算更加准确的整体统计量，作者采用下面的两个手段来接近真实的整体统计量：1. 应用同一个模型到多个mini-batch上收集批量的统计量；2. 将多个batch上收集到的统计量聚合为一个整体统计量。

假设总共有$N$个样本，batch size为$B$且有$N=k \times B$，定义一个样本$x_{i j} \in \mathbb{R}, i=1 \cdots k, j= 1 \cdots B$，表示第$k$批的第$j$个样本，想要利用每一批的均值$\mu_i$和每一批的方差$\sigma_i^2$来估计所有样本$X$的方差。首先有下面的定义。

$$
\mu_{i}=\sum_{j=1}^{B} \frac{x_{i j}}{B}, \sigma_{i}^{2}=\sum_{j=1}^{B} \frac{\left(x_{i j}-\mu_{i}\right)^{2}}{B}
$$

接着，使用下面公式来进行估计。

$$
\hat{\sigma}^{2}=\frac{N}{N-1}\left[\sum_{i=1}^{k} \frac{\mu_{i}^{2}+\sigma_{i}^{2}}{k}-\left(\sum_{i=1}^{k} \frac{\mu_{i}}{k}\right)^{2}\right]
$$

原始的BN论文中使用下面的式子进行估计。

$$
\hat{\sigma}^{2}=\frac{B}{B-1} \sum_{i=1}^{k} \frac{\sigma_{i}^{2}}{k}
$$

相比于EMA，这个策略有两点重要的属性：1.统计量是通过相同模型计算得到的，而EMA是通过多个历史模型计算得到的；2.所有样本的权重是相同的，而EMA不同样本的权重是不同的。由于EMA现在已成为BatchNorm的实际标准，因此当BatchNorm层在推理中使用这种更精确的统计信息时，使用名称“PreciseBN”以免与流行的BatchNorm实现混淆。但是，这种方法实际上BN论文里已经定义了，并且fvcore已经对此有了PyTorch实现。

作者使用常规的256的batch size在ImageNet上训练一个ResNet50网络。error曲线如下图所示。可以看到，使用PreciseBN的验证集结果是明显好于EMA的，并且收敛更加稳定，这证明了EMA的不准确给模型的性能带来了负面的影响。

![](./assets/ema_pbn_error.png)

接着，作者使用了高达8192的batch size训练网络，此时两种策略的差距就非常明显了。在学习率还较大的前30轮模型是非常不稳定的，这种不稳定主要归于两个因素：过大的学习率使得特征急剧变化；由于batch很大，因此EMA更新的次数更少了。**相反，PreciseBN的结果非常稳定。**

![](./assets/ema_pbn_error2.png)

上述两个实验的结果也列表如下，而且，PreciseBN只需要$10^3-10^4$个样本就能达到近似最优的结果。

![](./assets/exp1.png)

当计算整体统计值的时候，batch size的大小也是非常重要的，不同的batch会改变统计量并影响输出特征。这种特征上的不同会累计在深层BN中，导致不准确的统计量估计。作者对此也进行了实验，可以发现当batch size较小的时候（EMA必须使用和SGD同样的batch size），EMA并不能准确估计整体统计量。但是，PreciseBN中的batch size可以独立选择而且只需要更少的内存因为其不需要反向传播，在小batch下（如$B=2$）PreciseBN比EMA高了$3.5\%$的准确率。

![](./assets/exp2.png)

## Batch in Training and Testing

