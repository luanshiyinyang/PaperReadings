# LambdaNetworks 论文解读

> 最近有不少人和我提到 ViT 以及 DETR 以及商汤提出的 Deformable DETR，仿若看到了 Transformer 在计算机视觉中大放异彩的未来，甚至谷歌对其在自注意力机制上进行了调整并提出 Performer。但是，由于 Transformer 的自注意力机制对内存的需求是输入的平方倍，这在图像任务上计算效率过低，当输入序列很长的时候，自注意力对长程交互建模计算量更是庞大无比。而且，Transformer 是出了名的难训练。所以，想要看到其在视觉任务上有更好的表现，还需要面临不小的挑战，不过，LambdaNetworks倒是提出了一种新的长程交互信息捕获的新范式，而且在视觉任务中效果很不错。

## 简介
文章对于捕获输入和结构化上下文之间的长程交互提出了一种新的通用框架，该方法名为Lambda Layer。它通过将可用上下文转化为名为lambdas的线性函数，并将这些函数分别应用于每个输入。Lambda层是通用的，它可以建模全局或者局部的内容和位置上的信息交互。并且，由于其避开了使用“昂贵”的注意力图，使得其可以适用于超长序列或者高分辨率图像。由Lambda构成的LambdaNetworks在计算上是高效的，并且可以通过主流计算库实现。实验证明，LambdaNetworks在图像分类、目标检测、实例分割等任务上达到sota水准且计算更加高效。同时，作者也基于ResNet改进设计了LambdaResNet并且获得和EfficientNet相当的效果，快了4.5倍。

- 论文地址

    https://openreview.net/forum?id=xTJEN-ggl1b
- 论文源码

    https://github.com/lucidrains/lambda-networks


## 介绍

建模长程信息交互是机器学习领域很重要的课题，注意力机制是当前最主流的范式，然而，自注意力的二次内存占用不利于处理超长序列或者多维输入，比如包含数万像素的图像。论文中这里举了个例子，一批256个64x64的图像使用8head的多头注意力就需要32G的内存。

考虑到自注意力的局限性，论文提出了Lambda层，该层为捕获输入和结构化的上下文之间的长程信息交互提供了一种新的通用框架。Lambda层捕获信息交互的方式也很简单，它将可用上下文转化为线性函数，并将这些线性函数分别应用于每个输入，这些线性函数就是lambda。Lambda层可以成为注意力机制的替代品，注意力在输入和上下文之间定义了一个相似性核，而Lambda层将上下文信息总结为一个固定size的线性函数，这样就避开了很耗内存的注意力图。他俩的对比，可以通过下面的图看出来（左图是一个包含三个query的局部上下文，它们同处一个全局上下文中；中图是attention机制产生的注意力图；右图则是lambda层线性函数作用于query的结果）。

![](https://i.loli.net/2020/11/04/eDh6RuJ7BsMjgZx.png)

Lambda层用途广泛，可以实现为在全局、局部或masked上下文中对内容和基于位置的交互进行建模。由此产生的神经网络结构LambdaNetworks具有高效的计算能力，并且可以以较小的内存开销建模长程依赖，因此非常适用于超大结构化输入，如高分辨率图像。

后文也用实验证明，在注意力表现很好的任务中，LambdaNetworks表现相当，且计算更为高效且更快。

## 长程信息交互建模
论文在第二部分主要对一些Lambda的术语进行了定义，引入keys作为捕获queries和它们的上下文之间信息交互的需求，而且，作者也说明，Lambda layer采用了很多自注意力的术语来减少阅读差异，这就是为什么很多人觉得两者在很多名称定义上差异不大的原因。

### **queries、contexts和interactions**

$\mathcal{Q}=\left\{\left(\boldsymbol{q}_{n}, n\right)\right\}$和$\mathcal{C}=\left\{\left(\boldsymbol{c}_{m}, m\right)\right\}$分别表示queries和contexts，每个$\left(\boldsymbol{q}_{n}, n\right)$都包含内容$\boldsymbol{q}_{n} \in \mathbb{R}^{|k|}$和位置$n$，同样的，每个上下文元素$\left(\boldsymbol{c}_{m}, m\right)$都包含内容$\boldsymbol{c}_{m}$和位置$m$，而$(n, m)$指的是任意结构化元素之间的成对关系。举个例子，这个(n,m)对可以指被固定在二维栅格上的两个像素的相对距离，也可以指图（Graph）上俩node之间的关系。

下面，作者介绍了Lambda layer的工作过程。先是考虑给定的上下文$\mathcal{C}$的情况下通过函数$\boldsymbol{F}:\left(\left(\boldsymbol{q}_{n}, n\right), \mathcal{C}\right) \mapsto \boldsymbol{y}_{n}$将query映射到输出向量$\boldsymbol{y}_{n}$。显然，如果处理的是结构化输入，那么这个函数可以作为神经网络中的一个层来看待。将$\left(\boldsymbol{q}_{n}, \boldsymbol{c}_{m}\right)$称为基于内容的交互，$\left(\boldsymbol{q}_{n},(n, m)\right)$则为基于位置的交互。此外，若$\boldsymbol{y}_{n}$依赖于所有的$\left(\boldsymbol{q}_{n}, \boldsymbol{c}_{m}\right)$或者$\left(\boldsymbol{q}_{n},(n, m)\right)$，则称$\boldsymbol{F}$捕获了全局信息交互，如果只是围绕$n$的一个较小的受限上下文用于映射，则称$\boldsymbol{F}$捕获了局部信息交互。最后，若这些交互包含了上下文中所有$|m|$个元素则称为密集交互（dense interaction），否则为稀疏交互（sparse interaction）。

### **引入key来捕获长程信息交互**

在深度学习这种依赖GPU计算的场景下，我们优先考虑快速的线性操作并且通过点积操作来捕获信息交互。这就促使了引入可以和query通过点击进行交互的向量，该向量和query同维。特别是基于内容的交互$\left(\boldsymbol{q}_{n}, \boldsymbol{c}_{m}\right)$需要一个依赖$\boldsymbol{c}_{m}$的$k$维向量，这个向量就是key（键）。相反，基于位置的交互$\left(\boldsymbol{q}_{n},(n, m)\right)$则需要位置编码$\boldsymbol{e}_{n m} \in \mathbb{R}^{|k|}$，有时也称为相对key。query和key的深度$|k|$以及上下文空间维度$|m|$不在输出$\boldsymbol{y}_{n} \in \mathbb{R}^{|v|}$，因此需要将这些维度收缩为layer计算的一部分。因此，捕获长程交互的每一层都可以根据它是收缩查询深度还是首先收缩上下文位置来表征。

### **注意力交互**

收缩query的深度首先会在query和上下文元素之间创建一个相似性核，这就是attention操作。随着上下文位置$|m|$的增大而输入输出维度$|k|$和$|v|$不变，考虑到层输出是一个很小维度的向量$|v| \ll|m|$，注意力图（attention map）的计算会变得很浪费资源。


### **Lambda交互**

相反，通过一个线性函数$\boldsymbol{\lambda}(\mathcal{C}, n)$获得输出$\boldsymbol{y}_{n}=F\left(\left(\boldsymbol{q}_{n}, n\right), \mathcal{C}\right)=\boldsymbol{\lambda}(\mathcal{C}, n)\left(\boldsymbol{q}_{n}\right)$会更高效地简化映射过程（map）。在这个场景中，上下文被聚合为一个固定size的线性函数$\boldsymbol{\lambda}_{n}=\boldsymbol{\lambda}(\mathcal{C}, n)$。每个$\boldsymbol{\lambda}_{n}$作为一个小的线性函数独立于上下文并且被用到相关的query$\boldsymbol{q}_n$后丢弃。这个机制很容易联想到影响比较大的函数式编程和lambda积分，所以称为lambda层。


## Lambda层

一个lambda层将输入$\boldsymbol{X} \in \mathbb{R}^{|n| \times d_{i n}}$和上下文$\boldsymbol{C} \in \mathbb{R}^{|m| \times d_{c}}$作为输入并产生线性函数lambdas分别作用于query，返回输出$\boldsymbol{Y} \in \mathbb{R}^{|n| \times d_{o u t}}$。显然，在自注意力中，$\boldsymbol{C} = \boldsymbol{X}$。为了不失一般性，我们假定$d_{i n}=d_{c}=d_{o u t}=d$。在接下来的论文里，作者将重点放在了lambda层的一个具体实例上，并且证明lambda层可以获得密集的长程内容和位置的信息交互而不需要构建注意力图。

### **将上下文转化为线性函数**
首先，假定上下文只有一个query$\left(\boldsymbol{q}_{n}, n\right)$。我们希望产生一个线性函数lambda$\mathbb{R}^{|k|} \rightarrow \mathbb{R}^{|v|}$，我们将$\mathbb{R}^{|k| \times|v|}$称为函数。下表所示的就是lambda层的超参、参数以及其他相关的配置。

![](https://i.loli.net/2020/11/04/rRBmAzS2dYHeniW.png)

**生成上下文lambda函数**：lambda层首先通过线性投影上下文来计算keys和values，并且使用softmax操作跨上下文对keys进行标准化从而得到标准化后的$\bar{K}$。它的实现可以看作是一种函数式消息传递，每个上下文元素贡献一个内容function$\boldsymbol{\mu}_{m}^{c}=\overline{\boldsymbol{K}}_{m} \boldsymbol{V}_{\boldsymbol{m}}^{T}$和位置function$\boldsymbol{\mu}_{n m}^{p}=\boldsymbol{E}_{n m} \boldsymbol{V}_{\boldsymbol{m}}^{T}$，最终的lambda函数其实是两者的和，具体如下，式子中的$\boldsymbol{\lambda}^{c}$为内容lambda，而$\boldsymbol{\lambda}^p_n$为位置lambda。内容$\boldsymbol{\lambda}^{c}$对上下文元素的排列是不变的，在所有的query位置$n$之间共享，并仅基于上下文内容对$\boldsymbol{q}_{n}$进行编码转换。不同的是，位置$\lambda_{n}^{p}$基于内容$\boldsymbol{c}_{m}$和位置$(n, m)$对查询query进行编码转换，从而支持对结构化输入建模如图像。

$$
\begin{aligned}
\boldsymbol{\lambda}^{c} &=\sum_{m} \boldsymbol{\mu}_{m}^{c}=\sum_{m} \overline{\boldsymbol{K}}_{m} \boldsymbol{V}_{\boldsymbol{m}}^{T} \\
\boldsymbol{\lambda}_{n}^{p} &=\sum_{m} \boldsymbol{\mu}_{n m}^{p}=\sum_{m} \boldsymbol{E}_{n m} \boldsymbol{V}_{\boldsymbol{m}}^{T} \\
\boldsymbol{\lambda}_{n} &=\boldsymbol{\lambda}^{c}+\boldsymbol{\lambda}_{n}^{p} \in \mathbb{R}^{|k| \times|v|}
\end{aligned}
$$

**应用lambda到query**：输入被转化为query$\boldsymbol{q}_{n}=\boldsymbol{W}_{Q} \boldsymbol{x}_{n}$，然后lambda层获得如下输出。

$$
\boldsymbol{y}_{n}=\boldsymbol{\lambda}_{n} \boldsymbol{q}_{n}=\left(\boldsymbol{\lambda}^{c}+\boldsymbol{\lambda}_{n}^{p}\right) \boldsymbol{q}_{n} \in \mathbb{R}^{|v|}
$$

**Lambda的解释**：$\boldsymbol{\lambda}_{n} \in \mathbb{R}^{|k| \times|v|}$矩阵的列可以看作$|k| |v|$维上下文特征的固定size的集合。这些上下文特征从上下文内容和结构聚合而来。应用lambda线性函数动态地分布这些上下文特征来产生输出$\boldsymbol{y}_{n}=\sum_{k} q_{n k} \boldsymbol{\lambda}_{n k}$。这个过程捕获密集地内容和位置的长程信息交互，而不需要产生注意力图。

**标准化**： 实验表明，非线性或者标准化操作对计算是有帮助的，作者在计算的query和value之后应用batch normalization发现是有效的。

### **对结构化上下文应用Lambda函数**

在这一节，作者主要介绍如何将lambda层应用于结构化上下文。

**Translation equivariance**：在很多机器学习场景中，Translation equivariance是一个很强的归纳偏置。由于基于内容的信息交互是排列等变的，因此本就是translation equivariant。而位置的信息交互获得translation equivariant则通过对任意的translation $t$确保位置编码满足$\boldsymbol{E}_{n m}=\boldsymbol{E}_{t(n) t(m)}$来做到。实际中，我们定义一个相对位置编码的张量$\boldsymbol{R} \in \mathbb{R}^{|k| \times|r| \times|u|}$，其中$r$索引对所有的$(n,m)$对可能的相对位置，并将其重新索引为$\boldsymbol{E} \in \mathbb{R}^{|k| \times|n| \times|m| \times|u|}$，如$\boldsymbol{E}_{n m}=\boldsymbol{R}_{r(n, m)}$。

**Lambda 卷积**： 尽管有长程信息交互的好处，局部性在许多任务中仍然是一个强烈的归纳偏置。从计算的角度来看，使用全局上下文可能会产生噪声或过度。因此，将位置交互的范围限制到查询位置$n$周围的一个局部邻域，就像局部自注意和卷积的情况一样，可能是有用的。这可以通过对所需范围之外的上下文位置$m$的位置嵌入进行归零来实现。然而，对于较大的$|m|$值，这种策略仍然代价高昂，因为计算仍然会发生(它们只是被归零)。在上下文被安排在多维网格上时，可以通过常规卷积从局部上下文中生成位置lambdas，将$\boldsymbol{V}$中的$v$维视为额外的空间维度。考虑在一维序列上的大小为$|r|$的局部域上生成位置lambdas。相对位置编码张量$\boldsymbol{R} \in \mathbb{R}^{|r| \times|u| \times|k|}$可以被reshape到$\overline{\boldsymbol{R}} \in \mathbb{R}^{|r| \times 1 \times|u| \times|k|}$，并且被用作二维卷积核来计算需要的位置lambda，算式如下。

$$
\boldsymbol{\lambda}_{b n v k}=\operatorname{conv} 2 \mathrm{d}\left(\boldsymbol{V}_{b n v u}, \overline{\boldsymbol{R}}_{r 1 u k}\right)
$$

这个操作称为lambda卷积，由于计算被限制在一个局部范围，lambda卷积相对于输入只需要线性时间和内存复杂度的消耗。lambda卷积很容易和其他功能一起使用，如dilation和striding，并且在硬件计算上享受告诉运算。计算效率和局部自注意力形成了鲜明对比，如下表。

![](https://i.loli.net/2020/11/04/bmIGXYyWcRNJ23T.png)

### **multiquery lambdas减少复杂性**

这部分作者主要对计算复杂度进行了分析，设计了多query lambda，计算复杂度对比如下。

![](https://i.loli.net/2020/11/04/41S9hjkmUFdxiXb.png)

提出的multiquery lambdas可以通过einsum高效实现。

$$
\begin{aligned}
\boldsymbol{\lambda}_{b k v}^{c}=& \operatorname{einsum}\left(\overline{\boldsymbol{K}}_{b m k u}, \boldsymbol{V}_{b m v u}\right) \\
\boldsymbol{\lambda}_{b n k v}^{p} &=\operatorname{einsum}\left(\boldsymbol{E}_{k n m u}, \boldsymbol{V}_{b m v u}\right) \\
\boldsymbol{Y}_{b n h v}^{c} &=\operatorname{einsum}\left(\boldsymbol{Q}_{b n h k}, \boldsymbol{\lambda}_{b k v}^{c}\right) \\
\boldsymbol{Y}_{b n h v}^{p} &=\operatorname{einsum}\left(\boldsymbol{Q}_{b n h k}, \boldsymbol{\lambda}_{b n k v}^{p}\right) \\
\boldsymbol{Y}_{b n h v} &=\boldsymbol{Y}_{b n h v}^{c}+\boldsymbol{Y}_{b n h v}^{p}
\end{aligned}
$$

然后，对比了lambda 层和自注意力在resnet50架构上的imagenet分类任务效果。显然，lambda层参数量是很少的，且准确率很高。

![](https://i.loli.net/2020/11/04/fehAXKUOJM6Nas2.png)

## 实验

在大尺度高分辨率计算机视觉任务上进行了充分的实验，和SOTA的EfficientNet相比，可以说无论是速度还是精度都有不小的突破。

![](https://i.loli.net/2020/11/04/mWH1QgTY4JGOfdZ.png)

其长子检测任务上，LambdaResNet也极具优势。

![](https://i.loli.net/2020/11/04/1PTjFqlRaEcmJkK.png)


## 总结

作者提出了Lambda Layer代替自注意力机制，获得了较好的改进。并借此设计了LambdaNetworks，其在各个任务上都超越了SOTA且速度提高了很多。如果实践证明，Lambda Layer的效果具有足够的鲁棒性，在以后的研究中应该会被广泛使用。


## 参考文献

[1] Anonymous. LambdaNetworks: Modeling long-range Interactions without Attention[A]. 2020.