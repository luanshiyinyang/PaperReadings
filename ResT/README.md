# ResT解读

> 最近的一篇基于Transformer的工作，由南京大学的研究者提出一种高效的视觉Transformer结构，设计思想类似ResNet，称为ResT，这是我个人觉得值得关注的一篇工作。

## 简介

ResT是一个高效的多尺度视觉Transformer结构，可以作为图像识别的通用骨干网络，它采用类似ResNet的设计思想，分阶段捕获不同尺度的信息。不同于现有的Transformer方法只使用标准的Transformer block来处理具有固定分辨率的原始图像，ResT有着几个优势：提出一种内存高效的多头自注意力，使用深度卷积进行内存压缩，并且跨注意力头的维度投影交互同时保持多头的多样性能力；将位置编码构建为空间注意力，它可以以更加灵活的方式处理任意尺寸的输入而无需插值或者微调；不同于直接在每个阶段开始进行序列化，而是将patch embedding设计为一系列重叠的有stride的卷积操作。作者在图像分类以及下游任务中验证了ResT的性能，结果表明，ResT大幅度优于当前SOTA骨干网络，在ImageNet数据集上，同等计算量前提下，所提方法取得了优于PVT、Swin。

- 论文标题

    ResT: An Efficient Transformer for Visual Recognition
- 论文地址

    https://arxiv.org/abs/2105.13677
- 论文源码

    https://github.com/wofmanaf/ResT

## 介绍

用于提取图像特征的骨干网络（backbone）在计算机视觉任务中至关重要，好的特征有利于下游任务的展开，如图像分类、目标检测、实例分割等。如今，计算机视觉中主要有两种骨干网络结构，一种是**卷积神经网络结构**，一种是**Transformer结构**，它们都是堆叠多个块（block）来捕获特征信息的。

CNN block通常是一个bottleneck结构，可以定义为堆叠的1x1卷积、3x3卷积和1x1卷积配合一个残差连接，如下图的(a)所示。两个1x1卷积分别用于通道降维和通道升维，保证3x3卷积处理的特征图通道数不会太高。CNN骨干网络通常更快一些，这主要得益于参数共享、局部信息聚合以及维度缩减，然而，受限于有限且固定的感受野，卷积网络在那些需要长程依赖的场景中效果并不好，比如实例分割中，从一个更大的邻域中收集并关联目标间的关系是很重要的。

![](https://i.loli.net/2021/06/15/ZA3VkJpXv2yH5ma.png)

为了克服这些限制，能够捕获长程信息的Transformer结构最近被探索用于设计骨干网络。不同于CNN网络，Transformer网络首先是将图片切分为一系列块（patch，也叫token），然后将这些token和位置编码相加来表示粗糙的空间信息，最终采用堆叠的Transformer block来捕获特征信息。一个标准的Transformer block由一个多头自注意力（multi-head self-attention，MSA）和一个前馈神经网络（feed-forward network，FFN）构成，其中MSA通过query-key-value分解来建模token之间的全局依赖，FFN则用来学习更宽泛的表示。Transformer block的结构如上图的(b)所示，它能够根据图像内容自适应调整感受野。

虽然相比于CNN backbone。Transformer backbone潜力巨大，但它依然有四个主要的缺点如下。

1. 由于现有的Transformer backbone直接对原始输入图像中的块进行序列化，因此很难提取形成图像中一些基本结构（例如，角和边缘）的低级特征。
2. Transformer block中MSA的内存和计算与空间或通道维度成二次方扩展，导致大量的训练和推理开销。
3. MSA 中的每个head只负责输入token的一个子集，这可能会损害网络的性能，特别是当每个子集中的通道维度太低时，使得query和key的点积无法构成信息函数。
4. 现有 Transformer backbone中的输入token和位置编码都是固定规模的，不适合需要密集预测的视觉任务。

在这篇论文中，作者提出一种高效的通用backbone ResT（以ResNet命名），该结构可以解决上述的问题，这个结构会在下一节具体说明。

## ResT

![](https://i.loli.net/2021/06/15/VA5Wid3fX8LYaue.png)

上图所示的即为ResT的结构图，可以看到，它和ResNet有着非常类似的pipeline，即采用一个stem模块来提取底层特征，然后跟着四个stage捕获多尺度特征。每个stage由三个组件构成，一个**patch embedding**模块，一个**position encoding**模块以及L个**efficient Transformer block**。具体而言，在每个stage的开始，patch embedding模块用来减少输入token的分辨率并且拓展通道数。位置编码模块则被融合进来用于抑制位置信息并且加强patch embedding的特征提取能力。这两个阶段完成之后，输入token被送入efficient Transformer block。

### Rethinking of Transformer Block

标准的Transformer block包含两个子层，分别是MSA和FFN，每个子层包围着一个残差连接。在MSA和FFN前，先经过了一个layer normalization（下面简称LN）。假定输入token为$\mathrm{x} \in \mathbb{R}^{n \times d_{m}}$，这里的$n$和$d_m$分别表示空间维度和通道维度，每个Transformer block的输出表示如下。

$$
\mathrm{y}=\mathrm{x}^{\prime}+\mathrm{FFN}\left(\mathrm{LN}\left(\mathrm{x}^{\prime}\right)\right), \text { and } \mathrm{x}^{\prime}=\mathrm{x}+\mathrm{MSA}(\mathrm{LN}(\mathrm{x}))
$$

对上面的式子，我们先来看**MSA**，它首先通过三组线性投影获取query $\mathbf{Q}$、key $\mathbf{K}$和value $\mathbf{V}$，每组投影有$k$个线性层（即heads）将$d_m$映射到$d_k$的空间中，这里$d_{k}=d_{m} / k$。为了描述方便，后续所有的说明都是基于$k=1$，因此MSA可以简化为单头注意力（SA），token序列之间的全局关系可以定义为下式，每个head的输出concatenate到一起之后经过线性投影得到最终输出。可以得知，MSA的计算复杂度为$\mathcal{O}\left(2 d_{m} n^{2}+4 d_{m}^{2} n\right)$，它根据输入token的空间维度或者通道维度次方级变化。

$$
\mathrm{SA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Softmax}\left(\frac{\mathbf{Q K}^{\mathrm{T}}}{\sqrt{d_{k}}}\right) \mathbf{V}
$$

接着，来看**FFN**，它主要用于特征转换和非线性，通常由两个线性层和一个非线性激活函数构成，第一层将输入的通道数从$d_m$拓展到$d_f$，第二层则从$d_f$降到$d_m$。数学上表示如下式，其中$\mathbf{W}_{1} \in \mathbb{R}^{d_{m} \times d_{f}}$且$\mathbf{W}_{2} \in \mathbb{R}^{d_{f} \times d_{m}}$为两个线性层的权重，$\mathbf{b}_{1} \in \mathbb{R}^{d_{f}}$和$\mathbf{b}_{2} \in \mathbb{R}^{d_{m}}$则是相应的偏置项，$\sigma(\cdot)$表示GELU激活函数。标准的Transformer block中，通道数通常4倍扩大，即$d_{f}=4 d_{m}$。FFN的计算代价为$8 n d_{m}^{2}$。

$$
\mathrm{FFN}(\mathrm{x})=\sigma\left(\mathrm{x} \mathbf{W}_{1}+\mathbf{b}_{1}\right) \mathbf{W}_{2}+\mathbf{b}_{2}
$$

### Efficient Transformer Block

如上面所述，MSA有两个缺点，第一是其计算量是二次方倍的，这给训练和推理都带来了不小的负担；第二，MSA中的每个head只负责输入token序列的一个子集，当通道数比较少的时候这个会损害模型的表现。

![](https://i.loli.net/2021/06/15/bBcu1lzQHLgyq3U.png)

为了解决这些问题，作者提出了一种高效的多头自注意力模块，如上图所示。和MSA类似，EMSA首先采用一组投影获取query $\mathbf{Q}$。为了压缩内存，2D输入的token $\mathrm{x} \in \mathbb{R}^{n \times d_{m}}$会被沿着空间维度reshape为3D形式（$\hat{\mathrm{x}} \in \mathbb{R}^{d_{m} \times h \times w}$）然后送入深度可分离卷积中按照因子$s$降低宽高，为了简单，$s$根据$k$自适应为$s=8 / k$，卷积核尺寸、stride和padding分别是$s+1$、$s$和$s/2$。然后，下采样后的token map为$\hat{\mathrm{x}} \in \mathbb{R}^{d_{m} \times h / s \times w / s}$，它被reshape为2D的形式，也就是$\hat{\mathbf{x}} \in \mathbb{R}^{n^{\prime} \times d_{m}}, n^{\prime}=h / s \times w / s$，然后$\hat{x}$送入两组投影层获得key $\mathbf{K}$和value $\mathbf{V}$。再然后，采用下面的式子计算qkv之间的注意力函数，式子中的Conv表示标准的1x1卷积，它用于建模不同head之间的交互，通过这个方法attention的结果依赖于所有的key和query，然而，这将削弱 MSA 联合处理来自不同位置的不同表示子集的信息的能力。为了重建这种多样性能力，在点击矩阵后添加了一个LN，也就是Softmax之后。

$$
\operatorname{EMSA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{IN}\left(\operatorname{Softmax}\left(\operatorname{Conv}\left(\frac{\mathbf{Q} \mathbf{K}^{\mathrm{T}}}{\sqrt{d_{k}}}\right)\right)\right) \mathbf{V}
$$

最后，所有head的输出concatenate到一起经过投影得到最终输出。这就是整个EMSA块的计算过程，其实对照上图就能理解得很明白了，EMSA的计算代价为$\mathcal{O}\left(\frac{2 d_{m} n^{2}}{s^{2}}+2 d_{m}^{2} n\left(1+\frac{1}{s^{2}}\right)+d_{m} n \frac{(s+1)^{2}}{s^{2}}+\frac{k^{2} n^{2}}{s^{2}}\right)$，假定$s=1$的话这个复杂度是远远低于原始的MSA的，特别是较浅的stage时，$n$相对高一些。

当然，EMSA之后也添加了FFN以进行特征变换和非线性，因此最终effcient Transformer block的输出如下。

$$
\mathrm{y}=\mathrm{x}^{\prime}+\mathrm{FFN}\left(\mathrm{LN}\left(\mathrm{x}^{\prime}\right)\right), \text { and } \mathrm{x}^{\prime}=\mathrm{x}+\operatorname{EMSA}(\mathrm{LN}(\mathrm{x}))
$$

### Patch Embedding

知道了最核心的EMSA，接下来是关于Patch Embedding的内容。在标准的Transformer中，一个token序列的embedding作为输入，以ViT为例，3D图像$\mathrm{x} \in \mathbb{R}^{3 \times h \times w}$为输入，它被按照patch size为$p \times p$进行切分。这些patch被展平为2D然后被映射为隐嵌入$\mathrm{x} \in \mathbb{R}^{n \times c}$（其中$n=h w / p^{2}$）。然而，这种直接的标记化难以捕获底层特征信息(比如边缘、角点)。此外，ViT中的token序列长度是固定的，这使其难以进行下游任务(比如目标检测、实例分割)适配，因为这些任务往往需要多尺度特征图。

为了解决上述问题，作者构建了一种高效的多尺度backbone，名为ResT以进行密集预测任务。如上文所述，每个阶段的efficient Transformer block在一个确定的尺度和分辨率上跨空间和通道进行操作，因此，patch embedding模块需要减少空间分辨率的同时拓展通道维度。

和ResNet类似，stem模块（也就是第一个patch embedding模块）以4的缩减因子缩小高度和宽度。为了是圆通很少的参数高效捕获底层特征，作者引入了一个简单但有效的方式，即堆叠3个3x3卷积层（padding为1），stride分别为2、1、2，前两层紧跟一个BN和ReLU层。在stage2、stage3和stage4，patch embedding模块被用来4倍下采样空间维度并且2倍通道维度。这可以通过标准的3x3卷积以stride2和padding1实现。在stage2中，patch embedding模块将输入分辨率从$h / 4 \times w / 4 \times c$调整到$h / 8 \times w / 8 \times 2 c$。

### Position Encoding

位置编码对于序列顺序非常关键，在ViT中，一系列可学习参数被加（加法）到输入token上编码位置信息，假定$\mathrm{x} \in \mathbb{R}^{n \times c}$为输入，$\theta \in \mathbb{R}^{n \times c}$为位置参数，编码后的输入可以表示如下。

$\hat{\mathrm{x}}=\mathrm{x}+\theta$

但是，位置的长度需要和输入token的长度一致，这就限制了很多应用的场景，因此需要一种可以根据输入改变长度的位置编码。回顾上面的式子，其实相加操作非常类似逐像素对输入加权。假定$\theta$和$x$相关，即$\theta=\mathrm{GL}(\mathrm{x})$，这里的$\mathrm{GL}(\cdot)$表示组线性操作且组数为$c$，上式就被修改为下面的式子，$\theta$可以通过更灵活的注意力机制获得。

$$
\hat{\mathrm{x}}=\mathrm{x}+\mathrm{GL}(\mathrm{x})
$$

因此，作者这里提出了一种简单高效的像素级注意力（pixel-wise attention，PA）来编码位置。具体而言，PA采用3x3深度卷积操作来获得像素级权重，然后使用sigmoid激活，最终使用PA获得的位置编码如下式。

$$
\hat{\mathrm{x}}=\mathrm{PA}(\mathrm{x})=\mathrm{x} * \sigma(\operatorname{DWConv}(\mathrm{x}))
$$

由于每个stage的输入token通过卷积得到，可以将位置编码嵌入到patch embedding模块中，整体结果见下图。注：这里的PA可以采用任意空间注意力替换，这使得ResT中的PE极为灵活。

![](https://i.loli.net/2021/06/15/fNeH2Ll9ukXYGVg.png)

### Classification Head

分类head的设计非常简单，一个池化接线性层即可，在图像分类任务上的模型结构如下图所示。

![](https://i.loli.net/2021/06/15/Z89vQhcEMIb4qYx.png)

## 实验

图像分类、目标检测、实例分割的结果如下，超越了PVT、Swin等。

![](https://i.loli.net/2021/06/15/SMkGfRzt2pWBD5r.png)

![](https://i.loli.net/2021/06/15/K29hxrVRs4nvz5E.png)

![](https://i.loli.net/2021/06/15/WAGt3RMYmInkUbL.png)

此外作者还对各个模块进行了消融实验，具体可以查看论文。

## 总结

这篇文章作者提出了一种新的Transformer架构的视觉backbone，它可以捕获多尺度特征因而非常适用于密集预测任务。作者压缩了标准 MSA 的内存，并在保持多样性能力的同时对多头之间的交互进行建模。 为了处理任意输入图像，作者进一步将位置编码重新设计为空间注意力。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。



