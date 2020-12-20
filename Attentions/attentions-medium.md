# 视觉注意力机制(中)

## 简介
在[上篇文章](https://zhouchen.blog.csdn.net/article/details/111302952)中，我介绍了视觉注意力机制比较早期的作品，包括Non-local、SENet、BAM和CBAM，本篇文章主要介绍一些后来的成果，包括$A^2$-Nets、GSoP-Net、GCNet和ECA-Net，它们都是对之前的注意力模型进行了一些改进，获得了更好的效果。

![](https://i.loli.net/2020/12/16/ynLluegbMsiINTj.png)

本系列包括的所有文章如下，分为上中下三篇，本文是中篇。

1. NL(Non-local Neural Networks)
2. SENet(Squeeze-and-Excitation Networks)
3. CBAM(Convolutional Block Attention Module)
4. BAM(Bottleneck Attention Module)
5. **$\mathbf{A^2}$-Nets(Double Attention Networks)**
6. **GSoP-Net(Global Second-order Pooling Convolutional Networks)**
7. **GCNet(Non-local Networks Meet Squeeze-Excitation Networks and Beyond)**
8. **ECA-Net(Efficient Channel Attention for Deep Convolutional Neural Networks)**
9. SKNet(Selective Kernel Networks)
10. CC-Net(Criss-Cross Attention for Semantic Segmentation)
11. ResNeSt(ResNeSt: Split-Attention Networks)
12. Triplet Attention(Convolutional Triplet Attention Module)

## 回顾

我们首先还是来回顾一下卷积神经网络中常用的注意力，主要有两种，即**空间注意力和通道注意力**，当然也有融合两者的混合注意力。

我们知道，卷积神经网络输出的是维度为$C\times H \times W$的特征图，其中$C$指的是通道数，它等于作用与输入的卷积核数目，每个卷积核代表提取一种特征，所以每个通道代表一种特征构成的矩阵。$H \times W$这两个维度很好理解，这是一个平面，里面的每个值代表一个位置的信息，尽管经过下采样这个位置已经不是原始输入图像的像素位置了，但是依然是一种位置信息。如果，对每个通道的所有位置的值都乘上一个权重值，那么总共需要$C$个值，构成的就是一个$C$维向量，将这个$C$维向量作用于特征图的通道维度，这就叫**通道注意力**。同样的，如果我学习一个$H\times W$的权重矩阵，这个矩阵每一个元素作用于特征图上所有通道的对应位置元素进行乘法，不就相当于对空间位置加权了吗，这就叫做**空间注意力**。

上篇文章介绍的Non-local是一种空间注意力机制，而SENet是典型的通道注意力，BAM和CBAM则是混合注意力的代表。

## 网络详解

### **$A^2$-Nets**

$A^2$-Nets，也叫AA-Nets，指的就是Double Attention Networks，可以认为是Non-local的拓展工作，作为**长程交互的信息交互捕获**方法而言，相比于Non-local，论文中的方法精度更高、参数量更少。作者文中也直言，这篇文章基于了很多注意力的共工作，包括SENet、Non-local、Transformer。

论文的核心思想是首先将整个空间的关键特征收集到一个紧凑的集合中，然后自适应地将其分布到每个位置，这样后续的卷积层即使没有很大的接收域也可以感知整个空间的特征。第一次的注意力操作有选择地从整个空间中收集关键特征，而第二次的注意力操作采用另一种注意力机制，自适应地分配关键特征，这些特征有助于补充高级任务的每个时空位置。因为这两次注意力的存在，因此称为Double Attention Networks。

下面就来看看这篇文章的方法论，不过，在此提醒，这篇文章的数学要求很高，对注意力的建模也很抽象，需要多看几次论文才能理解，不过，结构上而言，这些花里胡哨的公式都是靠1x1卷积实现的。下图就是总体的pipeline设计，这张图分为左图、右上图和右下图来看。先看左图，这就是Double Attention的整体思路，一个输入图像（或者特征图）进来，先计算一堆Global Descriptors，然后每个位置会根据自己的特征计算对每个Global Descriptor的权重（称为Attention Vectors），从而是对自己特征的补充，如图中的红框中的球，所以它对球棒等周围物体的依赖很高，对球本身的依赖就很小。Global descriptors和Attention Vectors相乘就恢复输入的维度从而得到注意力结果。Global Descriptors和输出的计算细节就是右边两图，我们下面会具体数学分析。

![](https://i.loli.net/2020/12/20/olTdiOg1P7yIQEX.png)

上面的整个过程，数学表达如下式，显然，这是个两步计算，其计算流图如下图。

$$
\mathbf{z}_{i}=\mathbf{F}_{\text {distr }}\left(\mathbf{G}_{\text {gather }}(X), \mathbf{v}_{i}\right)
$$

![](https://i.loli.net/2020/12/20/1c2LN3BzGjifoYF.png)

上图有先后两个步骤，分别是Feature Gathering和Feature Distribution。先来看Feature Gathering，这里面用了一个双线性池化，这是个啥呢，如下图所示，由双线性CNN这篇文章提出来的，$A^2-Nets$只用到了最核心的双线性池化的思路，不同于平均池化和最大池化值计算一阶统计信息，双线性池化可以捕获更复杂的二阶统计特征，而方式就是对两个特征图$A=\left[\mathbf{a}_{1}, \cdots, \mathbf{a}_{ h w}\right] \in \mathbb{R}^{m \times  h w}$和$B=\left[\mathbf{b}_{1}, \cdots, \mathbf{b}_{ h w}\right] \in \mathbb{R}^{n \times h w}$计算外积，每个特征图有$h \times w$个位置的特征，每个特征是通道数$m$和$n$维度。下面这个公式其实就是对两个特征图的同样位置的两个向量进行矩乘然后求和，这里当然得到的是一个$m \times n$维的矩阵，将其记为$G=\left[\mathbf{g}_{1}, \cdots, \mathbf{g}_{n}\right] \in \mathbb{R}^{m \times n}$。 **有人问，这个$A$和$B$是怎么来的，是通过1x1卷积对$X$变换得到的，其中$A$发生了降维，而$B$没有。**

$$
\mathbf{G}_{\text {bilinear }}(A, B)=A B^{\top}=\sum_{\forall i} \mathbf{a}_{i} \mathbf{b}_{i}^{\top}
$$

![](https://i.loli.net/2020/12/20/uwf7sS5aidrDY8b.png)

作者这里给了一个解释，如果将特征图$B$改写为$B=\left[\overline{\mathbf{b}}_{1} ; \cdots ; \overline{\mathbf{b}}_{n}\right]$，那么$\mathbf{G}$的计算式可以写成下面的形式，此时每个$b_i$都是一个$h \times w$的行向量，$A$又是$m\times hw$的，所以$g_i$等同于$A$乘上一个注意力向量$b_i$得到$m$维的向量。从这个角度来看，$G$实际上是一个图片上视觉基元的集合，每个基元$g_i$都通过$\overline{\mathbf{b}}_{i}^{\top}$加权的局部特征聚合得到。
$$
\mathbf{g}_{i}=A \overline{\mathbf{b}}_{i}^{\top}=\sum_{\forall j} \overline{\mathbf{b}}_{i j} \mathbf{a}_{j}
$$
因此，自然诞生了一个新的注意力设计想法用来聚合信息，对$\overline{\mathbf{b}}_{i}^{\top}$应用softmax保证所有元素和为1（这就是上面提到的Attention Vectors），从而$G\in \mathbb{R}^{m \times n}$的计算如下式，$n$个$g$组成的就叫做Global Descriptors。

$$
\mathbf{g}_{i}=A \operatorname{softmax}\left(\overline{\mathbf{b}}_{i}\right)^{\top}
$$

**这第一步Feature Gathering我就解释到这里，具体哪里代表哪一个张量我在之前的图上进行了标注。** 下面，来看Feature Distribution这一步，将从整幅图得到的紧凑的Global Descriptors分发给各个位置，因此，后续处理使用小卷积核也能获得全局信息。受到SENet的启发，不过它是对每个位置分发同一个通道注意力值的，作者认为应该根据该位置的值的需求去有选择地分发视觉基元，因此对一个通道上地所有空间的$hw$个元素进行softmax然后和$G$相乘，就得到那个位置的注意力结果，显然$G$是$m\times n$的，而$v$是$n \times hw$维向量，他俩按照下式的计算结果为一个$m \times hw$维度的结果，再经过1x1卷积升维和原始的$X$相加都得到注意力后的结果。

$$
\mathbf{z}_{i}=\sum_{\forall j} \mathbf{v}_{i j} \mathbf{g}_{j}=\mathbf{G}_{\text {gather }}(X) \mathbf{v}_{i}, \text { where } \sum_{\forall j} \mathbf{v}_{i j}=1
$$

至此，完成了Double Attention Block的构建，它分两步进行，总的计算式如下（$A$、$B$和$V$都是1x1卷积实现的），它也可以改写为下面第二个式子，这两个式子数学上等价，但是复杂度不同，而且后面的计算方法空间消耗很高，一个32帧28x28的视频，第二个式子需要超过2GB的内存，而第一个式子只需要约1MB，所以在一般的$(dhw)^2 > nm$的情况下，建议采用式1。

$$
\begin{aligned}
Z &=\mathbf{F}_{\text {distr }}\left(\mathbf{G}_{\text {gather }}(X), V\right) \\
&=\mathbf{G}_{\text {gather }}(X) \operatorname{softmax}\left(\rho\left(X ; W_{\rho}\right)\right) \\
&=\left[\phi\left(X ; W_{\phi}\right) \operatorname{softmax}\left(\theta\left(X ; W_{\theta}\right)\right)^{\top}\right] \operatorname{softmax}\left(\rho\left(X ; W_{\rho}\right)\right)
\end{aligned}
$$

$$
Z=\phi\left(X ; W_{\phi}\right)\left[\operatorname{softmax}\left(\theta\left(X ; W_{\theta}\right)\right)^{\top} \operatorname{softmax}\left(\rho\left(X ; W_{\rho}\right)\right)\right]
$$

这篇文章算的上很会讲故事了，把注意力解释的非常抽象、层次很高，最后竟然全靠1x1卷积实现，这就是大繁至简吗（手动狗头）？Double Attention Block的优势应该是能用更少的层数达到与更多的层数带来的接近的大感受野的效果，这适用于轻量级网络。

### **GSoP-Net**

GSoP指的是Global Second-order Pooling，它是相对于GAP这种常用在网络末端的全局一阶池化的下采样方式，不过之前的GSoP都是放在网络末端，这篇论文将其放到了网络中间层用于注意力的学习，这也在网络的早期阶段就可以获得整体图像的二阶统计信息。按照这个思路设计的GSoP模块可以以微量的参数嵌入到现有网络中，构建的GSoP-Net如下图。它紧跟在GAP后面生成紧凑的全局表示时，称为GSoP-Net1（下图右侧），或者在网络中间层通过协方差矩阵表示依赖关系然后实现通道注意力或者空间注意力，这称为GSoP-Net2（下图左侧），本文介绍主要关注GSoP-Net2。

![](https://i.loli.net/2020/12/20/VrlvxB2FWhqpscC.png)

GSoP这种池化策略，它已经被证明在在网络末端使用会带来较好的性能提升，因为二阶统计信息相比一阶线性卷积核获取到的信息更加丰富。由此，作者从SENet开始思考改进，SENet中采用GAP来获取全局位置表示，然后通过全连接或者非线性卷积层来捕获逐通道依赖。CBAM则不仅仅沿着channel维度进行GAP，它也在空间上进行注意力实现了一种类似自注意力的机制。不过，SENet和CBAM都采用一阶池化，GSoP想到将二阶池化用过来，这就诞生了这篇文章。

![](https://i.loli.net/2020/12/20/g8rVIzNuRe2Zmaw.png)

上图就是GSoP模块的结构，它类似于SE模块，采用了压缩激励两个步骤。压缩操作是为了沿着输入张量的通道维度建模二阶统计信息。首先，输入的$h'\times w' \times c'$的张量（其实就是特征图）首先通过1x1卷积降维到$h'\times w' \times c$，然后通道之间两两之间计算相关性，得到$c \times c$的协方差矩阵，这个协方差矩阵意义鲜明，第$i$行元素表明第$i$个通道和其他通道的统计层面的依赖。由于二次运算涉及到改变数据的顺序，因此对协方差矩阵执行逐行归一化，保留固有的结构信息。SENet使用GAP只获得了每个通道的均值，限制了统计建模能力。

然后，激励模块，对上面的协方差特征图进行非线性逐行卷积得到$4c$的结构信息，再用卷积调整到输入的通道数$c'$维度，和输入进行逐通道相乘，完成通道注意力。

**如果，只是对SENet进行了这样的改进，其实创新度不高，因此作者进一步进行空间注意力的推广，用来捕获逐位置的依赖。**

空间GSoP模块和上面的通道GSoP思路类似，首先特征图降维到$h' \times w' \times c$，然后下采样到$h \times w \times c$，然后计算逐位置协方差矩阵，得到$hw \times hw$，它的含义和之前通道上类似。然后同样是两个非线性卷积，获得一个$h \times w$的注意力图，再被上采样到$h' \times w' \times c'$，再在空间位置上进行注意力乘法即可实现空间注意力。

具体如何将GSoP模块嵌入到网络中其实很简单，就不多赘述了。GSoP实现了一种二阶统计信息层面的通道和空间注意力，但其本质上其实和之前的一些自注意力结构类似，捕获了一种全位置的长程交互。

### **GCNet**

GCNet全名Non-local Networks Meet Squeeze-Excitation Networks and Beyond，听名字也看得出来，是Non-loca和SENet的结合成果，它深入分析了Non-local和SENet的优缺点，结合其优势提出了GCNet。

为了捕获长程依赖，主要产生两类方法：一种是采用自注意力策略进行逐对建模；另一种是不依赖query的全局上下文建模。Non-local就是采用的自注意力策略来建模像素对之间的关系，它对每个位置学习不受位置依赖的注意力图，存在大量的资源浪费；SENet则采用全局上下文对不同通道加权来调整通道依赖，但是这种利用加权进行的特征融合无法充分利用全局上下文。

作者通过大量实验分析发现Non-local的全局上下文在不同位置几乎是相同的，这表明学习到了无位置依赖的全局上下文，其实这算是自注意力的通病了。下图是可视化不同位置的注意力图，几乎是相同的。

![](https://i.loli.net/2020/12/20/EAjp9r6MeKu12zf.png)

既然如此，作者就想干脆全局共享一个注意力图，因此Non-local修改为如下结构。

![](https://i.loli.net/2020/12/20/aUcgF19z5Mtvp2P.png)

$$
\mathbf{z}_{i}=\mathbf{x}_{i}+W_{v} \sum_{j=1}^{N_{p}} \frac{\exp \left(W_{k} \mathbf{x}_{j}\right)}{\sum_{m=1}^{N_{p}} \exp \left(W_{k} \mathbf{x}_{m}\right)} \mathbf{x}_{j}
$$

简化版Non-local的第二项是不受位置依赖的，所有位置共享这一项。因此，作者直接将全局上下文建模为所有位置特征的加权平均值，然后聚集全局上下文特征到每个位置的特征上。它抽象为下面三个阶段：
1. 全局注意力池化，通过1x1卷积核softmax获取注意力权重，然后通过注意力池化捕获上下文特征；
2. 特征转换，通过1x1卷积进行特征变换；
3. 特征聚合，采用加法将全局上下文特征聚合到每个位置的特征上。

上述步骤可以抽象为下式所述的全局上下文建模框架，里外三层计算对应上面三个步骤，SENet也有类似的三个步骤：压缩、激励和聚合，SE模块的优势就是计算量很少，非常轻量。

$$
\mathbf{z}_{i}=F\left(\mathbf{x}_{i}, \delta\left(\sum_{j=1}^{N_{p}} \alpha_{j} \mathbf{x}_{j}\right)\right)
$$

为了对非常消耗算力的简化版Non-local进一步优化，对第二步的1x1卷积替换为SENet中先降维再升维的bottleneck transform模块，从而形成了下图的GC模块。

![](https://i.loli.net/2020/12/20/Tu2p14jnwVHMhAO.png)

最后，做个简单总结，Non-local和SENet本质上还是实现了一种全局上下文建模，这对于感受野局限的卷积神经网络是有效的信息补充，GCNet实现了Non-local的全局上下文建模能力和SENet的轻量，获得了相当不错的效果。



### **ECA-Net**

上面说了这么多种注意力方法，他们的出发点虽然不尽相同，但结果都是设计了更加复杂的注意力模块以获得更好的性能，但是即使精心设计，还是不可避免带来了不少的计算量，ECA-Net则为了克服性能和复杂度互相制约不得不做出权衡的悖论，提出了一种超轻量的注意力模块（Efficient Channel Attention，ECA），最终被收录于CVPR2020。下图是其和之前比较著名的注意力模块对比的结果图，其在精度和参数量上都实现了新的突破，同等层数，ECA-Net精度超越了之前的所有注意力模型且参数量最少。

![](https://i.loli.net/2020/12/20/3FpWd6vuQNtkPXY.png)

这篇文章如果直接看最后设计出来的ECA模块，可能不会感觉多么惊艳，但是其对之前很多注意力方法做的理论分析是非常具有开创性的，这也是我本人非常喜欢这篇论文的原因。

首先，回顾一下SENet，可以参考[我之前的文章](https://zhouchen.blog.csdn.net/article/details/110826497)，在SE模块中，GAP之后的特征经过降维后进行学习，这个操作其实破坏了通道之间的信息直接交互，因为经过投影降维，信息已经变化了。为了验证这个想法，作者设计了三个SE模块的变种并进行了实验对比，结果如下表，其中SE-Var1是不经过学习直接用GAP结果加权，SE-Var2是每个通道单独学习一个权重，这就已经超越了SE模块，SE-Var3是使用一层全连接层进行学习，相当于所有通道之间进行交互，也超过了之前的思路，计算式也相应改变。

![](https://i.loli.net/2020/12/20/IfrGtYURECSbuWg.png)

那么，第一个思路来了，不需要对原始的GAP后的通道特征进行降维，那么使用单层全连接层学习是否有必要呢？我们知道，全连接其实是捕获的每个通道特征之间的全局交互，也就是每个通道特征都和其他通道的存在一个权重，这个跨通道交互前人已经证明是必要的，但是这种全局交互并没有必要，局部范围的通道交互实际上效果更好。如下图所示，每个通道直接捕获自己邻域内通道的交互信息即可，这个操作采用一维卷积就能实现，卷积核采用通道数自适应获取。

![](https://i.loli.net/2020/12/20/aHNEhJXevkRsg6i.png)

下图就是ECA模块的PyTorch实现，是非常简单高效的。

![](https://i.loli.net/2020/12/20/q8EyvPiXwJlmu7N.png)

总的来说，这篇文章的工作还是和充分的，研究了SENet以及后面的通道注意力的问题，提出了降维有害理论并设计了局部通道交互的ECA模块，实验证明，ECA-Net是切实有效的。

## 总结

本文简单介绍了计算机视觉中几种比较新的采用注意力机制的卷积神经网络，它们都是基于前人的成果进行优化，获得了相当亮眼的表现，尤其是ECA-Net，后面我会介绍视觉注意力中最新的几个成果。
