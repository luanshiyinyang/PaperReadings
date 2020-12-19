# 视觉注意力机制(中)

## 简介
在[上篇文章](https://zhouchen.blog.csdn.net/article/details/111302952)中，我介绍了视觉注意力机制比较早期的作品，包括Non-local、SENet、BAM和CBAM，本篇文章主要介绍一些后来的成果，包括$A^2$-Nets、GSoP-Net、ECA-Net和GC-Net，它们都是对之前的注意力模型进行了一些改进，获得了更好的效果。

![](https://i.loli.net/2020/12/16/ynLluegbMsiINTj.png)

本系列包括的所有文章如下，分为上中下三篇，本文是中篇。

1. NL(Non-local Neural Networks)
2. SENet(Squeeze-and-Excitation Networks)
3. CBAM(Convolutional Block Attention Module)
4. BAM(Bottleneck Attention Module)
5. $A^2$-Nets(Double Attention Networks)
6. GSoP-Net(Global Second-order Pooling Convolutional Networks)
7. ECA-Net(Efficient Channel Attention for Deep Convolutional Neural Networks)
8. GC-Net(Global context network for medical image segmentation)
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

![](./assets/AA-Nets-pipeline.png)

上面的整个过程，数学表达如下式，显然，这是个两步计算，其计算流图如下图。

$$
\mathbf{z}_{i}=\mathbf{F}_{\text {distr }}\left(\mathbf{G}_{\text {gather }}(X), \mathbf{v}_{i}\right)
$$

![](./assets/AA-Nets-block.png)

上图有先后两个步骤，分别是Feature Gathering和Feature Distribution。先来看Feature Gathering，这里面用了一个双线性池化，这是个啥呢，如下图所示，由双线性CNN这篇文章提出来的，$A^2-Nets$只用到了最核心的双线性池化的思路，不同于平均池化和最大池化值计算一阶统计信息，双线性池化可以捕获更复杂的二阶统计特征，而方式就是对两个特征图$A=\left[\mathbf{a}_{1}, \cdots, \mathbf{a}_{ h w}\right] \in \mathbb{R}^{m \times  h w}$和$B=\left[\mathbf{b}_{1}, \cdots, \mathbf{b}_{ h w}\right] \in \mathbb{R}^{n \times h w}$计算外积，每个特征图有$h \times w$个位置的特征，每个特征是通道数$m$和$n$维度。下面这个公式其实就是对两个特征图的同样位置的两个向量进行矩乘然后求和，这里当然得到的是一个$m \times n$维的矩阵，将其记为$G=\left[\mathbf{g}_{1}, \cdots, \mathbf{g}_{n}\right] \in \mathbb{R}^{m \times n}$。 **有人问，这个$A$和$B$是怎么来的，是通过1x1卷积对$X$变换得到的，其中$A$发生了降维，而$B$没有。**

$$
\mathbf{G}_{\text {bilinear }}(A, B)=A B^{\top}=\sum_{\forall i} \mathbf{a}_{i} \mathbf{b}_{i}^{\top}
$$

![](./assets/bilinear-cnn.png)

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
### **ECA-Net**
### **GC-Net**