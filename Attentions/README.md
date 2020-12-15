# 视觉注意力机制(上)

## 简介

注意力机制（Attention Mechanism）是机器学习中的一种数据处理方法，起源于自然语言处理（NLP）领域，后来在计算机视觉中广泛应用。注意力机制本质上与人类对事物的观察机制相似：一般而言，我们在观察事物的时候，首先会倾向于观察事物一些重要的局部信息（如下图所示，我们会首先将注意力集中在目标而不是背景的身上），然后再去关心一些其他部分的信息，最后组合起来形成整体印象。

注意力机制能够使得深度学习在提取目标的特征时更加具有针对性，使得相关任务的精度有所提升。注意力机制应用于深度学习通常是对输入每个部分赋予不同的权重，抽取出更加关键及重要的信息使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销，这也是注意力机制广泛使用的原因。

![](./assets/human-attention.png)

在计算机视觉中，注意力机制主要和卷积神经网络进行结合，按照传统注意力的划分，大部分属于软注意力，实现手段常常是通过掩码（mask）来生成注意力结果。掩码的原理在于通过另一层新的权重，将输入特征图中关键的特征标识出来，通过学习训练，让深度神经网络学到每一张新图片中需要关注的区域，也就形成了注意力。说的更简单一些，网络除了原本的特征图学习之外，还要学会通过特征图提取权重分布，对原本的特征图不同通道或者空间位置加权。因此，按照加权的位置或者维度不同，将注意力分为**空间域、通道域和混合域**。

## 典型方法

卷积神经网络中常用的注意力有两种，即**空间注意力和通道注意力**，当然也有融合两者的混合注意力。

首先，我们知道，卷积神经网络输出的是维度为$C\times H \times W$的特征图，其中$C$指的是通道数，它等于作用与输入的卷积核数目，每个卷积核代表提取一种特征，所以每个通道代表一种特征构成的矩阵。$H \times W$这两个维度很好理解，这是一个平面，里面的每个值代表一个位置的信息，尽管经过下采样这个位置已经不是原始输入图像的像素位置了，但是依然是一种位置信息。如果，对每个通道的所有位置的值都乘上一个权重值，那么总共需要$C$个值，构成的就是一个$C$维向量，将这个$C$维向量作用于特征图的通道维度，这就叫**通道注意力**。同样的，如果我学习一个$H\times W$的权重矩阵，这个矩阵每一个元素作用于特征图上所有通道的对应位置元素进行乘法，不就相当于对空间位置加权了吗，这就叫做**空间注意力**。

下面我列举一些常见的使用了注意力机制的卷积神经网络，我在下面一节会详细介绍它们。

1. NL(Non-local Neural Networks)
2. SENet(Squeeze-and-Excitation Networks)
3. CBAM(Convolutional Block Attention Module)
4. BAM(Bottleneck Attention Module)
5. $A^2$-Nets(Double Attention Networks)
6. GSoP-Net(Global Second-order Pooling Convolutional Networks)
7. ECA-Net(Efficient Channel Attention for Deep Convolutional Neural Networks)
8. GC-Net(Global context network for medical image segmentation)
9. CC-Net(Criss-Cross Attention for Semantic Segmentation)
10. SKNet(Selective Kernel Networks)
11. ResNeSt(ResNeSt: Split-Attention Networks)
12. Triplet Attention(Convolutional Triplet Attention Module)

## 网络详解

### **NL**

Non-local Neural Networks应该算是引入自注意力机制比较早期的工作，后来的语义分割里各种自注意力机制都可以认为是Non-local的特例，这篇文章作者中同样有熟悉的何恺明大神😂。

首先聊聊Non-local的动机，我们知道，CV和NLP任务都需要捕获长程依赖（远距离信息交互），卷积本身是一种局部算子，CNN中一般通过堆叠多层卷积层获得更大感受野来捕获这种长程依赖的，这存在一些严重的问题：效率低；深层网络的设计比较困难；较远位置的消息传递，局部操作是很困难的。所以，收到非局部均值滤波的启发，作者设计了一个泛化、简单、可直接嵌入主流网络的non-local算子，它可以捕获时间（一维时序数据）、空间（图像）和时空（视频）的长程依赖。

![](./assets/non-local-1.png)

首先，Non-local操作早在图像处理中已经存在，典型代表就是非局部均值滤波，到了深度学习时代，在计算机视觉中，这种通过关注特征图中所有位置并在嵌入空间中取其加权平均值来计算某位置处的响应的方法，就叫做**自注意力**。

然后来看看深度神经网络中的non-local操作如何定义，也就是下面这个式子，这是一个通式，其中$x$是输入，$y$是输出，$i$和$j$代表输入的某个位置，可以是序列、图像或者视频上的位置，不过因为我比较熟悉图像，所以后文的叙述都以图像为例。因此$x_i$是一个向量，维数和通道一样；$f$是一个计算任意两点的相似性函数，$g$是一个一元函数，用于信息变换；$\mathcal{C}$是归一化函数，保证变换前后整体信息不变。所以，**下面的式子，其实就是为了计算某个位置的值，需要考虑当前这个位置的值和所有位置的值的关系，然后利用这种获得的类似attention的关系对所有位置加权求和得到当前位置的值。**

$$
\mathbf{y}_{i}=\frac{1}{\mathcal{C}(\mathbf{x})} \sum_{\forall j} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) g\left(\mathbf{x}_{j}\right)
$$

那么，确定了这个通式，在图像上应用，只需要确定$f$、$g$和$\mathcal{C}$即可，首先，由于$g$的输入是一元的，可以简单将$g$设置为1x1卷积，代表线性嵌入，算式为$g\left(\mathbf{x}_{j}\right)=W_{g} \mathbf{x}_{j}$。关于$f$和$\mathcal{C}$需要配对使用，作用其实就是计算两个位置的相关性，可选的函数有很多，具体如下。

- **Gaussian**
高斯函数，两个位置矩阵乘法然后指数映射，放大差异。
$$
f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=e^{\mathbf{x}_{i}^{T} \mathbf{x}_{j}}
$$
$$
\mathcal{C}(x)=\sum_{\forall j} f\left(\mathrm{x}_{i}, \mathrm{x}_{j}\right)
$$
- **Embedded Gaussian**
嵌入空间的高斯高斯形式，$\mathcal{C}(x)$同上。
$$
\theta\left(\mathbf{x}_{i}\right)=W_{\theta} \mathbf{x}_{i} \text { and } \phi\left(\mathbf{x}_{j}\right)=W_{\phi} \mathbf{x}_{j}
$$
$$
f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=e^{\theta\left(\mathbf{x}_{i}\right)^{T} \phi\left(\mathbf{x}_{j}\right)}
$$
论文中这里还特别提了一下，如果将$\mathcal{C}(x)$考虑进去，对$\frac{1}{\mathcal{C}(\mathbf{x})} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)$而言，这其实是一个softmax计算，因此有$\mathbf{y}=\operatorname{softmax}\left(\mathbf{x}^{T} W_{\theta}^{T} W_{\phi} \mathbf{x}\right) g(\mathbf{x})$，这个其实就是NLP中常用的自注意力，因此这里说，自注意力是Non-local的特殊形式，Non-local将自注意力拓展到了图像和视频等高维数据上。



### **SENet**

SENet应该算是Non-local的同期成果

