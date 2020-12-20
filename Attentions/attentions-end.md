# 视觉注意力机制(下)

## 简介
在[上篇文章](https://zhouchen.blog.csdn.net/article/details/111427558)中，我介绍了视觉注意力机制一些比较新的作品，包括$A^2$-Nets、GSoP-Net、GCNet和ECA-Net，本篇文章主要介绍一些近年以来最新的成果，包括SKNet、CCNet、ResNeSt和Triplet Attention。

![](https://i.loli.net/2020/12/16/ynLluegbMsiINTj.png)

本系列包括的所有文章如下，分为上中下三篇，本文是中篇。

1. NL(Non-local Neural Networks)
2. SENet(Squeeze-and-Excitation Networks)
3. CBAM(Convolutional Block Attention Module)
4. BAM(Bottleneck Attention Module)
5. $\mathbf{A^2}$-Nets(Double Attention Networks)
6. GSoP-Net(Global Second-order Pooling Convolutional Networks)
7. GCNet(Non-local Networks Meet Squeeze-Excitation Networks and Beyond)
8. ECA-Net(Efficient Channel Attention for Deep Convolutional Neural Networks)
9. **SKNet(Selective Kernel Networks)**
10. **CCNet(Criss-Cross Attention for Semantic Segmentation)**
11. **ResNeSt(ResNeSt: Split-Attention Networks)**
12. **Triplet Attention(Convolutional Triplet Attention Module)**

## 回顾

我们首先还是来回顾一下卷积神经网络中常用的注意力，主要有两种，即**空间注意力和通道注意力**，当然也有融合两者的混合注意力。

我们知道，卷积神经网络输出的是维度为$C\times H \times W$的特征图，其中$C$指的是通道数，它等于作用与输入的卷积核数目，每个卷积核代表提取一种特征，所以每个通道代表一种特征构成的矩阵。$H \times W$这两个维度很好理解，这是一个平面，里面的每个值代表一个位置的信息，尽管经过下采样这个位置已经不是原始输入图像的像素位置了，但是依然是一种位置信息。如果，对每个通道的所有位置的值都乘上一个权重值，那么总共需要$C$个值，构成的就是一个$C$维向量，将这个$C$维向量作用于特征图的通道维度，这就叫**通道注意力**。同样的，如果我学习一个$H\times W$的权重矩阵，这个矩阵每一个元素作用于特征图上所有通道的对应位置元素进行乘法，不就相当于对空间位置加权了吗，这就叫做**空间注意力**。

上篇文章介绍的GCNet是一种空间注意力机制，而ECA-Net是典型的通道注意力。

## 网络详解

### **SKNet**

SENet 设计了 SE 模块来提升模型对 channel 特征的敏感性，CVPR2019 的 SKNet 和 SENet 非常相似，它主要是为了提升模型对感受野的自适应能力，这种自适应能力类似 SENet 对各个通道做类似 attention，只不过是对不同尺度的卷积分支做了这种 attention。

![](https://i.loli.net/2020/12/20/QKsbW69f8v4djVF.png)

上图就是SK卷积的一个基础实现，为了方便描述，作者只采用了两个分支，事实上可以按需增加分支，原理是一样的。可以看到，从左往右分别是三个part：Split、Fuse和Select，下面我就一步步来解释这三个操作是如何获得自适应感受野信息的，解释是完全对照上面这个图来的。

**Split**：对给定的特征图$\mathbf{X} \in \mathbb{R}^{H^{\prime} \times W^{\prime} \times C^{\prime}}$，对其采用两种卷积变换$\widetilde{\mathcal{F}}: \mathbf{X} \rightarrow \tilde{\mathbf{U}} \in \mathbb{R}^{H \times W \times C}$和$\mathbf{X} \rightarrow \widehat{\mathbf{U}} \in \mathbb{R}^{H \times W \times C}$，它们只有卷积核size不同（这里以3和5为例），其余配置一致（卷积采用深度可分离卷积，5x5卷积采用3x3卷进进行膨胀）。这一步，通过两个变换构建了两个感受野的分支，形成了两个特征图$\tilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$，它们的维度都是$H\times W \times C$。


**Fuse**：这一步也就是自适应感受野的核心，这里采用最简单的gates机制控制进入下一层的多尺度信息流。因此，这个gates需要集成来自所有分支的信息，还要有权重的集成。首先，通过逐元素相加获得特征图$\mathbf{U}$（$\mathbf{U}=\tilde{\mathbf{U}}+\widehat{\mathbf{U}}$），然后采用SENet类似的思路，通过GAP生成逐通道的统计信息$\mathbf{s} \in \mathbb{R}^{C}$，计算式如下。

$$
s_{c}=\mathcal{F}_{g p}\left(\mathbf{U}_{c}\right)=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{U}_{c}(i, j)
$$

接着，为了更紧凑的表示，通过一个全连接层对$\mathbf{s}$进行降维，获得$\mathbf{z}=\mathcal{F}_{f c}(\mathbf{s})=\delta(\mathcal{B}(\mathbf{W} \mathbf{s}))$，这里先是和$\mathbf{W} \in \mathbb{R}^{d \times C}$相乘然后经过BN和ReLU，$d$作为一个超参使用下降比$r$来控制，不过$d$同时通过$d=\max (C / r, L)$约束下界（$L$设置为32）。接着，又一个全连接用于升维度，得到分支数个$C$维向量，论文中这里就是$a$和$b$，然后按照通道维度进行soft attention，也就是说$a_{c}+b_{c}=1$，这样可以反映不同尺度的特征的重要性，然后用$a$和$b$采用类似SE的方式对原始特征图$\tilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$进行逐通道相乘加权，得到有通道区分度的特征图，再相加到一起得到输出特征图$\mathbf{V}$。**这个特征图，就是自适应不同感受野获得的特征图。**

SKNet是尺度层面的注意力，虽然其本质还是通道注意力，但这种自适应卷积核的思路确实为卷积神经网络带来了很大的性能提升，被一些新的论文采用了进去。

### **CCNet**

CCNet，翻译应该是十字交叉注意力（Criss-Cross Attention），设计用于语义分割。这不是一篇新文章，2018年就出来了，收录于ICCV2019，后来扩充内容之后发表再TPAMI这个视觉顶级期刊上了。这篇文章主要针对的是Non-local进行改进的，Non-local主要用来捕获不同位置像素之间的长程依赖，具体解析[我之前的文章](https://zhouchen.blog.csdn.net/article/details/111302952)已经提到了，这里就不多赘述。

![](https://i.loli.net/2020/12/20/q2iPX37GsHTdpmN.png)

熟悉Non-local的，上图应该不会陌生，图a表示的就是Non-local的操作，它对输入特征图进行了两个分支处理，其中一个分支计算注意力图，一个分支对特征图进行信息变换$g$。图中的蓝色格子代表待处理像素位置$x_i$，红色格子代表结果$y_i$，绿色网格的分支代表求得的注意力图，每个格子(i,j)代表$x_i$和$x_j$之间的相关性，颜色越深，相关性越强。灰色网格的分支代表信息变换函数$g$处理，将处理后的结果和上面分支得到的注意力图相乘，就得到注意力后的结果。

图b就是论文提出的CC注意力块，可以发现论文题目的由来，它只计算了当前像素周围十字区域位置与自己的相关性，但是我们需要知道所有像素和它的相关性，所以这个结构需要堆叠，实验表明，循环堆叠两层即可覆盖所有的位置，所以作者设计了循环CC注意力模块RCCA。

在具体讲解CC模块的细节之前，我们先来比较抽象地解释一下怎么样两次循环就覆盖整个像素空间的，作者做了一个严格的数学证明，我这里直观上解释一下。最极端的情况下，$(u_x, u_y)$想要计算自己和$(\theta_x,\theta_y)$的相关性，但是第一轮循环的时候它是永远也不可能在$(\theta_x,\theta_y)$的十字路径上的，$(\theta_x,\theta_y)$的信息最多传给了$(u_x, \theta_y)$这个左上角的点和$(\theta_x, u_y)$这个右上角的点。但是，第二轮循环的时候，这两个点就在$(u_x,u_y)$的十字路径上了，且这两个点聚集了$(\theta_x,\theta_y)$的信息，因此左下角和右上角间接交互上了。其他点也类似，可以将信息通过两次循环传递给左下角的点，由此，左下角其实遍历了所有的点。

![](https://i.loli.net/2020/12/20/wg8aB3AMUbcuTtS.png)

CCA模块的设计使得Non-local的计算量从$(H * W)^{2}$减少为$(H * W) *(H+W-1)$，计算效率相比于Non-local到15%，内存占用仅为1/11。下图就是该模块的具体实现，首先，输入特征图$\mathbf{H} \in \mathbb{R}^{C \times W \times H}$经过两个1x1卷积得到通道降维的特征图$\{\mathbf{Q}, \mathbf{K}\} \in \mathbb{R}^{C^{\prime} \times W \times H}$。

![](https://i.loli.net/2020/12/20/J2L1hXmABFO9Wjg.png)

现在，按照设计我们需要得到一个注意力图$\mathbf{A} \in \mathbb{R}^{(H+W-1) \times(W \times H)}$（每个位置，都有$H+W-1$个位置要计算相关性，原始的Non-local中，这里的特征图维度为$(H * W)^{2}$），所以设计了一个下式表示的亲和度操作来获取注意力图。在$Q$的空间维度上，每个位置$\mathbf{u}$都有一个特征向量$\mathrm{Q}_{\mathrm{u}} \in \mathbb{R}^{C^{\prime}}$，同时，在$\mathbf{K}$上选择和$\mathbf{u}$同行同列的位置的特征向量，它们组成集合$\Omega_{\mathrm{u}} \in \mathbb{R}^{(H+W-1) \times C^{\prime}}$，$\boldsymbol{\Omega}_{i, \mathbf{u}} \in \mathbb{R}^{C^{\prime}}$是$\Omega_{\mathrm{u}}$的第$i$个元素。

$$
d_{i, \mathbf{u}}=\mathbf{Q}_{\mathbf{u}} \boldsymbol{\Omega}_{i, \mathbf{u}}^{\top}
$$

上式所示的亲和度操作中，$d_{i, \mathrm{u}} \in \mathbf{D}$表示$\mathrm{Q}_{\mathrm{u}}$和$\boldsymbol{\Omega}_{i, \mathbf{u}}, i=[1, \ldots, H+W-1]$之间的相关性程度，其中$\mathbf{D} \in \mathbb{R}^{(H+W-1) \times(W \times H)}$。此时，对$\mathbf{D}$沿着通道维度进行softmax就得到了注意力图$\mathbf{A}$。

接着，对$\mathbf{V}$进行类似的操作，在$\mathbf{V}$的每个位置$\mathbf{u}$都可以得到$\mathbf{V}_{\mathbf{u}} \in \mathbb{R}^{C}$和$\boldsymbol{\Phi}_{\mathbf{u}} \in \mathbb{R}^{(H+W-1) \times C}$，这个$\boldsymbol{\Phi}_{\mathbf{u}}$同样是$\mathbf{V}$中和$\mathbf{u}$同行同列的所有位置的特征向量集合。

至此，上下文信息通过下面的式子聚合，得到一个CCA模块的输出，重复堆叠两次即可。

$$
\mathbf{H}_{\mathbf{u}}^{\prime}=\sum_{i=0}^{H+W-1} \mathbf{A}_{i, \mathbf{u}} \boldsymbol{\Phi}_{\mathbf{i}, \mathbf{u}}+\mathbf{H}_{\mathbf{u}}
$$

此外，作者还设计了三维的CCA模块，这里我就不多赘述了。总之，CCNet通过独有的十字交叉注意力对Non-local进行了改进，并获得了更好的效果，在当时，是很有突破的成果。

### **ResNeSt**

ResNeSt，又叫Split-Attention Networks（分割注意力网络），是今年争议比较多的一篇文章，文章的质量其实还是不错的。这篇文章是基于ResNeXt的一个工作，在ResNeXt中，其对深度和宽度外增加了一个维度---基数（cardinality），对同一个特征图采用不同的卷积核进行卷积，最后将结果融合。

![](https://i.loli.net/2020/12/20/CoMX7FKWxEHDVaJ.png)

上图是SE模块、SK模块以及文章提出的ResNeSt的模块设计图，其将输入特征图切分为多路（共$k$路），每路的拓扑结构一致，又将每一路划分为$r$个组，特征图再次被切分，每组的拓扑结构也是一样的。

在每一路中，每个组会对自己切分得到的通道数为$c'/k/r$的特征图进行1x1卷积核3x3卷积，再将这一路的$r$个结果送入一个分割注意力模块（Split Attention，SA）中，每个路将自己注意力后的结果concat到一起，再经过1x1卷积恢复维度后和原始输入相加。

现在，这个Split Attention如何实现的呢，如下图所示，$r$个输入逐元素求和后通过全局平均池化得到$r$个$c$维度的向量，经过全连接层学习到通道注意力向量，再和各自的输入做乘法进行通道注意力，$r$个通道注意力的结果，最后相加。

![](https://i.loli.net/2020/12/20/HUnzjN3yQvmOBA4.png)

整体来看，每一路的处理都是一个SK卷积，所以ResNeSt其实是对ResNeXt、SENet和SKNet的结构的融合，但是融合并用好本身也是挺不错的工作了。

### **Triplet Attention**

最后，聊聊最近的一个注意力成果，Triplet Attention。这篇文章的目标是研究如何在不涉及任何维数降低的情况下建立廉价但有效的通道注意力模型。Triplet Attention不像CBAM和SENet需要一定数量的可学习参数来建立通道间的依赖关系，它提出了一个几乎无参数的注意机制来建模通道注意力和空间注意力。

![](https://i.loli.net/2020/12/20/XPpYw1E75eGDUuj.png)

如上图，顾名思义，Triplet Attention由3个平行的分支组成，其中两个负责捕获通道C和空间H或W之间的跨维交互。最后一个分支类似于CBAM，用于构建空间注意力。最终3个分支的输出通过求和平均进行聚合。

首先，为了解决此前注意力方法通道注意力和空间注意力分离的问题，论文提出跨维交互这个概念，下图是Triplet Attention的概念图，通过旋转输入张量实现不同维度之间的交互捕获。

![](https://i.loli.net/2020/12/20/FEV5ohUAvuasSzR.png)

为了实现一个维度上关键信息的捕获，Z-pool通过取最大值和平均值并将其连接的方式，将任意维度特征调整为2维，这样既能保留丰富的表示又能降维减少计算量。**这里之所以采用这个Z-pool降维而不是1x1卷积就是为了减少计算量。**

$$
Z \text { -pool }(\chi)=\left[\operatorname{MaxPool}_{0 d}(\chi), \text { AvgPool }_{0 d}(\chi)\right]
$$

给定输入张量$\chi \in R^{C \times H \times W}$，它被分配给三个分支进行处理。

首先，看Triplet Attention的第一个分支，其目的是对$H$维度和$C$维度之间建立交互。输入张量$\chi$先沿着$H$轴进行旋转90度，得到$\hat{\chi}_{1} \in R^{W \times H \times C}$。这个$\hat{\chi}$先是通过Z-pool降维到$(2 \times H \times C)$然后经过卷积和BN层，得到$(1 \times H \times C)$的特征图再经过Sigmoid输出注意力图，这个注意力图和$\hat{\chi}$相乘，得到$H$和$C$的交互注意力结果。然后，再反转90度回去，得到和输入维度匹配的注意力图。

![](https://i.loli.net/2020/12/20/cAfUlkRnIKqGzpw.png)

同样的，第二个分支类似，只是旋转方式的不同，因而得到的是$C$和$W$的交互注意力结果。**其实，这个文章的创新之处就在这个旋转操作上，这个所谓的跨维度交互就是旋转后最后两个维度上的空间注意力而已。**

![](https://i.loli.net/2020/12/20/PEOUoCKc2Srf4sD.png)

最后，第三个分支其实一模一样，只是没有旋转，那自然捕获的就是原本的$H$和$W$的空间注意力了。然后，简单的求和平均就得到了最终的特征图了。

![](https://i.loli.net/2020/12/20/1aYmjIls83nMvpJ.png)

上面说的这些，也就是Triplet Attention的全部了。它抓住了张量中各个维度特征的重要性并使用了一种有效的注意力计算方法，不存在任何信息瓶颈。实验证明，Triplet Attention提高了ResNet和MobileNet等标准神经网络架构在图像分类和目标检测等任务性能，而只引入了最小的计算开销。是一个非常不错的即插即用的注意力模块。

## 总结

本文简单介绍了计算机视觉中几种最新的采用注意力机制的卷积神经网络，它们都是基于前人的成果进行优化，获得了相当亮眼的表现，值得借鉴。