# SENet & SKNet 解读

## 简介

今年有很多基于 ResNet 的新成果诞生，包括由于代码实现错误等问题引起广泛关注却屠榜各个榜单的 ResNeSt，关于 ResNeSt 的好坏这里不多做评论，不过它基于的前人工作 SENet 和 SKNet 其实是很值得回顾的，本文就聊聊这两个卷积神经网络历史上标杆式的作品。

- SENet: [Squeeze-and-Excitation Networks](http://arxiv.org/abs/1709.01507)（CVPR2018）
- SKNet: [Selective Kernel Networks](http://arxiv.org/abs/1903.06586)（CVPR2019）

## SENet

SENet 获得了 ImageNet2017 大赛分类任务的冠军，这也是最后一届 ImageNet 比赛，论文同时获得了 CVPR2018 的 oral。而且，SENet 思路简单，实现方便，计算量小，模块化涉及，可以无缝嵌入主流的网络结构中，实践不断证明其可以使得网络获得更好的任务效果。

### **动机和思路**

我们知道，卷积操作是卷积神经网络的核心，卷积可以理解为在一个局部感受野范围内将空间维度信息和特征维度信息进行聚合，聚合的方式是加和（sum）。然后想要提高卷积神经网络的性能其实是很难的，需要克服很多的难点。为了获得更加丰富的空间信息，很多工作被提出，如使用多尺度信息聚合的 Inception（下面左图，图来自官方分享）、考虑空间上下文的 Inside-outside Network（下面右图）以及一些注意力机制。

![](https://i.loli.net/2020/12/07/jbxznFHyrITCYlR.jpg)

那么，自然会想到，能否在通道（channel）维度上进行特征融合呢？首先，其实卷积操作默认是有隐式的通道信息融合的，它对所有通道的特征图进行融合得到输出特征图（这就默认每个通道的特征是同权的），这就是为什么一个 32 通道的输入特征图，要求输出 64 通道特征图，需要 32x64 个卷积核。这方面也不是没人进行尝试，一些轻量级网络使用分组卷积和深度可分离卷积对 channel 进行分组操作，而这本质上只是为了减少参数，并没有什么特征融合上的贡献。

SENet 则从每个通道的特征图应该不同权角度出发，更加关注 channel 之间的关系，让模型学习到不同 channel 的重要程度从而对其加权，即显式建模不同 channel 之间的关系。为此，设计了 SE（Squeeze-and-Excitation）模块，我这边译作压缩激励模块，不过后文还是采用 SE 进行阐述。

### **结构设计**

下图就是 SE 模块的结构图，对输入$X$进行一系列卷积变换$\mathbf{F}_{t r}$后得到$U$，维度也从$(C',H',W')$变为$(C,H,W)$。这里假定$\mathbf{F}_{t r}$就是一个卷积操作，并且卷积核为$\mathbf{V}=\left[\mathbf{v}_{1}, \mathbf{v}_{2}, \ldots, \mathbf{v}_{C}\right]$，$\mathbf{v}_{c}$表示第$c$个卷积核的参数，那么输出$\mathbf{U}=\left[\mathbf{u}_{1}, \mathbf{u}_{2}, \ldots, \mathbf{u}_{C}\right]$，这种表示源于下式，这个不难理解，其中$*$表示卷积运算。

$$
\mathbf{u}_{c}=\mathbf{v}_{c} * \mathbf{X}=\sum_{s=1}^{C^{\prime}} \mathbf{v}_{c}^{s} * \mathbf{x}^{s}
$$

上面的式子说明了一个什么问题呢？一个卷积核到输入 feature map 的每一个 channel 上进行了操作然后加和在一起，channel 特征和卷积核学到的空间特征混杂纠缠在了一起。这就是说，默认卷积操作建模的通道之间的关系是隐式和局部的。作者希望显示建模通道之间的依赖关系来增强卷积特征的学习。因此，论文希望提供全局信息捕获的途径并且**重标定**卷积结果，所以设计了两个操作**压缩（squeeze）**和**激励（excitation）**。

![](https://i.loli.net/2020/12/07/OSTCu8DWXP5wyZm.png)

上图就是 SE 模块的结构，我们先来看看压缩操作。卷积是在一个局部感受野范围内操作，因此$U$很难获得足够的信息来捕获全局的通道之间的关系。所以这里通过压缩全局空间信息为一个通道特征的操作，论文中采用全局平均池化（GAP）来实现这个目的，具体的$\mathbf{z} \in \mathbb{R}^{C}$通过下式计算，通过在$H\times W$维度上压缩$U$得到。

$$
z_{c}=\mathbf{F}_{s q}\left(\mathbf{u}_{c}\right)=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_{c}(i, j)
$$

激励操作需要利用压缩操作得到的全局描述来抓取 channel 之间的关系，所以激励操作必须满足两个特性：灵活，它要能捕获通道间的非线性关系；不互斥，我们希望多个通道都被增强或者多个通道被抑制而不是要得到 one-hot 那样的结果。所以采用下述的简单 gating 机制，其中$W_{1} \in R^{\frac{C}{r} \times C}, W_{2} \in R^{C \times \frac{C}{r}}$。

$$
\mathbf{s}=\mathbf{F}_{e x}(\mathbf{z}, \mathbf{W})=\sigma(g(\mathbf{z}, \mathbf{W}))=\sigma\left(\mathbf{W}_{2} \delta\left(\mathbf{W}_{1} \mathbf{z}\right)\right)
$$

为了降低模型复杂度以及提升泛化能力，这里采用两个全连接层构成 bottleneck 的结构，其中第一个全连接层起到降维的作用，降维比例$r$是个超参数，然后采用 ReLU 激活，然后第二个全连接层用于恢复特征维度最后 Sigmoid 激活输出，显然输出$\mathbf{s}$是一个$C$维的向量，它表征了各个 channel 的重要性且值在$(0,1)$之间，将这个$\mathbf{s}$逐通道乘上输入$U$即可，这相当于对每个 channel 加权，这种操作类似什么呢？Attention。

$$
\widetilde{\mathbf{x}}_{c}=\mathbf{F}_{\text {scale}}\left(\mathbf{u}_{c}, s_{c}\right)=s_{c} \mathbf{u}_{c}
$$

### **结构应用**

SE 模块可以无缝嵌入到主流的网络结构中，以 Inception 和 ResNet 为例，改造前后的结构如下图，Inception 比较直白，右边的残差网络则将 SE 模块直接嵌入残差单元中。当然，其在 ResNetXt，Inception-ResNet，MobileNet 和 ShuffleNet 等结构基础上也可以嵌入。

而且，经过分析可以得知，SE 模块算力增加并不大，在 ResNet50 上，虽然增加了约 10%的参数量，但计算量（GFLOPS）却增加不到 1%。

![](https://i.loli.net/2020/12/07/Esir4h87akFveRZ.png)

作者也在主流任务上测试了 SE 模块的适用性，基本上在各个任务上都有涨点，感兴趣的可以查看原论文，不过其能获得 ImageNet2017 的冠军已经说明了 SE 模块的强大。而且其实现也是非常简单的，基本的 SE 模块的 PyTorch 实现如下（参考[开源链接](https://github.com/moskomule/senet.pytorch)）。

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

## SKNet

SENet 设计了 SE 模块来提升模型对 channel 特征的敏感性，CVPR2019 的 SKNet 和 SENet 非常相似，它主要是为了提升模型对感受野的自适应能力，这种自适应能力类似 SENet 对各个通道做类似 attention，只不过是对不同尺度的卷积分支做了这种 attention。

### **动机和思路**

我们设计卷积神经网络的时候参考了动物的视觉机制，但是一个重要的思路并没有在设计卷积神经网络时被过多关注，那就是感受野的尺寸是根据刺激自动调节的，SKNet就从这个思路出发，自适应选择更重要的卷积核尺寸（这个实现其实就是对不同尺度的特征图分别加权，这个权重由网络学习得到，这就是我为什么说和SENet很类似的原因）。

首先，我们考虑如何获得不同尺寸的感受野信息呢，一个非常直接的想法就是使用不同size的卷积核，然后把他们融合起来就行了，这个思路诞生了下图的Inception网络，不过，Inception将所有分支的多尺度信息线性聚合到了一起，这也许是不合适的，因为不同的尺度的信息有着不同的重要程度，网络应该自己学到这种重要程度（看到这是不是觉得和SENet针对的不同通道信息应该有不同重要程度有类似之处）。

![](https://i.loli.net/2020/12/07/hFZLVTjguEdtyqO.png)

所以，这篇论文，作者设计了一种非线性聚合不同尺度的方法来实现自适应感受野，引入的这种操作称为SK卷积（Selective Kernel Convolution），它包含三个操作：Split、Fuse和Select。Split操作很简单，通过不同的卷积核size产生不同尺度的特征图；Fuse操作则通过聚合不同尺度的信息产生全局的选择权重；最后的Select操作通过这个选择权重来聚合不同的特征图。SK卷积可以嵌入主流的卷积神经网络之中且只会带来很少的计算量，其在大规模的ImageNet和小规模的CIFAR-10上都超越了SOTA方法。

### **结构设计**

在具体了解SKNet之前，必须知道不少的前置知识，比如多分支卷积神经网络（不同size的卷积核特征融合）、分组卷积及深度可分离卷积及膨胀卷积（减少运算量的高效卷积方式）、注意力机制（增强网络的重点关注能力），这些我这边就不多做解释了。

![](https://i.loli.net/2020/12/07/gZNdSo9ImtO1HjE.png)
上图就是SK卷积的一个基础实现，为了方便描述，作者只采用了两个分支，事实上可以按需增加分支，原理是一样的。可以看到，从左往右分别是三个part：Split、Fuse和Select，下面我就一步步来解释这三个操作是如何获得自适应感受野信息的，解释是完全对照上面这个图来的。

**Split**：对给定的特征图$\mathbf{X} \in \mathbb{R}^{H^{\prime} \times W^{\prime} \times C^{\prime}}$，对其采用两种卷积变换$\widetilde{\mathcal{F}}: \mathbf{X} \rightarrow \tilde{\mathbf{U}} \in \mathbb{R}^{H \times W \times C}$和$\mathbf{X} \rightarrow \widehat{\mathbf{U}} \in \mathbb{R}^{H \times W \times C}$，它们只有卷积核size不同（这里以3和5为例），其余配置一致（卷积采用深度可分离卷积，5x5卷积采用3x3卷进进行膨胀）。这一步，通过两个变换构建了两个感受野的分支，形成了两个特征图$\tilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$，它们的维度都是$H\times W \times C$。


**Fuse**：这一步也就是自适应感受野的核心，这里采用最简单的gates机制控制进入下一层的多尺度信息流。因此，这个gates需要集成来自所有分支的信息，还要有权重的集成。首先，通过逐元素相加获得特征图$\mathbf{U}$（$\mathbf{U}=\tilde{\mathbf{U}}+\widehat{\mathbf{U}}$），然后采用SENet类似的思路，通过GAP生成逐通道的统计信息$\mathbf{s} \in \mathbb{R}^{C}$，计算式如下。

$$
s_{c}=\mathcal{F}_{g p}\left(\mathbf{U}_{c}\right)=\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{U}_{c}(i, j)
$$

接着，为了更紧凑的表示，通过一个全连接层对$\mathbf{s}$进行降维，获得$\mathbf{z}=\mathcal{F}_{f c}(\mathbf{s})=\delta(\mathcal{B}(\mathbf{W} \mathbf{s}))$，这里先是和$\mathbf{W} \in \mathbb{R}^{d \times C}$相乘然后经过BN和ReLU，$d$作为一个超参使用下降比$r$来控制，不过$d$同时通过下面的式子约束下界（$L$设置为32）。接着，又一个全连接用于升维度，得到分支数个$C$维向量，论文中这里就是$a$和$b$，然后按照通道维度进行soft attention，也就是说$a_{c}+b_{c}=1$，这样可以反映不同尺度的特征的重要性，然后用$a$和$b$采用类似SE的方式对原始特征图$\tilde{\mathbf{U}}$和$\widehat{\mathbf{U}}$进行逐通道相乘加权，得到有通道区分度的特征图，再相加到一起得到输出特征图$\mathbf{V}$。**这个特征图，就是自适应不同感受野获得的特征图。**

$$
d=\max (C / r, L)
$$

### **结构应用**

将SK卷积应用到ResNeXt50中，得到SKNet50，具体配置如下图，在主流任务上都有突破。SK卷积也适用于轻量级网络，因为其不会带来多少算力增加。

![](https://i.loli.net/2020/12/07/r9qWGCHwPD5Uueb.png)

SK卷积的PyTorch实现如下（参考[开源链接](https://github.com/ResearchingDexter/SKNet_pytorch)）。

```python
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        # the part of split
        for i,conv in enumerate(self.conv):
            output.append(conv(input))
        # the part of fuse
        U=reduce(lambda x,y:x+y,output)
        s=self.global_pool(U)
        z=self.fc1(s)
        a_b=self.fc2(z)
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
        a_b=self.softmax(a_b)
        # the part of select
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b))
        V=list(map(lambda x,y:x*y,output,a_b))
        V=reduce(lambda x,y:x+y,V)
        return V
```

## 总结

SENet和SKNet分别从通道信息和感受野自适应角度出发，设计了一个新的网络结构，获得了比较有突破的成果，SKNet是SENet基础上的工作，还集成了近几年卷积神经网络的一些主流技巧，可以说集众家之长，也可以说是人工设计卷积神经网络的集大成者了，SKNet后来的很多效果更好的卷积神经网络或多或少带有NAS技术的影子，不过，自动搜索也是不可阻挡的未来趋势。