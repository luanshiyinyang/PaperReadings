# Dynamic ReLU

> 其实一直在做论文阅读心得方面的工作，只是一直没有分享出来，这篇文章可以说是这个前沿论文解读系列的第一篇文章，希望能坚持下来。

## 简介

论文提出了动态线性修正单元（Dynamic Relu，下文简称 DY-ReLU），它能够依据输入动态调整对应分段函数，与 ReLU 及其静态变种相比，仅仅需要增加一些可以忽略不计的参数就可以带来大幅的性能提升，它可以无缝嵌入已有的主流模型中，在轻量级模型（如 MobileNetV2）上效果更加明显。

- 论文地址

  http://arxiv.org/abs/2003.10027

- 论文源码

  https://github.com/Islanna/DynamicReLU

## 介绍

ReLU 在深度学习的发展中地位举足轻重，它简单而且高效，极大地提高了深度网络的性能，被很多 CV 任务的经典网络使用。不过 ReLU 及其变种（无参数的 leaky ReLU 和有参数的 PReLU）都是静态的，也就是说他们最终的参数都是固定的。**那么自然会引发一个问题，能否根据输入的数据动态调整 ReLU 的参数呢？**

![](https://i.loli.net/2020/11/04/MYRCQ4ohxpecgnO.png)

针对上述问题，论文提出了 DY-ReLU，它是一个分段函数$f_{\boldsymbol{\theta}(\boldsymbol{x})}(\boldsymbol{x})$，其参数由超函数$\boldsymbol{\theta {(x)}}$根据$x$计算得到。如上图所示，输入$x$在进入激活函数前分成两个流分别输入$\boldsymbol{\theta {(x)}}$和$f_{\boldsymbol{\theta}(\boldsymbol{x})}(\boldsymbol{x})$，前者用于获得激活函数的参数，后者用于获得激活函数的输出值。超函数$\boldsymbol{\theta {(x)}}$能够编码输入$x$的各个维度（对卷积网络而言，这里指的就是通道，所以原文采用 c 来标记）的全局上下文信息来自适应激活函数$f_{\boldsymbol{\theta}(\boldsymbol{x})}(\boldsymbol{x})$。

该设计能够在引入极少量的参数的情况下大大增强网络的表示能力，本文对于空间和通道上不同的共享机制设计了三种 DY-ReLU，分别是 DY-ReLU-A、DY-ReLU-B 以及 DY-ReLU-C。

## Dynamic ReLU

### 定义

![](https://i.loli.net/2020/11/04/zenjQDE6FkM87hf.png)

原始的 ReLU 为$\boldsymbol{y}=\max \{\boldsymbol{x}, 0\}$，这是一个非常简单的分段函数。对于输入向量$x$的第$c$个通道的输入$x_c$，对应的激活函数可以记为$y_{c}=\max \left\{x_{c}, 0\right\}$。进而，ReLU 可以统一表示为带参分段线性函数$y_{c}=\max _{k}\left\{a_{c}^{k} x_{c}+b_{c}^{k}\right\}$，基于此提出下式动态 ReLU 来针对$\boldsymbol{x}=\left\{x_{c}\right\}$自适应$a_c^k$和$b_c^k$。

$$y_{c}=f_{\boldsymbol{\theta}(\boldsymbol{x})}\left(x_{c}\right)=\max _{1 \leq k \leq K}\left\{a_{c}^{k}(\boldsymbol{x}) x_{c}+b_{c}^{k}(\boldsymbol{x})\right\}$$

系数$\left(a_{c}^{k}, b_{c}^{k}\right)$由超函数$\boldsymbol{\theta (x)}$计算得到，具体如下，其中$K$为函数的数目，$C$为通道数目。且参数$\left(a_{c}^{k}, b_{c}^{k}\right)$不仅仅与$x_c$有关，还和$x_{j \neq c}$有关。

$\left[a_{1}^{1}, \ldots, a_{C}^{1}, \ldots, a_{1}^{K}, \ldots, a_{C}^{K}, b_{1}^{1}, \ldots, b_{C}^{1}, \ldots, b_{1}^{K}, \ldots, b_{C}^{K}\right]^{T}=\boldsymbol{\theta}(\boldsymbol{x})$

### 实现

DY-ReLU 的核心超函数$\boldsymbol{\theta {(x)}}$的实现采用 SE 模块（SENet 提出的）实现，对于维度为$C \times H \times W$的张量输入，首先通过一个全局池化层压缩空间信息，然后经过两个中间夹着一个 ReLU 的全连接层，最后一个标准化层用于标准化输出的范围在$(-1,1)$之间（采用 sigmoid
函数）。该模块最终输出$2KC$个元素，分别是$a_{1: C}^{1: K}$和$b_{1: C}^{1: K}$的残差，记为$\Delta a_{1: C}^{1: K}$和$\Delta b_{1: C}^{1: K}$，最后的输出为初始值和残差的加权和，计算式如下。

$$a_{c}^{k}(\boldsymbol{x})=\alpha^{k}+\lambda_{a} \Delta a_{c}^{k}(\boldsymbol{x}), b_{c}^{k}(\boldsymbol{x})=\beta^{k}+\lambda_{b} \Delta b_{c}^{k}(\boldsymbol{x})$$

其中，$\alpha^k$和$\beta^k$分别为$a_c^k$和$b_c^k$的初始值，$\lambda_a$和$\lambda_b$为残差范围控制标量，也就是加的权。$\alpha^k$和$\beta^k$以及$\lambda_a$、$\lambda_b$都是超参数。若$K=2$，有$\alpha^{1}=1, \alpha^{2}=\beta^{1}=\beta^{2}=0$，这就是原始 ReLU。默认的$\lambda_a$和$\lambda_b$分别为 1.0 和 0.5。

![](https://i.loli.net/2020/11/04/CRsN7KLjh4QIAHZ.png)

对于学习到不同的参数，DY-ReLU 会有不同的形式，它可以等价于 ReLU、Leaky ReLU 和 PReLU，也可以等价于 SE 模块或者 Maxout 算子，至于具体的形式依据输入而改变，是一种非常灵活的动态激活函数。

### 变种设计

主要提出三种不同的 DY-ReLU 设计，分别是 DY-ReLU-A、DY-ReLU-B 以及 DY-ReLU-C。DY-ReLU-A 空间和通道均共享，只会输出$2K$个参数，计算简单，表达能力较弱；DY-ReLU-B 仅空间上共享，输出$2KC$个参数；DY-ReLU-C 空间和通道均不共享，参数量极大。

### 实验结果

经过对比实验得出 DY-ReLU-B 更适合图像分类，DY-ReLU-C 更适合关键点检测任务，在几个典型网络上改用论文提出的 DY-ReLU，效果如下图，不难发现，在轻量级网络上突破较大。

![](https://i.loli.net/2020/11/04/OqLMQBNxGpcTdam.png)

### 源码解析

下面是 DY-ReLU-B 的 Pytorch 实现。

```python
import torch
import torch.nn as nn

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result
```

这个结构和上文我所说的 SE 模块是大体对应的，目前支持一维和二维卷积，要想使用只需要像下面这样替换激活层即可（DY-ReLU 需要指定输入通道数目和卷积类型）。

```python
import torch.nn as nn
from dyrelu import DyReluB

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.relu = DyReLUB(10, conv_type='2d')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
```

> 有空的话我会在 MobileNet 和 ResNet 上具体实验，看看实际效果是否如论文所述。

## 总结

论文提出了 DY-ReLU，能够根据输入动态地调整激活函数，与 ReLU 及其变种对比，仅需额外的少量计算即可带来大幅的性能提升，能无缝嵌入到当前的主流模型中，是一个涨点利器。本质上，DY-ReLU 就是各种 ReLU 的数学归纳和拓展，这对后来激活函数的研究有指导意义。


## 参考文献

[1] Chen Y, Dai X, Liu M, et al. Dynamic ReLU[J]. arXiv:2003.10027 [cs], 2020.
