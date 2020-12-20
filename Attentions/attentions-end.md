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
### **CCNet**
### **ResNeSt**
### **Triplet Attention**