# NFNet解读

> 针对BN做的一个工作，这段时间这样的工作还是不少的，当一个领域的技术发展得比较成熟时，我们往往就会考虑一些我们习以为常的东西的优化和改进，如之前的RepVGG、又如现在的NFNet。

## 简介

最近有不少文章介绍了NFNet，但是没怎么看到针对论文较为详细的解读，所以这边就结合论文谈谈个人的见解。NFNet（Normalizer-Free ResNets）是DeepMind提出了一种不需要Batch Normalization的基于ResNet的网络结构，其核心为一种AGC（adaptive gradient clipping technique，自适应梯度裁剪）技术。如下图所示，最小的NFNet版本达到了EfficientNet-B7的准确率，并且训练速度快了8.7倍，最大版本的模型实现了新的SOTA效果。

- 论文标题

    High-Performance Large-Scale Image Recognition Without Normalization
- 论文地址

    http://arxiv.org/abs/2102.06171
- 论文源码

    https://github.com/deepmind/deepmind-research/tree/master/nfnets


![](https://i.loli.net/2021/02/19/y2I9cK8GQAlsopH.png)

## 介绍

目前计算机视觉中很多网络网络都是基于ResNet的变种，使用Batch Normalization（下文简称BN）进行训练。BN和残差结构的组合已经被业界证明十分有效，可以很容易地训练深层网络。BN的存在可以平滑loss使得更大学习率更大batch size的训练稳定进行，BN也有一定的正则化效果。不可否认，BN是非常有效的，但是它存在问题，论文中总结了三个非常典型的缺点如下。
- **BN是带来了额外的不小的计算开销。** 计算均值等需要在内存中保存临时变量，导致了内存开销的增大，并且，在某些网络中增加了梯度评估的时间。
- **BN造成了模型训练和推理时的行为差异。** 在Pytorch中实现上时`model.train()`和`model.eval()`的差异，也就是说BN带来了需要调整的隐藏超参数。
- **BN打破了小批量样本之间的独立性。** 这就是说，其实选择哪些样本其实挺重要的。

这三个特性导致了一系列不好的后果。一方面，具体而言，研究者已经发现，使用BN的网络通常难以在不同的硬件设备之间精确复制，BN往往就是这种细微错误的原因，特别是分布式训练时。分布式训练时，如果数据并行，在不同的机器上都有BN层，那么就需要将信号发送到BN层，然后在BN层之间传递均值等统计信息，不这样的话这个批次就没有均值和方差，这使得网络可以欺骗某些损失函数，造成信息泄露等问题。另一方面，BN层对batch size是非常敏感的，bs过小BN网络就会效果很差，这是因为bs很小的时候样本很少，均值其实是噪声的近似。

![](https://i.loli.net/2021/02/19/MntigJhaZXAd9VB.png)

因此，尽管BN推动了深度学习的发展，但是从长远来看，它其实阻碍了深度网络的进步。其实如上图所示，业内已经出现了Layer Norm、Group Norm等BN的替代品，它们获得了更好的精度表现但也带来了不小的计算量。幸运的是，这些年也出现了一些具有代表性的无BN的网络，这些工作的核心思路就是通过抑制残差分支上隐藏激活层的尺度从而训练非常深的无BN的ResNet网络。最简单的实现方法就是在每个残差分支的末尾引入一个初始值为0的可学习标量，不过这种技巧的精度表现并不好。另一些研究表明，ReLU这个激活函数会带来均值偏移现象，这导致不同样本的隐藏激活值随着网络的深度增加越来越相关。此前已经有工作提出了Normalize-Free ResNets，它在初始化的时候抑制残差分支并且使用Scaled Weight Standardization来消除均值偏移现象，通过额外的正则化，这种网络在ImageNet上获得了和有BN网络相媲美的效果，但是它在大的bs时训练并不稳定并且其实距离目前的SOTA也就是EfficientNet还有一定距离。因此，这篇论文提出了解决它劣势的新方法，称为NFNet，论文的主要贡献如下。

- 提出Adaptive Gradient Clipping (AGC)模块，该方法基于逐单元梯度范数与参数范数的单位比例来裁剪梯度，实验表明，AGC允许NFNet以大的bs和强数据增强条件进行训练。
- 设计了一系列的Normalize-Free ResNets，称为NFNets，最简单版本的NFNet达到了EfficientNet-B7的精度，但训练速度是8.7倍。最优版本的NFNet在不适用额外数据的情况下实现了新的SOTA。
- 在3亿张带有标签的大型私有数据集进行预训练后，对ImageNet进行微调时，NFNet与批归一化网络相比，其验证准确率要高得多。最佳模型经过微调后可达到89.2％的top-1 accuracy。

这篇文章后面两节主要是叙述BN的效果，以及前人如何在去除了BN之后保留这些优势所做的工作，这里感兴趣的可以查看文章的第二节和第三节，我这里就直接来将这篇论文的方法论了。

## AGC（自适应梯度裁剪模块）

梯度裁剪技术常用于语言模型来稳定训练，最近的研究表明，与梯度下降相比，它允许以更大的学习率进行训练从而加速收敛。这对于条件较差的损失或大批量训练尤为重要，因为在这些设置中，最佳学习率往往会受到最大学习率的限制。因此作者假定梯度裁剪有利于NFNet的大批尺寸训练。梯度裁剪往往是对梯度的范数进行约束来实现的，对梯度向量$G=\partial L / \partial \theta$而言，$L$表示损失值，$\theta$则表示模型所有参数向量，标准的裁剪算法会在更新$\theta$之前以如下的公式裁剪梯度。

$$
G \rightarrow\left\{\begin{array}{ll}
\lambda \frac{G}{\|G\|} & \text { if }\|G\|>\lambda \\
G & \text { otherwise }
\end{array}\right.
$$

上式的$\lambda$是必须调整的超参数，根据经验，作者发现虽然这个裁剪算法能够以比以前更高的批尺寸进行训练，但训练稳定性对裁剪阈值$\lambda$的选择极为敏感，在改变模型深度、批尺寸或学习率时都需要精调阈值。

为了解决这个问题，作者引入了自适应梯度裁剪算法（AGC），下面详细叙述这个算法。记$W^{\ell} \in \mathbb{R}^{N \times M}$为第$\ell$层的权重矩阵，$G^{\ell} \in \mathbb{R}^{N \times M}$为对应于$W^{\ell}$的梯度矩阵，$\|\cdot\|_{F}$表示$F$范数，即有$\left\|W^{\ell}\right\|_{F}=\sqrt{\sum_{i}^{N} \sum_{j}^{M}\left(W_{i, j}^{\ell}\right)^{2}}$。AGC算法的动机源于观察到梯度与权重的范数比$\frac{\left\|G^{\ell}\right\|_{F}}{\left\|W^{\ell}\right\|_{F}}$，这其实是一个单次梯度下降对原始权重影响的简单度量。举个例子，如果使用无动量的梯度下降算法，有$\frac{\left\|\Delta W^{\ell}\right\|}{\left\|W^{\ell}\right\|}=h \frac{\left\|G^{\bar{\ell}}\right\|_{F}}{\left\|W^{\ell}\right\|_{F}}$，那么第$\ell$层的参数更新公式为$\Delta W^{\ell}=-h G^{\ell}$，其中$h$表示学习率。直观上，我们认为如果$\left\|\Delta W^{\ell}\right\| /\left\|W^{\ell}\right\|$很大那么训练就会变得不稳定，这就启发了一种基于$\frac{\left\|G^{\ell}\right\|_{F}}{\left\|W^{\ell}\right\|_{F}}$的梯度裁剪策略，然而实际上，逐单元的梯度范数和参数范数比会比逐层的效果好，因此定义第$\ell$层上第$i$个单元的梯度矩阵$G_{i}^{\ell}$（表示$G_{\ell}$的第$i$行）的裁剪公式如下，其中$\lambda$是一个标量超参数，定义$\left\|W_{i}\right\|_{F}^{\star}=\max \left(\left\|W_{i}\right\|_{F}, \epsilon\right)$（其中默认$\epsilon=10^{-3}$），这避免0初始化参数总是裁剪为0。对于卷积滤波器中的参数，我们在扇入范围(包括通道和空间维度)上评估逐单元范数。

$$
G_{i}^{\ell} \rightarrow\left\{\begin{array}{ll}
\lambda \frac{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}{\left\|G_{i}^{\ell}\right\|_{F}} G_{i}^{\ell} & \text { if } \frac{\left\|G_{i}^{\ell}\right\|_{F}}{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}>\lambda \\
G_{i}^{\ell} & \text { otherwise. }
\end{array}\right.
$$

使用上述的AGC模块，NFNet能够以高达4096的批尺寸训练同时使用RandAugment这样的数据增强策略，不适用AGC模块，NFNet是无法这样训练的。注意，最优裁剪参数$\lambda$可能取决于优化器的选择，学习率和批大尺寸。经验上，batch size越大，$\lambda$应该越小。

![](https://i.loli.net/2021/02/19/NfEXCLZdBQg2lOt.png)

上图是论文针对AGC做的两个消融实验，左图a表示使用BN的REsNet以及使用和不使用AGC的NFNet之间的对比，实验表明AGC使得NFNet有着媲美BN网络的效果，而且批尺寸越小，AGC收益越低。右图b则表示不同批尺寸不同$\lambda$选择的效果，结果表明，当批尺寸较大的时候，应该选择较小的$\lambda$以稳定训练。

后续作者也对AGC的作用层进行了消融实验，得到一些结论，比如对最终的线性层裁剪是不需要的等。

## NFNet

![](https://i.loli.net/2021/02/19/YJZMVeWQhfpnHok.png)

上一节提到了可以让网络以较大批尺寸和较强数据增强方法进行训练的梯度裁剪手段，同时为了配置这个AGC模块，论文也对模型结构进行了探索。EfficientNet是当前的分类SOTA网络，它基于NAS技术搜索而得，拥有很低的理论计算复杂度，但是实际硬件上的表现并不是很好，所以作者这边选择了手工探索模型空间以获得较好的表现，在SE-ResNeXt-D模型的基础上，对其先是应用了前人的Normalizer-Free配置，修改了宽度和深度模式，以及第二个空间卷积（下图表示这种配置上的修改，更具体的可以查看论文附加材料，给出的图如下）。接着，应用AGC到除了最后的线性层上的每一层，得到最终的NfNet配置。

![](https://i.loli.net/2021/02/19/URQyekCTVOMmtPh.png)

得到的NfNets如下表所示，构成一个网络系列。

![](https://i.loli.net/2021/02/19/FhokB3OlVtxATgK.png)

## 实验

下表是NFNet使用各种数据增强在ImageNet上和其他方法的对比，当之无愧的SOTA。

![](https://i.loli.net/2021/02/19/xBVNslkjWRPrILA.png)

也进行了预训练的实验，证明了Normalize-Free ResNets在迁移学习上效果也是能够很强的。

![](https://i.loli.net/2021/02/19/7zX9Dch8Fkxq1oK.png)


## 总结

关于BN这个结构我之前其实也有一些思考，虽然不否认它为深度学习做出的巨大贡献（我自己进行实验过程中深感BN结构的有效性），但是它也确实存在一些问题，DeepMind的这篇文章在保留BN优势克服BN劣势的基础上，实现了一个非常成功的Normalize-Free ResNets，是很值得关注的工作。