# RelationTrack解读

> MOT领域的一个新的SOTA方法，在FairMOT的基础上提出了特征图解耦和全局信息下的ReID Embedding的学习，前者和CSTrack思路类似，后者则采用了Deformable DETR里的deformable attention配合Transformer Encoder来捕获和目标相关的全图信息增强ReID的表示能力。

## 简介

现有的多目标跟踪方法为了速度通常会将检测和ReID任务统一为一个网络来完成，然而这两个任务需要的是不同的特征，这也是之前很多方法提到的任务冲突问题。为了缓解这个问题，论文作者设计了Global Context Disentangling（GCD）模块来对骨干网络提取到的特征解耦为任务指定的特征。此外，作者还发现，此前的方法在使用ReID特征为主的关联中，只考虑了检测框的局部信息而忽视了全局语义相关性的考虑。对此，作者设计了Guided Transformer Encoder（GTE）模块来学习更好的全局感知的ReID特征，这个模块不是密集相关性的而是捕获query节点和少量的自适应关键样本位置之间的相关性信息。因此非常高效。实验表明，由GCD和GTE构成的跟踪框架RelationTrack在MOT16和MOT17上均达到SOTA表现，在MOT20上更是超过此前的所有方法。

- 论文标题

    RelationTrack: Relation-aware Multiple Object Tracking with Decoupled Representation
- 论文地址

    http://arxiv.org/abs/2105.04322
- 论文源码

    暂未开源

## 介绍

目前主流的多目标跟踪方法主要包含两个子模型，即用于目标定位的检测模型和用于轨迹连接的ReID模型。分开训练检测和ReID两个模型可以在精度上获得较好的表现，然而推理速度较慢，很难达到实时跟踪的效果。一个较好的解决方案就是JDE首先提出的在一个网络中联合训练检测和ReID的思路。

![](https://i.loli.net/2021/05/12/xO56sENBFH3LZIa.png)

遗憾的是，直接将这两个任务放到一个网络中联合优化造成了精度大幅度的下降，这是因为这两个任务存在严重的优化矛盾。对检测分支而言，它期望同类目标之间的相似度尽量高，也就是网络能够最大化不同类目标之间的距离；但是，对ReID分支而言，它则希望最大化不同实例之间的距离（对行人跟踪而言这些不同实例是同类别的）。它们不一致的优化目标阻碍了当前的MOT框架向更高效的形式发展。

![](https://i.loli.net/2021/05/12/f9KUNvM8FRJLeBV.png)

为了缓解这个矛盾，作者设计了一个特征解耦模块称为**Global Context Disentangling** (GCD)，它将特征图解耦为检测任务指定和ReID任务指定的特征表示，如上图所示。这个模块的设计下文再阐述，不过经过实验验证，这个模块带来了1-2个点的收益，**可见它对解决任务冲突是很有效的**。

此外，作者发现，此前的方法通常利用局部信息来跟踪目标，然而，实际上，目标和周围目标以及背景之间的关系对于跟踪任务是非常重要的。为了捕获这种长程依赖，使用全局注意力是一个解决方案，但是全局注意力需要逐像素之间计算相关性以构成注意力图，这在计算上是代价极为昂贵的，也严重阻碍了实时MOT任务的进行。作者又发现，其实并不是所有的像素都对query node（查询节点）有影响的，因此只需要考虑和少数关键样本之间的关系可能是更好的选择，基于这个假设，作者使用deformable attention（源于Defomable DETR）来捕获上下文关系。相比于全局注意力，deformable attention是非常轻量的，计算复杂度从$O\left(n^{2}\right)$降低到了$O\left(n\right)$。而且，相比于基于图的受限邻域信息收集，deformable attention可以自适应选择整个图像上合适的关键样本来计算相关性。

接着，考虑到Transformer强大的建模能力，作者将deformable attention和Transformer Encoder进行了组合形成了Guided Transformer Encoder (GTE)模块，它使得MOT任务可以在全局感受野范围内捕获逐像素的相关性。

为了证明RelationTrack的效果，作者在MOT多个benchmark数据集上进行了实验，在IDF1上超越了此前的SOTA方法FairMOT3个点（MOT16）和2.4个点（MOT17）。

## RelationTrack

### 问题描述

RelationTrack旨在完成目标检测和基于ReID的轨迹关联任务，由三个部分组成，分别是检测器$\phi(\cdot)$、ReID特征提取器$\psi(\cdot)$以及关联器$\varphi(\cdot)$，它们分别负责目标的定位、目标的特征提取以及轨迹的生成。

形式上，输入图像$I_{t} \in \mathbb{R}^{H \times W \times C}$，不妨记$\phi\left(I_{t}\right)$和$\psi\left(I_{t}\right)$为$b_{t}$和 $e_{t}$，显然可以知道$b_{t} \in \mathbb{R}^{k \times 4}$且$e_{t} \in \mathbb{R}^{k \times D}$。上面的$H$、$W$和$C$分别表示输入图像的高、宽和通道数，$k$、$t$和$D$则表示检测到的目标数、图像的帧索引以及ReID embedding向量的维度。$b_t$和$e_t$分别指的是目标的边框坐标和相应的ReID特征向量。在完成检测和特征向量的提取之后，$\varphi(\cdot)$基于$e_t$对不同帧的$b_t$进行关联从而生成轨迹，目前的主流思路是只要检测和ReID足够准，使用一个简单的关联器即可，如匈牙利算法。

### 整体框架

下面的这个图就是RelationTrack的整体框架，总的来看分为五部分，分别是特征提取、特征解耦、ReID表示学习以及最后的数据关联，整个框架都是针对单帧进行处理的。首先，图像送入backbone中得到特征图，随后GCD模块将这个特征图解耦为两个特征图，分别为检测信息和ReID信息，检测分支通过检测信息进行类似于CenterNet的目标定位而GTE模块则负责生成判别性的目标特征表示。最后，有了目标框和对应的特征表示，就可以通过匈牙利算法进行关联从而得到最终的跟踪轨迹了。

![](https://i.loli.net/2021/05/12/ZEXjyvPrMwHVSuG.png)

### GCD

在上面的整体框架了解后，我们来看看GCD（Global Context Disentangling）模块具体是如何实现特征图解耦的。实际上，GCD分为两个阶段进行，**首先是全局上下文向量的生成然后利用这个向量去解耦输入特征图**，它的流程其实就是上图的中间部分。

记$x=\left\{x_{i}\right\}_{i=1}^{N_{p}}$为输入的特征图（就是backbone得到的，一般是64通道的），这里的$N_p$表示像素数目即$H^{\prime} \times W^{\prime}$，$H^{\prime}$和$W^{\prime}$分别表示特征图的高和宽。首先，第一个阶段先是计算全局上下文特征向量，可以用下面的式子表示，这里的$W_k$表示的是一个可学习的线性变换，文中采用1x1卷积实现。**这个过程其实是一个空间注意力图的形成。**

$$
z=\sum_{j=1}^{N_{p}} \frac{\exp \left(W_{k} x_{j}\right)}{\sum_{m=1}^{N_{p}} \exp \left(W_{k} x_{m}\right)} x_{j}
$$

**不过，作者这里似乎没有刻意提及利用这个注意力图更新原始输入在进行通道注意力得到一个特征向量，这个向量才是后续两个转换层的输入。（个人理解）**

接着，进入第二个阶段，两个转换层（对应上图中间部分的上下对称的两个结构，由卷积、LayerNorm、ReLU和卷积构成），它将上一个阶段的输出$z$结构成两个任务指定的特征向量，将这个向量和原始的特征图broadcast之后相加则可以获得检测任务特定的embedding $d=\left\{d_{i}\right\}_{i=1}^{N_{p}}$和ReID任务指定的$r=\left\{r_{i}\right\}_{i=1}^{N_{p}}$。这个过程可以通过下面的式子描述，其中的$W_{d 1}, W_{d 2}, W_{e 1}$ and $W_{e 2}$均表示可学习的参数矩阵，$\operatorname{ReL} U(\cdot)$和$\Psi_{\ln }(\cdot)$表示线性修正单元和层标准化操作。

$$
\begin{array}{l}
d_{i}=x_{i}+W_{d 2} R e L U\left(\Psi_{l n}\left(W_{d 1} z\right)\right) \\
r_{i}=x_{i}+W_{r 2} R e L U\left(\Psi_{l n}\left(W_{r 1} z\right)\right)
\end{array}
$$

如果考虑批量输入$I$，它的shape为$\left(B^{\prime}, H^{\prime}, W^{\prime}, C^{\prime}\right)$，则$\Psi_{l n}(\cdot)$这个标准化可以使用下面的式子定义，其中的$I_{b h w c}$和$\tilde{I}_{b h w c}$是输入和输出在$(b,h,w,c)$处的元素。

$$
\begin{array}{l}
\mu_{b}=\frac{1}{H^{\prime} W^{\prime} C^{\prime}} \sum_{1}^{H^{\prime}} \sum_{1}^{W^{\prime}} \sum_{1}^{C^{\prime}} I_{b h w c} \\
\sigma_{b}^{2}=\frac{1}{H^{\prime} W^{\prime} C^{\prime}} \sum_{1}^{H^{\prime}} \sum_{1}^{W^{\prime}} \sum_{1}^{C^{\prime}}\left(I_{b h w c}-\mu_{b}\right)^{2} \\
\tilde{I}_{b h w c}=\frac{I_{b h w c}-\mu_{b}}{\sqrt{\sigma_{b}^{2}+\epsilon}}
\end{array}
$$

从上面第一个式子可以看到，$z$的计算与$i$的选择是无关的，$d$和$r$的所有元素都可以通过同一个$z$计算得到。由此，其实GCD的复杂度只有$O\left(C^{2}\right)$，相比于此前那些$O\left(H W C^{2}\right)$复杂度的全局注意力方法，GCD是非常高效的，后面的实验也证明了其有效性。

### GTE

下面来看看GTE（Guided Transformer Encoder）模块是如何实现的。事实上，注意力作为学习判别性特征的有效手段已经被广泛使用，但是此前的方法都使用固定感受野的卷积操作来获得注意力图，这其实忽视了不同目标和背景区域之间的全局相关性。为了弥补这一点，每个像素之间的全局注意力是作者考虑采用的手段，但是常规的全局注意力太庞大了，对分辨率较高的特征图难以进行。

为此，作者采用deformable attention来捕获上下文的结构化信息，它只需要在query nodes和自适应选择的关键样本之间计算相似性而不需要使用所有特征图上的样本点，这个操作可以将复杂度从$O\left(H^{2} W^{2} C\right)$ 降到$O(H W C)$。

进一步，作者结合deformable attention和Transformer Encoder的优势，形成了GTE模块，如下图所示。结合Transformer出色的推理能力和可变形注意力的自适应全局感受野，GTE产生了非常有效的embedding。

![](https://i.loli.net/2021/05/12/hXJxpwFsVq4vdlM.png)

下面，对着上面的图我们来阐述一下Transformer encoder和deformable
attention的细节。

如上图所示，这里采用Transformer的encoder结构来进行特征的，它的结构和原始的Transformer的Encoder非常类似，我这里不多赘述，不过原始的Self-Attention操作计算量过大，作者这里采用Deformble Attention替换它。

deformable attention的思想如下图，对于下图a感兴趣区域的每个query node，deformable attention自适应在整个图上选择有价值的key样本点，如下图b所示，然后query和key进行交互即可得到下图c所示的注意力图。deformable attention的具体工作流程如上图的下半部分所示，给定输入特征图$I$，三个独立的encoder $\Phi_{a}(\cdot), \Phi_{b}(\cdot)$和$\Phi_{c}(\cdot)$分别编码生成offset map $F_a$、key map $F_b$以及query map $F_c$。若每个query选择$N_k$个key样本，那么$F_a$将包含$2N_k$个通道，分别表示$N_k$个key相对于query的横向和纵向的偏移。因此，对每个query节点$q \in I$而言，它的坐标$Z_q$以及key相对于$Z_q$的$F_a$上的偏移$\triangle Z_{k}=\left\{\triangle Z_{k}^{i}\right\}_{i=1}^{N_{k}}$是可以知道的。

![](https://i.loli.net/2021/05/12/qJnYgC39ysNOVlP.png)

接着，根据key的坐标$Z_{k}=\left\{Z_{k}^{i}\right\}_{i=1}^{N_{k}}$以及key map $F_b$，可以获得key 样本向量$V_{k}=\left\{V_{k}^{i}\right\}_{i=1}^{N_{k}}$，它进一步被$\Phi_{d}(\cdot)$转换。根据$Z_k$也可以对来自$F_c$的query attention map $V_{q}=\left\{V_{q}^{i}\right\}_{i=1}^{N_{k}}$进行裁剪。最终得到的特征图$F_o$可以通过下面的式子计算得到，这里的$W_m$是可学习参数，$\bullet$则是hadamard积。

$$
F_{o}=W_{m} \sum_{i=1}^{N_{k}} V_{q}^{i} \bullet F_{c}^{i}
$$

### Detection and Association

检测分支思路和CenterNet是一致的，跟踪方面也是和FairMOT一样的匈牙利算法，不过这里采用了MAT的轨迹填充策略来平衡正负样本。

### Optimization objectives

两个分支共同优化，损失函数加权求和，和FairMOT的优化方式几乎一样，如下所示，不详细展开了。

## 实验

实验数据集是MOT16、MOT17和MOT20，额外数据集和FairMOT一致，包括了CrowdHuman，预训练策略也和之前一样。

![](https://i.loli.net/2021/05/12/BHyk3MLhdsuTSrW.png)

各个模块的消融实验如下。

![](https://i.loli.net/2021/05/12/2HOwpEjyKGcWSZ5.png)

解耦前后特征图可视化如下，解耦效果还是很明显的。

![](https://i.loli.net/2021/05/12/1rQ5uBkWexIqfJs.png)

下面这个可视化则证明了RelationTrack的鲁棒性很强。

![](https://i.loli.net/2021/05/12/wGekpaUELBAxgl3.png)


## 总结

针对目前JDE范式下MOT方法的主流问题，即分支矛盾采用了特征图结构的策略进行缓解，利用Deformable DETR的思路进行reid的全局信息捕获，工作量还是挺大的，在MOT领域是值得关注的工作。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。



