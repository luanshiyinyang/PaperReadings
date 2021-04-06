# TransCenter解读

> 本文介绍这篇出自MIT的多目标跟踪方向的新论文，其将transformer和基于Center的目标检测与跟踪联合到一起，设计了一个全新的多目标跟踪框架，超越了此前的TransTrack和TrackFormer，一举冲到了SOTA方法精度的最前列，不过这篇文章从MOT Challenge上的指标来看，速度并不快。

## 简介

此前transformer方法已经被引入了多种任务并取得了很好的表现，计算机视觉当然也不例外，已经出现了很多在视觉中很不错的transformer结构并且有愈发流行的趋势。然而transformer结构仍然与多目标跟踪任务存在某种不兼容，这篇论文作者认为标准的边界框表示不适用于transformer的学习。受到最近研究的启发，作者提出了第一个基于transformer并且以点（center）的形式进行多个目标跟踪的MOT框架，名为TransCenter。方法论方面，作者提出在双解码器网络中使用密集query，这样可以非常可靠地推理目标中心heatmap并且沿着时间对其进行关联。实验证明，TransCenter在多个基准数据集上均获得了SOTA表现，包括典型地MOT17和MOT20。

- 论文标题

    TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
- 论文地址

    http://arxiv.org/abs/2103.15145
- 论文源码

    暂未开源

## 介绍

多目标跟踪任务（Multiple Object Tracking，MOT）简单来说，就是在一个场景中同时推理出不同目标的位置和ID，其在计算机视觉中属于一个核心任务。大量的基础数据集和评估方法的提出有助于MOT领域的快速发展，世界范围内的研究者可以通过MOT Challenge来评估自己团队提出的算法。

最近这段时间，大量的任务开始使用transformer来实现自己的框架，如行人检测、行人重识别、图像超分辨率等。基于transformer的结构是否适用于MOT还是一个值得斟酌的问题，TransTrack和TrackFormer已经对此进行了尝试并获得了不错的效果。然而，论文作者认为至今为止的基于transformer的MOT方法采用的行人表示方法是不适用于transformer结构的学习的。TransTrack和TrackFormer均采用边界框（bounding box，bbox），bbox表示法是非常直观的，它可以很好地和概率学方法和深度卷积神经网络相结合，因而在MOT中也广泛使用。在处理非常拥挤的场景时，使用bbox来跟踪多个目标的一个突出缺点就体现出来了，因为GT框经常相互重叠，所以遮挡很难处理。这是有问题的，因为这些bbox会在训练中使用，它不仅返回每个人的位置、宽度和高度，而且区分与每个轨迹相关的视觉外观。**然而，在这种情况下，重叠bbox意味着训练一个视觉外观表示，该表示结合了两个甚至更多的人的外观信息，这就会造成关联的歧义。** 当然，联合处理跟踪和分割任务（即进行MOTS任务）可以部分解决遮挡问题。然而，这需要有额外的像素级标注，这是非常繁琐和昂贵的。目前MOT的几个比较公认的基准数据集并没有分割标注。

那么，有没有缓解这种由bbox带来的混淆问题呢，其实此前的CenterTrack和FairMOT已经证明了用点来表示目标的高效性，因此论文作者提出了一种新的基于transformer的使用行人中心点进行跟踪的框架，即TransCenter。因此，相比于直接从transformer目标检测器Deformable DETR和DETR发展而来的TransTrack和TrackFormer，**TransCenter的主要区别在于，它可以缓解bbox跟踪固有的遮挡问题，而不需要额外的分割级别标注。** 虽然这种设计灵感非常简单，但是设计一个高效的基于transformer的结构来实现这种灵感还需要很多的路要走。

事实上，第一个挑战就是密集表示的推理，即中心热力图（center heatmaps）的生成。对此，作者提出了像素级的密集多尺度query，这个策略不仅仅实现了基于heatmap的MOT，同时也克服了只用少量query对decoder进行查询的限制（DETR和Deformable DETR就是使用的少量可学习的query）。和TransTrack类似，TransCenter也有两个不同的decoder，一个负责检测任务另一个负责跟踪任务。这两个decoder的query都是源于当前帧图像，不过通过不同的层学习得到。不过，当前帧的memory（transformer结构encoder的输出）会传给检测decoder，前一帧的memory才会传给跟踪decoder。基于这些设计，作者设计的TransCenter达到了目前MOT方法的SOTA精度，几种基于transformer或者center表示的MOT方法跟踪结果的比较如下图，可以看到基于transformer的结构采用稀疏query的话会造成粉色箭头那样的漏检，严重遮挡下也会造成绿色箭头那样的误检。以前的基于center的几个MOT方法有着同样的问题，因为center是相互独立进行估计的。TransCenter则通过使用密集的(像素级)多尺度query来实现基于热图的推理，并利用注意机制来引入预测center点之间的相互依赖（也就是每个center都和其他center通过自注意力建立了依赖），从而缓解这两种不利影响。

![](https://i.loli.net/2021/04/06/a19ohpFSQtmLfrN.png)

## TransCenter

针对此前transformer在MOT中的尝试，它们都将边界框作为目标的表示方式进行推理，作者认为这个选择是不合适的，并且探索了最近很流行的基于中心热力图的表示。但是，相比于bbox这种稀疏表示，热度图是一种密集表示，因此，作者引入了密集多尺度query到计算机视觉中的transformer结构中。事实上，据作者所知，这是第一个提出使用随输入图像大小缩放的密集query特征图的工作。在TransCenter这个工作中，query大概由14k个，因此使用密集query的一个缺点是相关的巨大内存消耗。为了减轻这种不良的影响，作者提出采用可变形的decoder，其灵感来自可变形卷积（DCN）。

具体而言，TransCenter将多目标跟踪问题分为两个独立的子任务：在$t$时刻的目标检测任务，以及在$t−1$时刻的与检测到的目标的数据关联任务。不过，和此前的原题相同的其他工作相比，TransCenter采用了不同的处理思路，通过一个全可变双解码器结构并行完成了这两个任务。detection decoder的输出用来估计目标的中心和大小，结合tracking decoder的输出来估计目标相对于前一帧图像的位移。将中心热图与双解码器架构结合使用的一个好处是，沿着时间的目标关联不仅取决于几何特征(如IoU)，还取决于来自解码器的视觉特征（即外观特征）。

### 结构简述

整个TransCenter的架构如下图所示，第$t$帧和第$t-1$帧的图像被送入提取特征的backbone中来产生多尺度特征并和Deformable DETR那样捕获更细粒度的细节信息，然后，这些特征进入可变自注意力encoder中，获得多尺度memory特征图$M_t$和$M_{t-1}$，它们分别对应第$t$帧图像和第$t-1$帧图像。接着，$M_t$被送入一个query学习网络，这是一个像素级的全连接网络，它输出一个包含密集多尺度**检测query**的特征图$DQ_t$。同时，$DQ_t$被送入另一个query学习网络产生密集多尺度**跟踪query**特征图$TQ_t$。接着，后面是两个可变decoder结构，图上从上向下分别是**可变跟踪解码器（Deformable Tracking Decoder）**和**可变检测解码器（Deformable Detection Decoder）**。我们先来看检测解码器，它通过检测query $DQ_t$从memory $M_t$中查询当前帧信息从而输出多尺度检测特征$DF_t$，跟踪解码器则利用跟踪query $TQ_t$从memory $M_{t-1}$中查询得到多尺度跟踪特征$TF_{t}$。接着，检测多尺度特征$DF_t$被用来进行中心热力图$C_t$的估计和边框尺寸$S_t$估计。另一个分支上，跟踪多尺度特征$TF_t$和上一帧的中心热力图$C_{t-1}$以及多尺度检测特征图$DF_t$用来进行跟踪位移$T_t$的估计。

![](https://i.loli.net/2021/04/06/XjZnFNSBePqHdmI.png)

上文整体上讲了讲TransCenter的pipeline，下文会更具体分别讲解各个组件，包括密集多尺度query的设计、全可变双解码器结构、三个任务分支以及训练损失。

### 密集多尺度query

传统的transformer结构中，输出的元素数目和decoder输入的query数目相同，而且，每个输出对应一个寻找的实体（如行人检测框）。当推断中心热力图时，在给定像素处有一个人的中心的概率成为这些寻求的实体之一，因此要求transformer解码器接受密集的query。这些query可以通过多尺度encoder生成的memory来生成，一个query learning network（QLN）通过像素级的前馈操作获得$DQ_t$，使用不同的query用于；两个不同的decoder，因此另有一个QLN通过$DQ_t$生成$TQ_t$，这两个query会送入两个不同的decoder中，这个decoder的设计后文详解，首先来分析一波这个密集query的好处。

事实上，密集query特征图的分辨率与输入图像的分辨率成正比有两个突出的优点。首先，query可以是多尺度的，并利用encoder的多分辨率结构，允许非常小的目标被这些query捕获。其次，密集query也使网络更加灵活，能够适应任意大小的图像。更普遍而言，QLN的使用避免了以前的视觉transformer架构中所做的那样进行手工调整query大小和预先选择最大检测数量。

### 全可变双解码器

为了找到目标的轨迹，一个MOT方法不仅仅需要检测出目标还需要进行跨帧关联。为此，TransCenter提出了一个全可变双解码器结构，这两个解码器并行处理检测和跟踪两个子任务。detection decoder利用注意力关联$DQ_t$到$M_t$上对第$t$帧的图像$I_t$进行目标检测，tracking decoder关联$TQ_t$与$M_{t-1}$来将检测到的目标根据它们在前一帧图像$I_{t-1}$的位置来进行数据关联。

具体来看，detection decoder利用多尺度$DQ_t$在多尺度特征图$M_t$上进行搜索，得到多尺度检测特征$DF_t$，它被后续head用于目标中心点定位和边框尺寸回归。tracking decoder做的事则不太一样，它在$M_{t-1}$特征图上寻找目标，并和帧$t$上的目标进行关联，要想实现这个功能，tracking decoder中的多头可变注意力其实是一个时序互相关模块，它寻找多尺度query $TQ_t$和$M_{t-1}$特征图之间的互相关系，并输出多尺度跟踪特征$TF_t$，这个特征包含了跟踪分支进行$t$到$t-1$的偏移预测需要的时序信息。

这两个decoder输入的都是密集query特征图所以输出的同样是密集信息，因此如果使用原始Transformer中的多头注意力模块在TransCenter中的话，意味着内存和复杂性增长是输入图像尺寸的二次方倍，即$O\left(H^{2} W^{2}\right)$。当然，这是不可取的，并且会限制方法的可伸缩性和可用性，尤其是在处理多尺度特性时。自然，作者考虑到可变形的多头注意力，从而形成了完全可变形的双解码器架构。

### 中心、尺存和跟踪分支

两个解码器的输出是两个多尺度特征集，分别为检测特征$DF_t$和跟踪特征$TF_t$，具体而言，它们包含四个分辨率的特征图，分别是原图尺寸的$1 / 64,1 / 32,1 / 16$和$1 / 8$，对于中心点定位和尺寸回归两个分支，不同分辨率的特征图通过可变形卷积核双线性插值组合到一起，结构如下图所示，可以看到，四个尺度的特征图输出进去通过逐渐上采样融合到一起，得到最大的特征图$\mathbf{C}_{t} \in[0,1]^{H / 4 \times W / 4}$，它用于中心热力图的生成。类似下图，尺寸回归也会产生一个类似的最终特征图$\mathbf{S}_{t} \in \mathbb{R}^{H / 4 \times W / 4 \times 2}$，它拥有两个通道，分别表示宽度核高度两个方向的尺寸。

![](https://i.loli.net/2021/04/06/1VB5lpW9jORwSMa.png)

接着考虑跟踪分支，和上面两个分支的处理思路一样（参数不同）得到两个多尺度特征图，尺寸也为$1/ 4$的原图分辨率。接着这两个特征图和上一帧中心热力图缩放后的结构联结到一起，依据这个特征图，类似其他分支，通过卷积操作计算最终输出即目标的位移$\mathbf{T}_{t} \in \mathbb{R}^{H / 4 \times W / 4 \times 2}$，它也是两个通道的，表示水平方向和垂直方向的位移预测。

### 训练损失

TransCenter的训练是**中心热力图分类、目标尺寸回归和跟踪位移回归**三个任务联合训练的，下面分别来看这三个分支。

首先是**中心预测分支**，为了训练这个分支首先需要构建GT热力图$\mathbf{C}^{*} \in[0,1]^{H / 4 \times W / 4}$，这里的思路和CenterTrack一致，通过考虑以每$K$个GT目标为中心的高斯核的最大响应来构建$\mathbf{C}^{*}$。公式层面看，对每个像素位置$(x,y)$而言，它的GT热力图通过下式计算，这里的$(x_k,y_k)$为GT目标中心，$G(\cdot, \cdot ; \sigma)$是扩展为$\sigma$的高斯核函数，在该工作中，$\sigma$与目标的大小成比例，如CornerNet的策略。

$$
\mathbf{C}_{x y}^{*}=\max _{k=1, \ldots, K} G\left((x, y),\left(x_{k}, y_{k}\right) ; \sigma\right)
$$

有了GT热力图$\mathbf{C}^{*}$和预测热力图$\mathbf{C}$，就可以计算中心focal loss $L_{\mathrm{C}}$了，计算式如下，缩放因子$\alpha = 2$，$\beta = 4$。

$$
L_{\mathrm{C}}=\frac{1}{K} \sum_{x y}\left\{\begin{array}{ll}
\left(1-\mathbf{C}_{x y}\right)^{\alpha} \log \left(\mathbf{C}_{x y}\right) & \mathbf{C}_{x y}^{*}=1 \\
\left(1-\mathbf{C}_{x y}^{*}\right)^{\beta}\left(\mathbf{C}_{x y}\right)^{\alpha} \log \left(1-\mathbf{C}_{x y}\right) & \text { otherwise }
\end{array}\right.
$$

接着，来看两个**回归分支**，从整个pipeline那个图上看，其实尺寸回归和位移回归分别对应$\mathbf{S}_t$和$\mathbf{T}_t$，下面为了叙述方便简称为$\mathbf{S}$和$\mathbf{T}$。对$\mathbf{S}$和$\mathbf{T}$的监督只发生在目标中心点上，即$C_{x y}^{*}=1$处，对尺寸的回归采用$L_1$损失，公式如下。

$$
L_{\mathrm{S}}=\frac{1}{K} \sum_{x y}\left\{\begin{array}{ll}
\left\|\mathbf{S}_{x y}-\mathbf{S}_{x y}^{*}\right\|_{1} & \mathbf{C}_{x y}^{*}=1 \\
0 & \text { otherwise }
\end{array}\right.
$$

跟踪分支的损失计算式$L_T$和$L_S$是类似的，但它作用于跟踪输出和相应的GT而不是目标的尺寸。同时，为了保证$L_S$和$L_T$的稀疏性，作者额外加了一个$L_1$损失，记为$L_R$，它是一个边框损失，边框来源于尺寸特征图$S_t$和GT中心点。

所以，整个TransCenter的训练损失如下式所示，它是各个损失的加权求和的结果，权重稀疏依据损失的数值尺度确定。

$$
L=L_{\mathrm{C}}+\lambda_{\mathrm{S}} L_{\mathrm{S}}+\lambda_{\mathrm{T}} L_{\mathrm{T}}+\lambda_{\mathrm{R}} L_{\mathrm{R}}
$$

## 实验

关于数据集的选择、优化器等参数的设置、推理的具体配置我这里就不多赘述，详细可以参考原论文，关于训练流程我简单说一下，作者这里首先使用COCO上的行人类别预训练，然后在具体的MOT数据集上fine-tuning。具体的，在两张RTX Titan GPU上以2为batch size，在MOT20和MOT17上一轮分别需要1h30min和1h。

下面两个表分别是TransCenter在MOT17和MOT20上和SOTA方法的对比，其中Data栏表示是否使用额外数据加入训练，CH表示使用CrownHuman数据集，PT表示使用PathTrack数据集，RE表示组合Market1501、CUHK01和CUHK03三个重识别数据集，5D1表示使用CrowdHuman, Caltech Pedestrian, CityPersons, CUHK-SYS,和PRW数据集，5D2表示同5D1只是将CrowdHuman换为ETH数据集，No则表示不使用额外数据。不难发现，忽略速度不谈，TransCenter在不使用额外数据集的情况下已经超越了很多SOTA方法，达到先进水平。

![](https://i.loli.net/2021/04/06/rV4IOstwNQmzXiP.png)

![](https://i.loli.net/2021/04/06/QOR7Pesj9poXSG8.png)

此外，作者还进行了几个消融实验，包括单解码器、重识别模块、纯检测、不加$L_R$等策略的效果，具体可以查看论文原文，下面放几张MOT20测试集上的跟踪可视化结果。

![](https://i.loli.net/2021/04/06/Qyz3d98afZ27vpN.png)

## 总结

TransCenter开创性地将基于Transformer的MOT和基于center的MOT结合到一起，构建了一个双解码器的跟踪框架，超越了诸多当前的SOTA方法，是很值得关注的工作。当然，论文里很多细节创新点很多，单从整体上来看，其实还是使用Transformer进行更好的特征提取，包括时序信息。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。




