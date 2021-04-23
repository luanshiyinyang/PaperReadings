# CSTrackV2解读

> 这篇文章是CSTrack原团队的一个新工作，核心出发点是引入时间信息来修正检测器结果以保证轨迹的连续，通过利用前后帧相关性来进行运动建模完善单帧检测的结果，从而使得跟踪更加合理。该方法从实验数据上来看效果是非常猛的，作者后续也会开放源代码。

## 简介

单阶段多目标跟踪方法联合检测和重识别任务，近年来取得了比较大的突破，诞生了非常有影响力的方法如JDE等。然而，当前单阶段跟踪器仅仅使用单帧输入来获得边界框预测，当遇到比较严重的视觉障碍如遮挡模糊等时，边界框可能是不可靠的。一旦一个目标框被检测器误分为背景类别，其对应轨迹段的时序一致性就将难以保持。这篇论文中，作者通过提出一个重检查网络来恢复错误分类的边界框，即虚假背景。重检查网络通过使用改进的互相关层探索跨帧时间线索与当前候选框之间的关系，从而将先前的轨迹段传播到当前帧。这种前后帧信息的传播有助于恢复虚假背景框并且最终修复被破坏的轨迹段。通过将设计的重检查网络插入到CSTrack模型中，在MOT16和MOT17上MOTA分别从70.7和70.6涨到了76.7和76.3，涨点是非常恐怖的。

- 论文标题

    One More Check: Making "Fake Background" Be Tracked Again
- 论文地址

    http://arxiv.org/abs/2104.09441
- 论文源码

    https://github.com/JudasDie/SOTS

## 介绍

当前多目标跟踪（Multiple Object Tracking，MOT）方法大体可以分为两大类，即**二阶段（two-step）方法**和**单阶段（one-shot）方法**。二阶段方法遵循TBD范式（即先检测再跟踪），将多目标跟踪任务解耦成了候选框预测和轨迹段关联两个任务。尽管二阶段方法在精度上取得了惊人的表现，然而它却需要巨大的计算资源，这主要是因为它通过一个额外的ReID网络来对每个候选框提取特征。近两年，单阶段方法受到了更多的关注，其通过集成检测和ReID特征提取到一个网络中，取得了较好的速度和精度的权衡。通过特征共享和多任务学习，它们可以接近实时运行。作者观察到，大多数现有的单阶段跟踪器都将高质量的检测作为一个默认假设，也就是说，**每帧中的每个目标都能被检测器正确定位**。然而，真实世界的各种情况可能使得这个假设并不成立，导致跟踪的效果较差。

下图所示即为单阶段跟踪器的典型失败案例，蓝色箭头代表帧连续的方向，图中红色框表示由于较小的前景概率而被当作背景的目标，从整个帧序列上看，由于这几个漏检目标，导致了轨迹段的时间一致性被破坏。作者经过思考发现这种虚假背景的根本原因其实是过于依赖基于图像的检测结果了。换句话说，检测器从背景中区分目标仅仅基于单帧的视觉线索。然而，实际跟踪的场景都是极具挑战的，比如遮挡、小目标、背景杂乱等，这些都会造成视觉特征的有效性下降，最终可能误导检测器将目标分类为背景。因此，仅仅依靠检测器得到目标的位置在跟踪中是不太可靠的。但是反观人类视觉的动态机制，它不仅仅考虑当前的视觉线索，而且能够连续感知移动目标的时间一致性。这启发了作者，可以通过探索时间线索在检测过程中仔细检查目标周围环境，来恢复由检测器引起的误分类目标。

![](https://i.loli.net/2021/04/22/gWAaIUBZQdrve75.png)

在这篇文章中，作者超越传统的单帧检测，通过设计二次检查检测机制来再次跟踪虚假背景目标。作者方法的本质是使用跨帧时序线索来作为单帧视觉线索的补充，来完成目标检测任务。和此前的工作直接使用时序特征来增强当前帧的视觉表示不同，作者提出了re-check网络通过学习转换之前的轨迹段来恢复误分类的目标。具体来说，考虑上一帧目标的位置，转换结果重新检查当前帧中的周围环境并预测一个候选框，这个框表示一个误分类的目标并且如果检测器没有检测出来它的话它会被重新加载。

作者这个re-check网络的灵感源于SOT领域的孪生网络，在这类方法中，互相关层（cross-correlation layer）被用来建模时序信息并且预测目标位置。但是，论文的方法中，作者修改了cross-correlation layer使其适配多轨迹段转换。具体来看，在re-check网络中，目标的时序信息依据其之前帧的ID embedding表示，即为一个$1 \times 1 \times C$的张量。随着目标在序列上移动，通过使用cross-correlation操作来评估ID embedding的相似性来进行目标位置的传递。最终，跟踪器能够感知目标在当前帧上的状态，因此可以重新检查检测器在当前帧上的预测框，以查看目标是否被错误丢弃。

作者将设计的re-check network应用到CSTrack上，得到一个新的跟踪器，SiamMOT。在MOT16、MOT17和MOT20上进行验证，该框架实现了新的SOTA。

## SiamMOT

这一节将解读整个SiamMOT框架，在详细理解re-check之前，先简要了解整个跟踪框架。

### 整体框架

基于CSTrack构建了一个新的JDE方法，首先来回顾一下JDE和CSTrack。下图所示即为JDE的核心思路，它通过共享模型完成目标检测和ReID特征提取来构建一个实时的单阶段多目标跟踪框架。

![](https://i.loli.net/2021/04/22/WOCYU5gZySPkjxE.png)

整体的pipeline可以参考下图。给定一帧输入$\mathbf{x}$，首先是特征提取器$\Psi$（即backbone和neck）对其进行处理，得到特征图$\mathbf{F_t}$（$\boldsymbol{F}_{t}=\Psi(\boldsymbol{x})$）。接着，$\mathbf{F_t}$被送入head网络$\Phi$中同时预测检测结果和ID embedding，表示如下式。

$$
\left[\boldsymbol{R}_{t}^{\mathrm{de}}, \boldsymbol{F}_{t}^{i d}\right]=\Phi\left(\boldsymbol{F}_{t}\right)
$$

![](https://i.loli.net/2021/04/22/qEIHWXPCd2ZwUvy.png)

如上式和上图所示，这里的$\boldsymbol{R}_{t}^{\mathrm{de}}$表示检测结果，它包含一个前景概率图$\boldsymbol{P}_{t}^{d e} \in \mathbb{R}^{H \times W \times 1}$（通道数为1表示只有行人这一类）和边框预测图$\boldsymbol{B}_{t}^{d e} \in \mathbb{R}^{H \times W \times 4}$，而$\boldsymbol{F}_{t}^{i d} \in \mathbb{R}^{H \times W \times C}$（$C=512$）表示ID embedding。最后，检测结果$\boldsymbol{R}_{t}^{\mathrm{de}}$经过greedy-NMS处理后得到基本检测结果$\boldsymbol{D}_{\text {base }}$。$\boldsymbol{D}_{\text {base }}$中的每个box都对应$\boldsymbol{F}_{t}^{i d}$中的一个$1 \times 1 \times C$的embedding。不妨记所有的$\boldsymbol{D}_{\text {base }}$中的box对应的embedding为一个集合$\boldsymbol{E}_{t}^{i d}$。最后，$\boldsymbol{D}_{\text {base }}$和$\boldsymbol{E}_{t}^{i d}$用于和之前的轨迹段关联，关联采用贪婪二分图匹配算法。

而在最近，CSTrack通过引入互注意力来缓解detection任务和ReID任务之间的竞争，从而以很少的开销大大改善了JDE，CSTrack就是这篇论文工作的baseline（毕竟是作者本人的工作），关于CSTrack想要了解更多的可以参考[我之前的解读博客](https://zhouchen.blog.csdn.net/article/details/109838203)。

作者在CSTrack的基础上，提出了一个re-check network来修复检测器得出的虚假背景。如上图所示，再次使用上一帧ID embedding $\boldsymbol{E}_{t-1}^{i d}$作为一个目标的时序线索，re-check network $\Pi$通过评估 $\boldsymbol{E}_{t-1}^{i d}$和$\boldsymbol{F}_{t}^{i d}$之间的相似度来转换之前的轨迹。具体而言，作者修改了孪生网络中的cross-correlation layer使之能够单次前向就跟踪多个目标。此外，作者通过实验发现，如果一个目标在当前帧出现，那么它倾向于在相似度图中引入一个假阳性响应。为了缓解这个问题，作者将视觉特征图$\boldsymbol{F}_{t}$和相似度图进行了融合，然后将它们精炼为更精细的guidance map（指导图）。为了简化描述，re-check network可以表述如下。

$$
\boldsymbol{M}_{p}=\Pi\left(\boldsymbol{F}_{t}^{i d}, \boldsymbol{E}_{t-1}^{i d}, \boldsymbol{F}_{t}\right)
$$

这个式子中的$\boldsymbol{M}_{p}$是网络最终的预测结果，它是之前轨迹段到当前帧的转换结果。将$\boldsymbol{M}_{p}$视为前景概率图，并将它和原始的边界框$\boldsymbol{B}_{t}^{d e}$一起送入greedy-NMS。NMS输出的结果，称为转换检测（transductive detections），记为$\boldsymbol{D}_{\text {trans }}$，和原本的检测器的检测框$\boldsymbol{D}_{\text {base }}$组合到一起送入作者提出的IOU投票机制来产生最终的候选边界框$\boldsymbol{D}_{\text {final}}$。此时，**才算真正完成了目标检测**，$\boldsymbol{D}_{\text {final}}$和其在ID特征图$\boldsymbol{F}_{t}^{i d}$上对应的ID embedding则用于后续的数据关联。当基本检测结果误分类目标为背景时，transductive detections能够重新检查虚假背景并恢复漏检的目标框。

*上面这个过程，简要叙述了整个SiamMOT的pipeline，略过了诸多细节，下面将详细分析其中几个关键的模块。*

### 重检查网络

为了改善由于虚假背景产生的时序一致性破坏，作者提出了一个轻量的re-check network来恢复检测器漏检的目标。更具体而言，re-check network包含两个模块，即下图所示的用于轨迹传播的transductive detection module（检测转换模块）和用于假阳性过滤的refinement module（精炼模块）。

![](https://i.loli.net/2021/04/22/hY4VIPQfEG8qdOm.png)

**Transductive Detection Module**

首先来看这个转换检测模块，它用于将之前的轨迹段传播到当前帧，换句话说，其实就是预测历史轨迹在当前帧的位置。具体来看，通过评估之前轨迹的embeddings $\boldsymbol{E}_{t-1}^{i d}=\left\{\boldsymbol{e}_{t-1}^{1}, \cdots, \boldsymbol{e}_{t-1}^{n}\right\}$和当前帧检测结果的embeddings特征图$\boldsymbol{F}_t^{id}$之间的相似度来预测目标的位置，这里的$n$表示历史轨迹的数目。对每个目标都通过cross-correlation算子$*$来获得一个位置响应图$m_i$，其计算式如下所示，可以看到，这其实就是一个普通的矩阵乘法，每个$\boldsymbol{m}_{i}$的维度就是上图所示的$H \times W \times 1$。

$$
\boldsymbol{m}_{i}=\left.\left(\boldsymbol{e}_{t-1}^{i} * \boldsymbol{F}_{t}^{i d}\right)\right|_{i=1} ^{n}
$$

在每个$\boldsymbol{m}_i$中，最大值的位置就是之前轨迹的预测状态，这样的$n$个$\boldsymbol{m}_i$组合到一起就形成了一个相似度图$\boldsymbol{M}=\left\{\boldsymbol{m}_{1}, \cdots, \boldsymbol{m}_{n}\right\}$，其中的每个元素表示之前轨迹的转换检测结果。需要注意的是，改进的cross-correlation操作可以很方便地通过矩阵乘法实现，通过主流的深度学习框架可以方便实现。这部分对应上图Transductive Detection Module部分的下面一部分，即$n$个响应图。

接着，通过缩小高响应的范围，将$\boldsymbol{m}_i$离散化为一个二值掩膜图$\hat{\boldsymbol{m}}_{i}$。进行这步操作的根本原因在于拥有相似外观的目标可能会带来较高的响应值，缩小高响应范围可以减少这种混淆的预测。形式上，这个二值掩膜的生成方式如下，这里的$\hat{\boldsymbol{m}}_{i}^{x y}$表示$\hat{\boldsymbol{m}}_{i}$上$(x, y)$处的值，$c_x$和$c_y$表示$\boldsymbol{m}_i$上最大值的位置，$r$表示缩放半径，在缩放半径构成的正方形内置为1否则置为0。接着，这个二值掩膜图和原始响应图相乘消除模糊的响应结果，这里就是普通的点乘。

$$
\hat{\boldsymbol{m}}_{i}^{x y}=\left\{\begin{array}{ll}
1 & \text { if }\left\|x-c_{x}\right\| \leq r,\left\|y-c_{y}\right\| \leq r \\
0 & \text { otherwise }
\end{array}\right.
$$

接着，将$n$个响应图沿着通道逐元素相加得到最终的相似度图$\boldsymbol{M}_{s}$，这个相似度图表示当前帧每个位置包含之前帧中目标的概率，如果一个位置有一个高置信度的响应得分，那么这里就有一个和之前轨迹相关的潜在边界框。

$$
\boldsymbol{M}_{s}=\sum_{i=1}^{n}\left(\hat{\boldsymbol{m}}_{i} \cdot \boldsymbol{m}_{i}\right)
$$

*这一部分对应上图的Transductive Detection Module部分。*

**Refinement Module**

但是上面这种轨迹预测也会带来一些问题，作者发现没有出现在当前帧的那些目标在tracklet transduction的过程中会带来一些假阳性样本。为了缓解这个问题，作者设计了一个Refinement Module来引入当前帧原始的视觉特征$\boldsymbol{F}_{t} \in \mathbb{R}^{H \times W \times C}(\mathrm{C}=256)$来提供语义信息以进行更精细的定位。作者首先将上面的$\boldsymbol{M_s}$通过inverted bottleneck模块进行编码，这是一个通过两个3x3卷积先升维再降维的过程，得到精炼的相似度图$\boldsymbol{M}_{s}^{\prime} \in \mathbb{R}^{H \times W \times 1}$和$\boldsymbol{F}_t$进行逐元素相乘得到增强的特征$\hat{\boldsymbol{F}} \in \mathbb{R}^{H \times W \times C}(\mathrm{C}=256)$，这个计算过程如下式。

$$
\hat{\boldsymbol{F}}=\boldsymbol{F}_{t} \cdot \boldsymbol{M}_{s}^{\prime}
$$

接着，这个增强的特征$\hat{\boldsymbol{F}}$经过几个卷积层得到最终的预测$\boldsymbol{M}_{p}$，这是一个精炼后的响应图，和输入Refinement Module之前的响应图shape相同。

**Optimization**

上面通过两个模块的分析介绍了re-check network，但是这个模块引入CSTrack这样的模型中是需要额外的监督的，因此作者设计了一个新的损失。相似度图$\boldsymbol{M}_p$的GT通过多个高斯分布的组合来定义，具体而言，对每个目标，它的监督信号是一个高斯掩膜如下所示，这里的$c_{i}=\left(c_{i}^{x}, c_{i}^{y}\right)$表示一个目标的中心位置而$\sigma_{i}$是目标尺寸自适应的标准差。

$$
\boldsymbol{t}_{i}=\exp \left(-\frac{\left(x-c_{i}^{x}\right)^{2}+\left(y-c_{i}^{y}\right)^{2}}{2 \sigma_{i}^{2}}\right)
$$

上面的式子生成一系列的GT mask $\boldsymbol{t}=\left\{\boldsymbol{t}_{1}, \ldots, \boldsymbol{t}_{n}\right\}$，然后沿着通道维度求和就得到了$\boldsymbol{M}_p$的监督信号$\boldsymbol{T}$。为了减少两个高斯分布之间的重叠，为$\sigma_{i}$设置一个值为1的上界。使用Logistic-MSE损失来训练re-check网络，损失公式如下式，$\boldsymbol{M}^{x y}$和$\boldsymbol{T}^{x y}$分别表示$\boldsymbol{M}_p$和$\boldsymbol{T}$上$(x,y)$位置的值。

$$
\mathcal{L}_{g}=-\frac{1}{n} \sum_{x v}\left\{\begin{array}{l}
\left(1-\boldsymbol{M}_{p}^{x y}\right) \log \left(\boldsymbol{M}_{p}^{x y}\right), \quad \text { if } \boldsymbol{T}^{x y}=1 \\
\left(1-\boldsymbol{T}^{x y}\right) \boldsymbol{M}_{p}^{x y} \log \left(1-\boldsymbol{M}_{p}^{x y}\right), \text { else }
\end{array}\right.
$$

### 检测框融合

通过re-check网络，得到了历史轨迹在当前帧上的预测框$\boldsymbol{D}_{trans}$和检测器在当前帧上的检测框$\boldsymbol{D}_{base}$，但是如何将这两个融合到一起用于最终的数据关联呢？首先，为$\boldsymbol{D}_{trans}$中的每一个边界框$\boldsymbol{b}_i$计算目标度得分（targetness score），得分的计算基于其与检测框的最大IOU，式子如下。

$$
s=1-\max \left(\operatorname{IOU}\left(\boldsymbol{b}_{i}, \boldsymbol{D}_{\text {base }}\right)\right)
$$

$s$值越大表明框$\boldsymbol{b}_i$并没有出现在检测器的检测结果中，因此它就可能是一个漏检框。若这个框的$s$值高于阈值$\epsilon$，那么这个框就作为检测框的补充加进去，作者设置的$\epsilon$为0.5。通过这个融合，可以将检测器漏掉的检测框找回来，保证轨迹的连续性。具体的算法如下图，比较清晰明了。

![](https://i.loli.net/2021/04/22/sQGMrN6wovFSgCb.png)

## 实验

我们知道在JDE和CSTrack中，anchor和GT之间的偏移是通过sigmoid函数约束在0到1之间的，这里记anchor的中心为$a=\left(a_{x}, a_{y}\right)$，GT的中心为$b=\left(b_{x}, b_{y}\right)$，两者的偏移通过下式计算，这里的$r$就是回归分支的输出。

$$
\boldsymbol{\Delta}=\boldsymbol{b}-\boldsymbol{a}=\operatorname{Sigmoid}(\boldsymbol{r})
$$

但是，作者发现，其实在图像的边界上，偏移量经常是大于1的。如下图所示，GT框的中心（绿色表示）已经超出了图像的边界，然而由于sigmoid函数的约束。预测框（红色表示）很难覆盖整个目标。当一个目标只出现部分身体，不完整的边界框预测将被视为假阳性样本，这是因为其与GT框的距离很远并且不完整，这使得跟踪的性能最终下降。为了缓解这个问题，作者将回归机制修改为边界感知回归（boundary-aware regression，BAR），它允许跟踪器通过可见身体推理出目标的全身区域。

![](https://i.loli.net/2021/04/22/cgukCn3SWUaKfdi.png)

具体来看，将上面那个式子修改如下，这里的$h$是一个可学习的尺度参数，这个尺度参数允许网络预测大于1的偏移。如上图的(c)所示，BAR能够通过可见部分预测出目标的不可见部分。

$$
\boldsymbol{\Delta}=\boldsymbol{b}-\boldsymbol{a}=(\operatorname{Sigmoid}(\boldsymbol{r})-0.5) \times h
$$

至于实验的设置和评估指标的配置之类的，就和CSTrack一样了，这里就不展开叙述了。下面来看一下SiamMOT在几个benchmark上的SOTA表现，可以看到，其精度是非常卓越的，虽然速度相比原来的CSTrack有所下降。

![](https://i.loli.net/2021/04/22/kUS9FYX6HI2Gzep.png)

此外，作者也进行了消融实验对比提出的re-check network（RCNet）和BAR的收益，如下表所示。可以看到，RCNet的效果是非常明显的，这也进一步说明，MOT这个任务其实是非常依赖显式的时序信息的。

![](https://i.loli.net/2021/04/22/hENalVgZSoIdmYO.png)

其他的组件的消融实验我这里不多说了，感兴趣的可以查看原文。可视化方面作者也做了不少，下面这个图第一列是原始图像，第二列是Transductive Detection Module输出，第三列则是Refinement Module输出的。从图上第二列可以看出来，之前轨迹的状态被有效转移到了当前帧，第三列则表示精炼模块确实有效过滤了假阳性。SiamMOT在遮挡严重的情况下依然可以保证跟踪的鲁棒性切处理较小的目标。

![](https://i.loli.net/2021/04/22/Fw7PJNMavfSc63o.png)

## 总结

这篇论文将重点放在了MOT的检测质量方面，设计了一个非常优雅的运动模型将历史轨迹的信息转换到当前帧上来补充检测器没能准确检测的目标，使得整个轨迹更加平滑连续，在CSTrack的基础上有了比较大的突破，是很值得关注的一个方法。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。







