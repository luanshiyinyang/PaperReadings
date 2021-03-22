# TraDeS解读

> 这篇TraDeS是MOT领域的新作，收录于CVPR2021，作者来自纽约州立大学等机构，在多个基准任务上均达到SOTA水平，包括2D跟踪、3D跟踪和分割级跟踪。

## 简介

大多数现有的online MOT方法的检测部分在整个网络中都是独立进行的，即没有任何输入来自跟踪任务。事实上，不仅仅检测可以为跟踪服务，跟踪其实也可以反过来辅助检测任务，因此在这篇文章中作者提出了一种新的联合检测和跟踪端到端模型，名为TraDeS（Track to Detect and Segment），它能利用跟踪线索来辅助检测。TraDeS通过cost volume来推断目标跟踪偏移量，该cost volume用于传播前帧的目标特征，以改善当前目标的检测和分割。在MOT、nuScenes、MOTS和Youtube-VIS数据集上，TraDeS均显示不错的有效性和优越性。

- 论文标题

    Track to Detect and Segment: An Online Multi-Object Tracker
- 论文地址

    http://arxiv.org/abs/2103.08808
- 论文源码

    https://github.com/JialianW/TraDeS

## 介绍

此前的在线多目标跟踪方法基本上遵循两个范式：**tracking-by-detection（TBD）**和**joint detection and tracking（JDT）**。TBD范式如下图的(a)所示，它将检测和跟踪视为两个独立的任务，通常是采用一个现有的检测器产生检测结果然后用另一个独立的网络来进行数据关联。显然，TBD范式分前后两个阶段，因此其效率并不高，也并非端到端优化的。为了解决TBD存在的问题，JDT范式应运而生，它在一次前向传播过程中同时解决检测和跟踪，如下图的(b)所示。

![](https://i.loli.net/2021/03/22/1s6DpW2dZA4uUrK.png)

JDT方法目前主要面临两个问题：一方面，尽管JDT方法中提取特征的backbone网络是共享的，但是检测通常是独立于跟踪进行的。而作者认为，检测是稳定和一致的轨迹的基石，反过来，跟踪线索将有助于检测，特别是在困难的场景，如部分遮挡和运动模糊。另一方面，从前人的工作和作者的实验中，其实不难发现，ReID跟踪损失其实和检测损失在同一个网络中是不兼容的，甚至跟踪损失会一定程度上损害检测任务的表现。原因早在FairMOT那时就已经发现，其实是ReID任务关注的是类内差异，而检测的目的是扩大类间差异，最小化类内差异。（**简单来说，对行人跟踪而言，ReID任务期待最大化每个人之间的差异，Detection任务则期待最大化不同类别之间目标的差异，这无形之中其实期待行人这一类在高层语义空间尽量接近。**）

论文中，作者提出了TraDeS方法，在该方法中，特征图上的每个点表示一个目标中心或者一个背景区域，这和CenterNet是类似的。TraDeS通过紧密集成跟踪到检测配合上一个精心设计的ReID学习策略来解决上文所说的两个问题。具体而言，作者设计了一个基于cost volume（代价空间，立体视觉中的一个概念，本文中可以理解为一个关联代价矩阵）的关联模块（CVA）和一个运动指导的特征变换模块（MFW）。CVA模块通过backbone提取逐点的ReID embedding特征来构建一个cost volume，这个cost volume存储两帧之间的embedding对之间的匹配相似度。继而，模型可以根据cost volume推断跟踪偏移（tracking offset），即所有点的时空位移，也就得到了两帧间潜在的目标中心。跟踪偏移和embedding一起被用来构建一个简单的两轮长程数据关联。之后，MFW模块以跟踪偏移作为运动线索来将目标的特征从前帧传到当前帧。最后，对前帧传来的特征和当前帧的特征进行聚合进行当前帧的检测和分割任务。

在CVA模块中，cost volume被用来监督ReID embedding，不同的目标类别和背景区域被隐式考虑了进去，这就是说，**ReID分支的学习目标考虑了类间差异**。因此，它不仅仅可以学到有效的embedding而且能够兼容检测损失从而不会损害检测性能。此外，由于tracking offset是基于外观特征相似度得到的，因此它可以匹配一个大幅度运动的目标或者在低帧率下工作，它甚至能跟踪不同数据集之间的目标（即训练数据集和测试数据集不同，如下图所示）。因此，这个预测的tracking offset可以作为鲁棒的运动线索来指导MFW模块中的特征传播。当前帧被遮挡或者模糊的目标在之前的帧中可能是清晰的，所以通过MFW模块，前帧特征传播可以支撑当前帧特征去恢复潜在的不可见目标。

![](https://i.loli.net/2021/03/22/MjPeZ4KD16NJd89.png)

总的来说，这篇论文作者设计了一个新的在线多目标跟踪器，TraDeS，该模型深度集成跟踪线索从而辅助检测，进而检测又会带来更好的跟踪效果。TraDeS是一个适用于2D跟踪、3D跟踪和分割级跟踪的通用跟踪器，多个数据集上的实验表明，TraDeS是目前多目标跟踪各个任务上的SOTA方法。

## CenterNet

和之前FairMOT、CenterTrack等方法一样，TraDeS也是基于CenterNet的工作，事实上基于keypoint的anchor-free检测方法似乎在MOT领域取得了相当不错的结果。CenterNet以图片$\boldsymbol{I} \in \mathbb{R}^{H \times W \times 3}$并且通过backbone网络$\phi(\cdot)$产生基本的特征图$\boldsymbol{f}=\phi(\boldsymbol{I})$，这里的$\boldsymbol{f} \in \mathbb{R}^{H_{F} \times W_{F} \times 64}$，其中$H_{F}=\frac{H}{4},$和$W_{F}=\frac{W}{4}$。

![](https://i.loli.net/2021/03/22/oxNmeW28jis6Qcp.png)

如上图所示，在基本特征图之后，一系列head被设计用于各个任务，比如heatmaps head经过卷积变换产生特征图$\boldsymbol{P} \in \mathbb{R}^{H_{F} \times W_{F} \times N_{c l s}}$以及2D尺寸特征图和偏移特征图等。其中$N_{cls}$指的是类别的数目，CenterNet通过$P$上的峰值点作为待检测目标的中心，通过预测距离中心点的距离确定目标的尺寸。所以，整体来看，CenterNet其实是一种非常简约的检测框架，TraDeS这篇文章中，作者构建baseline跟踪器是基于CenterTrack的，它是在CenterNet的基础上添加了一个位移预测分支$\boldsymbol{O}^{B} \in \mathbb{R}^{H_{F} \times W_{F} \times 2}$来进行数据关联，这个$\boldsymbol{O}^{B}$计算的是当前帧$t$到之前帧$t-\tau$的时空位移。

## TraDeS跟踪器

作者认为检测和跟踪不该独立进行，跟踪可以辅助检测，提升困难场景下的检测质量，继而又带来更好的跟踪。为了实现这个目标，作者设计了Cost Volume based Association（CVA）模块和Motion-guided Feature Warper（MFW）模块，前者用于学习ReID embedding和产生目标运动，后者则用于利用CVA产生的跟踪线索来传播和增强目标的特征。

![](https://i.loli.net/2021/03/22/8Vt9CIgT3HXfuWN.png)

后面的内容基本上都是围绕上面这个pipeline来叙述的，这个图看起来复杂，但是一步步看其实是非常清晰的，下面的讲解主要针对$t$和$t-1$帧的处理。

### 基于cost volume的数据关联

给定两帧图像$\boldsymbol{I}^{t}$和$\boldsymbol{I}^{t-\tau}$，它们通过backbone产生的特征图为$\boldsymbol{f}^{t}$和$\boldsymbol{f}^{t-\tau}$，这个backbone在上面的pipeline图中对应$\phi$网络（即原图右侧权重共享的蓝色梯形框）。随即，特征图被送入一个三层卷积构成的模块中提取ReID embedding，得到的特征图为$\boldsymbol{e}^{t}=\sigma\left(\boldsymbol{f}^{t}\right) \in \mathbb{R}^{H_{F} \times W_{F} \times 128}$，这里的$\sigma$就是三层卷积网络构成的模块，在上图中对应黄色部分，这个ReID提取网络每帧之间依然是权值共享的。

接下来，就是最关键的计算一个cost volume来保存两帧之间特征图上每两个点之间的匹配相似度。那么具体怎么算的呢？首先，为了降低计算量先用最大池化对ReID embedding特征图进行下采样，得到$\boldsymbol{e}^{\prime} \in \mathbb{R}^{H_{C} \times W_{C} \times 128}$，其中$H_{C}=\frac{H_{F}}{2}$，$W_{c} = \frac{W_F}{2}$，也就是说每个特征图上有$H_C \times W_C$个目标的embedding向量，因此需要计算得到两个特征图上任意两个点之间的相似度矩阵，即cost volume，它表示为$\boldsymbol{C} \in \mathbb{R}^{H_{C} \times W_{C} \times H_{C} \times W_{C}}$，代表图像$\boldsymbol{I}^{t}$和$\boldsymbol{I}^{t-\tau}$之间的cost volume，具体的计算方法为下面的矩阵乘法式子，其中的元素$C_{i, j, k, l}$表示帧$t$上的点$(i,j)$和帧$t-\tau$上的点$(k,l)$之间的embedding相似度。这一部分其实对应到上图的Cost Volume Map这部分。 

$$
C_{i, j, k, l}=\boldsymbol{e}_{i, j}^{\prime t} \boldsymbol{e}_{k, l}^{\prime t-\tau \top}
$$

有了上面的$C$就可以通过$C$计算跟踪偏移矩阵$\boldsymbol{O} \in \mathbb{R}^{H_{C} \times W_{C} \times 2}$，这个矩阵存储$t$时刻上的每个点相对于其在$t-\tau$时刻的那个点的时空位移。后面为了方便叙述，聚焦到一个点上来看它是怎么算出来的，其$O_{i, j} \in \mathbb{R}^{2}$。

如上图$O^C$部分所示，对帧$t$上的中心在$(i,j)$的目标$x$而言，可以从cost volume即$C$中得到其对应的二维cost volume map $\boldsymbol{C}_{i, j} \in \mathbb{R}^{H_{C} \times W_{C}}$，它表示$x$和帧$t-\tau$上所有点之间的匹配相似度，通过$\boldsymbol{C}_{i, j}$计算$\boldsymbol{O}_{i, j} \in \mathbb{R}^{2}$需要下面的两个步骤。**首先**，两个方向的以为最大池化分别作用于$C_{i,j}$，其中池化核分别为$H_{C} \times 1$和$1 \times W_C$，然后经过softmax函数，继而得到两个向量$\boldsymbol{C}_{i, j}^{W} \in[0,1]^{1 \times W_{C}}$和$\boldsymbol{C}_{i, j}^{H} \in[0,1]^{H_{C} \times 1}$。这里的$\boldsymbol{C}_{i, j}^{W}$和$\boldsymbol{C}_{i, j}^{H}$分别包含对象$x$出现在$t-\tau$帧上出现在指定的水平位置和竖直位置的概率。举例而言，$C_{i, j, l}^{W}$即为目标$x$出现在位置帧$t-\tau$上$(*,l)$位置的概率。**接着**，由于$\boldsymbol{C}_{i, j}^{W}$和$\boldsymbol{C}_{i, j}^{H}$提供了$x$在之前帧上指定位置的概率，为了获得最终的偏移，作者定义了两个偏移模板对应两个方向，它表明$x$实际出现在那些位置的偏移值。记这两个模板为$\boldsymbol{M}_{i, j} \in \mathbb{R}^{1 \times W_{C}}$和$\boldsymbol{V}_{i, j} \in \mathbb{R}^{\hat{H}_{C} \times 1}$，其通过下面的式子计算得到。

$$
\left\{\begin{array}{ll}
M_{i, j, l}=(l-j) \times s & 1 \leq l \leq W_{C} \\
V_{i, j, k}=(k-i) \times s & 1 \leq k \leq H_{C}
\end{array}\right.
$$

上式中的$s$为特征图相对于原图的下采样倍率，在TraDeS中为8。$M_{i,j,l}$指的则是目标$x$在$t-\tau$帧上出现在$(*,l)$位置的偏移量。最终的跟踪偏移可以通过相似度和实际偏移值的点积计算得到，如下式所示。由于$O$的维度为$H_C \times W_C$因此通过两倍上采样变为$\boldsymbol{O}^{C} \in \mathbb{R}^{H_{F} \times W_{F} \times 2}$作为MFW模块的运动线索。

$$
\boldsymbol{O}_{i, j}=\left[\boldsymbol{C}_{i, j}^{H \top} \boldsymbol{V}_{i, j}, \boldsymbol{C}_{i, j}^{W} \boldsymbol{M}_{i, j}^{\top}\right]^{\top}
$$

所以上面说了这么多，CVA模块中唯一可学习的部分就是三层卷积的$\sigma$部分，CVA模块的优化目标就是学到有效的ReID embedding $\boldsymbol{e}$。但是CVA并不直像很多ReID模型那样通过损失直接监督$e$，而是监督cost volume。那么如何构建训练监督的标签$Y$呢，当$t$帧上的$(i,j)$位置的目标出现在帧$t-\tau$上的$(k,l)$位置时，令$Y_{i j k l}=1$，否则$Y_{i j k l}=0$。CVA的训练损失记focal loss形式的逻辑回归损失来计算，如下式所示。下式中的$\alpha_{1}=\left(1-C_{i, j, l}^{W}\right)^{\beta}$ 和$\alpha_{2}=\left(1-C_{i, j, k}^{H}\right)^{\beta}$，其中的$\beta$为focal loss中的超参数。

$$
L_{C V A}=\frac{-1}{\sum_{i j k l} Y_{i j k l}} \sum_{i j k l}\left\{\begin{array}{cl}
\alpha_{1} \log \left(C_{i, j, l}^{W}\right)  +\alpha_{2} \log \left(C_{i, j, k}^{H}\right) &  { if } Y_{i j k l}=1 \\
0 & \text { otherwise }
\end{array}\right.
$$

由于$C_{i, j, l}^{W}$和$C_{i, j, k}^{H}$是通过softmax计算得到的，它们不仅仅包含点$(i,j)$和点$(k,l)$之间的嵌入相似度，还包含$(i,j)$和之前帧上所有点的相似度。也就是说，当$C_{i, j, l}^{W}$和$C_{i, j, k}^{H}$被优化到接近1时，它强制一个目标不仅接近前一帧中的自己，而且排斥其他目标和背景区域。

综上所述，不同于传统的ReID损失，CVA损失不仅仅要求学习的embedding考虑类内差异，也要求其考虑类间差异。这种处理方式不会损害到detection任务的学习。此外，由于tracking offset是基于外观相似度计算得到的，所以它能在较大的运动范围内跟踪目标，因此也能作为非常有效的运动线索。而且，同时使用外观嵌入和跟踪偏移，可以保证更加准确的数据关联。

### 运动指导的特征变换

上一节对应的CVA模块，本节所讲的MFW模块则根据CVA生成的跟踪偏移$\boldsymbol{O}^C$来将跟踪线索从特征图$\boldsymbol{f}^{t-\tau}$变换传播到当前帧来完善增强$\boldsymbol{f}^{t}$。为了实现这个目的，作者通过单个可变形卷积来进行高效的时序传播，继而聚合传播的特征来增强$\boldsymbol{f}^{t}$。

首先是如何计算传播的特征。首先，记$\boldsymbol{O}^{D} \in \mathbb{R}^{H_{F} \times W_{F} \times 2 K^{2}}$为DCN两个方向的输入偏移，取$K=3$作为DCN核的宽和高。为了生成$\boldsymbol{O}^{D}$，首先经过3x3卷积对$\boldsymbol{O}^{C}$进行变换，变换为$\gamma(\cdot)$，对应pipeline图上的红色部分。为了获得更多的运动线索，采用$\boldsymbol{f}^{t}-\boldsymbol{f}^{t-\tau}$作为$\gamma(\cdot)$的输入。考虑到检测核分割任务是基于目标中心特征进行的，相比于直接变换$\boldsymbol{f}^{t-\tau}$，作者这里将其先计算为一个中心注意力图$\overline{\boldsymbol{f}}^{t-\tau} \in \mathbb{R}^{H_{F} \times W_{F} \times 64}$，其计算公式如下。

$$
\overline{\boldsymbol{f}}_{q}^{t-\tau}=\boldsymbol{f}_{q}^{t-\tau} \circ \boldsymbol{P}_{a g n}^{t-\tau}, \quad q=1,2, \ldots, 64
$$

上式中的$q$为通道索引，$\circ$表示Hadamard积，$\boldsymbol{P}_{a g n}^{t-\tau} \in \mathbb{R}^{H_{F} \times W_{F} \times 1}$则是类别无关的中心热度图，它通过CenterNet中类似的$\boldsymbol{P}^{t-\tau}$获得，然后通过DCN可以计算传播特征$\hat{\boldsymbol{f}}^{t-\tau}=D C N\left(\boldsymbol{O}^{D}, \bar{f}^{t-\tau}\right) \in \mathbb{R}^{H_{F}} \times W_{F} \times 64$。

下面，考虑到当前帧目标可能遮挡或者模糊，之前帧传播的特征可以融合到当前帧特征上来增强表示，考虑当前帧特征为$\boldsymbol{f}^{t}$，之前帧传播的特征为$\hat{\boldsymbol{f}}^{t-\tau}$，加权求和的增强特征通过下式计算得到，$\boldsymbol{w}^{t} \in \mathbb{R}^{H_{F} \times W_{F} \times 1}$为帧$t$的自适应权重且$\sum_{\tau=0}^{T} \boldsymbol{w}_{i, j}^{t-\tau}=1$，其中$T$表示用于聚合的之前帧数目，这里的$\boldsymbol{w}$通过两层卷积加softmax预测得到，且实验表明加权求和比加权平均效果好一些。

$$
\tilde{\boldsymbol{f}}_{q}^{t}=\boldsymbol{w}^{t} \circ \boldsymbol{f}_{q}^{t}+\sum_{\tau=1}^{T} \boldsymbol{w}^{t-\tau} \circ \hat{\boldsymbol{f}}_{q}^{t-\tau}, \quad q=1,2, \ldots, 64
$$

融合后的特征$\tilde{\boldsymbol{f}}^{t}$被用于后续的head部分，生成检测框或者掩膜完成检测和分割任务。因此，相比于CenterNet其实整个TraDeS只多了一个CVA损失，总损失为$L=L_{C V A}+L_{d e t}+L_{\text {mask }}$。

### 轨迹生成

TraDeS可以适应三种不同的head以处理检测和分割任务，对于2D和3D检测，head设计和CenterNet类似，对于分割任务则和CondInst类似。

下面聊聊这篇文章设计的两轮数据关联，对于一个检测框或者分割部分$d$，处在位置$(i,j)$，首先进行**DA-Round(1)**，首先将其和$t-1$帧上未匹配的检测框进行关联，不过该检测必须在以$r$为半径的中心点在$(i, j)+\boldsymbol{O}_{i, j}^{C}$的范围内，其中的$r$是检测框宽高的几何平均值。这里的$\boldsymbol{O}_{i, j}^{C}$仅仅表示图像$\boldsymbol{I}^{t}$和$\boldsymbol{I}^{t-1}$的跟踪偏移。接着，进行**DA-Round(2)**，这里考虑$d$在第一轮没有匹配上任何目标的情况，将其embedding$\boldsymbol{e}_{i, j}^{t}$与未匹配的或者历史轨迹段的embedding计算余弦相似度，$d$将和具有最高相似度且高于某个阈值的轨迹段关联上。第二轮匹配适用于长期关联，如果两轮$d$都没有匹配上，则将其视为新轨迹的产生。

## 实验

首先是消融实验部分，作者先是验证了各个模块的有效性，实验证明，CVA模块和MFW模块都是行之有效的，具体对比如下图。

![](https://i.loli.net/2021/03/22/jpbLlwICVJ8qUck.png)

之后，是2D任务（MOT数据集）、3D任务（nuScenes数据集）和分割任务（MOTS、YouTube-VIS数据集）与SOTA方法的比较，可以看到TraDeS都是具有很大的优势的。

![](https://i.loli.net/2021/03/22/1BwWNDXJRnMHk9p.png)

![](https://i.loli.net/2021/03/22/eGnqjsCcropVdmX.png)

![](https://i.loli.net/2021/03/22/24HqThm3ykp9Qez.png)

![](https://i.loli.net/2021/03/22/yxueKr7qUSpVoJs.png)

## 总结

论文作者设计了一种新的online MOT框架，它是基于JDT范式的，致力于让跟踪辅助检测任务的进行，设计的CVA模块和MFW模块都有效改善了JDT范式目前的问题，取得了相当不错的表现。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。

