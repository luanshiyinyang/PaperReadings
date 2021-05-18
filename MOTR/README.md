# MOTR解读

> 最新的一篇MOT领域基于Transformer的工作，是第一个真正严格意义上端到端的多目标跟踪框架，超越了此前的TransTrack和TrackFormer等工作。

## 简介

多目标跟踪的关键挑战在于轨迹上目标的时序建模，而现有的TBD方法大多采用简单的启发式策略，如空间和外观相似度。尽管这些方法具有通用性，但它们过于简单，不足以对复杂的变化进行建模，例如通过遮挡进行跟踪。 **本质上，现有方法缺乏时间建模的能力。** 这篇论文中，作者提出了MOTR，这是一个真正的完全端到端的跟踪框架。MOTR能够学习建模目标的长程时间变化，它隐式地进行时间关联，并避免了以前的显式启发式策略。基于Transformer和DETR，MOTR引入了track query这个概念，一个track query负责建模一个目标的整个轨迹，它可以在帧间传输并更新从而无缝完成目标检测和跟踪任务。时间聚合网络（temporal aggregation network，TAN）配合多帧训练被用来建模长程时间关系。实验结果表明MOTR达到了SOTA效果。

- 论文标题

    MOTR: End-to-End Multiple-Object Tracking with TRansformer
- 论文地址

    http://arxiv.org/abs/2105.03247
- 论文源码

    https://github.com/megvii-model/MOTR

## 介绍

多目标跟踪是在视频序列的每一帧中定位所有的目标并确定它们的移动轨迹的一个任务。多目标跟踪是极具挑战的一个任务，因为每帧中的目标都可能因为环境的变化而被遮挡，而且跟踪器要想进行长期跟踪或者低帧率的跟踪是比较困难的。这些复杂多样的跟踪场景为MOT方法的设计带来了诸多挑战。

现有的多目标跟踪方法基本上都遵循tracking-by-detection（TBD）范式，它将轨迹的生成分为两个步骤：**目标定位**和**时序关联**。对目标定位而言，使用检测器逐帧检测目标即可。而对于时序关联，现有的方法要么使用空间相似性（即基于IoU关联）要么使用外观相似性（即基于ReID关联）。对于基于IoU的方法，计算两帧检测框的两两之间的IoU矩阵，若两个目标之间的IoU高于某个阈值则赋予同一个ID，如下图的(a)所示。基于ReID的方法思路类似，两帧之间目标两两计算特征相似度，具有最高相似度的两个目标赋予同一个ID，不过，单独训练一个检测器和ReID模型代价太大，最近的主流思路是一个模型联合训练检测和ReID分支，这类方法称为JDT（joint detection and tracking）方法，如下图(b)所示。

![](https://i.loli.net/2021/05/18/X4mROEBF2eo8ivA.png)

上述的时序关联方法都是启发式的，并且是相当简单的，因此它们难以建模长时间的物体复杂的空间和外观变化。本质上看，其不具备对时间变化建模的能力，这和深度学习“端到端学习”的理念是不一致的。这些方法也许大多数情况下工作正常，但是对于一些挑战性的场景缺乏鲁棒性，这些场景下IoU和外观是不可信的，而这些复杂场景才是MOT任务的关键。因此，作者为了解决这个问题，构建了一个不需要任何数据关联处理的端到端跟踪框架。

最近，DETR提出了一套端到端目标检测的策略，它开创性地提出了“object query”这个概念，这是目标的显式解耦表示（representation），这种表示简化了Transformer框架的学习。首次启发，论文作者拓展“object query”到目标跟踪领域，形成了“track query”，每个track query负责预测一个对象的完整轨迹。如上图(c)所示，和分类和回归分支并行，MOTR对每一帧预测track query集合，这个track query集输入到decoder网络中以产生当前帧的跟踪预测和更新的track query，更新后的track query传递到下一帧的decoder中。这个query传递和预测的处理过程会在整个视频序列上一帧帧重复进行，因此称为连续query传递（continuous query passing）。由于一个query一旦和某个目标匹配，这个query就会一直跟随这个目标，因此continuous query passing可以很自然地消除数据关联和一些手工操作，如NMS。

接着，为了建模长期时间关系，作者进一步引入了多帧训练以及时间聚合网络（TAN），TAN建立了一个query memory bank来收集历史帧中和已跟踪目标对应的query，当前帧的track query会和memory 板块中的每个query通过多头注意力进行交互。

作者认为，MOTR是简单且高效的，并且严格意义上它是第一个端到端的MOT框架。通过多帧时间聚合学习的track query有强大的时间建模能力，因而MOTR不需要时间关联处理或者手工操作。相反，此前基于Transformer的工作并非完全端到端的，它们依然需要IoU匹配或者Tracking NMS等操作。在MOT16和MOT17上的实验结果表明，MOTR达到SOTA性能并且比之前的方法具有更低的复杂度。

## MOTR

### Deformable DETR

DETR是Transformer在目标检测领域的经典成功案例之一。在DETR中，object query是固定数量的可学习位置嵌入，表示可能的目标的proposal。一个object query只会对应一个目标，这种对应关系是通过二分图匹配构建的。考虑到DETR较高的复杂度和较慢的收敛速度，Deformable DETR将Transformer中的self-attention替换为了multi-scale deformable attention。为了表示object query如何通过decoder与特征交互，作者重新设计了Deformable DETR的解码器。

令$q \in R^{C}$表示object query集且$f \in R^{C}$表示从encoder获得的特征图，这里的$C$表示特征的维度。解码器的处理过程可以描述如下式，其中$k \in 1, \ldots, K$（$K$为decoder的层数），$q^k$表示第$k$个decoder层的output query。$G_{sa}$表示DETR中的self-attention操作，$G_{ca}$则表示multi-scale deformable attention操作。

$$
q^{k}=G_{c a}\left(G_{s a}\left(q^{k-1}\right), f\right)
$$

### Framework

MOTR中，作者引入track query和continuous query passing来进行端到端的跟踪，temporal aggregation network进一步被提出用来增强多帧的时间信息。下面就逐一阐述这几个核心模块。

#### **End-to-End Track Query Passing**

**Track Query:**

![](https://i.loli.net/2021/05/18/ExyRcAlIL1Q29ph.png)

DETR中的object query并不负责预测特定的目标，因此一个object query在不同的帧上可能预测不同的目标。如上图所示，DETR对MOT数据集进行检测，同一个object query在两个不同帧中预测的目标是不一样的（如上图中(a)绿色框所示），因此将query的id用于轨迹关联是不合理的。（**作者这里其实想表述的是，object query只是对一个区域目标负责，并不具体到id级别。**）

针对上述问题，作者拓展object query进行跟踪任务，拓展后的称为track query，每个track query负责一个目标的整个轨迹的预测。如上图(b)所示，一旦一个track query在某一帧和某个目标匹配之后，这个track query将一直预测这个目标直到该目标消失。因此，同一个track query的所有预测结果就形成了一个目标的轨迹，不需要显式的数据关联步骤，至于track query之间的order-preserving则是依据特定目标的track query监督的。当然，还有一个问题，那就是新目标的产生，因此作者进一步提出了empty query来负责新目标的检测。

**Continuous Query Passing:**

基于上述提出的track query，作者进一步提出了continuous query passing机制，在这个机制下，track query随着帧的变化而改变representation和匹配的目标的localization。然后，作者提出了基于continuous query passing的端到端MOT框架，即MOTR。在MOTR中，目标的时间变化建模是编码器的多头注意力隐式学习的，因此不需要显式的数据关联。

![](https://i.loli.net/2021/05/18/q8WrZLundsOxXg2.png)

MOTR的整体结构如上图所示，视频序列首先会送入CNN中，随后进入Deformable DETR的编码器提取基本特征$f=\left\{f_{0}, f_{1}, \ldots, f_{N}\right\}$，这里的$f_0$表示第$T_0$帧的特征。对第$T_0$帧而言，特征$f_0$和empty query set $q_e$被输入到decoder网络中定位所有初始化目标并生成原始的track query set $q_{ot}^1$。**到这里，对应到上图就是最左侧的那部分，需要注意，这里的continuous query passing其实将$T_0$的$q^0_{ot}$传递到了下一帧。** $q^0_{ot}$通过QIM模块生成了下一帧（
$T_1$）的track query输入，也就是图上对应的$q^1_t$。所以从上图可以看出来，整个模型迭代式地处理每一帧$T_i$（$i \in[1, N]$），QIM根据上一帧地输出产生的$q^i_t$会和empty query set $q_e$级联到一起，级联后的query set $q_e$回合特征$f_i$一起送入decoder中直接产生当前帧的预测结果，并且更新query set $q_{ot}^{i+1}$将其送入到下一帧。

#### **Query Interaction Module**

在上面的叙述中，QIM负责接受上一帧的track query输出并生成当前帧的track query输入，在这一节将具体阐述Query Interaction Module（QIM）。QIM主要包括目标进出机制（object entrance and exit mechanism）和时间聚合网络（temporal aggregation network）。

**Object Entrance and Exit:** 首先来看目标进出机制，我们知道，每个track query表示一个完整轨迹，然而，一些目标可能在中间某一帧出现或者消失，因此MOTR需要输出一些边界框$\left\{b o x_{i}, \ldots, b o x_{j}\right\}$假定目标在$T_i$帧出现但在$T_j$帧消失。

MOTR是如何处理目标进出的情况呢？在训练时，track query的学习可以通过二分图匹配的GT来监督。但是，在推理时，使用跟踪得分预测来决定一个轨迹的出现和消失。来看下图，这是QIM的结构图，对$T_i$帧而言，track query set $q^i_t$通过QIM从$T_{i-1}$帧生成，然后和empty query set $q_e$级联到一起，级联的结果继而输入到decoder并产生原始的包含跟踪得分的track query set $q_{ot}^i$。$q_{ot}^i$随机被分割为两个query set，即$q_{e n}^{i}=q_{\text {ot }}^{i}\left[: d_{e}\right]$和$q_{c e}^{i}=q_{o t}^{i}\left[d_{e}:\right]$，这里的$d_e$是$q_e$中query的数量。$q_{en}^i$包含进入的目标而$q_{ce}^i$包含跟踪着的和离开的目标。对目标的进入，$q_{en}^i$中的query（下图的“3”）如果跟踪得分大于进入阈值$\tau_{e n}$则被保留，其余的被移除。表示如下式，$s_k$表示$q_{en}^i$的第$k$个query $q_k$的分类得分。

$$
\bar{q}_{e n}^{i}=\left\{q_{k} \in q_{e n}^{i} \mid s_{k}>\tau_{e n}\right\}
$$

![](https://i.loli.net/2021/05/18/cFhCkfBerJE7URA.png)

对目标的退出，$q_{ce}^i$的query（上图的“2”）如果跟踪得分连续M帧低于退出阈值$\tau_{ex}$，将被移除，剩下的query（上图的“1”）则被保留，式子如下，实验中设置$\tau_{e n}=0.8, \tau_{e x}=0.6$且$M=5$。

$$
\bar{q}_{c}^{i}=\left\{q_{k} \in q_{c e}^{i} \mid \max \left\{s_{k}^{i}, \ldots, s_{k}^{i-M}\right\}>\tau_{e x}\right\}
$$

**Temporal Aggregation Network:** 接着，来看时间聚合网络TAN，TAN的目的是增强时间相关性并为跟踪目标提供上下文先验信息。还是上面那张图，首先构建了一个memory bank以便于跟踪目标的时间聚合。memory bank $q_{bank} = \left\{\widetilde{q}_{c}^{i-M}, \ldots, \tilde{q}_{c}^{i}\right\}$首先收集历史query（也就是和已跟踪目标相关联的query），然后这些memory bank中的query将被级联到一起，如下式所示。

$$
t g t=\widetilde{q}_{c}^{i-M} \oplus \cdots \tilde{q}_{c}^{i-1} \oplus \tilde{q}_{c}^{i}
$$

级联之后，这些query被送入多头注意力模块，它们既是value也是key，继而生成注意力权重。$\bar{q}_{c}^{\imath}$被作为MHA（多头注意力）的query，因此有下面的点积注意力的式子，其中的$\sigma_s$表示softmax函数而$d$表示track query的维度。

$$
q_{s a}^{i}=\sigma_{s}\left(\frac{t g t \cdot \operatorname{tg} t^{T}}{\sqrt{d}}\right) \cdot \bar{q}_{c}^{i}
$$

之后，$q_{sa}^i$被FFN网络进一步调整，这里的FC表示线性投影层而LN表示层标准化，$\sigma_r$表示ReLU激活函数。

$$
\begin{array}{c}
t \tilde{g} t=L N\left(q_{s a}^{i}+\bar{q}_{c}^{i}\right) \\
\hat{q}_{c}^{i}=L N\left(F C\left(\sigma_{r}(F C(\tilde{g g} t))+t \tilde{g} t\right)\right.
\end{array}
$$

TAN的输出$\hat{q}_{c}$和$\bar{q}_{e n}^{i}$串联到一起来产生$T_{i+1}$帧的track query set $q_t^{i+1}$。

#### **Overall Optimization**

既然是端到端的结构，那么优化就显得非常重要，MOTR是以视频序列作为输入并且逐帧计算跟踪损失的，整体的跟踪损失是每一帧的损失的和并根据GT数目标准化的结果，如下式所示，这里的$N$表示视频序列的长度，$Y_{i}$和$\hat{Y}_{i}$分别表示$T_i$帧的预测和对应的GT。$V_{i}=V_{t}^{i}+V_{e}^{i}$表示$T_i$帧上所有的GT数目，$V_{t}^{i}$和$V_{e}^{i}$分别是已跟踪的目标数目和新轨迹的数目。

$$
L_{o t}(Y, \hat{Y})=\frac{\sum_{n=0}^{N}\left(L_{t}\left(Y_{i}, \hat{Y}_{i}\right)\right)}{\sum_{n=0}^{N}\left(V_{i}\right)}
$$

$L_t$是单帧的跟踪损失，和Deformable DETR中的检测损失是类似的，单帧损失$L_t$可以用下面的式子表示，其中$L_{c l s}$是focal loss，$L_{l_{1}}$表示L1损失而$L_{\text {giou }}$表示GIoU损失，$\lambda_{c l s}, \lambda_{l_{1}}$和$\lambda_{\text {giou }}$则是相应的权重系数。

$$
L_{t}\left(Y_{i}, \hat{Y}_{i}\right)=\lambda_{c l s} L_{c l s}+\lambda_{l_{1}} L_{l_{1}}+\lambda_{\text {giou }} L_{\text {giou }}
$$

MOTR的跟踪损失和Deformable DETR的检测损失主要的不同在于标签分配的方式不同，对检测损失而言，标签的分配完全由匈牙利的分配结果决定，但是对跟踪损失而言，每个track query负责特定的目标，因此其GT是由对应ID的目标决定的，对empty query而言，它的GT才是通过匈牙利匹配得到的。

## 实验

首先是和其他SOTA的比较，MOTR确实取得了相当不错的效果，相比此前基于Transformer的方法也有了不小的提高。

![](https://i.loli.net/2021/05/18/PNgZcRo3vquLsQ7.png)

也进行了一些模块的消融实验，如下。

![](https://i.loli.net/2021/05/18/JkIKzHyPi5D4dTc.png)

## 总结

论文基于Transfomer提出了一个真正的端到端的多目标跟踪框架MOTR并超越了之前的方法，该框架真正意义上不需要数据关联和NMS等后处理操作，并且在基准数据集上达到了SOTA效果，是很值得关注的工作。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。



