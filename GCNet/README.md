# GCNet解读

> 收录于ICCV2021的一篇多目标跟踪的工作，提出了一个比较简洁的范式，虽然由于侧重点在自动驾驶上因此没有在MOT Challenge赛道上比拼，不过在其他基准数据集上的效果还是很不错的。和另一篇新的SOTA方法TransCenter类似，它们都是基于CenterNet的新SOTA，这也一定程度上验证了以点为目标的检测确实比以框为目标的检测更适合于遮挡场景下的多目标跟踪。

## 简介

此前的多目标跟踪（MOT）方法多遵循TBD（Tracking by Detection）范式，将检测和跟踪分为两个组件。早期的TBD方法针对检测任务和跟踪任务需要分别进行特征提取，典型的如DeepSORT方法。近段时间，不少方法将两个任务的特征提取融入一个网络中形成了JDT（Joint Detection and Tracking）范式，典型的是JDE方法。不过，JDE方法的跟踪部分依然依赖于数据关联并且需要复杂的后处理来进行轨迹的生命周期管理，因此它们并没有将检测和跟踪组合得很好。因此，GCNet（Global Correlation Network，全局相关网络）被提了出来，它以端到端得方式实现了联合多目标检测和跟踪。与大多数目标检测方法不同的是，GCNet引入了全局相关层来回归目标的绝对大小和边框坐标，而不是偏移量预测。GCNet的pipeline概念上是非常简洁的，它不需要非极大值抑制、数据关联以及其他复杂的跟踪策略。在UA-DETRAC这个车辆跟踪数据集上进行评估，GCNet达到了SOTA的检测和跟踪表现。


- 论文标题

    Global Correlation Network: End-to-End Joint Multi-Object Detection and Tracking
- 论文地址

    http://arxiv.org/abs/2103.12511
- 论文源码

    暂未开源

## 介绍

多目标跟踪是计算机视觉中的一个基本问题，它的目的是从连续的图像帧中得出所有感兴趣的目标的轨迹。MOT具有广泛的应用场景，如自动驾驶、运动姿态分析、交通监控等，因此近年来受到越来越多的关注。

传统的多目标跟踪方法遵循TBD范式，分为两个子模块，分别是检测和跟踪。随着目标检测的快速发展，这些算法取得了良好的性能，几乎统治了整个MOT领域。TBD范式的跟踪模块主要有三个部分：特征提取、数据关联和轨迹生命周期管理。早期的跟踪方法使用如位置、形状和速度这类的简单特征来进行数据关联，不过这些特征存在明显的缺陷。因此，后来的方法开始采用外观特征，特别是深度神经网络提取的高级语义特征。外观特征大大提高了关联的准确性和鲁棒性，但是也带来的比较大的计算开销增长。因此，当前这个环境下，一些MOT方法将特征提取集成到了检测模块内，这就形成了JDT范式，其中比较主流的思路就是添加一个额外的ReID head来获得实例级别的特征以进行数据关联。虽然这类方法减少了计算的开销，但是数据关联仍旧需要进行运动预测、设置复杂的跟踪策略，这就会导致引入太多的超参数和繁琐的推理流程。

在这篇论文中，作者提出了一种新的端到端联合检测和跟踪的网络，该网络通过同样的方式完成边界框回归和跟踪，这种方式称为全局相关操作（global correlation）。我们都知道边界框的回归通常利用局部特征去估计anchor与ground truth之间的偏移量，或者估计框大小以及关键点与特征位置之间的偏移量。GCNet中，网络直接回归边界框的绝对坐标和尺寸而不是相对坐标或偏移量，但是传统卷积神经网络中，由于感受野限制，局部特征无法包含全局信息，因此回归全局的绝对坐标是比较困难的。自注意力机制允许每个位置的特征包含全局的信息，但是它的计算复杂度太高以至于并不能用于高分辨率特征图，因此作者提出了**global correlation layer**来编码全局信息到每个位置的特征里。

![](https://i.loli.net/2021/04/08/FCk19PQUIx3r4zY.png)

关于global correlation layer的细节我会在下一节详细讲解，GCNet通过精细的设计联合实现了检测和跟踪并且不需要复杂的后处理步骤，是一个非常简洁的pipeline。由于其关注的主要是自动驾驶领域，因此作者只在自动驾驶数据集UA-DETRAC上验证了方法的可行性，检测方面获得了74.04\%的AP和36的FPS，在跟踪上获得了19.10\%的PR-MOTA和34的FPS，上图是一些跟踪结果的示例。

## GCNet

GCNet设计用于解决online MOT问题，为了后面叙述方便，这里首先给定一些说明。在$t$时刻，现有的目标轨迹集为$\left\{T_{1}, T_{2}, \ldots, T_{n}\right\}$，其中每个$T_{i}=\left[B_{i, 1}, B_{i, 2}, \ldots, B_{i, t-1}\right]$表示第$i$个目标在各个时刻的边框信息。这里的$B_{i j}$表示第$i$个目标在时间$j$时刻的边界框。考虑当前帧图像$I_{t} \in R^{h \times w \times 3}$，应当将当前帧上的目标框$B_{x, t}$安排给历史轨迹，或者成为新的新的轨迹。下文会详细讲述整个GCNet的整个pipeline。

### Global Correlation Layer

首先，来看作者提出的global correlation layer是如何对全局信息进行编码的。对于特征图$F \in R^{h \times w \times c}$，通过两个线性变换计算特征图$\mathbf{Q}$和特征图$\mathbf{K}$，计算式如下。

$$
Q_{i j}=W_{q} F_{i j}, K_{i j}=W_{k} F_{i j}
$$

上式中的下标表示行列的位置，如$X_{i j} \in R^{c}$表示$X$在第$i$行第$j$列位置的特征向量。

接着，对每个特征向量$Q_{ij}$，计算其与所有的$K_{ij}$之间的余弦距离，然后再通过矩阵$W$进行线性变换即可得到相关向量$C_{ij} \in R^{c'}$，它的形式如下。

$$
C_{i j}=W \cdot \text { flatten }\left(\left[\begin{array}{ccc}
\frac{Q_{i j} K_{11}}{\left|Q_{i j}\right|\left|K_{11}\right|} & \cdots & \frac{Q_{i j} K_{1 w}}{\left|Q_{i j} \| K_{1 w}\right|} \\
\vdots & \ddots & \vdots \\
\frac{Q_{i j} K_{h 1}}{\left|Q_{i j}\right|\left|K_{h 1}\right|} & \cdots & \frac{Q_{i j} K_{h w}}{\left|Q_{i j} \| K_{h w}\right|}
\end{array}\right]\right)
$$

因此，每个$C_{ij}$都编码了局部特征向量$Q_{ij}$和全局特征图$K$之间的相关性，所以它可以用于图像中相应位置的目标的绝对边界框。所有相关向量可以构建一个相关特征图$C \in R^{h \times w \times c^{\prime}}$，因此可以通过简单的1x1卷积来得到边框预测$B \in R^{h \times w \times 4}$。这个Global Correlation Layer操作在GCNet的检测和跟踪结构中被使用，不过，在检测时，$Q$和$K$来自同一帧图像，而在跟踪时，$Q$来自前一帧图像$K$来自当前帧图像。

**这个Global Correlation Layer操作，其实是作者定义的一套计算全局相关性特征图的计算方式，下面的模块中会用到。**

### Global Correlation Network

整个GCNet分为**检测模块（Detection module）和跟踪模块（Tracking module）** 两部分，我们首先来看**检测模块**，它的结构如下图所示，包含三个部分分别是backbone、分类分支、回归分支。backbone用于高级语义特征的提取，由于分类分支采用的是和CenterNet一样的思路，特征图的每个位置对应一个目标的中心点，因此特征图的分辨率其实对网络的性能影响非常大。为了得到高分辨率大感受野的特征图，作者采用和FPN一样的跳跃连接结构，但是只输出最精细级别的特征图$F$，它的尺寸为$h^{\prime} \times w^{\prime} \times c$，等价于$\frac{h}{8} \times \frac{w}{8} \times c$，这里的$h$和$w$表示原始图像的高和宽，这个分辨率是DETR的四倍。

![](https://i.loli.net/2021/04/08/ArHthDQsUlFkMmO.png)

到这里，通过backbone得到了原始图像的特征图$F$，利用$F$上图最下面的分类分支通过卷积层运算得到分类置信度图$Y_{d} \in R^{h^{\prime} \times w^{\prime} \times n}$，它的数值均在0和1之间，因此它其实是一个热力图。$Y_{d}$的第$i$个通道的峰值对应于第$i$类的目标的中心。

然后，上面的回归分支输入backbone的特征图$F$和分类置信度图$Y_d$，然后计算三个特征图，即$Q$、$K$和$V$。它们的计算式如下式，这里$Conv(F,a,b,c)$表示核尺寸为$a$、步长为$b$、核数目为$c$的卷积层，$BN$表示batch
normalization层。

$$
\begin{aligned}
Q &=B N_{Q}\left(\operatorname{Conv}_{Q}(F, 1,1, c)+P\right) \\
K &=\operatorname{Gate}\left[B N_{K}\left(\operatorname{Conv}_{K}(F, 1,1, c)+P\right), Y_{d}\right] \\
V &=\operatorname{Conv}_{V}(F, 1,1, c)
\end{aligned}
$$

上式其实也对应上图的三条之路，应该是不难理解的，唯一的疑惑就是这里的$\operatorname{Gate}(X, Y)$的含义，它其实是下图所示的一种空间注意力结构，比较简单，我就不展开叙述了。此外，在$Q$和$K$计算之前，特征图$F$加上了一个shape一样的位置编码$P$，它的计算方式如下。

![](https://i.loli.net/2021/04/08/aqEvXwjiZYMfoRn.png)

![](https://i.loli.net/2021/04/08/nkBOElbF6hDPHVR.png)

通过这个位置编码，位置上距离较近的两个嵌入向量余弦相似度较大，距离较远的两个嵌入向量余弦相似度较小，因而可以减少跟踪时类似对象的负面影响。按照之前我们分析的Global Correlation Layer，有了$Q$和$K$就可以计算处相关性特征图$C$了，然后，就可以利用下面的式子计算最终的边界框$B_{d, i j}= \left[x_{i j}, y_{i j}, h_{i j}, w_{i j}\right]$。需要注意的是，GCNet直接回归目标边界框的绝对坐标和尺寸，这和当前主流的目标检测思路是不一样的，尤其和基于anchor的方法不同。

$$
B_{d, i j}=W \cdot B N\left(\left[\begin{array}{ll}
C_{i j} & V_{i j}
\end{array}\right]\right)
$$

接着，我们来看**跟踪模块**，其实所谓跟踪就是为当前帧的目标赋予ID的过程，它可以通过和历史目标关联也可以作为新的轨迹的开端。跟踪模块的结构如下图所示，它的输入由三部分组成，分别是当前帧的特征图$K$、当前帧的检测置信度图（即热力图）、历史轨迹的特征向量。跟踪模块为每个历史轨迹输出一个跟踪置信度和边界框。从下图不难看出，它的结构其实和检测模块是很类似的，它的大部分参数也是和检测模块共享的，除了计算跟踪置信度的全连接层（下图的绿色块）。跟踪框和检测框在表达上是一致的，都是形如$B_{i}=\left[x_{i}, y_{i}, h_{i}, w_{i}\right]$的绝对坐标的格式。跟踪置信度则表明这个目标仍旧在当前帧上的概率。此外，由于跟踪模块以逐目标的方式进行（也就是每个历史轨迹都会逐一处理到），因此它能够很自然地进行ID传递而不需要关联步骤，这和并行SOT方法类似。

![](https://i.loli.net/2021/04/08/cw7W2Mza1HEyrQk.png)

### Training

GCNet是可以端到端训练的，不过为了更好的效果，作者采用先训练检测模块再整个网络微调的二阶段训练思路。

分类分支的训练策略和CornerNet一致，GT热力图的产生方式如下式，通过2D高斯核函数产生GT heatmap $Y_{g t} \in R^{h^{\prime} \times w^{\prime} \times n}$ 。

$$
\begin{array}{l}
Y_{g t, i j k}=\max _{1 \leqslant n \leqslant N_{k}}\left(G_{i j n}\right) \\
G_{i j n}=\exp \left[-\frac{\left(i-x_{n}\right)^{2}}{2 \sigma_{x, n}^{2}}-\frac{\left(i-y_{n}\right)^{2}}{2 \sigma_{y, n}^{2}}\right]
\end{array}
$$

上面式子中的$N_k$表示类别$k$的目标数目，$\left[x_{n}, y_{n}\right]$则表示目标$n$的中心，$\sigma^2$依据目标的尺寸而定，$\sigma_x$和$\sigma_y$的表达式如下，其中的IoU阈值设置为0.3。

$$
\begin{aligned}
\sigma_{x} &=\frac{h(1- IoU\_threshold)}{3(1+IoU\_threshold)} \\
\sigma_{y} &=\frac{w(1-IoU\_threshold )}{3(1+IoU\_threshold )}
\end{aligned}
$$

分类损失是像素级的focal loss，式子如下。

$$
\begin{array}{c}
L_{d, c l a}=-\frac{1}{h^{\prime} w^{\prime} n} 
\sum_{i j k}\left\{\begin{array}{ll}
\left(1-Y_{d, i j k}\right)^{2} \log \left(Y_{d, i j k}\right), & Y_{g t, i j k}=1 \\
\left(1-Y_{g t, i j k}\right)^{2} Y_{d, i j k}^{2} \log \left(1-Y_{d, i j k}\right), & Y_{g t, i j k} \neq 1
\end{array}\right.
\end{array}
$$

回归损失采用CIoU loss，算式如下。

$$
L_{d, r e g}=\sum_{[i j]=1} \beta_{i j} \cdot L_{C I o U}\left(B_{g t, i j}, B_{d, i j}\right)
$$

这里的$[i j]=1$表示相应的$B_{d, i j}$被分配给一个GT框，只有当$G_{i j n}>0.3$ and $\sum_{n} G_{i j n}- \max _{n} G_{i j n}<0.3$满足时一个$B_{d, i j}$才会被分配给一个GT，表示如下。

$$
[i j]=\left\{\begin{array}{ll}
1, \exists_{n} G_{i j n}>0.3 \& \sum_{n} G_{i j n}-\max _{n} G_{i j n}<0.3 \\
0, & \text { otherwise }
\end{array}\right.
$$

并且对于$\max _{n} G_{i j n}=1$的$B_{ij}$，设置回归损失的权重为2，其余为1，这是为了加强中心点的边框精准度。

首先预训练检测模块，然后在整个网络上微调，微调这一步训练时，一次输入两个图像$I_{t-i}$和$I_t$，这里的$i$在1和5之间。损失包含两部分，分别是$I_{t-i}$的检测损失和两幅图像的跟踪损失，跟踪损失又由两项组成，回归CIoU损失和分类focal loss。跟踪的GT由目标的ID确定，当$I_{t-i}$中的$[i j] = 1$时，相对应的目标也存在$I_t$中，$B_{t, i j}$ 和 $Y_{t, i j}$为正例。最终的损失如下式所示。

$$
\operatorname{Loss}=L_{d, c l a}+L_{t, c l a}+0.1 \times\left(L_{d, r e g}+L_{t, r e g}\right)
$$

### Inference

![](https://i.loli.net/2021/04/08/SRC3nU1LzOBW2NP.png)

上图所示为整个推理算法，输入为连续的图像帧序列$I_{1} \sim I_{t}$，记录每个目标的轨迹$T_i$、置信度$Y_i$、特征向量及候选向量$\left[V_{i}, Q_{i}\right]$为四个集合，$\mathcal{T}, \mathcal{O}, \mathcal{Y}, \mathcal{C}$，它们均初始化为空。在每个时刻，对当前帧图像进行检测并在轨迹集合候选集之间进行跟踪。通过跟踪置信度来更新集合$\mathcal{Y}, \mathcal{C}$的所有置信度，其中$Y_{i}=\min \left(2 \times Y_{i} \times Y_{t, i}, 1.5\right)$。轨迹和候选的置信度低于$p_2$将会被删除，然后轨迹集、候选集和特征集也会相应更新。检测结果中，IoU高于$p_3$或者置信度低于$p_2$的将被忽略。对于剩下的检测框，如果置信度高于阈值$p_1$则会用于生成新的轨迹，此外的框加入候选集合$\mathcal{C}$中。这里不难发现，整个检测和跟踪都能以稀疏模式完成，因此整体计算复杂度并不高。

## 实验

关于一些实验的配置这里忽略，感兴趣的可以查看原论文。作者首先是对各个模块的有效性进行了消融实验，得出的结果如下表。可以看到，无论是Gate还是V亦或是位置编码，都是实实在在有效的组件。

![](https://i.loli.net/2021/04/08/gDduYEonAKmqxsL.png)

在基准数据集上和其他方法比较检测的结果如下，改方法优势还是比较明显的。

![](https://i.loli.net/2021/04/08/fQX6VdWeF3a7AiS.png)

GCNet在多目标跟踪任务上的效果如下图，达到SOTA水准，且速度很快，达到34FPS。

![](https://i.loli.net/2021/04/08/a1TNIYe2KuCw8Js.png)

## 总结

这篇论文提出了一种新的联合检测和跟踪的框架，GCNet，通过引入全局相关计算操作来捕获全局信息实现绝对尺寸和坐标的回归，并且通过双帧输入无需复杂的关联策略即可达到SOTA表现。实验表明，GCNet速度较快且达到实时应用的需求，满足自动驾驶等场景的应用。总的来说，是JDT领域的一个新的方法，也留有了不少改进的空间，是一个该方向强有力的baseline。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。