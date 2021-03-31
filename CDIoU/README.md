# CDIoU解读

## 简介

这篇文章针对之前基于IoU的边框评估和回归机制，在前人工作的基础上设计了Control Distance IoU（CDIoU）和Control Distance IoU loss（CDIoU loss），在几乎不增加计算量的前提下有效提升了模型性能，对常见的多个SOTA模型都有不错的涨点效果（如Faster R-CNN、YOLOv4、RetinaNet和ATSS）。

- 论文标题

    Control Distance IoU and Control Distance IoU Loss Function for Better Bounding Box Regression
- 论文地址

    http://arxiv.org/abs/2103.11696
- 论文源码

    https://github.com/Alan-D-Chen/CDIoU-CDIoUloss

## 介绍

近年来，针对设计精度更高速度更快的目标检测算法出现了大量的工作。新的数据增强、更深层的神经网络、更复杂的FPN结构乃至更多的训练迭代次数，这些策略不断刷新检测的SOTA表现。不可否认，这些方法确实获得了显著的成功，然而它们也伴随着大量的参数和计算消耗，这是有损算法的高效性的。这篇论文的作者则将关注的重心放在region proposals的评估系统和反馈机制上（说的通俗一点就是**IoU模块**和**损失函数**，它们组合在一起称为评估反馈模块）。

评估反馈模块有三个主要部件。1. 根据ground truth评估proposal的质量；2. 对同一个ground truth对应的一批proposal排序；3. 将proposal和ground truth之间的差距输入神经网络以便于下一个评估模型的修正。 既然这个评估反馈模块如此重要，那么它应当是高效且参数较少的。一个优秀的评估反馈模块应当满足下面三个条件。
1. 评估覆盖区域；
2. 良好的区分能力和差异程度的度量(有时理解为质心距离和长宽比)；
3. IoU计算可以与损失函数相关联。

此前，很多研究都比较关注于反馈机制的设计（即设计更好的损失函数），却忽略了评估系统的优化。因此这篇论文，作者提出了CDIoU和CDIoU loss，它们同等重要。CDIoU具有良好的连续性和可导性，通过统一测量RP（Region Proposal）和GT（Ground Truth）之间的距离，简化了计算，优化了DIoU和CIoU对质心距离和长宽比的计算，并快速完成了边框质量评估。CDIoU损失函数可以与CDIoU计算相关联，使反馈机制能够更准确地表征RP与GT之间的差异，从而使深度学习网络的目标函数更快收敛，提高整体效率。与传统的IoU模块和损失函数相比，CDIoU和CDIoU loss函数具有很强的适应性，在主流数据集上，CDIoU和CDIoU loss在几种不同的模型上都有显著的改进。

## 传统IoU和损失函数

在目标检测中，IoU的计算通常用来评估RP和GT之间的相似性，IoU计算值的高低通常也是正负样本挑选的依据。在评估反馈模块中，最具代表性的方法是IoU、GIoU、DIoU loss和CIoU loss，它们对目标检测的发展进程起到了巨大的推动作用，但是仍留有许多的优化空间。

### Traditional IoUs

IoU是一个常见的评估方法，如下图所示，四种情况下RP和GT之间的相对位置明显不同，我们人类可以很清晰地看出优劣，但是它们之间的IoU值是一样的。

![](https://i.loli.net/2021/03/31/OW3N4vRDKcMmLly.png)

基于原始的IoU，产生了很多从各个方面其针对其的优化，丰富了评估的维度。IoU只考虑了重叠区域，但是GIoU同时考虑了重叠区域和非重叠区域，引发了评估模块的进一步思考。虽然很遗憾，GIoU并没有考虑RP和GT之间的“**差异评估**”，这个差异评估包括中心点之间的距离（centroid）和长宽比（aspect ratio）。DIoU考虑了中心距离但是忽略了长宽比，如下面两图所示，DIoU并不能区分高瘦的proposal和矮胖的proposal之间的不同，并且给出了同样的计算结果，但是显然，它们是存在明显的好坏之分的。

![](https://i.loli.net/2021/03/31/9jGNMKZEY14r78Q.png)

![](https://i.loli.net/2021/03/31/ljMkvrp1BG3m4WK.png)

### Smooth L1 Loss and IoU Loss

Faster R-CNN提出了平滑损失初步解决了边框损失的表征问题。假定$x$为RP和GT之间的数值差，那么$L_1$和$L_2$损失的定义如下。

$$
\begin{array}{c}
\mathcal{L}_{1}=|x| \frac{d \mathcal{L}_{2}(x)}{x}=2 x \\
\\
\mathcal{L}_{2}=x^{2}
\end{array}
$$

它们的导数分别如下，从中可以发现，$L_1$损失的导数是常数，在训练的后期，当$x$非常小的时候，若学习率也是常数，那么损失函数将在某个稳定值附近波动导致难以收敛到更高的精度。而$L_2$损失的导数和输入$x$是正相关的，当训练初期$x$非常大的时候它的导数也很大使得训练非常不稳定。

$$
\begin{array}{c}
\frac{d \mathcal{L}_{1}(x)}{x}=\left\{\begin{array}{ll}
1 & , \text { if } x \geq 0 \\
-1 & , \text { otherswise }
\end{array}\right. \\
\\
\frac{d \mathcal{L}_{2}(x)}{x}=2 x
\end{array}
$$

$smooth L_1$则完美的解决了$L_1$和$L_2$的的缺陷，它的计算式和导数式如下。

$$
\begin{array}{ll}
\operatorname{smooth}_{\mathcal{L}_{1}}(x)=\left\{\begin{array}{ll}
0.5 x^{2} & , \text { if }|x|<1 \\
|x|-0.5 & , \text { otherswise }
\end{array}\right. \\
\\
\frac{d s m o o t h_{\mathcal{L}_{1}(x)}}{x}=\left\{\begin{array}{ll}
x & , \text { if }|x|<1 \\
\pm 1 & , \text { otherswise }
\end{array}\right.
\end{array}
$$

然而，实际的目标检测任务中，边框回归任务的损失表示如下，这里的$v=\left(v_{x}, v_{y}, v_{w}, v_{h}\right)$表示GT框的四个坐标而$t^{u}=\left(t_{x}^{u}, t_{y}^{u}, t_{w}^{u}, t_{h}^{u}\right)$表示预测框的坐标，下面的计算式表明总的边框损失其实是四个坐标损失分别计算然后加和得到的，这就会带来一些问题。

$$
\mathcal{L}_{l o c}\left(t^{u}, v\right)=\sum_{i \in\{x, y, w, h\}} \operatorname{smooth}_{\mathcal{L}_{1}}\left(t_{i}^{u}-v_{i}\right)
$$

**问题主要有两点。**
1. 分别计算四个坐标的损失然后加到一起作为边框损失，这是有假设前提的，那就是这四个坐标点彼此独立，然而实际上它们存在理所当然的相关性。正是这个原因导致平滑损失和IoU计算并不统一，从而导致反馈机制和评估系统存在一定的偏差。
2. 实际的评估指标是IoU，这和L1这类损失是不等价的，而如下的IoU loss又不能上面提到的“不同proposal相同反馈结果”这一现象。

$$
\begin{array}{l}
\mathcal{L}_{I o U}=-\ln (I o U) \\
\\
\mathcal{L}_{I o U}=1-I o U
\end{array}
$$

### GIoU and GIoU Loss

GIoU在不增加计算时间的前提下，初步优化了重叠区域IoU的计算并减小了计算误差，但是GIoU依然没有考虑上文所说的差异评估。GIoU和GIoU Loss的计算式如下所示。

$$
\begin{array}{c}
G I o U=I o U-\frac{|C \backslash(A \cup B)|}{|C|}, \\
\\
\mathcal{L}_{G I o U}=1-G I o U
\end{array}
$$

上式中的$C$表示能够包围两个框的最小闭包区域的面积，也就是下图中的灰色虚线框的区域。

![](https://i.loli.net/2021/03/31/J5WmMPNbBsEXgUV.png)

实践证明，其实GIoU loss仍然有着收敛缓慢、回归不准确的问题，研究发现，这是因为GIoU首先试图去扩大RP框使得其和GT框接近，然后使用IoU loss去最大化覆盖区域。当两个边框彼此包含的时候，GIoU loss就退化为了IoU loss，此时边框对其会更加困难，从而收敛缓慢。

### DIoU loss and CIoU loss

DIoU（Distance IoU）损失和CIoU（Complete IoU）损失极大地丰富了IoU计算结果的内涵，二者分别考虑了差异评估中的中心距离和长宽比。

首先来看DIoU损失，它的计算公式如下所示，其中$b$和$b^{gt}$分别表示预测框和真实框的中心点，$\rho^2()$表示两点之间的欧氏距离，$c$则表示能够同时包含预测框和真实框之间的最小闭包区域的对角线距离。

$$
\mathcal{L}_{D I o U}=1-I o U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}
$$

然而，按照这个计算方式，DIoU loss并不能区分多个中心点重合的proposal谁与GT更加相似。当两个框完全重合时，存在$\mathcal{L}_{I o U}=\mathcal{L}_{G I o U}=\mathcal{L}_{D I o U}=0$，当两个框不相交时，它又不能准确区分proposal，因为此时存在$\mathcal{L}_{G I o U}=\mathcal{L}_{D I o U} \rightarrow 2$。

如下图所示，由于DIoU的惩罚项是中心点距离，所以只要proposal的中心在圆O的弧上，那么惩罚项是一模一样的，这就导致DIoU丧失了评估系统的准确性。

![](https://i.loli.net/2021/03/31/KWp9FvUrJOQf5TP.png)

因为上面的问题存在，在DIoU loss的基础上继续添加长宽比，从而形成了下式的CIoU损失，相比于上面的CIoU损失它多了一项，即$\alpha v$，下面详细解释一下这一项。

$$
\mathcal{L}_{C I o U}=1-I o U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}+\alpha v
$$

首先，$\alpha$和$v$的计算如下，前者为权重系数（在非重叠情况下，对重叠面积因子给予较高的优先级），而$v$度量长宽比的相似性。

$$
\begin{array}{c}
\alpha=\frac{v}{(1-I o U)+v} \\
\\
v=\frac{4}{\pi^{2}}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right)^{2}
\end{array}
$$

由于计算CIoU损失涉及到反三角函数，通过对比实验可知，计算CIoU损失的过程比较耗时，最终会拖低整体训练时间，而带来的精度收益确不是很高，有点得不偿失，因此使用不是特别广泛。

### CDIoU and CDIoU Loss

基于之前IoU的种种问题，作者提出了CDIoU和CDIoU Loss，在不增加计算开销的前提下运行效率和精度显著提升。CDIoU损失收敛更快并且大大减少了计算复杂度。

CDIoU是RP和GT的一个新的评估方式，它不直接计算中心点距离或者形状相似性，如下图所示，其中的diou计算式如下，其中MBR为包围两个框的最小矩形区域，这个diou的计算和DIoU计算是类似的。

![](https://i.loli.net/2021/03/31/6LkzaVE5DXKb8RW.png)

$$
\begin{aligned}
\text { diou } &=\frac{\|R P-G T\|_{2}}{4 \mathrm{MBR}^{\prime} s \text { diagonal }} \\
&=\frac{A E+B F+C G+D H}{4 W Y}
\end{aligned}
$$

然后，就可以定义CDIoU了，如下式所示，虽然CDIoU没有显式考虑中心点距离和长宽比，但是最终的计算结果反映了RP与GT的差异程度，CDIoU值越高，差异程度越低;CDIoU的值越高，相似性越高。

$$
C D I o U=I o U+\lambda(1-\text { diou })
$$

随后，可以定义CDIoU loss，如下式，通过观察这个公式，可以直观地感觉到，在反向传播之后，深度学习模型倾向于将RP的四个顶点拉向GT的四个顶点，直到它们重叠为止，具体算法如下图所示。

$$
\mathcal{L}_{C D I o U}=\mathcal{L}_{I o U_{s}}+\text { diou }
$$

![](https://i.loli.net/2021/03/31/KTwXb8CYo9asfhc.png)

![](https://i.loli.net/2021/03/31/tGHMjpliSefXLTF.png)

CDIoU和CDIoU loss具有如下特性：第一， $0 \leq$ diou $<1$，$\mathcal{L}_{I o U_{s}}$是$\mathcal{L}_{C D I o U}$的下界。第二， diou拓展了$\mathcal{L}_{C D I o U}$的范围，当$\mathcal{L}_{I o U_{s}}=\mathcal{L}_{I o U}$ 有$0 \leq \mathcal{L}_{C D I o U}<2$，当$\mathcal{L}_{I o U_{s}}=\mathcal{L}_{G I o U}$有$-1 \leq \mathcal{L}_{C D I o U}<2$。

## 实验

作者为了验证CDIoU和CDIoU的有效性很泛化性，在多个方法上尽量不使用trick的情况下进行了对比实验，实验配置可以参照原文，这里不多赘述，从下表的消融实验不难发现，CDIoU在几乎对速度没有影响的前提下，对多个模型都有不错的AP提升。

![](https://i.loli.net/2021/03/31/bsBP4rdekZ2OwyL.png)

当然，致敬前辈，既然是改进，总要都比一比，下表就是和其他IoU loss的比较，速度精度都有改善。

![](https://i.loli.net/2021/03/31/yKhTbfeOjL8cm2q.png)

后面作者还有一些实验分析以及一些涨点trick的分享，我这里就略过了，感兴趣可以查看原论文。


## 总结

虽然作者没有把CDIoU的由来说得特别清楚，但是从结果上看，本文设计的CDIoU和CDIoU loss几乎对当前主流的检测器无痛涨点，是很值得关注的工作。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。