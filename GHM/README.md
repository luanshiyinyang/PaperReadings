# GHM解读
> 收录于AAAI2019 oral的一篇文章，主要是解决目标检测中的**样本不平衡问题**，包括正负样本不平衡、难易样本不平衡，和著名的Focal Loss类似，也是基于交叉熵做的改进。此外，本文成文过程参考了[知乎上一个精简的分析](https://zhuanlan.zhihu.com/p/80594704)，欢迎访问。

## 简介

单阶段跟踪以一个更优雅的方式对待目标检测问题，然而它也存在困扰已久的问题，那就是样本的不均衡从而导致模型训练效果不好，这包括正负样本的不平衡和难易样本的不平衡。这两种不平衡本质上都可以从梯度的层面解释，因此作者提出了GHM（梯度调和机制， gradient harmonizing mechanism）来处理这一现象，以此为基础构建的GHM分类损失（GHM-C）和GHM回归损失（GHM-R）可以轻松嵌入到如交叉熵的分类损失和如Smooth L1的回归损失中，这两种损失分别用于anchor的分类和边界框的修正，实验表明，在不经过费力挑战超参数的前提下，GHM-C和GHM-R可以为单阶段检测器带来实质性的改进，并且超过了使用Focal Loss和Smooth L1的SOTA方法。

- 论文标题

    Gradient Harmonized Single-stage Detector
- 论文地址

    https://arxiv.org/abs/1811.05181
- 论文源码

    https://github.com/libuyu/GHM_Detection

## 介绍

单阶段检测的出现为目标检测带来了一种更加高效且优雅的范式，但是单阶段方法和二阶段方法的精度还是存在不小的差距的。造成这种差距的主要原因之一就是单阶段检测器的训练存在正负样本和难易样本的不平衡问题，**这里需要注意的是，正负样本和难易样本是两回事，因此存在正易、正难、负易和负难四种样本。** 大量的简单样本和背景样本使得模型的训练不堪重负，但是二阶段检测则不存在这个问题，因为其是基于proposal筛选的。

针对这个问题，OHEM是一个基于难样本挖掘的方法被广泛使用，但是这种方法直接放弃了大多数样本并且使得训练的效率不高。后来，Focal Loss横空出世，它通过修改经典的交叉熵损失为一个精心设计的形式来解决这个问题。但是，Focal Loss采用了两个超参数，这两个参数需要花费大量的精力对其进行调整，并且Focal Loss是一种静态损失，它不能自适应于数据分布的变化，即随训练过程而改变。

这篇论文中，作者指出类别不平衡最终可以归为梯度范数分布的不平衡。如果一个正样本被很好地分类，那么它是一个简单样本，模型从中受益不多，换句话说，该样本产生的梯度很小。而错误分类的样本无论属于哪个类都应该受到模型足够的关注。因此，从全局来看，大量的负样本是容易分类的并且难样本往往是正样本。因此这两种不平衡可以粗略概括为属性不平衡。

![](https://i.loli.net/2021/04/15/au86Syt7CHQBdlA.png)

作者认为梯度范数的分布可以隐式表达不同属性的样本的不平衡。相对于梯度范数的样本密度（下文简称梯度密度），梯度密度如上图的最左侧所示，变化幅度是很大的。首先看图像的最左侧，这里的样本的梯度范数比较小但是密度很大，这表示大量的负易样本。再看图像最右侧，这里梯度范数大，数量相对于左侧的容易样本少得多，这表示困难样本。尽管一个容易样本相对于困难样本对整体损失的贡献少，但是架不住数量大啊，大量的容易样本的贡献可能超越了少数困难样本的贡献，导致训练效率不高。而且，难样本的密度比中性样本的密度稍高。这些非常难的样本可以视作离群点，因为即使模型收敛，它们依然稳定存在。离群点可能会影响模型的稳定性，因为它们的梯度可能与其他一般样本有很大差异。

经过上面的梯度范数分布的分析，作者提出了梯度调和机制（GHM）来高效训练单阶段目标检测器，该策略关注于不同样本梯度分布的平衡。GHM首先对具有相似属性但没有梯度密度的样本进行统计，然后依据密度将调和参数附加（以乘法的形式）到每个样本的梯度上。上图最右侧即为平衡后的结果，可以看到，GHM可以大大降权简单样本累积的大量梯度同时也可以相对降权离群点梯度，这使得各类样本的贡献度平衡且训练更加稳定高效。

## GHM

### 前人工作

首先，来回顾一下单阶段检测的一个关键问题，样本类型的极度不平衡。对一个候选框而言，$p \in[0,1]$表示模型预测的概率分布，$p^{*} \in\{0,1\}$为一个确定的类真实标签，那么考虑二分交叉熵的标准形式如下，它通常用于分类任务。

$$
L_{C E}\left(p, p^{*}\right)=\left\{\begin{array}{ll}
-\log (p) & \text { if } p^{*}=1 \\
-\log (1-p) & \text { if } p^{*}=0
\end{array}\right.
$$

但是，这个形式的交叉熵用于正负样本分类是不合理的，因为单阶段检测器通常会产生高达100k的目标，能和GT框匹配的正样本少之又少，因此出现了正负样本不平衡的问题，为了解决这个问题，下面的加权交叉熵被提出。

$$
L_{C E}\left(p, p^{*}\right)=\left\{\begin{aligned}
-\alpha \log (p), & \text { if } p^{*}=1 \\
-(1-\alpha) \log (1-p), & \text { if } p^{*}=0
\end{aligned}\right.
$$

虽然加权交叉熵平衡了正负样本，但是就像上文所说的，样本还有难易之分，目标检测的候选框其实大量是易分样本，加权交叉熵对难易样本的平衡并没有太大作用。易分样本虽然损失很低，但是数量极多，梯度的累积造成其主导了优化的方向，显然这是不合理的。因此，Focal Loss作者认为，易分样本（置信度高的样本）对模型效果的提升影响非常小，模型应该主要关注那些难分样本。这个假设其实是有一些问题的，GHM就是针对次做了探索，下文再细说。

那么，Focal Loss如何平衡难易样本呢，其实很简单，**把高置信度的简单样本的损失再降低不就行了**，于是有了下面这个中间结果，通过$\gamma$的设置可以有效衰减高置信度样本的损失值。

$$
F L=\left\{\begin{aligned}
-(1-p)^{\gamma} \log (p), & \text { if } p^{*}=1 \\
-p^{\gamma} \log (1-p), & \text { if } p^{*}=0
\end{aligned}\right.
$$

这个公式再结合上加权交叉熵，不就同时解决了正负样本不平衡和难易样本不平衡两个问题嘛，于是就有了下面这个Focal Loss的常见形式。实验表明，$\gamma=2$且$\alpha=0.25$时效果最佳。

$$
F L=\left\{\begin{array}{ccc}
-\alpha(1-p)^{\gamma} \log (p), & \text { if } & p^{*}=1 \\
-(1-\alpha) p^{\gamma} \log (1-p), & \text { if } & p^{*}=0
\end{array}\right.
$$

下面是mmdet实现的pytorch版本的focal loss，这个代码也比较容易理解，主要对focal loss公式进行了拆分实现。

```python
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return 
```

### GHM损失

然而，Focal Loss虽然极大的推动了单阶段检测和anchor-free检测的发展，但是它也是存在问题的。首先，就是最核心的一个问题：让模型过分关注特别难分的样本肯定是不合适的，因为样本中存在离群点，模型可能已经收敛，但是这些离群点的存在模型仍旧判断失误难以正常训练，即使拟合了这样的样本模型也是不合理的。其次，Focal Loss的两个超参数需要人为设计，且需要联合调参，因为它俩会互相影响。为了解决这两个问题，GHM被提了出来，不同于Focal Loss关注于置信度来衰减损失，GHM从一定置信度的样本量的角度出发衰减损失。

那么GHM是如何做的呢？首先，假定$x$是模型输出且通过sigmoid激活，那么可以可以得到关于$x$的交叉熵梯度如下，这其实可以视为损失的梯度即为预测和真值的差距。

$$
\begin{aligned}
\frac{\partial L_{C E}}{\partial x} &=\left\{\begin{array}{ll}
p-1 & \text { if } p^{*}=1 \\
p & \text { if } p^{*}=0
\end{array}\right.\\
&=p-p^{*}
\end{aligned}
$$

利用上式，可以定义梯度的模$g$如下，这个$g$等于相对于$x$的梯度范数，它的值表示样本的难易属性（越大样本越难），同时也隐含了样本对全局梯度的影响。这里梯度范数的定义数学上并不严格，只是作者为了方便的说法。

$$
g= \left|\frac{\partial L_{C E}}{\partial x}\right| = \left|p-p^{*}\right|=\left\{\begin{array}{ll}
1-p & \text { if } p^{*}=1 \\
p & \text { if } p^{*}=0
\end{array}\right.
$$

下图所示即为一个收敛的单阶段模型的$g$分布情况，横轴表示梯度的模，纵轴表示样本量比例。很直观地可以看到，$g$很小的样本数量非常多，这部分为大量的容易样本，随着$g$增加，样本量迅速减少，但是在$g$接近1的时候，样本量也不少，这部分属于困难样本。**需要注意的是，这是一个收敛的模型，也就是说即使模型收敛，这些样本还是难易区分，它们属于离群点，如果模型强行拟合这类样本会导致其他正常样本分类精度降低。**

![](https://i.loli.net/2021/04/15/NVjnOT98UuzRS7e.png)

依此，GHM提出了自己的核心观点，的确不应该关注那些数量很多的容易样本，但是非常困难的样本也是不正常的，也不应该过分关注，而且容易样本和困难样本的数量相比于中性样本都比较多。那么如何同时衰减这两类样本呢，其实只要从它们数量多这个角度出发就行，**就衰减那些数量多的样本**。那么如何衰减数量多的呢，那就需要定义一个变量，来衡量某个梯度或者某个梯度范围内的样本数量，这个概念其实很类似于“密度”这个概念，因此将其称为梯度密度，这就有了下面这个这篇文章最核心的公式，梯度密度$GD(g)$的定义。

$$
G D(g)=\frac{1}{l_{\epsilon}(g)} \sum_{k=1}^{N} \delta_{\epsilon}\left(g_{k}, g\right)
$$

这个式子需要做一些说明。$g_k$表示第$k$个样本的梯度的模，$\delta_{\epsilon}(x, y)$和$l_{\epsilon}(g)$的定义式如下，**前者**表示$x$是否在$y$的一个邻域内，在的话则为1否则为0，上式中求和后的含义就是$g_k$在$g$的范围内的样本数目，**后者**则表示计算样本量的这个邻域的区间长度，它作为标准化因子。**因此，梯度密度$GD$的含义可以理解为单位模长在$g$附近的样本数目。**

$$
\delta_{\epsilon}(x, y)=\left\{\begin{array}{rr}
1 & \text { if } y-\frac{\epsilon}{2} <= x < y+\frac{\epsilon}{2} \\
0 & \text { otherwise }
\end{array}\right.
$$

$$
l_{\epsilon}(g)=\min \left(g+\frac{\epsilon}{2}, 1\right)-\max \left(g-\frac{\epsilon}{2}, 0\right)
$$

有了梯度密度，下面定义最终用在损失调和上的梯度密度调和参数，这里的$N$表示样本总量，为了方便理解，其可以改写为$\beta_{i}=\frac{1}{G D\left(g_{i}\right) / N}$，分母$G D\left(g_{i}\right) / N$表示第i个样本具有邻域梯度的所有样本分数的归一化器。如果样本关于梯度均匀分布，对任何的$g_i$的$GD(g_i) = N$并且每个样本的$\beta_i = 1$。相反，具有较大密度的样本则会被归一化器降权表示。

$$
\beta_{i}=\frac{N}{G D\left(g_{i}\right)}
$$

利用$\beta_i$可以集成到分类损失和回归损失中从而构建新的损失GHM-C Loss，这里的$\beta_i$则作为第$i$个样本的损失加权，最终梯度密度调和后的分类损失如下式。其实从最后的结果来看，就是原本的交叉熵乘以该样本梯度密度的倒数。

$$
\begin{aligned}
L_{G H M-C} &=\frac{1}{N} \sum_{i=1}^{N} \beta_{i} L_{C E}\left(p_{i}, p_{i}^{*}\right) \\
&=\sum_{i=1}^{N} \frac{L_{C E}\left(p_{i}, p_{i}^{*}\right)}{G D\left(g_{i}\right)}
\end{aligned}
$$

使用GHM-C和交叉熵以及Focal Loss的梯度范数调整效果如下图，显然，GHM-C的抑制更加明显合理。

![](https://i.loli.net/2021/04/15/tCTpU8Swhu9lAke.png)

这里作者还提到了EMA处理的技巧，此外魔改smooth l1可以得到下面的GHM-R损失函数，这里就不展开了。

$$
\begin{aligned}
L_{G H M-R} &=\frac{1}{N} \sum_{i=1}^{N} \beta_{i} A S L_{1}\left(d_{i}\right) \\
&=\sum_{i=1}^{N} \frac{A S L_{1}\left(d_{i}\right)}{G D\left(g r_{i}\right)}
\end{aligned}
$$

最后，补充一个mmdet关于GHM Loss的实现，它是先将梯度模长划分为bins个范围，然后计算每个区域的便捷edges，接着就很容易判断梯度模长落在哪个区间里了，然后按照公式计算权重乘以原来的交叉熵损失即可（实现思路和上面的Focal Loss差不多，都是对原本的交叉熵加权）。

```python
class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight
```

## 实验

实验配置等细节查看原文即可，我这里就给出SOTA方法的涨点结果图，可以看到，相比于Focal Loss，涨点效果还是挺明显的。

![](https://i.loli.net/2021/04/15/QtcKMDPSOC4pzNf.png)

## 总结

这篇论文针对单阶段目标检测的样本不均衡问题从梯度出发，提出了GHM这一策略，从损失的角度有效改善了此前的Focal Loss。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。
