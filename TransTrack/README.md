# TransTrack解读

## 简介

Transformer已经在计算机视觉各个任务上获得了不错的效果，当然也不会放过多目标跟踪，香港大学、字节跳动AI实验室等机构在2020年最后一天挂到Arxiv上的这篇文章就是第一个将Transformer用于多目标跟踪的文章，尽管紧随其后就出现了TrackFormer这样类似的作品，不过本人觉得TransTrack效果目前看来好一些，本文就自己的理解谈谈这篇文章。

- 论文标题

    TransTrack: Multiple-Object Tracking with Transformer

- 论文地址

    http://arxiv.org/abs/2012.15460

- 论文源码

    https://github.com/PeizeSun/TransTrack

## 介绍

回顾MOT的核心思路，如下图所示，目前TBD范式的多目标跟踪方法依然受限于复杂的pipeline，这带来了大量的计算代价，如下图(a)所示，一如当年的DeepSORT，检测和跟踪任务分开进行，这会带来一些比较严重的问题：一方面，这种两个任务分开进行会造成它们不能共享有效的信息带来额外的算力消耗；另一方面，连续两帧间的无序目标对和每帧中不完整的检测都为跟踪算法带来了极大的挑战。**因此，JDE范式的产生其实是MOT领域的发展非常重要的一步。** 

![](https://i.loli.net/2021/01/12/CZYSFnKIE24GprH.png)

上图的(b)其实是SOT中常见的孪生网络，这种结构本质上就是Query-Key机制，目标对象是query而各个图像区域是keys。直接将这个思路引入MOT中也是可行的，前一帧的目标特征作为query，当前帧的图像特征作为key，这就是上图(c)的由来。然而，仅仅将Query-Key机制引入MOT中效果非常差，主要是FN指标上的表现（在检测中的直观含义就是漏检），原因也很简单，因为当前帧出现的新目标的特征肯定不存在于query中，因此当然无法获取key，这也就造成了新目标的缺失。

因此，回到这篇论文的初衷，能否设计一种基于Query-Key机制的MOT框架，它能输出有序目标集也能检测新目标的出现呢？这就诞生了**TransTrack**，一个联合检测和跟踪（JDE范式）的新框架，它利用Query-Key机制来跟踪当前帧中已存在的目标并且检测新目标。

![](https://i.loli.net/2021/01/12/hczZfslYqBxeMyK.png)

先简单地看一下这个pipeline的设计，它基于Query-Key目前最火热的Transformer架构构建。最中间的key来自骨干网络对当前帧图像提取的特征图，而query按照两个分支的需求分别来自上一帧的目标特征query集和一个可学习的目标query集。**这两个分支都很有意思，我们先看下面这个检测分支，这里这个learned object query思路来自于DETR，是一种可学习的表示，它能逐渐学会从key中查询到目标的位置从而完成检测，想知道得更明白得可以去看看DETR论文。可以很明显地看明白，这个检测分支完成了当前帧上所有目标的检测得到detection boxes。然后我们看上面这个跟踪框分支，这个object feature query其实就是上一帧的检测分支产生的目标的特征向量，这个object feature query从key中查询目标当前帧中位置，用CenterTrack的思路来理解，这可以认为是一个位移预测分支，它最终得到tracking boxes。最后，由于跟踪框和检测框都在当前帧上了，进行简单的IOU匹配就能完成跟踪了，至此，MOT任务完成。**

回顾上面这个TransTrack的设计，其实很清晰地发现其优势，那就是JDE范式下地同时优化两个子网络，速度很快。不过，这个基于检测框和跟踪框匹配的思路，让人有点梦回DeepSORT啊😂。

## TransTrack

论文里还回顾了一下Query-Key机制在跟踪中的发展以及TBD范式和JDE范式的创新，我这里就不多赘述了，直接来看TransTrack的一些细节。

首先，作者认为，一个理想的跟踪模型的输出目标集应该是完整且有序的，因此TransTrack使用learned object query和上一帧的object feature query作为输入query。前者经过decoder之后变为当前帧的检测框，后者经过decoder之后变为跟踪框（上一帧目标在当前帧的位置预测）。因此TransTrack在当前帧完成了数据关联步骤，这就允许其才有简单的metric作为关联指标，文中采用IOU。

![](https://i.loli.net/2021/01/12/yVX8JuPELAlKQ6s.png)

上图是整个网络的结构图，它和TransFormer很像，由一个产生复合特征图的encoder和两个平行的decoder构成。encoder和decoder的具体结构这里不详细讲解了，可以去阅读Transformer论文了解，简单来说它其实由堆叠的多头注意力和全连接层组成。在Transformer的设计中，encoder生成大量的key，而decoder接收指定任务的query对key进行查询得到想要的输出。

上图所示，encoder将backbone对当前帧提取的特征图和上一帧处理时保留的特征图组合到一起作为输入，这就是keys。然后两个decoder接收这些keys，他俩分别进行目标检测和目标传播任务。首先来看检测这个decoder，它采用DETR的集合预测思路完成检测，可学习的query从全局特征图上查询目标的位置得到检测框。接着，来看看这个目标传播的decoder，它和检测分支的结构类似，不过输入不同，是前帧目标的特征向量，这个向量包含了历史目标的位置和外观信息，所以能够在当前帧上查询到跟踪框。

最后的Matching模块就很简单了，既然跟踪框和检测框都在当前帧上，那么相同目标只会有很微小的偏移，基于IOU进行匹配即可（这个思路和CTracker类似），匹配算法采用常用的KM算法。未被匹配上的检测框初始化为新目标。

## 训练和推理

### 训练

训练数据集采用了CrowdHuman和MOT Challenge数据集，损失方面，可以认为检测框和跟踪框的获得都是在当前帧进行目标检测，因此两个decoder可以采用同一个训练损失，这是一个二分匹配的集合预测损失，公式如下。${L}_{c l s}$表示框间的类别损失，使用focal loss实现；L1和GIOU loss用于监督框的定位准确率，具体的训练就是采用DETR类似的方式训练。

$$
\mathcal{L}=\lambda_{c l s} \cdot \mathcal{L}_{c l s}+\lambda_{L 1} \cdot \mathcal{L}_{L 1}+\lambda_{g i o u} \cdot \mathcal{L}_{g i o u}
$$

### 推理

推理第一帧的时候没有历史目标因此拷贝一份特征图即可，后面都是相邻帧之间的处理，需要注意的是，这里引入了Track Rebirth策略以增强针对遮挡和短时间目标消失的鲁棒性，具体是指如果一个跟踪框未被匹配上，它暂时不被移除，只有确定连续K帧都没有匹配上时才真正认为该目标消失，文中取K=32。

## 实验

作者进行了大量的消融实验，首先验证了额外训练数据确实对效果有不小的帮助。

![](https://i.loli.net/2021/01/12/NybrvUouR53tEYF.png)

接着对比了不同的Transformer架构的效果，原始的Transformer训练非常耗时，Deformble DETR训练快效果好。

![](https://i.loli.net/2021/01/12/MVznJ5DpZe6U8js.png)

此外，还验证了两个解码层的有效性，两种query都是必须的。

![](https://i.loli.net/2021/01/12/zCxha43WJ8NSmY9.png)

最后，来看看下面和SOTA方法的比较，在MOTA、MOTP、FN等指标上达到了SOTA效果，这主要归功于检测器的强大，然而在IDS这个关乎跟踪效果的指标上表现并不好。

![](https://i.loli.net/2021/01/12/PJolZkEpfRFAjKM.png)


## 总结
TransTrack第一次将Transformer引入到了MOT领域并获得了不错的效果，这里也推荐一篇最新的视觉Transformer综述[Transformers in Vision: A Survey](http://arxiv.org/abs/2101.01169)，其涉及了Transformer在视觉各个任务的应用效果。回到TransTrack上来，这篇文章思路上还是很有创新性的，将一个东西用到一个全新的任务上来要做的远远不是1+1=2这么简单，需要做很多很多的工作，而TransTrack做的都不错，在多个MOT的指标上都达到了SOTA水平，不过在MOT很关键的IDS指标上距离SOTA还有一些距离，期待正式版本的论文效果会更好。

## 参考文献

[1]Sun P, Jiang Y, Zhang R, et al. TransTrack: Multiple-Object Tracking with Transformer[J]. arXiv:2012.15460 [cs], 2020.