# TPAGT 解读

## 简介

浙江大学和达摩院前不久提出的一个 MOT 新方法，目前在 MOT Challenge 常用的几个数据集上名列前茅。论文标题 Tracklets Predicting Based Adaptive Graph Tracking 其实已经表明本文最大的两个创新点，基于轨迹预测的特征提取以及基于自适应图网络的特征聚合。大多数现存的多目标跟踪方法将当前帧的检测结果链接到历史轨迹段都是采用基于特征余弦距离和目标边界框 IOU 的线性组合作为度量的，这其实有两个问题：**一是两个不同帧（当前帧和上一帧）上同一个目标提取到的特征往往会出现不一致的问题；二是特征提取只考虑外观而不考虑位置关系、轨迹段信息是不合理的。**

因此，论文提出了一种新的高精度端到端多目标跟踪框架 TPAGT（上一个版本叫 FGAGT，感觉 TPAGT 更加贴合论文的工作），该方法解决了上述的两个问题，在多个数据集上实现了新的 SOTA。

- 论文标题<br>
  Tracklets Predicting Based Adaptive Graph Tracking
- 论文地址<br>
  http://arxiv.org/abs/2010.09015
- 论文源码<br>
  暂未开源

## 介绍

首先说明的是，TPAGT 按照一般 MOT 的方法划分是一个**二阶段框架**，也就是先完成检测，再按照检测结果到相应的位置提取目标特征，最后利用关联算法得到结果，关联一般采用匈牙利算法。单阶段方法融合了检测和特征提取，是为了速度做出的精度妥协，所以精度相比二阶段有些低。所以，作为一个二阶段方法，TPAGT 的精度应该有所创新，但是相应的速度比较慢，具体推理速度，论文没有提及，只能等源码开放后测试了。

先来说一说 MOT 现有方法没解决的几个问题。

1. 特征不一致问题<br>
   这个问题怎么来的呢，其实是因为轨迹段（tracklet）上目标的特征都是来自于之前帧，而不是当前帧（这很容易理解，当前帧只有当前帧的检测结果确定目标位置来提取特征嘛），但是呢，其实在移动过程中，目标的姿态、光强度、视角都可能发生变化，这导致来自不同图像的同一目标的特征即使检测准确也会不太一致，这种不一致对数据关联来说负面影响比较大。
2. 特征融合问题<br>
   事实上，从 DeepSORT 开始，特征提取器主要关注的就是外观信息，因为这对忽略了运动建模的一些 MOT 方法至关重要，因此特征提取分支也成为 ReID 分支，主要就是因为重识别模型关注的就是外观信息。但是，目标之间的位置关系、tracklet 的历史信息对 MOT 任务也是很重要的。
3. 样本不平衡问题<br>
   一个 tracklet 只能匹配一个检测框，那这个 tracklet 就是个连续的正例，没有匹配上的 tracklet 就是连续的负例。显然，正例数量是远远少于负例的，而且由于少量的新目标的产生和旧目标的消失，进一步加剧了不同类型的样本的不均衡问题。

上述的问题 TPAGT 都逐一解决了，其中最主要的一个问题就是 traklets 中的特征和当前帧是不一致的，那么如何解决呢，到当前帧上**重提取**特征就行，但是显然不能直接把上一帧的 bbox（边界框，包含目标的位置区域等信息）用于当前帧，因为目标在图像上不可能静止，使用上一时刻的位置很不合理，所以需要对上一帧进行运动估计得到目标在当前帧预测的 bbox 位置然后提取特征。然后是特征融合的问题，考虑到目标之间的联系近似一个图表示，作者采用了**GNN**（图神经网络）来进行信息的聚合，为了更好获取全局时空信息，GNN 的边权自适应学习。最后，样本不平衡的问题采用了**Balanced MSE Loss**，这是一个加权 MSE，属于常用思路。

## 框架设计

### **Tracklets predicting based feature re-extracting**

![](https://i.loli.net/2020/12/04/tfmMKJpOq5BgcnR.png)

上面这个图就是整体框架的设计，我先大体介绍一下网络的 pipeline。首先，网络的输入有当前帧图像、当前帧检测结果、历史帧检测结果；接着，图像被送入 backbone 中获得特征图（这里 backbone 最终采用 ResNet101+FPN 效果最好），然后将 bbox（这里当前帧用的是检测的 bbox，上一帧用的光流预测的 bbox）映射到特征图上通过 RoI Align 获得 region 外观特征继而送入全连接（这个操作类似 Faster R-CNN 的 proposal 提取特征，不理解的可以查阅我的[博客](https://zhouchen.blog.csdn.net/article/details/110404238)），然后结合当前帧的位置信息、历史帧信息，让图网络自适应学习进行特征融合从而计算相似度，有了相似度矩阵匈牙利就能计算匹配结果了。

**上面的叙述有个容易误解的地方，它将过去一帧预测的 bbox 和历史帧的非预测的 bbox 都在当前特征图上提取了特征，事实上，不是的，一来实际上，$t-2$帧的特征在处理$t-1$帧的时候已经重提取过了，在当前帧上用当时的 bbox 提取肯定存在严重的不对齐问题；二来，这样会大大加大网络计算的复杂性，完全没有必要。论文这个图画的稍微有些让人误解，等开源后可以再细细研究。**

我们知道，此前的 MOT 方法对运动的建模主要采用卡尔曼滤波为代表的状态估计方法、光流法和位移预测法，这篇论文使用稀疏光流法预测 bbox 的中心点运动，由于目标的运动有时候是高速的，为了应对这种运动模式，必须采用合适的光流方法，文章采用金字塔光流，该方法鲁棒性很强，具体想了解的可以参考[这篇博客](https://blog.csdn.net/gh_home/article/details/51502933)，下图是金字塔光流预测的目标当前帧位置（b 图），c 图是 GT 的框，可以看到，预测还是很准的。

![](https://i.loli.net/2020/12/04/cg1P6rNB49bGSyH.png)

### **Adapted Graph Neural Network**

![](https://i.loli.net/2020/12/04/NtzZBPnAXgE6l7U.png)

下面聊一聊这个自适应图神经网络。将 tracklets 和 detections 作为二分图处理不是什么新鲜的事情，但是用来聚合特征 TPAGT 应该是为数不多的工作，**要知道此前我们聚合运动和外观特征只是人工设计的组合，作者这种借助图网络自适应聚合特征是很超前的思路。** 每个检测目标和每个 tracklet 都是节点，如上图所示，detection 之间没有联系，tracklet 之间也没有联系，但是每个 tracklet 和每个 detection 之间都有连接。图网络的学习目的就是每个节点的状态嵌入$\mathbf{h}_{v}$，或者说聚合其他信息后的特征向量。最终，这个$\mathbf{h}_{v}$包含了邻居节点的信息。

需要学习的状态嵌入通过下面的公式更新，第一行表示 detections 的节点更新，第二行表示 tracklets 的节点更新，共有$N$个 detection 和$M$个 tracklet。下面讲解第一行的几个符号含义，第二行类似。$f$表示神经网络运算，可以理解为网络拟合函数；$h_{t, c}^{j}$表示第$c$层第$i$个 detection 的状态嵌入。在一开始，$c=0, h_{d, 0}^{i}=f_{d}^{i}, h_{t, 0}^{i}=f_{t}^{j}$，$e_{d, c}^{i, j}$则表示第$i$个检测和第$j$个 tracklet 在第$c$层的图上的边权。本文作者只使用添加自适应的单层 GNN，所以下面具体阐述单层学习的情况。

$$
\begin{aligned}
h_{d, c+1}^{i} &=f\left(h_{d, c}^{i},\left\{h_{t, c}^{j}, e_{d, c}^{i, j}\right\}_{j=1}^{N}\right), i=1,2, \cdots, M \\
h_{t, c+1}^{j} &=f\left(h_{t, c}^{j},\left\{h_{d, c}^{i}, e_{t, c}^{j, i}\right\}_{i=1}^{M}\right) j=1,2, \cdots, N
\end{aligned}
$$

首先，边权的初始化不采用随机初始化，而是采用节点的特征和位置先验信息，具体如下，主要是计算每个节点特征向量之间的归一化距离相似度。具体图信息聚合步骤如下。<br>

1. 计算初始相似度
   $$
   \begin{array}{c}
   s_{i, j}=\frac{1}{\left\|f_{d}^{i}-f_{t}^{j}\right\|_{2}+1 \times 10^{-16}} \\
   s_{i, j}=\frac{s_{i, j}}{\sqrt{s_{i, 1}^{2}+s_{i, 2}^{2}+\cdots s_{i, j}^{2}+\cdots+s_{i, N}^{2}}}\\
   \mathbf{S}_{\mathrm{ft}}=\left[s_{i, j}\right]_{M \times N}, i=1, \cdots M, j=1, \cdots N
   \end{array}
   $$
2. 通过 IOU 和上面的初始相似度组成边权（$w$可学习，表示位置和外观信息的相对重要性）

$$
\mathrm{E}=w \times \mathrm{IOU}+(1-w) \times \mathrm{S}_{\mathrm{ft}}
$$

3. 根据上述的自适应权重聚合节点特征（$\odot$表示点积）

$$
\mathbf{F}_{\mathrm{t}}^{\mathrm{ag}}=\mathrm{EF}_{t}=\mathrm{E}\left[f_{t}^{1}, f_{t}^{2}, \cdots, f_{t}^{N}\right]^{T}
$$

$$
\mathbf{H}_{\mathrm{d}}=\sigma\left(\mathbf{F}_{d} W_{1}+\operatorname{Sigmoid}\left(\mathbf{F}_{d} W_{a}\right) \odot \mathbf{F}_{\mathrm{t}}^{\mathbf{a g}} W_{2}\right)
$$

$$
\mathbf{H}_{\mathrm{t}}=\sigma\left(\mathbf{F}_{t} W_{1}+\operatorname{Sigmoid}\left(\mathbf{F}_{t} W_{a}\right) \odot \mathbf{F}_{\mathrm{d}}^{\mathbf{a g}} W_{2}\right)
$$

现有的图跟踪方法需要额外的全连接层降维特征向量，然后通过欧式距离计算相似度。TPAGT 的方法只要标准化来自单隐层图网络的特征，然后矩乘它们即可得到相似度决战，如下式。最终得到的相似度矩阵值介于 0 和 1 之间，越大代表两个目标越相似。学习的目的是使得同一个目标的特征向量尽量接近，不同目标的特征向量尽量垂直，这等价于三元组损失，但是更加简单。

$h_{d}^{i}=\frac{h_{d}^{i}}{\left\|h_{d}^{i}\right\|_{2}}, h_{t}^{j}=\frac{h_{t}^{j}}{\left\|h_{t}^{j}\right\|_{2}}, \mathbf{S}_{\mathrm{out}=\mathbf{H}_{\mathrm{d}} \mathbf{H}_{\mathbf{t}}^{\mathrm{T}}}$

### **Blanced MSE Loss**

得到最终的相似度矩阵就可以进行监督训练了，不过 GT 的标签为相同目标为 1，不同的目标为 0，下图是作者做的可视化，每行代表一个 detection，每列代表一个 tracklet，绿行表示 detection 没有匹配上任何 tracklet，所以是新目标；相对的，红列表示消失的目标。1 表示正例，0 表示负例，显然正负例严重不均衡，所以这里对 MSE 按照目标类型进行了加权（超参），如下式。

![](https://i.loli.net/2020/12/04/NGSUCpjo8mQa7xl.png)

$$
\begin{aligned}
\mathcal{L} &=\alpha E_{c 0}+\beta E_{c 1}+\gamma E_{n e}+\delta E_{d}+\varepsilon E_{w} \\
&=\sum_{i=1}^{M} \sum_{j=1}^{N}\left[\begin{array}{c}
\alpha\left(\hat{S}_{i, j}-S_{i, j}\right)^{2} \cdot \mathbb{I}_{\text {continue }} \cdot \mathbb{I}_{S_{i, j}=0}+\beta\left(\hat{S}_{i, j}-S_{i, j}\right)^{2} \cdot \mathbb{I}_{\text {continue }} \cdot \mathbb{I}_{S_{i, j}=1} \\
+\gamma\left(\hat{S}_{i, j}-S_{i, j}\right)^{2} \cdot \mathbb{I}_{n e w}+\delta\left(\hat{S}_{i, j}-S_{i, j}\right)^{2} \cdot \mathbb{I}_{\text {disap }}+\varepsilon\|W\|_{2}^{2}
\end{array}\right]
\end{aligned}
$$

## 推理设计

推理时，我们会得到相似度矩阵，那么如何利用这个矩阵呢？假设有$N$个 detection 和$M$个 tracklet，矩阵就是$M\times N$的，此时在后面补充一个$M\times M$的增广矩阵，矩阵中每个值都是一个阈值，如下图，匈牙利算法就成了带筛选的匹配方法，下图由于第 3 行和第 8 行没有高于阈值（0.2）的相似度，所以成为了新目标。

![](https://i.loli.net/2020/12/04/ZQHhlGDUcg2o9XC.png)

## 实验及分析

检测部分采用 FairMOT 的检测结果，也就是采用 CenterNet 作为检测器。特征提取部分，文章使用 ResNet101-FPN 作为 backbone，在 COCO 上预训练过，然后在 MOT 数据集上 fine tune 30 轮。其他训练细节可以自行查阅论文，我这里就不多说了，在 Public 和 private 两个赛道进行了测试，结果分别如下，超越了之前的 SOTA 方法如 FairMOT 等，精度突破很大，速度比较慢。

![](https://i.loli.net/2020/12/04/cE86gYS5t3oKBDj.png)

![](https://i.loli.net/2020/12/04/QbR6ngGePwxSYWh.png)

此外，作者还进行了丰富的消融实验，证明了 TPAGT 的鲁棒性。

## 总结

开创性地提出了特征重提取策略，并引入 AGNN 进行特征融合，从而构建了 TPAGT 框架，这是一个端到端的学习框架，可以直接输出相似度矩阵。在 MOT Challenge 两个赛道都获得了 SOTA 表现。

## 参考文献

[1]Shan C, Wei C, Deng B, et al. Tracklets Predicting Based Adaptive Graph Tracking[J]. arXiv:2010.09015 [cs], 2020.
