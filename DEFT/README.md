# DEFT解读

## 简介

最近不少2D多目标跟踪（Multiple Object Tracking，MOT）的成果表明，使用SOTA检测器加上简单的基于空间运动的帧间关联就可以获得相当不错的跟踪表现，它们的效果优于一些使用外观特征进行重识别丢失轨迹的方法。Uber等最近提出的DEFT（Detection Embeddings for Tracking）是一种联合检测和跟踪的模型，它是一个以检测器为底层在上层构建基于外观的匹配网络的框架设计。在二维跟踪上，它能达到SOTA效果并具有更强的鲁棒性，在三维跟踪上它达到了目前SOTA方法的两倍性能。

- 论文标题

    DEFT: Detection Embeddings for Tracking
- 论文地址

    http://arxiv.org/abs/2102.02267
- 论文源码

    https://github.com/MedChaabane/DEFT


## 介绍

卷积神经网络的发展推动了目标检测领域的进步，TBD范式（Tracking by Detection）的MOT也取得巨大的突破，最近的研究表明，在SOTA跟踪器上添加简单的跟踪机制就可以比依赖旧检测架构的复杂跟踪器效果更好。TBD范式的跟踪框架通常有两步：**检测**：在当前帧检测出所有目标；**关联**：将当前帧上的目标和之前帧上的目标进行链接。跨帧关联的方法有很多，但是那些特征可学习关联方法通常更有趣一些，因为它们有望解决建模和启发式方法失败的情况。

即使有了可学习关联，二阶段方法也可能产生准确性和效率方面的次优结果。因此最近的一个趋势就是在一个网络中联合学习检测和跟踪任务，这会带来性能上的提升。可以假设，一个可学习的目标匹配模块可以添加到主流的CNN检测器中，从而产生高性能的多目标跟踪器，进而通过联合训练检测和跟踪（关联）模块，两个模块彼此适应实现更好的性能。相比于那种将检测作为黑盒模型输入到关联模块中，这种让目标检测和帧间关联共享backbone的思路会有更好的速度和精度。

所以这篇论文的作者提出了DEFT这个新的方法，在该方法中每个目标的embedding（我这里翻译为嵌入）通过多尺度检测backbone获得，并且这个embedding作为后续的object-to-track关联子网络的外观特征。DEFT可以灵活用于多个常见的目标检测backbone上，并且由于检测和跟踪共享特征的特性，这种使用外观和运动信息的方法在速度上相比于那些使用更简单的关联策略的方法在速度上页不遑多让。

## DEFT

下面来看整个DEFT的网络设计，总体来说，它是非常类似JDE和FairMOT的一个工作。基于TBD范式，DEFT提出应该使用目标检测器（本文将目标检测器作为backbone）的中间特征图来提取目标的embedding从而用于目标匹配子网络。如下图所示，单看上半部分其实就是整个网络的结构，图像进入上面的Detector分支，同时检测器的不同stage的特征图用于下面的Embedding Extractor模块的外观特征学习，不同帧间的检测目标的外观特征送入Matching Head中获得关联相似度得分矩阵。

在DEFT中检测器和目标匹配网络是联合训练的，训练时，目标匹配网络的损失会反传给检测backbone从而优化外观特征的提取和检测任务的表现。此外，DEFT还在使用了一个低维的LSTM模块来为目标匹配网络提供几何约束，以避免基于外观的但在空间变化上不可能发生的帧间关联。虽说可以基于多个检测器，DEFT在CenterNet上做了主要的工作，获得了SOTA并且比其他类似方法要快，这种速度上的快时来自于在DEFT中，目标关联只是额外的一个小模块，相比于整个检测任务只会有很小的延时。

![](https://i.loli.net/2021/02/10/5XGa7qQdn9f34HI.png)

在介绍各个模块之前，我先讲述一下整个DEFT推理时的pipeline，这和上面的训练其实有所不同。如下图所示，Embedding Extractor使用检测backbone多个阶段的特征图和检测框作为输入来获得获得每个目标的外观嵌入。Matching Head使用这些嵌入来计算当前帧目标和历史帧目标（也就是当前轨迹）来计算相似度。之后，一个使用LSTM的运动预测模块会对相似度矩阵进行限制，从而保证那些物理上不可能产生的轨迹的链接。最后，匈牙利算法会基于相似度矩阵计算最终的在线匹配结果，将检测结果链接到轨迹上。

![](https://i.loli.net/2021/02/10/PHdiypFRKQO6qkh.png)

下面我们按照论文第三章的叙述思路来详细理解这个网络，下面的叙述还是基于下图，不过我会忽略一些细节上的数学表示。

![](https://i.loli.net/2021/02/10/5XGa7qQdn9f34HI.png)

### Object Embeddings

在图中对应的为Embedding Extractor，该模块从检测backbone的中间特征图提取具有表示意义的embedding，在跟踪的过程中帮助关联（或者叫re-identify）。从图中可以看到，这个模块的输入是检测器的多个stage的特征图（每个特征图有不同的尺度，即有不同的感受野），这种策略会提高单感受野策略的鲁棒性。DEFT网络的输入是视频中的一帧图像，检测head会输出包含多个目标的检测结果的bbox集$\mathrm{B}_{t}=\left\{b_{1}^{t}, b_{2}^{t}, \ldots, b_{N_{t}}^{t}\right\}$，不妨用$N_{t}=\left|\mathrm{B}_{t}\right|$来表示这一帧检测框的数目。

对每个检测到的目标，依据其2D检测框中心点来提取object embedding，对于3D检测框，则将其中心点投影到二维空间。那么如何将这个中心点在各stage的特征图上找到特征呢，其实这里就是做了一个简单的比例映射。对于在size为$W\times H$的图像上第$i$个目标的中心点坐标为$(x, y)$，对当前图像又存在$M$个选择的stage的特征图，该目标在第$m$个size为$W_{m} \times H_{m} \times C_{m}$的特征图上映射的中心点坐标为$\left(\frac{y}{H} H_{m}, \frac{x}{W} W_{m}\right)$，在这个第$m$个特征图上包含一个$C_m$维度的特征向量，这就是这个stage特征图上目标的object embedding，记为$f_i^m$，然后将$M$个stage通过这种方式得到的特征图级联（concatenate）到一起，得到该目标的object embedding，记为$f_{i}=f_{i}^{1} \cdot f_{i}^{2} \ldots f_{i}^{M}$，它是一个$e$维向量。**不过，由于不同stage特征图上通道数是不同的，一般是浅层channel少，高层channel多，所以对第$m$个特征图处理之前，作者使用1x1卷积对浅层特征图进行升维对高层特征图进行降维，使得不同stage含有的特征量在最终的object embedding中分布合理。**

### Matching Head

关于最后这个Matching Head，作者思路采用的是DAN（Deep Affinity Network，是端到端数据关联的一个著名成果）的思路，使用object embedding作为输入来估计两帧之间两两相似度得分。首先，对每一帧设定最大目标个数的限制$N_{max}$，那么就可以构建张量$E_{t, t-n} \in \mathbb{R}^{N_{\max } \times N_{\max } \times 2 e}$，它通过将$t$帧中的每个object embedding和$t-n$帧的每个object embedding沿着深度维度级联。为了保证$E_{t, t-n}$维度固定，会进行补零操作。这个汇总得到的$E_{t, t-n}$会被输入Matching Head，这个匹配模块由几层1x1卷积构成，该Head的输出就是相似度矩阵$A_{t, t-n} \in \mathbb{R}^{N_{\max } \times N_{\max }}$。

尽管我们学习到了嵌入之间的相似性，然而不能保证沿着帧前向和后向的相似度得分是对称的，即$t$和$t-n$之间的相似度与$t-n$和$t$之间的相似度是不同的。因此对两个方向分别计算相似度矩阵，用下标'fwd'和'bwd'表示前向和后向。同时为了考虑有些目标不参与关联（新目标或者离开场景的目标），在相似度矩阵$A_{t,t-n}$添加一列填充常数值$c$，对$A_{t,t-n}$每一行应用softmax可以得到矩阵$\hat{A}^{b w d}$，它表示包括非匹配分数在内的最终相似度。$c$的选择并不是特别敏感的，网络会学会为大于$c$的进行真实匹配。

每个$\hat{A}^{b w d}[i, j]$表示两个目标框$b_i^t$和$b_j^{t-n}$的估计关联概率，$\hat{A}^{b w d}\left[i, N_{\max }+1\right]$则表示$b_i^t$是一个不在$t-n$帧中出现的目标的概率。类似的，前向相似度矩阵$\hat{A}^{f w d}$通过对原始相似度矩阵的转置$A_{t, t-n}^{T}$添加列并逐行softmax得到。在推理过程中，目标框$b_i^t$和$b_j^{t-n}$之间的相似度为$\hat{A}^{b w d}[i, j]$和$\hat{A}^{f w d}[j, i]$的平均值。

### Online Data Association

在DEFT中，保留了最近$\delta$帧中存在的轨迹的每个目标的object embedding，这些组成了memory。一个新bbox和已有轨迹的关联需要计算这个bbox对象和memory中所有轨迹目标的相似度。为了应对遮挡和漏检，track memory会维持几秒，以便新检测和之前的目标高度相关时，进行轨迹恢复。如果一个轨迹$N_{age}$帧都没有匹配上，那么它会被删除。

对于轨迹$T$的定义是一个相关联的检测框集合，来自于$t-n$帧到$t-1$帧，注意，不是每一帧该轨迹都有检测框，可能某一帧没有对应的检测结果。轨迹的长度显然为$T$，它表示检测框数目或者object embedding数目。那么，可以定义当前的第$t$帧上的第$i$个检测框$b_i^t$与第$j$个轨迹$T_j$之间的距离为下面的式子。

$$
d\left(b_{i}^{t}, \mathrm{~T}_{j}\right)=\frac{1}{\left|\mathrm{~T}_{j}\right|} \sum_{b_{k}^{t-n} \in \mathrm{T}_{j}} \frac{\hat{A}_{t, t-n}^{f w d}[k, i]+\hat{A}_{t, t-n}^{b w d}[i, k]}{2}
$$

检测框和轨迹T_j的匹配满足互斥对应原则，是一个二分图匹配问题，记$K = {T_j}$为当前轨迹集，构建检测集到轨迹集的相似度矩阵$D \in \mathbb{R}^{|K| \times\left(N_{t}+|K|\right)}$，它来自于所有检测和轨迹的size为$|K| \times N_t$逐对距离矩阵追加一个size为$K \times K$的矩阵$X$，这个$X$表示当一个轨迹不和任何当前帧的检测关联的情况。$X$对角线上的元素为轨迹中所有检测的平均不匹配得分，非对角线元素设置为$-\infty$，具体而言，$D$按照下式构建。

$$
\begin{aligned}
D &=[S \mid X] \\
S[j, i] &=d\left(b_{i}^{t}, \mathrm{~T}_{j}\right) \\
X[j, k] &=\left\{\begin{array}{ll}
\frac{1}{\left|\mathrm{~T}_{j}\right|} \sum_{b_{k}^{t-n} \in \mathrm{T}_{j}} \hat{A}_{t, t-n}^{f w d}\left[k, N_{\max }+1\right] & j=k \\
-\infty, & j \neq k
\end{array}\right.
\end{aligned}
$$

接着，只要将$D$送入匈牙利算法中进行求解即可。只有亲和度大于阈值$\gamma_1$会被关联，为被匹配的检测会作为新的轨迹，连续$N_age$帧没被匹配到的轨迹会被认为离开场景，从而删除。

### Motion Forecasting

如果仅仅使用学习到的外观嵌入进行帧间匹配，那么很可能出现两个目标其实真的外观空间上很相似，从而造成匹配上的问题。常见的手段是添加一个几何或者时间约束来限制匹配，常用的是卡尔曼滤波或者LSTM。论文采用的是LSTM设计运动预测模块，该模块会依据过去$\Delta T_{\text {past}}$帧预测未来$\Delta T_{\text {pred }}$帧轨迹所在的位置。这个运动预测模块用来约束那些物理上不可能存在的关联，它将距离轨迹预测位置太远的检测框的相似度距离置为$-\infty$。这个模块的具体设计细节可以查看原论文的补充材料。

### Training

![](https://i.loli.net/2021/02/10/5XGa7qQdn9f34HI.png)

如上图所示，为了训练网络的匹配模块，训练时将间隔$n$帧的一个两帧组成的帧对（pair）输入网络。图像对被$1≤n≤n_{gap}$的随机数帧分开，以鼓励网络学习对临时遮挡或漏检具有鲁棒性。对每个输入帧对，会有两个ground truth的匹配矩阵$M^{fwd}$和$M^{bwd}$分别表示前向和后向关联，这个矩阵每个元素$[i, j] \in\{0,1\}$并且维度为$N_{\max } \times\left(N_{\max }+1\right)$来允许未被关联的目标。矩阵中为$1$的位置表示一个关联或者未被关联的目标，其他情况为$0$。

匹配损失由前向匹配损失$\mathcal{L}_{\text {matcl }}^{b w d}$和后向匹配损失$\mathcal{L}_{\text {matcl }}^{b w d}$平均组成，前者表示$t$帧和$t-n$帧的匹配误差，后者相反。具体定义如下。

$$\begin{aligned} \mathcal{L}_{\text {match }}^{*} &=\sum_{i=1}^{N_{\max }} \sum_{j=1}^{N_{\max }+1} M^{*}[i, j] \log \left(\hat{A}^{*}[i, j]\right), \\ \mathcal{L}_{\text {match }} &=\frac{\mathcal{L}_{\text {match }}^{f w d}+\mathcal{L}_{\text {match }}^{b w d}}{2\left(N_{t}+N_{t-n}\right)} \end{aligned}$$

整个网络的训练损失由匹配损失和检测器损失构成，它俩的相对权重网络自动学习得到。

$$\mathcal{L}_{\text {joint }}=\frac{1}{e^{\lambda_{1}}}\left(\frac{\mathcal{L}_{\text {detect }}^{t}+\mathcal{L}_{\text {detect }}^{t-n}}{2}\right)+\frac{1}{e^{\lambda_{2}}} \mathcal{L}_{\text {match }}+\lambda_{1}+\lambda_{2}$$

## 实验

数据集使用的是MOT16、MOT17和KITTI作为2D评估，nuScenes作为3D评估标准数据集。

首先是对比多个backbone检测器，效果如下图，发现CenterNet效果最好，所以后面都采用CenterNet作为backbone。

![](https://i.loli.net/2021/02/10/eg8FmGyZpckjEqB.png)

在MOT17、KITTI和nuScenes上的与其他方法对比如下，均达到了SOTA表现，作者还进行了效果分析和消融实验，感兴趣的可以查看原论文（包括补充材料）。

![](https://i.loli.net/2021/02/10/Qljqs2OTNoFvbht.png)

![](https://i.loli.net/2021/02/10/N6pkR2BD3tdZYFE.png)

![](https://i.loli.net/2021/02/10/zUVAYCOmIXDdoj3.png)

## 总结

论文提出了一种新的联合检测和跟踪的MOT方法，在多个基准上达到SOTA表现，可以在主流的目标检测器的基础上构建，非常灵活高效，是值得关注的MOT新方法。