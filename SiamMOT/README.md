# SiamMOT解读

> AWS的一篇新的MOT工作，将孪生跟踪器引入多目标跟踪中进行运动建模并获得了SOTA表现。

## 简介

通过引入一个基于区域的孪生多目标跟踪网络，设计了一个新的online多目标跟踪框架，名为SiamMOT。SiamMOT包含一个运动模型来估计两帧之间目标的移动从而关联两帧上检测到的目标。为了探索运动建模如何影响其跟踪能力，作者提出了孪生跟踪器的两种变体，一种隐式建模运动，另一种显式建模运动。在三个MOT数据集上进行实验，作者证明了运动建模对于多目标跟踪的重要性并验证了SiamMOT达到SOTA的能力。而且，SiamMOT是非常高效的，它可以在单个GPU上720P视频上达到17的FPS。

- 论文标题

    SiamMOT: Siamese Multi-Object Tracking
- 论文地址

    http://arxiv.org/abs/2105.11595
- 论文源码

    https://github.com/amazon-research/siam-mot

## 介绍

多目标跟踪（Multiple Object Tracking，MOT）任务指的是检测出每一帧上的所有目标并跨帧在时间上将其关联起来形成轨迹。早期的一些工作将数据关联视为TBD范式下的图优化问题，在这类方法中一个节点代表一个检测框而一条边则编码两个节点链接到一起的可能性（或者说相似度）。实际上，这些方法往往采用视觉线索和运动线索的组合来表示一个节点，这通常需要比较大的计算量。而且，他们通常会构建一个很大的离线图，基于这个图做求解并不容易，这就限制了这类方法在实时跟踪上的可能性。

![](https://i.loli.net/2021/06/04/n32RFQpwEYC5ZLP.png)

最近，online方法开始兴起，它们在实时跟踪场景中更受欢迎。它们更加关注于改进相邻帧上的关联而不是基于较多帧构建离线图进行关联。Tracktor等方法的诞生将online MOT的研究推向了一个新的高峰，使得这个领域的研究如火如荼了起来。

这篇论文中，作者探索了以SORT为基础的一系列online多目标跟踪方法中运动建模的重要性。在SORT中，一个更好的运动模型是提高跟踪精度的关键，原始的SORT中采用基于简单几何特征的卡尔曼滤波进行运动建模，而最近的一些SOTA方法学习一个深度网络来基于视觉和几何特征进行位移预测，这极大地提高了SORT的精度。

作者利用基于区域的孪生多目标跟踪网络来进行运动建模的探索，称其为**SiamMOT**。作者组合了一个基于区域的检测网络（Faster R-CNN）和两个思路源于孪生单目标跟踪的运动模型（分别是隐式运动模型（IMM）和显式运动模型（EMM））。不同于CenterTrack基于点的特征进行隐式的目标运动预测，SiamMOT使用基于区域的特征并且开发了显式的模板匹配策略来估计模板的运动，这在一些挑战性的场景下更加具有鲁棒性，比如高速运动的场景下。

作者还通过额外的消融实验证明目标级别的运动建模对鲁棒多目标跟踪至关重要，特别是在一些挑战性场景下。而且，实验表明，SiamMOT的运动模型可以有效提高跟踪的性能，尤其是当相机高速运动或者行人的姿态变化较大时。

最后，说明一下，论文中提到的孪生跟踪器和通常所说的孪生网络是不一样的，孪生网络的目的是学习两个实例之间的**亲和度函数**，而孪生跟踪器则学习一个**匹配函数**，该函数用于在一个较大的上下文区域内找到一个匹配的检测框。

## SiamMOT

SiamMOT是基于Faster R-CNN构建的，Faster R-CNN是一个非常流行的目标检测器，它包含一个区域推荐网络（RPN）和一个基于区域的检测网络。在标准的Faster R-CNN上，SiamMOT添加了一个基于区域的孪生跟踪器来建模实例级别的运动。下图是整个SiamMOT的框架结构，它以两帧图像$\mathbf{I}^{t}, \mathbf{I}^{t+\delta}$作为输入，并且已有第$t$帧上的检测框集合$\mathbf{R}^{t}= \left\{R_{1}^{t}, \ldots R_{i}^{t}, \ldots\right\}$。在SiamMOT中，检测网络输出第$t+\delta$帧的检测结果集$\mathbf{R}^{t+\delta}$，跟踪器则将$\mathbf{R}^{t}$传播到$t+\delta$帧上以生成预测框集合$\tilde{\mathbf{R}}^{t+\delta}$。接着，$t$帧上的目标在$t+\delta$帧上的预测框和$t+\delta$帧上的检测框进行匹配，从而关联起来形成轨迹，这个思路和SORT中是一样的。

![](https://i.loli.net/2021/06/04/TYWjO4BJZmikVzp.png)

从上面的叙述不难看出来，和SORT相比作者最核心的一个工作实际上是构建了一个更好的运动模型，因此下面我们首先来看看这个孪生跟踪器是如何建模目标的运动的，以及它的两种变种，接着再叙述一下训练和推理的一些细节。

### Motion modelling with Siamese tracker

给定第$t$帧的实例$i$，孪生跟踪器根据其在$t$帧中的位置在$t+\delta$帧的一个局部窗口范围内搜索对应的实例，形式上表述如下。这里的$\mathcal{T}$表示参数为$\Theta$的孪生跟踪器，$\mathbf{f}_{R_{i}}^{t}$则是根据检测框$R_{i}^{t}$在$t+\delta$帧上获得的搜索区域$S_{i}^{t+\delta}$提取的特征图。而搜索区域$S_{i}^{t+\delta}$的获得则通过按照比例因子$r$(r>1)来扩展检测框$R_{i}^{t}$来获得，拓展前后具有同样的集合中心，如上图中的黄色实线框到黄色虚线框所示。当然，不管是原来的检测框还是拓展后的预测框，获得其特征$\mathbf{f}_{R_{i}}^{t}$和$\mathbf{f}_{S_{i}}^{t+\delta}$的方式都是不受大小影响的RoIAlign层。孪生跟踪器输出的结果有两个，其中$\tilde{R}_{i}^{t+\delta}$为预测框，而$v_{i}^{t+\delta}$则是预测框的可见置信度，若该实例在区域$S_{i}^{t+\delta}$是可见的，那么$\mathcal{T}$将会产生一个较高的置信度得分$v_{i}^{t+\varepsilon}$，否则得分会较低。

$$
\left(v_{i}^{t+\delta}, \tilde{R}_{i}^{t+\delta}\right)=\mathcal{T}\left(\mathbf{f}_{R_{i}}^{t}, \mathbf{f}_{S_{i}}^{t+\delta} ; \Theta\right)
$$

这个公式的建模过程其实和很多孪生单目标跟踪器的思路非常类似，因此在多目标跟踪场景下只需要多次使用上式即可完成每个$R_{i}^{t} \in \mathbf{R}^{t}$的预测。重要的是，SiamMOT允许这些操作并行运行，并且只需要计算一次骨干特征（由于RoIAlign），这大大提高了在线跟踪推理的效率。

作者发现，运动建模对于多目标跟踪任务至关重要，一般在两种情况下$R^{t}$和$R^{t+\delta}$会关联失败，一是$\tilde{R}^{t+\delta}$没有匹配上正确的$R^{t+\delta}$，二是对于$t+\delta$帧上的行人得到的可见得分$v_{i}^{t+\delta}$太低了。

此前的很多工作为了实现$R_{i}^{t}$到$\tilde{R}^{t+\delta}$的预测，一般采用将两帧的特征输入网络中，所以它们都是隐式建模实例的运动。然而，很多单目标跟踪的研究表明，在极具挑战性的场景下，细粒度的空间级监督对于显式学习一个鲁棒性的目标匹配函数是非常重要的。因此，作者提出了两种不同的孪生跟踪器，一个是隐式运动模型一个是显式运动模型。

### IMM

![](https://i.loli.net/2021/06/04/p4riskjYRzKNGcd.png)

首先来看隐式运动模型（Implicit motion model，IMM），它通过MLP来估计目标两帧间的运动。具体而言，它先将特征$\mathbf{f}_{R_{i}}^{t}$和$\mathbf{f}_{S_{i}}^{t+\delta}$按通道连接在一起然后送入MLP中预测可见置信度$v_i$和相对位置和尺度偏移，如下所示，其中$\left(x_{i}^{t}, y_{i}^{t}, w_{i}^{t}, h_{i}^{t}\right)$为目标框的四个值，通过$R_{i}^{t}$和$m_{i}$我们可以轻易解出$\tilde{R}^{t+\delta}$。

$$
m_{i}=\left[\frac{x_{i}^{t+\delta}-x_{i}^{t}}{w_{i}^{t}}, \frac{y_{i}^{t+\delta}-y_{i}^{t}}{h_{i}^{t}}, \log \frac{w_{i}^{t+\delta}}{w_{i}^{t}} \log \frac{h_{i}^{t+\delta}}{h_{i}^{t}}\right]
$$

给定$\left(R_{i}^{t}, S_{i}^{t+\delta}, R_{i}^{t+\delta}\right)$，IMM的训练可以采用下面的损失来进行，其中$v_{i}^{*}$和$m_{i}^{*}$是根据$R_{i}^{t+\delta}$计算的Gt标签，$\mathbb{1}$为指示函数，$\ell_{focal}$为分类损失，$\ell_{\text {reg }}$为常用的smooth l1回归损失。

$$
\mathbf{L}=\ell_{\text {focal }}\left(v_{i}, v_{i}^{*}\right)+\mathbb{1}\left[v_{i}^{*}\right] \ell_{\text {reg }}\left(m_{i}, m_{i}^{*}\right)
$$

### EMM

![](https://i.loli.net/2021/06/04/8FrmCIDi9qGSdoX.png)

接着来看显式运动模型（Explicit motion model，EMM）。具体而言，如上图所示，它通过逐通道的互相关操作（*）来生成像素级别的响应图$\mathbf{r}_{i}$，该操作已被证明对于密集光流估计和实例级别的运动建模是很有效的。在SiamMOT中，这个操作用目标特征图$\mathbf{f}_{R_{i}}^{t}$和搜索图像特征图$\mathbf{f}_{S_{i}}^{t+\delta}$的每个位置计算相关性，得到$\mathbf{r}_{i}=\mathbf{f}_{S_{i}}^{t+\delta} * \mathbf{f}_{R_{i}}^{t}$，所以每个$\mathbf{r}_{i}[k,:,:]$表示一个相似程度。受到FCOS的启发，EMM使用全卷积网络$\psi$来检测$\mathbf{r}_{i}$中匹配的目标。

详细来看，$\psi$预测一个密集的可见置信度图$\mathbf{v}_{i}$来表示每个像素包含目标图像的可能性，再预测一个密集的定位图$\mathbf{p}_{i}$来编码该像素位置到框的左上角和右下角的偏移量。因此，处在$(x,y)$的目标可以通过$\mathcal{R}(\mathbf{p}(x, y))=[x-l, y-t, x+r, y+b]$解出边界框，其中$\mathbf{p}(x, y)=[l, t, r, b]$（两个角点的偏移）。最终，特征图可以通过下面的式子解码，此处的$\odot$表示逐元素相乘，$\boldsymbol{\eta}_{i}$是一个惩罚图，为每一个候选区域设置非负的惩罚得分。

$$
\begin{array}{r}
\tilde{R}_{i}^{t+\delta}=\mathcal{R}\left(\mathbf{p}_{i}\left(x^{*}, y^{*}\right)\right) ; \quad v_{i}^{t+\delta}=\mathbf{v}_{i}\left(x^{*}, y^{*}\right) \\
\text { s.t. }\left(x^{*}, y^{*}\right)=\underset{x, y}{\operatorname{argmax}}\left(\mathbf{v}_{i} \odot \boldsymbol{\eta}_{i}\right)
\end{array}
$$

惩罚得分的计算如下式，其中$\lambda$是一个加权系数（$0 \leq \lambda \leq 1$），$\mathcal{C}$则是关于目标区域$\mathcal{R}_{i}^{t}$几何中心的余弦窗口函数，$\mathcal{S}$是关于候选区域$\mathbf{p}(x, y)$和$R_{i}^{t}$之间的相对尺度变化的高斯函数。惩罚图$\boldsymbol{\eta}_{i}$的引入是为了阻止跟踪过程中剧烈的运动。

$$
\boldsymbol{\eta}_{i}(x, y)=\lambda \mathcal{C}+(1-\lambda) \mathcal{S}\left(\mathcal{R}(\mathbf{p}(x, y)), R_{i}^{t}\right)
$$

考虑$\left(R_{i}^{t}, S_{i}^{t+\delta}, R_{i}^{t+\delta}\right)$，EMM的训练损失如下式子所示。其中$(x,y)$表示$S_{i}^{t+\delta}$中的所有有效位置，$\ell_{r e q}$是用于回归的IoU损失，$\ell_{\text {focal }}$是用于分类的损失。$\mathbf{v}_{i}^{*}$和$\mathbf{p}_{i}^{*}$是像素级的GT图。如果$(x,y)$在$R_{i}^{* t+\delta}$的范围内那么$\mathbf{v}_{i}^{*}(x, y)=1$，否则为0；$\mathbf{p}_{i}^{*}(x, y)=\left[x-x_{0}^{*}, y-y_{0}^{*}, x_{1}^{*}-\right. \left.x, y_{1}^{*}-y\right]$，其中的$\left(x_{0}^{*}, y_{0}^{*}\right)$和$\left(x_{1}^{*}, y_{1}^{*}\right)$表示GT框的两个角点的坐标。此外，作者还修改了回归任务，添加了一个$w(x, y)$表示中心度，有$w(x, y)=\sqrt{\frac{\min \left(x-x_{0}, x_{1}-x\right)}{\max \left(x-x_{0}, x_{1}-x\right)} \cdot \frac{\min \left(y-y_{0}, y_{1}-y\right)}{\max \left(y-y_{0}, y_{1}-y\right)}}$。

$$
\begin{aligned}
\mathbf{L} &=\sum_{x, y} \ell_{\text {focal }}\left(\mathbf{v}_{i}(x, y), \mathbf{v}_{i}^{*}(x, y)\right) 
+\sum_{x, y} \mathbb{1}\left[\mathbf{v}_{i}^{*}(x, y)=1\right]\left(w(x, y) \cdot \ell_{r e g}\left(\mathbf{p}_{i}(x, y), \mathbf{p}_{i}^{*}(x, y)\right)\right)
\end{aligned}
$$

相比于IMM，EMM有两点改进。第一，它使用通道分离的相关性操作来允许网络显式学习相邻帧上同一个目标的相似性；第二，它采用一个细粒度的像素级监督，这有效减少了错误匹配。

### Training and Inference

训练和推理就不多细说了，主要就是在Faster R-CNN的基础上添加了一个运动预测损失，这里的运动损失上文已经说的很明白了，对多个样本区域累计求和即可。

$$
\ell=\ell_{r p n}+\ell_{\text {detect }}+\ell_{\text {motion }}
$$

## 实验

实验作者采用MOT17、TAO-person和Caltech Roadside Pedestrians (CRP) ，分别获得如下的实验结果，具体的消融实验可以参考原论文。

![](https://i.loli.net/2021/06/04/dKjZSH8yiE9lLVv.png)

![](https://i.loli.net/2021/06/04/A5zKVTMh1qxCZek.png)

![](https://i.loli.net/2021/06/04/tFCGn9pIBJcbmae.png)

## 总结

设计了一种基于区域的多目标跟踪框架SiamMOT，它可以同时完成检测和关联任务，这也是孪生结构在MOT领域的一次很不错的尝试，是很值得关注的工作。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。






