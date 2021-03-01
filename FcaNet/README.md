# FcaNet解读

## 简介

注意力机制，特别是通道注意力机制在计算机视觉中取得了巨大的成功，很多工作着重于设计更加高效的通道注意力结构，却忽略了一个重要的问题，那就是为了得到每个通道全局表示所使用的全局平均池化（Global Average Pooling，GAP）真的合适吗？FcaNet这篇文章，作者从频域角度重新思考GAP，为了弥补了现有通道注意力方法中特征信息不足的缺点，将GAP推广到一种更为一般的2维的离散余弦变换（DCT）形式，通过引入更多的频率分量来充分的利用信息。设计的高效Fca模块甚至在现有的通道注意力方法基础上，只需要修改一行代码即可实现。

- 论文标题

    FcaNet: Frequency Channel Attention Networks

- 论文地址

    http://arxiv.org/abs/2012.11879

- 论文源码
	
	https://github.com/cfzd/FcaNet


## 介绍

注意力机制在计算机视觉中受到了广泛的关注，它让网络更加关注于部分重要的信息，按照作用维度的不同，我们将注意力分为空间注意力、通道注意力和自注意力，其中，由于简单高效，下图所示的通道注意力直接对不同的通道加权，成为一种主流的注意力范式。

![](https://i.loli.net/2021/01/18/lC34dQHptugA6M7.png)

SENet和ECANet致力于设计不同的通道加权函数，如全连接或者一维卷积，然后这些函数的输入都是每个通道一个标量，这个标量默认都是来自于GAP，这是因为GAP相当的简洁高效，但是GAP也有不可忽略的问题，那就是GAP没办法捕获丰富的输入表示，这就导致了经过GAP得到的特征缺乏多样性，这主要是因为GAP对一个通道所有空间元素取其均值，而这个均值其实不足以表达不同通道的信息。

作者对全局平均池化即GAP进行了理论上的分析，最终得出如下结论：首先，不同的通道有极大概率出现相同的均值，然而它们的语义信息是不同的，换句话说，GAP抑制的通道之间的多样性；其次，从频域角度来看，作者证明了GAP其实是离散余弦变换（DCT）的最低频分量，这其实忽略了很多其他有用的分量；最后，CBAM的成功也佐证了只使用GAP得到的信息是不足够的。

在这些结论的基础上，作者设计了一种新的高效多谱通道注意力框架。该框架在GAP是DCT的一种特殊形式的基础上，在频域上推广了GAP通道注意力机制，提出使用有限制的多个频率分量代替只有最低频的GAP。通过集成更多频率分量，不同的信息被提取从而形成一个多谱描述。此外，为了更好进行分量选择，作者设计了一种二阶段特征选择准则，在该准则的帮助下，提出的多谱通道注意力框架达到了SOTA效果。

## 方法论

### 通道注意力

通道注意力的权重学习如下式所示，它表示输入经过GAP处理后由全连接层学习并经过Sigmoid激活得到加权的mask。

$$
a t t=\operatorname{sigmoid}(f c(\operatorname{gap}(X)))
$$

然后，mask与原始输入经过下式逐通道相乘得到注意力操作后的输出。

$$
\tilde{X}_{:, i,:,:}=a t t_{i} X_{:, i,:, :}, \text { s.t. } i \in\{0,1, \cdots, C-1\}
$$

### 离散余弦变化

DCT是和傅里叶变换很相似，它的基本形式如下，$f \in \mathbb{R}^{L}$为DCT的频谱，$x \in \mathbb{R}^{L}$为输入，$L$为输入的长度。

$$
f_{k}=\sum_{i=0}^{L-1} x_{i} \cos \left(\frac{\pi k}{L}\left(i+\frac{1}{2}\right)\right), \text { s.t. } k \in\{0,1, \cdots, L-1\}
$$

进而，我们推广得到二维DCT如下，$f^{2 d} \in \mathbb{R}^{H \times W}$是二维DCT的频谱，$x^{2 d} \in \mathbb{R}^{H \times W}$是输入，$H$和$W$是输入的高和宽。

$$
\begin{array}{l}
f_{h, w}^{2 d}=\sum_{i=0}^{H-1} \sum_{j=0}^{W-1} x_{i, j}^{2 d} \underbrace{\cos \left(\frac{\pi h}{H}\left(i+\frac{1}{2}\right)\right) \cos \left(\frac{\pi w}{W}\left(j+\frac{1}{2}\right)\right)}_{\text {DCT weights }} \quad  
\text { s.t. } h \in\{0,1, \cdots, H-1\}, w \in\{0,1, \cdots, W-1\}
\end{array}
$$

同样，逆DCT变换的公式就如下了。
$$
\begin{array}{l}
x_{i, j}^{2 d}=\sum_{h=0}^{H-1} \sum_{w=0}^{W-1} f_{h, w}^{2 d} \underbrace{\cos \left(\frac{\pi h}{H}\left(i+\frac{1}{2}\right)\right) \cos \left(\frac{\pi w}{W}\left(j+\frac{1}{2}\right)\right)}_{\text {DCT weights }} \quad
\text { s.t. } i \in\{0,1, \cdots, H-1\}, j \in\{0,1, \cdots, W-1\} 
\end{array}
$$

上面两个式子中，为了简单起见，移除了一些常数标准化约束因子。DCT变换属于信号处理领域的知识，是JPEG图像压缩的核心算法，相当于是对重要信息的聚集。其实从这里可以看出来，DCT变换其实也是一种对输入的加权求和，式子中的余弦部分就是权重。因此，GAP这种均值运算可以认为是输入的最简单频谱，这显然是信息不足的，因此作者引出了下面的多谱通道注意力。

### 多谱通道注意力

这里作者首先按证明了GAP其实是二维DCT的特例，其结果和二维DCT的最低分量成比例。这个证明作者是令$h$和$w$都为0得到的，其中$f_{0,0}^{2 d}$表示二维DCT最低频分量，显然，结果来看它与GAP成正比。

$$
\begin{aligned}
f_{0,0}^{2 d} &=\sum_{i=0}^{H-1} \sum_{j=0}^{W-1} x_{i, j}^{2 d} \cos \left(\frac{0}{H}\left(i+\frac{1}{2}\right)\right) \cos \left(\frac{\theta}{W}\left(j+\frac{1}{2}\right)\right) \\
&=\sum_{i=0}^{H-1} \sum_{j=0}^{W-1} x_{i, j}^{2 d} \\
&=g a p\left(x^{2 d}\right) H W
\end{aligned}
$$

通过上面的结论，自然会想到将其他分量引入通道注意力中，首先，为了叙述方便，将二维DCT的基本函数记为$B_{h, w}^{i, j}=\cos \left(\frac{\pi h}{H}\left(i+\frac{1}{2}\right)\right) \cos \left(\frac{\pi w}{W}\left(j+\frac{1}{2}\right)\right)$，继而将逆二维DCT变换改写如下，

$$
\begin{array}{l}
x_{i, j}^{2 d}=\sum_{h=0}^{H-1} \sum_{w=0}^{W-1} f_{h, w}^{2 d} \cos \left(\frac{\pi h}{H}\left(i+\frac{1}{2}\right)\right) \cos \left(\frac{\pi w}{W}\left(j+\frac{1}{2}\right)\right) \\
\stackrel{简写为B}{=} f_{0,0}^{2 d} B_{0,0}^{i, j}+f_{0,1}^{2 d} B_{0,1}^{i, j}+\cdots+f_{H-1, W-1}^{2 d} B_{H-1, W-1}^{i, j} \\
\stackrel{GAP特殊形式结论}{=} g a p\left(x^{2 d}\right) H W B_{0,0}^{i, j}+f_{0,1}^{2 d} B_{0,1}^{i, j}+\cdots+f_{H-1, W-1}^{2 d} B_{H-1, W-1}^{i, j} \\
\text { s.t. } i \in\{0,1, \cdots, H-1\}, j \in\{0,1, \cdots, W-1\}
\end{array}
$$

由这个式子其实不难发现，此前的通道注意力只应用了第一项的最低频分量部分，而没有使用下式表示的后面其他部分，这些信息都被忽略了。

$$
X=\underbrace{g a p(X) H W B_{0,0}^{i, j}}_{\text {utilized }}+\underbrace{f_{0,1}^{2 d} B_{0,1}^{i, j}+\cdots+f_{H-1, W-1}^{2 d} B_{H-1, W-1}^{i, j}}_{\text {discarded }}
$$

基于此，作者设计了多谱注意力模块（Multi-Spectral Attention Module，），该模块通过推广GAP采用更多频率分量从而引入更多的信息。

首先，输入$X$被沿着通道划分为多块，记为$\left[X^{0}, X^{1}, \cdots, X^{n-1}\right]$，其中每个$X^{i} \in \mathbb{R}^{C^{\prime} \times H \times W}, i \in\{0,1, \cdots, n-1\}, C^{\prime}=\frac{C}{n}$，每个块分配一个二维DCT分量，那么每一块的输出结果如下式。

$$
\begin{aligned}
F r e q^{i} &=2 \mathrm{DDCT}^{u, v}\left(X^{i}\right) \\
&=\sum_{h=0}^{H-1} \sum_{w=0}^{W-1} X_{:, h, w}^{i} B_{h, w}^{u, v} \\
& \text { s.t. } i \in\{0,1, \cdots, n-1\}
\end{aligned}
$$

上式中的$[u, v]$表示2DDCT的分量下标，这就对每一块采用不同的频率分量了，因此下式得到最终的输出$Freq \in \mathbb{R}^{C}$就是得到的多谱向量，然后再将这个向量送入通道注意力常用的全连接层中进行学习得到注意力图。

$$
\text { Freq }=\operatorname{cat}\left(\left[\text { Fre } q^{0}, \text { Fre } q^{1}, \cdots, \text { Freq }^{n-1}\right]\right)
$$

$$
m s_{-} a t t=\operatorname{sigmoid}(f c(\text { Freq }))
$$

这就是全部的多谱注意力模块的设计了，现在，下图这个FcaNet整体框架中间的一部分就看得明白了，唯一留下的问题就是对分割得到的每个特征图块，如何选择$[u,v]$呢？事实上，对空间尺寸为$H\times W$的特征图，会有$HW$个频率分量，由此频率分量的组合共有$CHW$种，遍历显然是非常费时的，因此，文中设计了一种启发式的两步准则来选择多谱注意力模块的频率分量，其主要思想是先得到每个频率分量的重要性再确定不同数目频率分量的效果。具体而言，先分别计算通道注意力中采用各个频率分量的结果，然后，根据结果少选出topk个性能最好的分量。

![](https://i.loli.net/2021/01/18/6rVxBHFSzWh1y2L.png)

到这里，整个FcaNet的方法论就说明完了，下面作者还进行了一些讨论，包括计算复杂度和修改难度。复杂度方面，相比于SENet没有引入额外参数，因为2DDCT权重是预先计算好的，相比于SENet增加的计算量几乎可以忽略不记。此外，2DDCT可以认为是对输入的加权求和，因此它可以通过逐元素乘加实现，在原本通道注意力代码基础上前向计算只需要修改一行，其PyTorch实现如下图。

![](https://i.loli.net/2021/01/18/Buk6VcWI94Hsvy8.png)

## 实验

作者首先经过实验对比了单个频率分量的有效性，具体的实验配置可以查看原论文，不过，实验最终验证当$[u,v]$为[0,0]时效果是最好的，这就是GAP操作，下图同时也验证了深度模型对低频分量更关注的事实。但是，虽然其他分量效果不及GAP，但也是含有大量有效信息的，因此加进去是合理的。

![](https://i.loli.net/2021/01/18/pX18sD5VUnTjAP7.png)
接着，在知道了哪些分量更好的前提下，作者对分量的数目进行了实验，结果如下表，Number=1的GAP效果弱于其他各种Number选择的多谱注意力，Number=16时效果最好。

![](https://i.loli.net/2021/01/18/GRj4bQWqHakvB6d.png)

最后，作者也将本文设计的FcaNet和其他注意力网络进行对比，在分类、检测和分割等benchmark上和其他方法对比，在分类和检测上都有比较明显的提高，下图所示为ImageNet上分类任务结果，其他的我这里就不展示了，可以查看原文。

![](https://i.loli.net/2021/01/18/CwsgWMS1rPOKG9b.png)


## 总结

本文是依据可靠数学进行的理论创新，一改此前通道注意力的结构设计思路，将关注重心放在了GAP是DCT的特例这一推导下，从而设计了多谱通道注意力，取得了相当不错的成果。将思路转到频域还是非常新颖的，虽然一行代码的说法略显牵强，最终的性能也是坐等开源再说，总体来看，FcaNet还是值得了解的注意力文章。

