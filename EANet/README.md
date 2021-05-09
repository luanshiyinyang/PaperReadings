# EANet解读

> 最近关于MLP的工作还是蛮多的，先是MLP-Mixer为视觉开了个新思路，接着EANet（即External Attention）和RepMLP横空出世，乃至最新的《Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet》质疑Transformer中attention的必要性。由于我个人对自注意力是比较关注的，因此EANet通过线性层和归一化层即可替代自注意力机制还是很值得关注的。

## 简介

自注意力其实早已在计算机视觉中有所应用，从较早的Non-Local到最近的Transformer，计算机视觉兜兜转转还是有回到全局感知的趋势。相比于卷积这种局部感知的操作，自注意力计算每个位置与其他所有位置信息的加权求和来更新当前位置的特征，从而捕获长程依赖（这在NLP中至关重要）获取全局信息。但是自注意力的劣势也是非常明显的，一方面自注意力对每个位置都要计算与其他位置的关系，这是一种二次方的复杂度，是非常消耗资源的；另一方面，自注意力对待每个样本同等处理，其实忽略了每个样本之间的潜在关系。针对此，清华计图团队提出了一种新的external attention（外部注意力），仅仅通过两个可学习的单元即可取代现有的注意力并将复杂度降到和像素数目线性相关，而且由于这个单元全数据集共享所以它能隐式考虑不同样本之间的关系。实验表明，由external attention构建的EANet在图像分类、语义分割、图像生成、点云分类和分割等任务上均达到甚至超越自注意力结构的表现，并且有着小得多的计算量和内存开销。

- 论文标题

    Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks
- 论文地址

    https://arxiv.org/abs/2105.02358v1
- 论文源码

    https://github.com/MenghaoGuo/-EANet

## EA

论文中关于自注意力机制的发展就不多赘述了，这一节按照论文的思路来阐述一下External Attention。

### Self-Attention

首先，不妨来回顾一下原始的自注意力机制，如下图a所示，给定输入特征图$F \in \mathbb{R}^{N \times d}$，这里的$N$表示像素数目而$d$则表示特征的维度也就是通道数目。自注意力首先通过三个线性投影将输入变换为query矩阵$Q \in \mathbb{R}^{N \times d^{\prime}}$、key矩阵$K \in \mathbb{R}^{N \times d^{\prime}}$和value矩阵$V \in \mathbb{R}^{N \times d}$，从而按照下面的式子计算自注意力的输出，**这两个式子分别表示根据Query和Key计算相似性权重，然后将权重系数作用于Value得到加权求和的注意力结果。** 至于这里的点积相似性只是相似性计算的一种方式而已。

$$
\begin{aligned}
&A =(\alpha)_{i, j}=\operatorname{softmax}\left(Q K^{T}\right) \\
&F_{\text {out }} =A V
\end{aligned}
$$

在上面这个式子中，$A \in \mathbb{R}^{N \times N}$表示注意力矩阵并且$\alpha_{i, j}$就是第$i$个像素和第$j$个像素之间的逐对相似性。

![](https://i.loli.net/2021/05/09/LbHqerKGm5kCdUP.png)

既然QKV的计算都是线性变换，因此可以省略这个过程直接对输入特征进行点积相似度的计算，从而得到注意力图，因而简化版的自注意力如下式，这对应上图的b。然而，这种算法虽然简化了，但是$\mathcal{O}\left(d N^{2}\right)$的计算复杂度大大限制了自注意力的使用，特别是用于图像这种像素众多的数据上，因此很多的工作都是patch之间计算自注意力而不是在pixel上计算。

$$
\begin{aligned}
&A =\operatorname{softmax}\left(F F^{T}\right) \\
&F_{\text {out }} =A F
\end{aligned}
$$

### External Attention

论文作者通过可视化注意力图，发现大部分像素其实只和少量的其他像素密切相关，因此构建一个N-to-N的注意力矩阵其实是非常冗余的。因此自然产生一个想法，能否通过少量需要的值来精炼原始的输入特征呢？这就诞生了external attention模块来替代原始的注意力，它在输入特征和一个外部单元$M \in \mathbb{R}^{S \times d}$计算注意力。

$$
\begin{aligned}
&A =(\alpha)_{i, j}=\operatorname{Norm}\left(F M^{T}\right) \\
&F_{\text {out }} =A M
\end{aligned}
$$

上面的式子表示EA的计算过程，这里的M是一个独立于输入的可学习参数，它相当于整个数据集的memory，因此它是所有样本共享的，可以隐式考虑所有样本之间的影响。A是通过先验推断出的注意力图，它类似于自注意力那样进行标准化，然后根据A和M对输入特征更新。

实际上，为了增强EA的表达能力，与自注意力相似，论文采用了两个记忆单元分别是$M_k$和$M_v$分别作为key和value，因此External Attention可以表示为下面修改后的式子。

$$
\begin{aligned}
&A =\operatorname{Norm}\left(F M_{k}^{T}\right) \\
&F_{\text {out }} =A M_{v}
\end{aligned}
$$

从这个式子和上面的图c不难发现，EA的计算过程其实就是线性运算和Norm层，源码中线性层采用的是1x1卷积实现的，Norm层的实现下面会提到。整个EA的计算复杂度其实为$\mathcal{O}(d S N)$，S和d都是超参数，这个复杂度因此和像素数目线性相关了，通过修改S的大小可以方便地控制计算的复杂度，事实上，实验中，作者发现S设置为64效果就已经很好了。因此，EA通过线性层就替代了自注意力机制并且非常高效轻量，非常适用于大尺度的输入特征图。

### Normalization

在自注意力中，Softmax常被用于注意力图的标准化以实现$\sum_{j} \alpha_{i, j}=1$，然而注意力图常常通过矩阵乘法计算得到，不同于余弦相似度，这种算法的相似度是尺度敏感的。因此作者对double-attention优化得到了下面这个分别行列进行标准化的方法。

$$
\begin{aligned}
(\tilde{\alpha})_{i, j} &=F M_{k}^{T} \\
\alpha_{i, j} &=\frac{\exp \left(\tilde{\alpha}_{i, j}\right)}{\sum_{k} \exp \left(\tilde{\alpha}_{k, j}\right)} \\
\alpha_{i, j} &=\frac{\alpha_{i, j}}{\sum_{k} \alpha_{i, k}}
\end{aligned}
$$

下图是EANet的结构示意图，事实上就是在head之前添加了一个EA模块，关于EA模块的实现可以参考下面官方开源的Jittor的实现代码。

![](https://i.loli.net/2021/05/09/5r2HRbVB6ZP781D.png)

```python
class External_attention(Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm(c))        

        self.relu = nn.ReLU()

    def execute(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        attn = nn.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x
```

## 实验

为了证明方法的有效性，作者进行了大量任务的实验，包括图像分类、语义分割、图像生成、点云分类和点云分割，我这里就只列一下图像分类数据集ImageNet上的结果，在其他任务上EANet均能达到SOTA效果。

![](https://i.loli.net/2021/05/09/YLMjBPZlHwaenIK.png)

此外，关于自注意力及其变种也进行了计算消耗的比较，EA真的是非常轻量。

![](https://i.loli.net/2021/05/09/479WPK63BmiSsEV.png)

## 总结

论文提出了一种新的轻量级视觉注意力结构External Attention，克服了自注意力的超大计算量和缺乏样本多样性考虑的缺点，通过简单的线性层即可实现自注意力，并且效果相当不错，是很值得关注的一个方法。本文也只是我本人从自身出发对这篇文章进行的解读，想要更详细理解的强烈推荐阅读原论文。最后，如果我的文章对你有所帮助，欢迎一键三连，你的支持是我不懈创作的动力。