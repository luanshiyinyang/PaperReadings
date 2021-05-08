# EANet解读

> 最近关于MLP的工作还是蛮多的，先是MLP-Mixer为视觉开了个新思路，接着EANet（即External Attention）和RepMLP横空出世，乃至最新的《Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet》质疑Transformer中attention的必要性。由于我个人对自注意力是比较关注的，因此EANet通过线性层和归一化层即可替代自注意力机制还是很值得关注的。

## 简介

清华计图

- 论文标题

    Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks
- 论文地址

    https://arxiv.org/abs/2105.02358v1
- 论文源码

    https://github.com/MenghaoGuo/-EANet