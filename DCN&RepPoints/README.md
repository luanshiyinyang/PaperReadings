# DCN & RepPoints解读

## 简介

近几年，Anchor-free的目标检测方法受到了很大的关注，究其原因，该类方法不需要像Anchor-base方法那样受限于anchor的配置（anchor的设置要求开发者对数据很了解）就可以获得不错的检测结果，大大减少了数据分析的复杂过程。Anchor-free方法中有一类方法是基于关键点的，它通过检测目标的边界点（如角点）来配对组合成边界框，RepPoints系列就是其代表之作，这包括了RepPoints、Dense RepPoints和RepPoints v2。不过，回顾更久远的历史，从模型的几何形变建模能力的角度来看，RepPoints其实也是对可变形卷积（Deformable Convolutional Networks，DCN）系列的改进，所以本文会从DCN开始讲起，简单回顾这几个工作对几何建模的贡献，其中，DCN系列包括DCN和DCN v2。

- [DCN](http://arxiv.org/abs/1703.06211)（ICCV2017）
- [DCN v2](http://arxiv.org/abs/1811.11168)（CVPR2019）
- [RepPoints](http://arxiv.org/abs/1904.11490)（ICCV2019）
- [Dense RepPoints](http://arxiv.org/abs/1912.11473)（ECCV2020）
- [RepPoints v2](http://arxiv.org/abs/2007.08508)（暂未收录）


## DCN

首先，我们来看DCN v1。在计算机视觉中，同一物体在不同的场景或者视角中未知的几何变化是识别和检测任务的一大挑战。为了解决这类问题，通常可以在数据和算法两个方面做文章。从数据的角度看来，通过充分的**数据增强**来构建各种几何变化的样本来增强模型的尺度变换适应能力；从算法的角度来看，设计一些**几何变换不变的特征**即可，比如SIFT特征。

上述的两种方法都很难做到，前者是因为样本的限制必然无法构建充分数据以保证模型的泛化能力，后者则是因为手工特征设计对于复杂几何变换是几乎不可能实现的。所以作者设计了Deformable Conv（可变形卷积）和Deformable Pooling（可变形池化）来解决这类问题。

### **可变形卷积**

顾名思义，可变形卷积的含义就是进行卷积运算的位置是可变的，不是传统的矩形网格，以原论文里的一个可视化图所示，左边的传统卷积的感受野是固定的，在最上层的特征图上其作用的区域显然不是完整贴合目标，而右边的可变形卷积在顶层特征图上自适应的感受野很好的捕获了目标的信息（这可以直观感受得到）。

![](https://i.loli.net/2020/11/13/19Arvci4ldpDfna.png)

那么可变形卷积是如何实现的呢，其实是通过针对每个卷积采样点的偏移量来实现的。如下图所示，其中淡绿色的表示常规采样点，深蓝色的表示可变卷积的采样点，它其实是在正常的采样坐标的基础上加上了一个偏移量（图中的箭头）。

![](https://i.loli.net/2020/11/13/HMiJ4mwNU9fPzjS.png)

我们先来看普通的卷积的实现。使用常规的网格$\mathcal{R}$在输入特征图$x$上进行采样，采样点的值和权重$w$相乘加和得到输出值。举个例子，一个3x3的卷积核定义的网格$\mathcal{R}$表示如下式，中心点为$(0,0)$，其余为相对位置，共9个点。

$$
\mathcal{R}=\{(-1,-1),(-1,0), \ldots,(0,1),(1,1)\}
$$

那么，对输出特征图$y$上的任意一个位置$p_0$都可以以下式进行计算，其中$\mathbf{p}_n$表示就是网格$\mathcal{R}$中的第$n$个点。

$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}\right)
$$

而可变形卷积干了啥呢，它对原本的卷积操作加了一个偏移量$\left\{\Delta \mathbf{p}_{n} \mid n=1, \ldots, N\right\}$，也就是这个偏移量使得卷积可以不规则进行，所以上面的计算式变为了下式。不过要注意的是，这个偏移量可以是小数，所以偏移后的位置特征需要通过双线性插值得到，计算式如下面第二个式子。

$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}+\Delta \mathbf{p}_{n}\right)
$$

$$
\mathbf{x}(\mathbf{p})=\sum_{\mathbf{q}} G(\mathbf{q}, \mathbf{p}) \cdot \mathbf{x}(\mathbf{q})
$$

至此，可变卷积的实现基本上理清楚了，现在的问题就是，这个偏移量如何获得？不妨看一下论文中一个3x3可变卷积的解释图（下图），图中可以发现，上面绿色的分支其实学习了一个和输入特征图同尺寸且通道数为$2N$的特征图（$N$为卷积核数目），这就是偏移量，之所以两倍是因为网格上偏移有x和y两个方向。

![](https://i.loli.net/2020/11/13/uemUOJDXLHWwcEi.png)

### **可变形RoI池化**

![](https://i.loli.net/2020/11/13/V1DoF9PZqim2jwd.png)

理解了可变形卷积，理解可变形RoI就没有太大的难度了。原始的RoI pooling在操作时将输入RoI划分为$k\times k = K$个区域，这些区域叫做bin，偏移就是针对这些bin做的。针对每个bin学习偏移量，这里通过全连接层进行学习，因此deformable RoI pooling的输出如下式（含义参考上面的可变卷积即可）。

$$
\mathbf{y}(i, j)=\sum_{\mathbf{p} \in \operatorname{bin}(i, j)} \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}+\Delta \mathbf{p}_{i j}\right) / n_{i j}
$$

**至此，关于DCN的解读就完成了，下图是一个来自原论文对的DCN效果的可视化，可以看到绿点标识的目标基本上被可变形卷积感受野覆盖，且这种覆盖能够针对不同尺度的目标。这说明，可变形卷积确实能够提取出感兴趣目标的完整特征，这对目标检测大有好处。**

![](https://i.loli.net/2020/11/13/xS8pHOFe72Wf3Jq.png)


## DCN v2

DCNv1尽管获得了不错的效果，然而还是存在着不少的问题，DCNv2中进行了一个可视化，对比了普通卷积、DCNv1和DCNv2的区别，下图中每个图片都有从上到下三个可视化，分别是采样点、有效感受野、有效分割区域。可以看出来，DCNv1虽然能覆盖整个目标，但是这种覆盖不够“精细”，会带来不少的背景信息的干扰。

![](https://i.loli.net/2020/11/13/kHfNovIC9XZcwy1.png)

为此，DCNv2提出了一些改进，总结下来就是：
1. 使用更多的可变卷积层
2. Modulated Deformable Modules（调制变形模块）
3. RCNN特征模仿指导训练

### **更多的可变卷积层**

DCNv1中，只在ResNet50的conv5中使用了3个可变形卷积，DCNv2认为更多的可变形卷积会有更强的几何变化建模效果，所以将conv3到conv5都换为了可变形卷积。之前之所以没有采用更多的可变形卷积，是因为当时没有在大型检测数据集上进行验证，致使精度提升不高。

### **调制可变卷积模块**

这才是本文比较大的突破之一，设计了调制可变卷积来控制采样点权重，这样就可以忽略掉不重要甚至有负效果的背景信息。下式是引入偏移量的可变卷积输出特征图的计算式，这是DCNv1的思路。

$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}+\Delta \mathbf{p}_{n}\right)
$$

上面我们不是发现DCNv1采样了很多无效区域吗，DCNv2则认为，从输入特征图上不仅仅需要学习偏移量，还需要学习一个权重来表示采样点区域是否感兴趣，不感兴趣的区域，权重为0即可。所以原来的计算式变为下式，其中我们所说的权重$\Delta m_{k} \in [0,1]$称为调制因子。**结构上的实现就是原来学习的offset特征图由2N个通道变为3N个通道，这N个通道的就是调制因子。**

$$
y(p)=\sum_{k=1}^{K} w_{k} \cdot x\left(p+p_{k}+\Delta p_{k}\right) \cdot \Delta m_{k}
$$

相应的，可变形池化也引入这样的调制因子，计算式变为下式，结构上的实现类似上面的调制可变卷积，这里就不详细展开了。

$$
y(k)=\sum_{j=1}^{n_{k}} x\left(p_{k j}+\Delta p_{k}\right) \cdot \Delta m_{k} / n_{k}
$$

### **RCNN特征模仿**

作者发现，RCNN和Faster RCNN的分类score结合起来，模型的表现会有提升。这说明，RCNN学到的关注在物体上的特征可以解决无关上下文的问题。但是将RCNN融入整个网络会降大大降低推理速度，DCNv2这里就采用了类似知识蒸馏的做法，把RCNN当作teacher network，让DCNv2主干的Faster RCNN获得的特征去模拟RCNN的特征。

![](https://i.loli.net/2020/11/13/mxRNAUnickufZOt.png)

整个训练设计如上图，左边的网络为主网络（Faster RCNN），右边的网络为子网络（RCNN）。用主网络训练过程中得到的RoI去裁剪原图，然后将裁剪到的图resize到224×224作为子网络的输入，最后将子网络提取的特征和主网络输出的特征计算feature mimicking loss，用来约束这2个特征的差异（实现上就是余弦相似度）。同时子网络通过一个分类损失（如下式）进行监督学习，因为并不需要回归坐标，所以没有回归损失。推理阶段因为没有子网络，所以速度不会有缺失。

$$
L_{\operatorname{mimic}}=\sum_{b \in \Omega}\left[1-\cos \left(f_{\mathrm{RCNN}}(b), f_{\mathrm{FRCNN}}(b)\right)\right]
$$

很多人不理解RCNN的有效性，其实RCNN这个子网络的输入就是RoI在原输入图像上裁剪出来的图像，这就导致不存在RoI以外区域信息（背景）的干扰，这就使得RCNN这个网络训练得到的分类结果是非常可靠的，以此通过一个损失函数监督主网络Faster RCNN的分类路线训练就能够使网络提取到更多RoI内部特征，而不是自己引入的外部特征。**这是一种很有效的训练思路，但这并不能算创新，所以DCNv2的创新也就集中在调制因子上了。而且，从这个训练指导来看，DCNv2完全侧重于分类信息，对采样点没有监督，因此只能学习分类特征而没有几何特征。**

DCNv2对DCNv1进行了一些粗暴的改进并获得了卓有成效的效果，它能将有效感受野更稳定地聚集在目标有效区域，尽管留下了一些遗憾，不过前人的坑总会被后人解决，那就是RepPoints的故事了。


## RepPoints

这篇文章发表于ICCV2019，新颖地提出使用点集的方式来表示目标，这种方法在不使用anchor的前提下取得了非常好的效果。很多人认为RepPoints是DCNv3，这是由于RepPoints也采用了可变形卷积提取特征，不过它对采用点有相应的损失函数（追求更高的可解释性），可以对DCN的采样点进行监督，让其学习有效的RoI内部信息。此外，它也弥补了之前DCNv2的遗憾，可以学习到目标的几何特征，，而不仅仅是分类特征。

总结一下，个人觉得，RepPoints主要针对了之前DCNv2的offset（偏移量）学习过于black box，难以解释，所以采用了定位和分类损失直接监督偏移量的学习，这样定位和识别任务会更加“精致”一些。从这个角度来看（如果仅仅是将RepPoints看作一种新的目标表示法未免太低估其价值了），它其实是对DCNv2的改进，至于作者是不是这个出发点，，可以参考作者的[知乎回答](https://www.zhihu.com/question/322372759/answer/798327725)，，总之，其对DCNv2的改进可以总结如下：

1. 通过定位和分类的损失直接监督可形变卷积的偏移量的学习，使得偏移量具有可解释性；
2. 通过采样点来直接生成伪框 (pseudo box)，不需要另外学习边界框，这样分类和定位建立起了联系。

当然，上面这些看法是从DCNv2的角度来看的，下面我们回归作者论文的思路来看看RepPoints是如何实现的。

![](https://i.loli.net/2020/11/13/s5pudHOwIBvNKYL.png)

首先，RepPoints（representative points，表示点）具体是怎样表示目标的呢，其实就像下图这样。显然，这和传统的bbox表示法（bounding box，边界框）不同。在目标检测任务中，bbox作为描述检测器各阶段的目标位置的标准形式。然而，其虽然容易计算，但它们仅提供目标的粗略定位，并不完全拟合目标的形状和姿态（因为bbox是个矩形，而目标往往不可能如此规则，所以bbox中必然存在冗余信息，这会导致特征的质量下降从而降低了检测器性能）。对此，提出了一种更贴合目标的细粒度目标表示RepPoints，其形式是一组通过学习自适应地置于目标之上的点，这种表示既限制了目标的空间位置又捕获了精确的语义信息。

RepPoints是一种如下的二维表示（我们一般用2D表示和4D表示来区分anchor-free和anchor-base方法），一个目标由$n$个点确定，在该论文中$n$设置为9。

$$
\mathcal{R}=\left\{\left(x_{k}, y_{k}\right)\right\}_{k=1}^{n}
$$

这种点表示可以通过转换函数$\mathcal{T}$转换为伪框（pseudo box），三种转换函数具体可以参考原论文3.2节。然后就是RepPoints的监督学习了，这里也不难实现，通过pseudo box就可以和GT的bbox进行定位监督了，而分类监督和之前的方法没有太大的区别。

![](https://i.loli.net/2020/11/13/d8LgoFJzXxHNGvk.png)

借此，作者设计了一个anchor-free检测框架RPDet，该框架在检测的各个阶段均使用RepPoints表示。通过下图的Pipeline其实就能理解这个网络架构：首先，通过FPN获得特征图；不同于其他单阶段方法一次分类一次回归得到最终结果，RPDet通过两次回归一次分类得到结果（作者自称1.5阶段），分类和第二次回归均采用可变形卷积，可变形卷积可以很好的和RepPoints结合。因为它是在不规则点上进行的。可形变卷积的偏移量是通过第一次回归得到的，也就意味着偏移量在训练过程中是有监督的，而第一次回归的偏移量通过对角点监督得到。采用这种方式，后续的分类和回归特征均是沿着目标选取的，特征质量更高。

![](https://i.loli.net/2020/11/13/QDGWAlZou4HNOCn.png)


## Dense RepPoints

Dense RepPoints是RepPoints之后的一个成果，它将RepPoints拓展至实例分割任务，而方法就是采用更加密集的点集表示目标，如下图。

![](https://i.loli.net/2020/11/13/GhBfVzujlgDYObw.png)

Dense RepPoints在RepPoints的基础上进行了如下拓展，使用了更多的点并赋予了点属性。而在表示物体方面，Dense RepPoints采用上图第四个边缘掩码的方式表示目标，它综合了轮廓表示（表示边缘）和网格掩码（前后景区分，利于学习）的优点。

$$
\mathcal{R}=\left\{\left(x_{i}+\Delta x_{i}, y_{i}+\Delta y_{i}, \mathbf{a}_{i}\right)\right\}_{i=1}^{n}
$$

![](https://i.loli.net/2020/11/13/X7esbqK1hxBDYtQ.png)

最后pipeline和常规的思路很类似，唯一的问题就是采用点集在目标检测中尚可，用于分割会因为点数太多导致计算量大增，所以设计Group pooling;、Shared offset fields;、Shared attribute map来减少计算量，此外还发现点集损失比点损失更加合适等问题，具体可以查阅论文。

## RepPoints v2

最后，我们来看看最新的RepPoints v2，这是对原本RepPoints目标检测任务上的改进。验证（这里指的是分割任务）和回归是神经网络两类通用的任务。验证更容易学习并且准确，回归通常很高效并且能预测连续变化。因此，采用一定的方式将两者组合起来能充分利用它们的优势。RepPoints v2就是在RepPoints的基础上增加了验证模块，提升了检测器性能。

![](https://i.loli.net/2020/11/13/uyBAhK13zkHdGSb.png)

这篇文章细节不少，这里不展开了，直接对着上面的pipeline讲讲整体的思路。RepPoints V2在RepPoints方法基础上添加了一个验证（分割）分支，该分支主要包含两部分，一部分是角点预测，另一部分是目标前景分割。

如上图所示，训练阶段得到分割heatmap后，这个分割图和原始特征相加，作为回归特征的补充。不过，在推理阶段，在回归分支获取到目标位置后，利用该分割heatmap来对结果进行进一步修正。

总的来说，这个分支的添加对任务提升不小：一方面多任务往往会带来更好的效果；另一方面，利用分割图增强特征确实可以增强回归效果。


RepPoints v2工作有一定的可拓展性，它证明了在基于回归的方法上加上这个模块确实可以提升性能，后人要做的就是权衡精度和速度了。


## 总结

从DCN到RepPoints，本质上其实都是更精细的特征提取演变的过程。点集（RepPoints）方式只是显式的表现了出来而已，不过其确实能在精度和速度上取得非常好的平衡。以RepPoints或者类似的思路如今已经活跃在目标检测和实例分割任务中，推动着计算机视觉基础任务的发展，，这是难能可贵的。而且，跳出bbox的范式也诠释着，有时候跳出固有的范式做研究，会获得意想不到的效果。