# R-CNN 系列解读

## 简介

最近香港大学和加州伯克利开放了一篇新的基于 R-CNN 的工作 Sparse R-CNN，这是一个类似于 DETR 的稀疏端到端目标检测框架，使用少量可学习的 proposal 即可达到 SOTA 性能。这为稀疏端到端目标检测开辟了一条新的路，也将更多研究者的目标吸引了过来。本文从 R-CNN 开始，逐步讲解 R-CNN 系列目标检测算法的发展优化之路。，它们都是沿着 region proposal 这个思路的，只是处理方式大不相同。下面的图是我非常喜欢的目标检测综述《Object detection in 20 years: A survey》中归纳的目标检测里程碑式作品，可以看到，R-CNN 系的三个算法是目标检测发展史上避不开的话题。

- [R-CNN](https://arxiv.org/abs/1311.2524)（CVPR2014）
- [Fast R-CNN](https://arxiv.org/abs/1504.08083) （ICCV2015）
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)（NIPS2015）
- [Sparse R-CNN](http://arxiv.org/abs/2011.12450)（暂未收录）

![](https://i.loli.net/2020/11/30/I6oZrcKxTzlfQ8q.png)

## 目标检测思路

在聊具体的算法之前，我们首先要知道目标检测永不过时的核心思路：**目标定位**+**目标分类**，这是任务本身决定的，后来所谓的 two-stage 方法和 one-stage 方法其实都还是这个思路，只是处理的技巧不同罢了。

我们知道，**目标分类**这个任务已经基本上被 CNN 所解决，所以只要在 pipeline 中引入卷积分类模型就能确保不错的分类精度。留给目标检测的核心问题其实就是**目标定位**。我们当然会想到很直接很粗暴的想法：遍历图片中所有可能的位置，搜索不同大小、宽高比的所有区域，逐个检测其中是否存在某个目标，以概率较大的结果作为输出。这个就是传统的滑窗方法，这个方法显然是一种密集采样的方式，类似后来的 anchor 策略，R-CNN 系列则采用了一种候选框提名的方式减少滑窗的复杂性形成了 Dense-to-Sparse 方法，Sparse R-CNN 则完全采用 Sparse 策略，获得了低维的 proposal 输入。

![](https://i.loli.net/2020/11/30/cQYRqUzH1SvgA5I.png)

## **R-CNN**

R-CNN 是 2014 年出现的一篇目标检测方法，为后来目标检测的研究奠定了基础，R-CNN 结合了传统的滑窗法和边框回归思路，开创性地提出了候选区的概念（Region proposals），先从输入图像中找到一些可能存在对象的候选区，这个过程称为 Selective Search（根据输入图像的特征如颜色等提取出候选区）。在 R-CNN 中这些候选区大概会有 2000 个，对这些区域进行分类和微调即可完成目标检测。候选区的提出大大减少了目标定位的时间，提高了目标检测的效率。然而由于深度学习分类器的存在，非常耗费算力，且 proposal 的生成都是在 CPU 上完成的，因此 R-CNN 处理一张图片大概需要 49 秒，这和实时检测差的还很远。

![](https://i.loli.net/2020/11/30/c4A15kyVlfmbi9v.png)

下面我们来理解一下 R-CNN 的具体训练流程，它的整体思路如上图，具体细节如下。

- **Region proposals**：使用 Selective Search 方法生成大约 2000 个候选框（proposals），与 ground truth box（以下简称 GT box）的 IOU（交并比）大于 0.5 则认为这个 proposal 为值得学习的 positive（正例），否则为 negative（负例，即 background）。
- **Supervised pre-training and Domain-specific fine-tuning**：在 ImageNet 上预训练 VGG16 这样的视觉深度模型用于提取图像特征，然后在 VOC 数据集上 fine tune 为 21 类（包括背景），这样就得到了一个优质的特征提取器，只要将 proposal 区域 resize 到这个卷积模型的需要尺寸就能提取特征，提取的特征用于目标的分类和边框回归。
- **Object category classifiers and Bounding box regression**：分类器使用 SVM 进行分类。分类完成后，边框回归模型用于精调边框的位置，论文对每一类目标使用了简单的线性回归模型。具体训练的细节如优化器等配置这里就不细细展开了。

测试时（推理时）思路类似，先对图像产生大量的 proposal，然后进行分类和边框回归，然后使用 NMS 算法（非极大值抑制）去除冗余的边框得到最终的检测结果。

**至此，我们理解了 R-CNN 的思路，尽管它不像传统方法那样穷举可能的目标位置，然而 Selective Search 方法提取的候选框多达 2000 个，这 2000 个框都需要 resize 后送入 CNN 和 SVM 中，这就是说，对一幅图像的检测需要对 2000 个图像进行 CNN 特征提取，这在这个算力爆炸的时代都是很费时的，何况在当时，这就导致 R-CNN 平均处理一张图片需要 49 秒。**

那么，有没有办法提高检测的速度呢？其实很容易发现，这 2000 个 proposal 都是图像的一部分，完全可以对图像提取一次特征，然后只需要将 proposal 在原图的位置映射到特征图上，就得到该 proposal 区域的特征了。这样，对一幅图像只需要一次得到卷积特征，然后将每个 proposal 特征送入全连接进行后续任务的端到端学习。

## **Fast R-CNN**

在上面一节的最后，得到一个优化 R-CNN 的思路，SPPNet 沿着这个思路走了下去，其中最关键的一个问题就是 crop/warp 操作使得 proposal 尺寸固定送入了 CNN 中，这是为了全连接层固定输入的要求，但是导致了图像的失真，因此作者设计了金字塔池化来使得任意输入的图像获得同维的特征。有了这个方法，就能对 R-CNN 进行优化了，只对原图进行一次卷积计算，便得到整张图的卷积特征 feature map，然后找到每个候选框在 feature map 上的映射 patch，将此 patch 作为每个候选框的卷积特征输入到 SPP layer（空间金字塔池化层） 和之后的层，完成特征提取工作。如此，对一幅图，SPPNet 只需要提取一次卷积特征，提速 R-CNN 百倍左右。

Fast R-CNN 吸取了 SPPNet 的思路，对 R-CNN 进行了改进，获得了进一步的性能提升。它采用了单层空间金字塔池化（文中为 RoI Pooling）代替原来的池化操作使得不同的 proposal 在最后一层 feature map 上映射得到的区域上经过池化后得到同维特征。原本的 proposal 位置经过下采样倍率除法就可以得到其在特征图上的位置从而获得特征，这个不难理解，遇到除不尽的就取整即可。**问题是，RoI Pooling 是如何保证任意尺度的特征图都能得到同维的输出特征呢 2？RoI Pooling 采用了一个简单的策略，那就是网格划分。**

假设我有个$w=7,h=5$的 RoI 特征矩阵，我需要得到$W=2,H=2$的输出特征，那么其实只需要将$7\times 5$的矩阵划分为四个部分，每个部分取最大元素值就行，不过，由于非方阵，，难以等分，所以 RoI Pooling 会先进行取整，这样$w$被划分为 3 和 4，$h$被划分为 2 和 3。这样就完成了 RoI 区域的固定维度特征提取，有意思的是，**上述过程其实发生了两次取整，一次计算 proposal 在 feature map 上的坐标，一次计算 feature map 上的 RoI 区域的网格划分，这个过程是有问题的，所以后来提出了 RoI Align。**

![](https://i.loli.net/2020/11/30/wVenWRkUbPrCfY3.png)

上图所示的就是 Fast R-CNN 的 pipeline 设计，现在一次卷积特征提取就能得到 2000 个 proposal 的特征了，后续的任务作者也做了改变。彼时，**多任务学习**已经初有起色，原先在 R-CNN 中是先提 proposal，然后 CNN 提取特征，之后用 SVM 分类器，最后再做 bbox regression 的，在 Fast R-CNN 中则将对 proposal 的分类和边框回归同时通过全连接层完成，利用一个上图所示的多任务学习网络。实验也证明，这两个任务能够共享卷积特征，并相互促进。

其他方面，和 R-CNN 处理思路类似，其实，除了 proposal 的生成，整个 Fast R-CNN 已经是端到端训练的了，Fast-RCNN 很重要的一个贡献是为 Region Proposal + CNN 这种实时检测框架提供了一丝可能性，这也为后来的 Faster R-CNN 埋下伏笔。

最后提一下，Fast R-CNN 单图平均处理时间缩短到了 2.3 秒，这是一个巨大的突破，不过，对实时检测而言还是有点慢，而下一节的 Faster R-CNN 则将实时检测的可能性化为了现实。

## **Faster R-CNN**

在 Fast R-CNN 中留下了一个遗憾，也是制约网络速度和端到端训练的部件，那就是 Selective Search 求得候选框，这个方法其实非常耗时，Faster R-CNN 就设计了一个 Region Proposal Network（RPN，区域推荐网络）代替 Selective Search，同时也引入了 anchor box 应对目标形状的变化问题（anchor 可以理解为位置和大小固定的 bbox，或者说事先设置好的固定的 proposal）。

![](https://i.loli.net/2020/11/30/LUgclEvd8bXsipn.png)

具体而言，就像上图一样，RPN 网络会在最后一层特征图上对大量的预定义的 anchor 进行正例挑选（这是个分类任务）以及边界框回归（这是个回归任务），通过这两个任务会得到较为准确的 proposal，这就完成了 Fast R-CNN 中 Selective Search 得到的 proposal 了。接着，这些 proposal 会被用于和 Fast R-CNN 中一样的目标分类和边框回归来得到精确的检测结果。

![](https://i.loli.net/2020/11/30/KWs7YZC2EbJtmFv.png)

RPN 的引入使得端到端目标检测框架得以实现，这大大提高了目标检测的速度，它单图检测时间只需要 0.2 秒。直到今天，Faster R-CNN 这种基于候选框的二阶段目标检测框架仍然是目标检测的一个重要研究领域，Faster R-CNN 也成为目标检测领域绕不过的里程碑。

## **Sparse R-CNN**

距离 Faster R-CNN 的出现已经过去了 5 年的时间，这 5 年是目标检测高速发展的 5 年，在 Faster R-CNN 之后出现了 anchor-based 的单阶段目标跟踪范式，它主张去掉 RPN 直接对 anchor 进行处理，在速度上获得了卓越的表现，如 SSD、RetinaNet 等，它们成为了目标检测的新 SOTA。再后来，受到工业界普遍关注的高效[anchor-free 方法](https://zhouchen.blog.csdn.net/article/details/108032597)为目标检测开拓了全新的方向，如 FCOS、CenterNet 等。今年，FaceBook 发布了使用 Transformer 构建的 DETR 框架，它只需要少量的低维输入即可获得 SOTA 检测表现，这为稀疏检测带来了一丝契机。

最近，沿着目标检测中 Dense 和 Dense-to-Sparse 的框架的设计思路，Sparse R-CNN 建立了一种彻底的稀疏框架，它完全脱离了 anchor、RPN 和 NMS 后处理等设计。

首先，我们来回顾一下目标检测的范式。最著名的为**Dense范式**，密集检测器的思路从深度学习时代之前沿用至今，这种范式最典型的特征是会设置大量的候选（candidates），无论是anchor、参考点还是其他的形式，最后，网络会对这些候选进行过滤以及分类和边框微调。接着，取得过突出成果的**Dense-to-Sparse范式**最典型的代表就是上文所说的R-CNN系列，这种范式的典型特点是对一组较为稀疏的候选进行分类和回归，这组稀疏的候选其实还是来自密集检测中那种。

![](https://i.loli.net/2020/11/30/cQYRqUzH1SvgA5I.png)

很多人认为目标检测已经是个几乎solved problem，然而Dense范式总有一些问题难以忽略：较慢的NMS后处理、多对一的正负样本分类、candidates的设计等。这些问题可以通过Sparse范式解决，DETR提出了一个解决方案：DETR中，candidates是一组sparse的可学习的object queries，正负样本分配是一对一的二分图匹配，而且，不需要NMS就能获得检测结果。然而，DETR使用的Transformer无形之中使得每个object query与全局特征图进行了注意力交互，这还是Dense的交互。

Sparse R-CNN作者认为，Sparse的检测框架不仅仅是稀疏的候选，也应该是稀疏的特征交互，因此提出了Sparse R-CNN。

![](https://i.loli.net/2020/11/30/bRAIPkMeNhSv7L9.png)

Sparse R-CNN的出发点是通过少量的（如100个）proposal代替RPN产生的动辄上千的proposal。不妨看一下上面的pipeline设计，Sparse R-CNN的proposal是一组可学习参数，就是上面绿色框中的$N*4$向量，其中$N$表示proposal的数目，一般几百个就够了，4代表物体边框的4个属性，为归一化后的中心点坐标$(x,y)$以及宽高（$w$和$h$），这个向量作为可学习参数和其他参数一起被网络优化，这就是整体的设计。

但是，如果这也就可以了那么前人早就实现了，这个学习到的proposal可以理解为什么呢？其实就是根据图像信息推理得到的可能出现物体的统计值，这种“肤浅”的proposal确定的RoI特征显然不足以用来精确定位和恶分类目标，事实上作者在Faster R-CNN基础上换成这种可学习的proposal后精度下降了20个点。于是，，作者提出了上图蓝色框中的特征层面的proposal，它和proposal box数目一致，是一个高维特征向量（256维左右）。**proposal box和proposal feature一一对应，proposal feature和proposal box提取出来的RoI feature做一对一的交互，使得RoI特征更利于分类和回归。这种交互如下图所示，类似于注意力机制，这种交互模块可以多个拼接以进一步精炼特征，所以该模块为Dynamic Instance Interactive Module，堆叠多个这样的模块构成了该框架的Dynamic Instance Interactive Head。** 由于交互是一对一的，所以特征的交互和proposal一样是稀疏的。

![](https://i.loli.net/2020/11/30/HopeQ7qyCt5wNul.png)

关于整体框架：backbone采用基于ResNet的FPN，Head是一组Dynamic Instance Interactive Head，上一个head的output features和output boxes作为下一个head的proposal features和proposal boxes。proposal features在与RoI features交互之前做self-attention。整体训练的损失函数是基于二分图匹配的集合预测损失。

![](https://i.loli.net/2020/11/30/n7b5JhgqB9PLIjy.png)

实验结果上看，在COCO数据集达到了SOTA表现。

## 总结

R-CNN和Fast R-CNN出现后的一段时期内，目标检测领域的一个重要研究方向是提出更高效的候选框，其中Faster R-CNN开创性提出RPN，产生深远影响。Sparse R-CNN以一组稀疏输入即可获得比肩SOTA的检测性能，为真正的端到端检测开拓了一条路。