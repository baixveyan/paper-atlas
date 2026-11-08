# 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds

作者：Ryousuke Yamada, Kohsuke Ide, Yoshihiro Fukuhara, Hirokatsu Kataoka, Gilles Puy, Andrei Bursuc, Yuki M. Asano。

分类：3d-vision；交付形式：Markdown 中文论文讲解。

核对日期：2026-11-08。以下实验均为作者报告；本地完成原文阅读与讲解自检，未复现实验。

## 一句话总结

LAM3C用网页房屋视频重建的有噪点云进行自监督预训练，以局部平滑和噪声一致性弥补真实三维扫描稀缺。

## 研究背景与动机

室内扫描采集昂贵，而看房视频丰富。核心问题不是视频能否重建，而是有缺洞、重影和尺度不一致的伪点云能否学出可迁移三维特征。无需扫描指预训练点云来源，不代表下游微调或评估不使用真实扫描和标签（第1–2页）。

## 方法总览

![LAM3C：视频点云、教师学生与抗噪图约束](../../../../site/figures/lam3c-roomtours-method-v2.svg)

本方法先离线构造RoomTours视频生成点云，再训练PTv3教师—学生。聚类目标学可迁移表示；Laplacian项让局部邻点特征稳定，noise-consistency项让噪声/遮蔽视图中对应点保持一致。重建模型原名是**π³**；旧稿的“ω3”是抽取/转写误差，本轮已对PDF目视更正。[Fig.2–5、§3–4，PDF3–5]

## 核心创新与关键设计：数据构造和表示学习是两条阶段

### 1. 先从视频得到点云，损失并不把点直接修成真扫描

自收3462个看房视频，加RealEstate10k、YouTube House Tours、HouseTours等已有源。CLIP先逐帧分室内/室外，丢弃室外，再以living room/bedroom/bathroom三个粗类别切片；0.5秒内保持标签一致以稳定边界。不是给每个三维点贴三类语义标签，而是控制送入重建器的视频片段范围。[§4，PDF4–5]

每片段均匀抽帧，长片段降到约200–400帧，保持长宽比调整像素数，由π³一次混合精度前向产生每帧几何，置信度过滤、边缘抑制、离群点清理并融合为彩色点云。不做SLAM/BA迭代不代表没有预训练几何先验或没有成本；作者报每场平均约5分钟。动态物体、快速镜头和遮挡仍可能留下重墙、缺洞等伪结构。

进一步随机采点、SOR统计离群过滤，检测主平面对齐Z-up并用SVD精修；失败可能把墙/天花板当主面导致方向翻转，作者保留这类场景而非全部自动丢弃。然后以包围盒对角线s_current定义尺度，从ScanNet统计分布采s_target，以a=s_target/s_current缩放点坐标，局部PCA估法向。教学s_current=10、s_target=6时坐标乘.6；这不是把场景恢复成量测的真实米制长度，只有分布对齐。法向也需要方向/旋转一致性，正文未给完整失败阈值和符号消歧。[PDF5]

全RoomTours有49,219伪场景，其中自采视频子集15,921场景。说“无需真实3D扫描预训练样本”是数据主体来源，仍有ScanNet尺度统计、预训练CLIP/π³和室内视频筛选先验；下游微调/评估也使用真实扫描与标签，不能省略这些边界。

### 2. 输入点与特征、两种view、教师和学生

每个点的9维输入是xyz、RGB、normal；一场可记P∈R^(N×3)、X∈R^(N×9)。不同crop/mask使教师N_t点、学生N_s点不同，各自产生点级embedding H_t/H_s，宽度C由骨干配置决定，主文没完整列出。为避免混淆，下文h是骨干特征、l是prototype logits；论文同用z表示不同位置的输出，阅读时不能把二者直接代换。

| 模块/状态 | 做什么 | 如何更新 |
| --- | --- | --- |
| PTv3学生f_θ | 局部/遮蔽global输入，预测特征与原型分布 | 对总损失反向梯度更新 |
| EMA教师g_φ | global/noise增强输入，提供更稳定目标 | φ←mφ+(1−m)θ，不由该轮目标反传更新 |
| 原型head/logits | 每点对K个原型的分数 | 温度/assignment配置依Sonata，完整实现待核 |
| 点对应集合 | 不同crop/增强间的kNN匹配索引 | 根据点坐标关系构造，不是语义GT |
| 几何kNN边 | 同一伪点云内近邻关系与距离权重 | 用于Laplacian；与跨view对应集合不是同一对象 |

EMA是参数随训练平滑更新，教师不是永远冻结的初始网络；在计算学生损失时教师目标需要stop-gradient，更新后再EMA是常见顺序解释，主文未给m日程和精确step顺序。π³和视频预处理在离线数据阶段，不由PTv3损失反向优化。

### 3. 聚类主目标：对原型分布做交叉熵，不直接回归点坐标

学生logits除温度τ_s再softmax成p_s，教师logits除τ_t经Sinkhorn形成q_t。Sinkhorn在批量原型分配上施加熵/均衡约束，避免简单所有样本都只占一个原型的局面；不是输入人工语义标签，也不保证每个原型自然对应一个物体类别。[式1–2，PDF3]

\[p_s=\operatorname{softmax}(l_s/\tau_s),\qquad q_t=\operatorname{Sinkhorn}(l_t/\tau_t),\qquad L_{CE}=-\sum_{points}\sum_{k=1}^{K}q_{t,k}\log p_{s,k}.\]

同一个点的q作为目标，交叉熵鼓励学生把概率移向教师认为重要的原型。教学q=[.8,.2]，学生p=[.5,.5]给loss≈.693147，若学生变成[.8,.2]则≈.500402；目标相同并不要求loss为0，因为soft target本身有熵。q与p都要按同一原型字典和对应点匹配，不能错把不同空间索引相减。

Sonata基础目标有三项，权重4:2:2：L_unmask用学生local-view与教师global-view做kNN匹配；L_mask用完整global教师监督被mask的学生global；L_roll交换两global-view构造目标，强化跨view一致。加权合成L_clustering=4L_unmask+2L_mask+2L_roll。具体crop数量/比例、mask形状、原型数K、温度、Sinkhorn迭代和loss归约正文没列全，不应补造可运行配方。

### 4. Laplacian：几何邻近边如何把特征拉在一起

对每个伪场景的点坐标建kNN图E，距离d_ij=||p_i−p_j||，按w_ij=exp(−d_ij²/σ²)给边权，σ取该场景kNN距离中位数。过远邻居还会被阈值过滤。论文式3先写平方L2形式：

\[R_{Lap}^{basic}=\sum_{(i,j)\in E}w_{ij}\|h_i-h_j\|_2^2,\qquad w_{ij}=e^{-\|p_i-p_j\|^2/\sigma^2}.\]

坐标决定“谁影响谁及强度”，梯度作用在embedding差上，不直接移动p_i以修复几何。近距离点的特征差会被较强惩罚，远离群点影响较弱；但同样几何近邻也可能跨物体边缘或两层重影，平滑过强会抹掉语义边界。

教学σ=.1，距离.1的边权e⁻¹≈.367879，距离1.9的边权e⁻³⁶¹接近0，说明kNN即使强制为离群点找到邻居，距离权重还能削弱其影响。σ随场景密度调整；重复点使中位数0、空图或极稀场景仍需数值处理，主文没有给epsilon/回退。

作者实际实现还加入**degree normalization和Huber惩罚**，不是直接运行上述basic式。主文未给采用哪种归一化、图是否对称/边是否重复计、k和距离截断、Huber阈值以及按向量范数或逐通道Huber。标准标量Huber教学为h_δ(r)=.5r²（|r|≤δ），否则δ(|r|−.5δ)；δ=1、r=10时9.5，比.5r²=50增长慢。这个例子说明减弱大误差影响，不表示作者δ=1。不能自己挑D⁻¹W或D⁻¹/²WD⁻¹/²后声称二者与原实现完全一致。[PDF3]

### 5. Noise consistency：对齐的是同一场景跨增强的点级特征

图3/式4把noise-augmented global-view x^(a)送EMA教师，masked global-view x^(b)送学生，以跨view kNN对应集合P比较骨干embedding：

\[R_{cons}=|\mathcal P|^{-1}\sum_{(i,j)\in\mathcal P}\|g_\phi(x^{(a)})_j-f_\theta(x^{(b)})_i\|_2^2.\]

这是平方L2，不是原型交叉熵，也不是全场景一个均值向量。教师第j点和学生第i点应指向相同空间内容；若增强包含旋转/尺度，要在可比较坐标系恢复对应，不能用已经随机变换后的绝对坐标直接盲配。被裁掉或mask后不存在的点怎样找对应、距离阈值与是否插值主文未给，需要Sonata/源码细节。

例如对应teacher h=[1,0]、student=[.8,.2]，平方差=.08；梯度促学生靠近teacher。对应关系如果跨到墙后另一物体，这个loss反而传递错误约束，所以它提升抗噪的前提不是所有邻近点天然语义相同。

### 6. 总目标、系数日程与坍塌边界

总损失为L_total=L_clustering+λR_Lap+μR_cons。λ从2×10⁻⁴线性增至3×10⁻³，μ=.05固定；先较弱平滑有利于不过早强迫随机embedding全部相同，具体线性区间/学习率配合仍需训练配置。[式5，PDF4]

**平滑本身不能防止坍塌**。若所有点h_i=c，则所有差0、R_Lap=0；若teacher/student都输出相同c，R_cons也为0。主文将“仅依赖关系结构”与防坍塌联系起来，不构成数学保证。必须连同teacher/student的原型学习、Sinkhorn均衡、增强与温度机制理解整个系统；即使如此实际是否语义退化仍需实验监测，而不是靠两个loss为0判断成功。

```text
离线：获许可室内视频 → CLIP切片 → π³点云 → 过滤/朝向/尺度/法向
预训练step：同一场景构造global/local、noise与mask视图
            student/teacher前向 → prototype CE（4:2:2基础项）
            伪场景几何kNN → degree-normalized Huber平滑
            跨view对应点 → teacher/student embedding平方差
            总损失反向更新student；按EMA更新teacher
下游：使用预训练点云骨干，按LP或Full-FT协议训练评估
```

上面是算法依赖关系，不是已运行的作者训练脚本；global teacher不走学生梯度，EMA也不是反向传播。下游LP一般固定预训练骨干只训练线性头，Full-FT解冻更多参数；论文比较时还需按其任务设置，不能混成同一个无标签测试结果。

### 7. 可扩展性与代价

图有约Nk条边，算embedding边差约NkC，点匹配/建图和PTv3前向另计；不会显式建立N²全连接图，但不意味着百万点免费。教师还需要额外global-view前向；场景点数、裁剪、稀疏序列化、图缓存决定峰值显存。8张H200、batch16、145600步是默认预训练资源，Large/大数据的更多步数需要分别列出。

离线49,219场景若机械乘作者每场平均5分钟，约4102小时串行处理量，仅为数量级换算，非作者实际总墙钟或硬件时长。新增视频需要版权/可用性核对与重建算力，不是增加条目就无成本扩展数据。与直接真实扫描相比，获取来源更广但引入重建偏差；与只增加聚类数据相比，两个抗噪项有针对性但可能过平滑，最后仍须看干净/有噪、LP/Full-FT的不同结果。

## 实验与关键结果

第6页：Base/49k的ScanNet语义Full-FT mIoU77.7，随机初始化74.7，Sonata全真实78.5；Large/49k为79.5，但模型及步数均更大。S3DIS实例Full-FT mAP45.7，Large47.2，Sonata真实+合成45.5。不能把Large优势全归于新损失。预训练8张H200、batch16，默认145600步，大模型437k步（第5–6页）。

## 消融实验与证据强度

第7页1k设置中去正则LP51.1→加入57.1；去平滑55.0、去一致性55.3。Z轴对齐LP51.8→57.1，但Full-FT76.1→75.1反而下降；尺度对齐也主要惠及LP。第8页干净ScanNet上Sonata67.1、LAM3C66.8，无通用胜出；噪声RoomTours上51.1→57.1，支持针对噪声的设计。

## 局限与讨论

点云尺度通过ScanNet统计分布对齐，因此“无真实扫描预训练样本”不等于完全没有真实数据统计先验。网页视频权利、重建模型权重来源和室内域偏置仍需核查。PCA彩色可视化不是带语义标签的零样本分类精度；数据生成每场约5分钟也有总计算成本。

## 复现提示

记录视频可用性和授权、π³版本、对齐失败处理及数据划分。作者为公平比较移除Sonata预训练中的验证/测试输入，虽原方法没用其标签，也会影响比较。论文给出lam3c仓库入口，此次未运行下载或预训练，未确认数据总体积及发行许可。

## 相关工作与阅读顺序

先读DINO教师学生聚类、Sonata与PTv3，再读VGGT/π³的前馈重建。本文贡献需要同时看数据源扩展与抗噪目标，不能只把它概括成“视频转点云”。

## 来源与校验

- 2026-11-08重读PDF3–5并逐页渲染，目视确认π³名称、两个平方范数与实际Huber/degree-normalization说明。新SVG经Chrome整图检查；CE、图权重、Huber、对应差与尺度算例仅独立教学验证。未下载视频/点云、运行π³或训练PTv3。使用哈希一致的既有官方PDF缓存，不声称本轮重新下载。
- [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Yamada_3D_sans_3D_Scans_Scalable_Pre-training_from_Video-Generated_Point_Clouds_CVPR_2026_paper.html) · [所读 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yamada_3D_sans_3D_Scans_Scalable_Pre-training_from_Video-Generated_Point_Clouds_CVPR_2026_paper.pdf)。页码均为该 PDF 的页序。
- 阅读范围：正文、公式与实验表；补充材料、上游代码与权重未逐项核验。许可证和固定代码 commit 尚未确认，不复制原文全文或数据。
- PDF SHA-256：`1cb78bbf3478b3620347884762f7f95862aa73cb69a5bcd31f232765bfe5c70e`；版本为本次下载的 CVF 接收稿，核对日期 2026-11-08。
