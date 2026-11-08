# 3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image

作者：Ze-Xin Yin, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, Jin Xie。

分类：generation；交付形式：Markdown 中文论文讲解。

核对日期：2026-11-08。以下实验均为作者报告；本地完成原文阅读与讲解自检，未复现实验。

## 一句话总结

3D-Fixer在单图估计的局部几何原位置补全物体，通过粗到细边界预测减少独立生成后再配准造成的布局偏差。

## 研究背景与动机

直接生成整场景易受训练域限制；逐物体生成再优化位姿成本高且容易错配。可见点云虽不完整，却已含位置、尺度和朝向线索，适合充当生成的空间锚点（第1–3页）。

## 方法总览

![3D-Fixer 中文边界、条件投影和教师对齐详解](../../../../site/figures/3d-fixer-method-v2.svg)

场景分割与单图几何提供每物体的可见碎片；生成发生在这些碎片确定的场景位置，先估完整边界，再在更紧的范围细化几何和纹理。TRELLIS先验中的可训练条件分支适应遮挡场景，ORFA在训练时用干净视图teacher作中间特征指导。[Fig.2–3、§3，PDF3–5]

## 核心创新与关键设计：为何要先找边界再生成细节

### 1. 输入碎片是空间锚点，不是完整资产真值

输入单幅RGB I，经实例分割得到K个mask M_i，经几何foundation model估计可见点云。按每个mask提取G_frag^(i)∈R^(N_i×3)，保留同一场景坐标系；对象、桌子与其他物体的相对位置来自该估计。生成输出是每个完整对象及其材质，组装回场景，而不是一张新视图RGB。[§3.1，PDF4]

“in-place”省去先生成canonical资产后再搜索姿态/尺度的显式优化，但并不保证完全不用坐标归一化。需要把场景局部盒映到生成器支持的网格，再按对应映射返回；初始尺度和方向如果错，补全仍会被带偏。分割把两物体误并成一个、透明面深度失败、严重遮挡只剩极小碎片，都不能靠锚点自动修正。

### 2. 从可见AABB到宽松立方盒

令碎片各轴最小/最大为b_min/b_max，C_vis=(b_min+b_max)/2，s_max=max(b_max−b_min)。主文宽松盒B_exp以C_vis为中心，**三个方向边长都取4s_max**，不是每轴各自扩大4倍的长方体。[§3.3，PDF5]

\[B_{exp}=\prod_{a\in\{x,y,z\}}[C_{vis,a}-2s_{max},\ C_{vis,a}+2s_{max}].\]

教学可见范围x=[2,3]、y=[0,.5]、z=[4,4.2]，中心(2.5,.25,4.1)，s_max=1，扩展盒为x=[.5,4.5]、y=[−1.75,2.25]、z=[2.1,6.1]。假如真实不可见部分延伸到x=8，这个盒仍装不下，所以原文称保守扩张“确保覆盖”应理解为经验设计，不能当所有遮挡程度的数学保证。零尺寸碎片、离群极值与不可靠轴长怎样回退，正文没有完整规则。

### 3. 两阶段生成各管什么：粗形状定范围，细形状恢复局部结构

TRELLIS基础流程先在压缩自64³网格的latent域用DiT生成coarse结构，再在活跃体素上生成SLAT并解码mesh。3D-Fixer把它适配成**宽松盒内coarse contour→预测完整B_full→B_full内fine shape**。coarse的关键输出是估计对象完整范围，而不是要求第一阶段就生成高质量最终网格。[PDF3、5]

细阶段以较准确B_full减少无效空间和尺度不确定性。同样格分辨率下，盒越大单格代表的场景长度越大；教学64格轴、宽松边长4时格宽.0625，若fine盒轴长1.6，格宽.025，约2.5倍更细的空间采样。这里只说明分辨率关系，不是作者fine固定64格或固定生成精度。

归一化读法可用教学u=(x−C)/s+.5、逆映射x=C+s(u−.5)，实际需要逐轴长度/盒策略与源实现一致。换fine盒后必须同步变换fragment、voxel位置与图像投影；不能只缩放mesh而忘记条件坐标。B_full如何从coarse occupancy估计、是否滤离群点/加padding/再次生成latent，主文未逐步给出，需保留复现边界。

### 4. 几何条件分支：不是把碎片硬复制为最终表面

base prior本来主要面对干净单物体图，无法自然适应场景遮挡。本文额外输入partial geometry、mask/occlusion条件、位置embedding及全局几何特征；图3的条件分支自注意力带depth-ratio信息，cross-attention读全局特征，再与prior的生成表示交互。原文把细节放补充，未给depth ratio计算、量化/embedding、各层token数量和注入系数，不能只凭名字编一个深度比公式。[§3.2、Fig.3，PDF4–5]

可用工作形状理解：噪声SLAT为L×C，partial voxel token为L_vis×C_g，全局condition为L_global×C_c。Query来自待生成/条件流，Key/Value来自相应上下文，使生成利用可见几何；各维经projector后必须匹配。不是每个partial点都与一个最终顶点一一对应，也没有在主文给出“可见面完全固定不动”的硬约束；学习补全仍可能修改观察区域。

### 5. GAFP纹理条件：把二维外观放到正确可见体素附近

基础TRELLIS使用DINOv2全局cross-attention，但全局语义不保证把椅子腿纹理放在正确3D位置。GAFP从MoGe v2取高分辨率二维图像特征，投到G_frag对应可见体素上，形成空间对齐的局部appearance condition；visibility-ratio self-attention再结合MoGe全局token，并逐层注入纹理DiT。[PDF5]

投影对应的**基础阅读公式**：若场景点X经相机变换X_c=RX+t，则像素齐次p~KX_c，除第三维后采样2D特征，再将它关联到对应voxel索引。单图几何往往已有像素↔点映射，实际可利用其已有索引；这是解释GAFP含义，不是声称作者必须重新用K/R/t做一次投影。depth不一致时遮挡/可见性需检查，否则前景颜色可能被投到后方物体。

一个体素对应多个像素如何聚合、图像特征采样插值、不可见体素如何填值、visibility ratio的分母和阈值、MoGe全局特征来源层，主文未展开。不可见面没有观测纹理，应由先验生成，不能把GAFP写成完整背面纹理“直接复制原图”。

### 6. ORFA：同一噪声状态、不同视图条件，比较中间特征

训练可取得同对象干净视图与遮挡场景条件。固定TRELLIS teacher用干净图，3D-Fixer student用遮挡输入，但二者共享同一noised latent z_t与noise level t，这样中间特征差主要反映条件/适配差异，而不是比较完全不同生成阶段。[§3.4，PDF5]

\[L_{AL}=-\mathbb E\left[\frac1N\sum_{n=1}^{N}\operatorname{sim}(h_s,h)\right].\]

减少负相似度等于提高teacher/student中间特征相似，不能误读为把特征推远。主文称逐层alignment，却未明确式中N是全部层/位置组合怎样取、sim具体归一化、是否加projector或所有层都等权，所以这里只按其定义解释梯度方向，不给伪造逐层实现。

教学如果sim采用余弦、teacher h=(1,0)，student h_s=(.8,.6)，cos=.8，对齐损失−.8；调整student为(1,0)可到−1。这个例子强调方向一致，不证明特征模长也一致，更不把余弦假定登记成作者已确认metric。只有损失下降也不保证最终mesh拓扑完全正确。

### 7. Flow matching与冻结边界怎样配合

除L_AL还训练标准flow-matching L_FM，以生成正确的latent流场。常见直线路径教学可令z_t=(1−t)ε+t z_data、目标速度z_data−ε，预测v_θ并做平方差；本文未在正文给完整概率路径/噪声日程/solver/步数，不能把这个基础例子当作者精确配方。与仅做特征对齐相比，FM直接要求生成器学到目标资产分布，避免只对齐中间方向而不产生正确输出。

Fig.3/caption把预训练生成prior视作固定分支，新scene条件分支训练；ORFA teacher也固定。参数不更新不等于student路径的prior运算可以全部no_grad：注入的条件表示需要通过后续固定层影响输出，并收到对应Jacobian传回的梯度。哪些projector、output层、复制的DiT block逐项可训练，图与正文没有完整参数表，须源码核查。不能笼统写整套TRELLIS全量微调，也不能说训练只修改文本prompt。

### 8. 完整训练与推理顺序

```text
训练：从有完整asset的合成scene取得遮挡图/visible fragment/完整资产目标
      保持scene坐标与局部盒变换一致，构造几何或纹理条件
      相同z_t/t：固定teacher读clean图，student读遮挡条件
      student输出学习FM目标；中间特征加ORFA；只更新可训适配模块
推理：单图→instance masks + visible scene geometry
      每instance：AABB→4s_max宽松盒→coarse completion→预测B_full
                   →fine geometry→GAFP等条件引导纹理化
      将生成资产按各自局部/scene映射放回原场景
```

teacher/clean图只用于训练，不要求真实推理知道不可见的完整物体；各instance可并行，但不代表无限物体在一张GPU上无显存竞争。每次噪声去噪仍需数值生成过程，feed-forward框架不等于所有场景只需单层前向一次。

### 9. 取舍、资源和失败边界

粗阶段先估边界能减少fine采样范围；若coarse B_full切掉真实部分，fine通常难在范围外挽回；扩展过大会稀释空间分辨率。depth/visibility-ratio机制试图处理不可靠碎片，但单图几何和mask错到完全错误对象时，没有额外观察可保证救回。不同instance独立生成也可能产生彼此穿插、支撑不真实或阴影不一致；in-place降低配准搜索，不自动解决物理场景一致性。

K个物体每个L_i token时，单物体全注意力部分约∑L_i²，而全体联合注意力可能约(∑L_i)²；这解释独立并行避免跨物体二次交互的潜在收益，不是实际总体时间比，因为还有条件编码、coarse/fine多轮、可见性和纹理计算。增加可训练层数量改善一部分表面，也可能提高成本，需结合作者12/18层消融而非宣称越多越好。

复现应分别留存mask、fragment、B_vis/B_exp/B_full、归一/逆变换、噪声seed和纹理投影中间结果，再比较布局、形状、材质三个误差；用“看起来完整”不能替代盒IoU、场景/对象CD及遮挡区域验证。

## 实验与关键结果

ARSG-110K由180k+资产、110k+场景、每景30视角组成，约3M图，提供实例mask、相机和物体网格。第6页MIDI测试：场景CD0.069、F-score78.67、对象CD0.032，对比MIDI0.080/50.19/0.103；盒IoU0.492低于MIDI0.518。RTX5090推理30秒对MIDI40秒。ScanNet子集场景CD0.130对MIDI0.163，GT为MetaScenes估计，不是完美实测资产。

## 消融实验与证据强度

第8页12层加入粗到细后场景CD0.276→0.266，再加对齐0.264；对象CD0.271→0.329先变差，加对齐后0.283。18层场景0.252优于12层，但作者选择效率与完整性折中。混合几何源对象CD0.329→0.262，场景CD0.266→0.272却略差，说明收益有口径差异。

## 局限与讨论

单图不可见部分仍是生成猜测，受分割和初始几何错误影响。实拍小样本验证，正文对15场景/15视图表述不同；自建测试仅选可见度最高5物体计算定量，全部物体主要用于可视化，不能混同。资产素材许可与实际公开情况尚待核验。

## 复现提示

需分别保存mask、碎片点云、坐标归一化、两阶段边界及纹理路径。训练配置在补充材料，未读部分不推断。论文称代码数据将公开，不能据此写成已下载可用；此次未运行其代码，也未镜像ARSG数据。

## 相关工作与阅读顺序

先读TRELLIS与MoGe，再比较MIDI和Gen3DSR。重点看“在已有场景坐标补全”相对“生成规范坐标资产再配准”的差别。

## 来源与校验

- 2026-11-08重读PDF3–5，PDF4–5渲染核对Fig.2–3与ORFA式1。新SVG经Chrome整图检查；扩展盒、体素尺度、坐标映射和余弦例只为独立教学验算。未运行分割/几何/生成/纹理模型，未镜像数据。使用哈希一致的既有官方PDF缓存，不声称本轮重新下载。
- [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Yin_3D-Fixer_Coarse-to-Fine_In-place_Completion_for_3D_Scenes_from_a_Single_CVPR_2026_paper.html) · [所读 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yin_3D-Fixer_Coarse-to-Fine_In-place_Completion_for_3D_Scenes_from_a_Single_CVPR_2026_paper.pdf)。页码均为该 PDF 的页序。
- 阅读范围：正文、公式与实验表；补充材料、上游代码与权重未逐项核验。许可证和固定代码 commit 尚未确认，不复制原文全文或数据。
- PDF SHA-256：`5d4bd59de94147d24712a58229019046ef08cb721dfda06506d2b1c0228a461e`；版本为本次下载的 CVF 接收稿，核对日期 2026-11-08。
