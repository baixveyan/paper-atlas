# 3D-IDE: 3D Implicit Depth Emergent

作者：Chushan Zhang, Ruihan Lu, Jinguang Tong, Yikai Wang, Hongdong Li。

分类：multimodal；交付形式：Markdown 中文论文讲解。

核对日期：2026-11-08。以下实验均为作者报告；本地完成原文阅读与讲解自检，未复现实验。

## 一句话总结

把深度、跨视图与三维教师约束放在训练阶段，让单个RGB视频编码器学到几何信息，部署时移除辅助分支。

## 研究背景与动机

显式深度/位姿输入增加部署依赖，双编码器又增加参数与延迟。作者希望几何成为共享视觉token可解码的信息，避免测试阶段继续外接三维模型（第1–3页）。

## 方法总览

![3D-IDE：共享视觉token、置信权重与深度warp](../../../../site/figures/3d-ide-method-v2.svg)

同一套SigLIP视觉token既供Qwen2-7B视频语言回答，又在训练阶段供弱DPT式validator读出深度/权重。深度GT、已知相机位姿和固定VGGT全局teacher提供三种训练辅助；部署时移除validator/teacher，只保留原RGB视频与语言路径。[Fig.3、§3.2–3.3，PDF4–6]

## 核心创新与关键设计：监督要迫使几何留在共享token里

### 1. 输入、读出和训练专属信息

每帧I_t∈R^(H×W×3)经可微调视觉encoder f_E得到F_t∈R^(M×C)。语言支路把F_t线性投影到LLM宽度，作为soft visual prompt与文本token拼接；不额外拼接GT三维坐标token，也不新增一套在线几何encoder。F_t^3D≡F_t是**同一共享表示的设计定义**，不意味着每个feature分量天然就是某个xyz坐标。[式4，PDF4–5]

| 中间量 | 工作形状/含义 | 是否部署保留 |
| --- | --- | --- |
| F_t | M×C的RGB视觉token | 是 |
| 语言projector/LLM | token映射与标准自回归输出 | 是 |
| f_P validator | F_t→Dhat_t、Σhat_t，两个H×W图 | 否 |
| D_gt、相对pose/有效mask | 深度与视图一致监督 | 仅训练使用 |
| f_G、f_a | 固定VGGT场景descriptor | 否 |
| f_b | 对学生全序列表示做对齐投影后的descriptor | 全局训练分支，不需在线teacher |

M对应patch网格而非H×W个独立RGB像素，DPT读出要将低分辨率token恢复到密集图。具体token重排、上采样层、Σ正值activation和描述子聚合/投影维数主文没给完整表；不能假设视觉和teacher维数相同就直接做cos。

### 2. 弱validator为什么从零训练

validator是低容量DPT式decoder，输入F_t，输出深度Dhat和像素权重Σhat。若一个很强的预训练深度模型自己就能完成全部几何任务，辅助损失可能主要被head吸收，而共享视觉token的增益不明显。限制head容量并从零初始化，是为了让“简单读出也能得到深度”的训练压力传到视觉encoder。[§3.2.2，PDF5]

这是一种优化归因假设，不是互信息或能力分配的严格定理：弱head仍可学数据偏差，encoder也可能只把容易解码的训练域线索写入token。作者比较相同架构的预训练/随机初始化head，差距较小，证据支持“不需要强预训练head”而非证明所有几何都已经唯一存于encoder。部署移除head后，几何需要通过语言任务表现/冻结探测等间接验证，不能把可读深度等同精确场景重建。

### 3. 深度与梯度残差：Σ的作用方向要按公式理解

对有效像素p，主文式5–6：

\[\ell_p=\|\widehat\Sigma_p(\widehat D_p-D_p^{gt})\|+\|\widehat\Sigma_p(\nabla\widehat D_p-\nabla D_p^{gt})\|-\alpha\log\widehat\Sigma_p,\qquad L_{geometry}=|\Omega|^{-1}\sum_{p\in\Omega}\ell_p.\]

第一项拟合绝对深度，第二项拟合空间深度梯度，鼓励边缘/局部结构不被过度平滑。∇不是时间差分；若目标深度相邻像素从2m跳到3m，预测都为2.5m，平均深度可能对但边界梯度错，第二项可惩罚这种情况。有效mask应过滤缺失深度，梯度算子也要处理相邻无效值，主文没有展开差分模板和mask边界。

Σ必须为正以计算log。固定预测残差且把其两个范数和记r≥0，在正标量权重下ℓ(Σ)=Σr−αlogΣ。求导r−α/Σ=0，若r>0则最优Σ*=α/r：残差大时倾向较小权重，残差小时倾向较大权重。它更像**置信权重/逆尺度**，不能按名字“uncertainty map”直接当越大越不确定的方差。[PDF5；此处为公式的阅读推导]

教学α=.1，r=.5得Σ*=.2；r=.05得2，权重差10倍。α实际值和Σ范围/正值映射正文未给。若r=0而Σ无上界，单个形式项−αlogΣ可不断降低，所以需要核验实际clamp、activation和数值正则，而不是看到log就断言此项永远有有限最优。该边界不代表已确认作者代码发散。

### 4. cross-view：把源预测深度变换成目标相机深度再比较

对帧t随机选邻帧t'，使用已知相对pose P(t'→t)把预测Dhat_t'重投影到t得到Dhat_t'→t；在有效warps集合上平均L1差。是预测对预测的一致约束，不是拿GT深度同时当两边输入。[式7，PDF6]

\[L_{cross}=|\Omega_{t'\to t}|^{-1}\sum_{p\in\Omega_{t'\to t}}|\widehat D_t(p)-\widehat D_{t'\to t}(p)|.\]

基础几何过程为：源像素p'与深度d'反投影X'=d'K_t'⁻¹p'；再X=R X'+T；目标p~K_t X，写入的深度是目标相机X的z分量。不能仅把源深度数值当普通图像纹理warp而不转换深度坐标。[几何细节为阅读补充，正文未给完整算法]

教学源中心点(0,0,2)，相对平移T=(0,0,1)，R=I，变换后目标深度3；若目标预测3，正确残差0，直接搬源值2则误罚1。这个例子忽略真实遮挡/插值，仅解释单位与坐标。

有效warp还应考虑出界、负深度、多个源点投同像素的z-buffer、遮挡/反遮挡、相机intrinsics变化与动态物体；主文只给valid mask M，没有写如何构造这些规则、空集合处理或是否stop某侧梯度。两侧一致也可能共同错误，例如尺度/平滑偏差，故它不能替代GT深度项。随机相邻对降低N²全对监督成本，同时不能保证所有远距离帧都满足严格几何闭环。

### 5. 全局teacher：用方向对齐补充局部视图约束

固定VGGT为全序列提场景descriptor f_a，学生共享encoder序列表征经投影得到f_b，主文用：

\[L_{global}=1-\frac{f_a^\mathsf Tf_b}{\|f_a\|_2\|f_b\|_2}.\]

教学f_a=(1,0)、f_b=(.8,.6)，两者单位范数，loss=.2；若f_b=(2,0)，loss为0，说明cos只限制方向，不限制模长。局部像素depth准确和全局descriptor方向相似是不同目标；teacher语义/几何偏差也可能被复制，global loss小不证明所有空间细节正确。

主文称f_a/f_b“defined above”，但正文没有完整给全局pooling、是否投影head、teacher特征层/坐标与维度；不能替它补成唯一mean-pooling公式。全局descriptor提供紧凑训练指导，不是在线把VGGT特征加到每个视觉token。teacher冻结，无学生梯度更新；f_b路径仍需反向训练其投影/encoder。[式8，PDF6]

### 6. 总目标与各参数的梯度路径

L_total=L_ce+L_geometry+L_cross-view+L_global。L_ce为标准答案token自回归负log概率；三个辅助项均从当前RGB特征取得student输出，直接把几何压力传到f_E。SigLIP明确end-to-end fine-tune，validator从零训练；语言支路按LLaVA-Next-Video/Qwen2-7B配置训练，VGGT固定。[式3、9，PDF3、5–6]

这一点与GAP/4D-RGPT固定视觉encoder的设置不同：3D-IDE的目的正是修改共享视觉token，不能误写“只训depth head”。L_geometry和cross-view先经f_P Jacobian传到F_t，再到θ_E；L_global经学生描述子/投影到θ_E；L_ce还训练语言projector与LLM。主文未给全部语言参数/LoRA与optimizer分组细则，不能仅凭end-to-end声称每个parameter都更新。

```text
训练：RGB帧 → 可微调SigLIP共享token F
      路1：projector + LLM + 问题 → 答案CE
      路2：弱DPT validator → depth/正权重 → GT几何loss
           相邻预测depth + 已知pose → valid warp → cross-view loss
      路3：学生全局descriptor对固定VGGT descriptor → cosine loss
      总loss反向更新相应student/validator/语言参数，teacher不更新
部署：移除DPT validator、全局teacher/辅助读出；只RGB → 共享视觉+语言 → 输出
```

“privileged information”指训练可用GT深度/相机/teacher，而测试不再输入它们；不是训练也无需几何监督。针对测试深度缺失的方便性，不能扩展成对无标定或动态场景具有严格几何保证。

### 7. 贯穿例子、资源与设计取舍

对“柜子相对于移动观察者怎样变化”的问题，RGB提供外观与多帧视角，深度/梯度监督鼓励token区分柜子表面远近，cross-view让静态几何在已知相机变换下相容，teacher把整场结构方向写入共享表征，LLM用这些token生成答案。没有新增GT坐标输入，不代表自动生成可验证的全部三维轨迹；若物体运动破坏静态warp假设，需要mask/模型处理，而本文主要室内ScanNet验证未覆盖所有情形。

教学32帧224²，depth和Σ两通道单套float32约12.25MiB；还要GT、warped depth、mask、DPT激活、teacher和LLM训练内存，不能用此数估训练峰值。每参考帧抽一邻帧大约O(NHW)重投影而非全部O(N²HW)，仍需前向/反向与teacher计算；部署去掉辅助分支所以相对同主干没有这些推理成本，并非总推理时延为0。

强预训练head可快速降低depth loss，却不保证encoder获得更可迁移几何；弱head可能把压力转给encoder，也可能难以优化，所以作者lr=2×10⁻⁶、初始化与学习率配合要按配置验证。显式深度输入能提供测试观测，但增加传感/估计依赖；额外几何encoder提供强表示却增加在线模型。本文把成本前移到有监督训练，取舍不等于消除全部资源开销。

## 实验与关键结果

第7页表2：ScanRefer Acc@0.25/0.5为60.9/54.5，去3D输入的Video-3D LLM基线53.7/47.8；Multi3DRefer F1为59.8/54.9。Scan2Cap CIDEr79.0低于VID-LLM81.5，非所有任务最优。32帧、8×H100、batch16，一轮约23小时。第8页VG-LLM8B与本文参数9.25B/8.06B、平均时间3.60/1.61秒，55.3%是该配置延迟下降。

## 消融实验与证据强度

第8页表3：仅全局监督ScanRefer56.9/50.8，加入从零训练validator59.8/53.3，再加跨视图60.9/54.5；预训练validator完整模型60.5/54.1，差距较小且无误差条。表3将ScanRefer列头写F1、Multi3DRef写Acc，与主表常用口径对调，应以任务和表2定义核查。几何对应Recall@2cm40.15→42.27，法向RMSE32.26→31.36。

## 局限与讨论

“零额外延迟”指辅助分支相对保留主干不进入测试，不是总推理零耗时。训练明确用深度GT、相机位姿及冻结三维教师，不能说无需几何监督。多数评估来自ScanNet，尚不证明开放环境、运动场景或任意尺度泛化。

## 复现提示

核查深度有效mask、置信项正值约束、帧采样、全局教师与场景坐标约定；validator学习率2×10⁻⁶，其他模块需对应基线配置。论文提供3D-IDE仓库地址，但未审查代码及权重许可，未本地运行。

## 相关工作与阅读顺序

先读Video-3D LLM与VG-LLM，再看特权信息训练、DPT深度读出及几何表征探测。将训练监督和部署输入分开，是理解本文贡献的关键。

## 来源与校验

- 2026-11-08重读PDF4–6并逐页渲染核对深度权重、warp/global与训练图；PDF3补读语言目标。新SVG经Chrome整图检查。置信最优/深度变换/余弦/内存仅独立教学验证，未运行SigLIP、DPT、VGGT或训练LLM。使用哈希一致的既有官方PDF缓存，不声称本轮重新下载。
- [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_3D-IDE_3D_Implicit_Depth_Emergent_CVPR_2026_paper.html) · [所读 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_3D-IDE_3D_Implicit_Depth_Emergent_CVPR_2026_paper.pdf)。页码均为该 PDF 的页序。
- 阅读范围：正文、公式与实验表；补充材料、上游代码与权重未逐项核验。许可证和固定代码 commit 尚未确认，不复制原文全文或数据。
- PDF SHA-256：`5df74903205e6983dc2c0164157517a5d7630a441898a947894c54c885486dce`；版本为本次下载的 CVF 接收稿，核对日期 2026-11-08。
