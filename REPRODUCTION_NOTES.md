# DyFCLT 复现记录：证据、推断与验证

论文：Chaolang Li, Pengwen Dai, Jingyu Li, Siyuan Yao, Yuchen Jiang, Zhuoran Zheng.
*DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection.*
CVPR 2026（CVF open access 版，pp. 11313–11323）。

标记约定：**[P]** 论文明确给出 · **[F]** 由论文图示读出 · **[R]** 沿用 RT-DETR 官方默认（论文称“其余设置与 RT-DETR 保持一致”）· **[O]** 来自数据集官方代码 · **[I]** 论文未给出、由复现者推断（可能影响结果）。

## 1. 代码可得性（2026-10-10 检索）

| 检索 | 结果 |
|---|---|
| 论文 PDF 中的代码/项目链接 | 无 |
| GitHub 仓库搜索 `DyFCLT`、论文标题关键词 | 0 个结果 |
| GitHub 用户搜索作者名 / 邮箱前缀 | 无匹配 |
| 网页检索（论文标题 + github/code） | 仅论文摘要页，无代码 |

结论：没有公开的官方或第三方实现（按 paper-reproduction 技能的分级相当于 L3）。本仓库是**依据论文文字与图示的独立实现**，检测框架基于论文所述基线 RT-DETR（参照 lyuwenyu/RT-DETR 的 Apache-2.0 实现重写）。因此复现结果与论文的差距，可能部分来自下文的 [I] 项。

## 2. 模型结构对照

| 组件 | 论文证据 | 本实现 | 标记 |
|---|---|---|---|
| 双骨干 | “pretrained ResNet50 for both the visible and infrared branches”, L=3 | 两个独立的 PResNet50-vd（RT-DETR-R50 骨干），输出 S3/S4/S5；冻结 BN 与 stem | [P]+[R] |
| 顶层 Self-Attn | 图 2(a)：IR 分支顶层输入为 (F_ir, F_ir, F_ir) | RT-DETR AIFI（1 层 Transformer encoder，8 头，FFN 1024，sincos 位置编码），模态内自注意力 | [F]+[R] |
| DFCA 位置 | 图 2(a)：l 与 l−1 两层使用 DFCA | S4、S3 使用 DFCA，S5 使用 AIFI | [F] |
| DFCA 的 Q/K/V | IR 分支 Q 来自 F_vis，K/V 来自 F_ir；1×1 卷积 + 3×3 深度卷积 | 相同；RGB 分支对称 | [P] |
| 频带分解 FBD | 式 (1)(2)：FFT 后按径向频率 √(u²+v²) 取二值掩膜，B=3，k₀=0，k_B=1/2 | rfft2（ortho 归一化），频率网格 fftfreq，B=3 | [P] |
| 动态频带 | 内边界为累积正增量，按倍频程初始化 {0,1/8,1/4,1/2} | k_b = Σ softplus(θ_i)，初始化精确为 {1/8, 1/4} | [P] |
| 频带掩膜的可导性 | 未说明（二值掩膜对边界不可导） | 默认 `soft`：差分 sigmoid，过渡宽度 1 个频率格点；可选 `ste`/`hard` | [I] |
| 最高频带上界 | k_B = 1/2 | 最高频带包含 r ≥ k_{B−1} 的全部频率（含 r>1/2 的角点），保证各频带之和恰为全频谱 | [I] |
| 频带交互 | 式 (3)：A_b = IFFT(F_{q,b}·conj(F_{k,b})) | 相同（逐通道循环互相关，ortho 归一化相当于 1/√N 缩放） | [P] |
| 调制 | 式 (4)：σ(Conv3×3(A_b)) ⊙ IFFT(F_{v,b}) | 每个频带一个独立的 3×3 卷积（C→C，全连接通道） | [P]+[I] |
| 聚合 | 式 (5)：Proj(LN(Σ_b R_b)) | 通道 LayerNorm + 1×1 卷积；**无残差**（论文未提） | [P]+[I] |
| Table 5 变体 | None / Q / Q&K / K&V / Q&K&V | `dfca.decouple` = none/q/qk/kv/qkv；None 即 B=1（与 Table 6 的 B=1 数值相同） | [P] |
| IBS 掩膜 | 卷积预测前景二值掩膜，focal Tversky loss 监督 | 3×3 卷积 + sigmoid（前向用软掩膜）；GT 为与 GT 框有重叠的特征格；FTL α=0.7, β=0.3, γ=4/3 | [P]+[I] |
| IBS 平滑 | 式 (7)：两个 3×3 卷积先压缩到 C/r 再恢复 | Conv-BN-SiLU ×2，r=2 | [P]+[I]（r 未给出，取与 85.5M 参数量最接近的值） |
| FREF | 式 (9)–(15)：F_bgs 预测 K×K 动态核 → softmax → PixelUnshuffle 成 4 组 → 作用于 F^{l+1} → PixelShuffle → 加双线性上采样 → 与 F_bgs 拼接后 CSPBlock | 完全按公式实现（einsum 实现式 12），K=5 | [P]+[I]（K 未给出，取 CARAFE 默认） |
| 自顶向下路径 | 图 2(a)：Self-Attn → SSE(l) → SSE(l−1) | 相同；**无自底向上 PAN**（图中未画） | [F] |
| 跨模态融合 | “F_out 再由另一个 CSPBlock 融合”，随后拼接、展平送入解码器 | 每层 concat(IR, RGB) → RT-DETR CSPRepLayer(512→256) | [P]+[R] |
| 解码器 / 头 | 与 RT-DETR 相同的解码层数与 query 数 | 6 层可变形注意力解码器，300 queries，IoU-aware query selection，100 个对比去噪 query | [P]+[R] |
| 基线 | “在 RT-DETR 上增加一个模态分支” | `baseline.yaml`：每个分支是完整的 RT-DETR HybridEncoder（AIFI + FPN + PAN），再逐层 CSPBlock 融合 | [P]+[I] |
| Table 4 第 1 行 | 三个模块全关 | `t4_none.yaml`：DyFCLT 骨架去掉 DFCA/IBS/FREF（无 PAN）。与上面的 `baseline` 二者孰是论文的“baseline”无法判定，两者都提供 | [I] |

参数量（`python tools/count_params.py configs/rgbt_tiny/dyfclt.yaml`，7 类）：

| | 本实现 | 论文 |
|---|---|---|
| DyFCLT | **84.08M** | 85.5M（差 −1.7%） |
| 双分支 RT-DETR 基线 | 84.86M | 未报告 |

组成：双骨干 46.95M，输入投影+AIFI 3.42M，DFCA 8.17M，SSE/自顶向下 11.53M，跨模态融合 6.70M，解码器 7.32M。
推断项对参数量的敏感度：r=4 时 82.9M；若 DFCA 调制卷积改为深度卷积则约 77M；若保留 PAN 则约 95M。r=2 + 全通道调制卷积 + 无 PAN 最接近论文，作为默认值。

## 3. 训练设置对照

| 项 | 论文 | 本实现 | 标记 |
|---|---|---|---|
| 学习率 | 0.00025（三个数据集相同） | AdamW lr 2.5e-4，骨干 2.5e-5（RT-DETR 比例 0.1），wd 1e-4，norm/bias 不衰减 | [P]+[R] |
| 训练轮数 | RGBT-Tiny、RGBTDronePerson 20；FLIR 50 | 相同 | [P] |
| 学习率策略 | 未说明 | 前 2000 次迭代线性 warm-up，之后恒定（RT-DETR v1 为恒定，v2 带 warm-up） | [I] |
| batch size | 未说明（单张 A100） | 总 batch 8 | [I] |
| 数据增强 | 仅随机缩放、裁剪、翻转 | RandomIoUCrop(p=0.8) + 水平翻转 + Resize 640×640 + RT-DETR 多尺度 [480…800]；RGB/IR 几何变换完全相同；未用颜色扰动、ZoomOut | [P]+[R] |
| 输入归一化 | 未说明 | 与 RT-DETR 一致，[0,1] 不做均值方差归一化 | [R] |
| EMA / 梯度裁剪 | 未说明 | EMA 0.9999（2000 次 warm-up），clip 0.1 | [R] |
| 损失 | RT-DETR 损失 + 掩膜的 focal Tversky loss | VFL 1、L1 5、GIoU 2（含辅助层与去噪组）；掩膜损失权重 1.0，取 2 层 × 2 分支平均 | [R]+[I] |
| 训练标注 | — | 三个数据集都用红外标注训练与评测（见第 4 节） | [O]/[I] |

## 4. 数据集与评测协议

**RGBTDronePerson（Table 2）**
- 来源：作者 Google Drive（RGBTDronePerson.zip，1,573,829,771 字节）+ `train_thermal.json`、`val_thermal.json`。4,900 训练对 / 1,225 验证对（共 6,125，与论文一致），70,880 个框（与论文一致）。
- 类别：标注文件含 person / rider / crowd / uncertain 4 类（即论文所说的“four types”）。官方 QFDet 代码只训练与评测前 3 类（`CLASSES = ('person', 'rider', 'crowd')`），本实现相同。[O]
- 只有红外标注；可见光与红外**并非像素对齐**（视场与尺度不同，见下图描述：红外框在可见光图上明显错位）。这是数据集自身属性，不是预处理错误（已在原图上核实）。
- 评测：从官方 QFDet 仓库（NNNNerd/mmdet-rgbtdroneperson）的 `cocoeval.py` 移植。IoU 阈值 {0.25, 0.5, 0.75}，maxDets 1000，面积段 tiny [1,20²]、small [20²,32²] 等；落在被忽略 GT 内且 IoD ≥ 阈值的未匹配检测视为忽略。**已用合成数据与官方评测代码逐项比对，21 个 AP 数值完全一致**（`tests/test_pipeline.py::test_tinyperson_matches_official`）。
- 两处与官方 json 的差异（均不改变官方协议的语义）：标注 id 从 1 重新编号（官方 json 的 id 0 会被 pycocotools 当作“未匹配”，即 1/14,566 个框评测失真）；3 张训练图的 width/height 写成 630×500，实际文件是 640×512，已按文件修正。

**RGBT-Tiny（Table 1、4、5、6）**
- 需填写申请表单获取（github.com/XinyiYing/RGBT-Tiny），本地未获取。
- 论文 Table 1 中 DINO、UA-CMDet、QFDet、CALNet 等基线数值与 RGBT-Tiny 论文（TPAMI 2025）**基于 SAFit 的 Table III 热红外一栏逐项相同**，因此 DyFCLT 的 Table 1 是 SAFit 指标、红外标注。注意命名错位：DyFCLT 的 “tiny” = [1², 8²)，对应 RGBT-Tiny 论文的 “extremely tiny”；DyFCLT 的 “extremely small” = [8², 16²)。
- SAFit = σ·IoU + (1−σ)·NWD，σ = sigmoid((√A−C)/C)，C=32。按 RGBT-Tiny 官方代码（AAAI989/RGBT-Tiny `codes/loss/loss.py`），NWD 的高斯参数为 [x1, y1, w/2, h/2]（左上角而非中心，与论文文字不同），默认按官方代码实现，可用 `eval.nwd_anchor=center` 切换。官方评测脚本本身未公开（在作者网盘的 baseline 压缩包里），maxDets 等细节无法核对：本实现同时输出 maxDets=100（主结果）与 1000（`@1000` 后缀）。
- 85/30 个序列的训练/测试划分：`tools/prepare_rgbt_tiny.py` 需要官方划分列表。

**FLIR-aligned（Table 3）**
- Zhang et al.（ICIP 2020）对齐版，4,129 训练 / 1,013 测试，类别 person/car/bicycle，标准 COCO 指标。本地未下载；`tools/prepare_flir_aligned.py` 按公开的 VOC 结构编写，**尚未在真实文件上运行**。

## 5. 已完成的验证（本机）

`python tests/test_pipeline.py` 全部通过：

1. 三种评测协议下，把 GT 当作检测结果输入，AP50 = 100。
2. `coco` 协议与原版 pycocotools 的 AP、AP50、AR 完全一致。
3. `tinyperson` 协议与官方 QFDet 评测代码在 21 个 AP 指标上完全一致。
4. SAFit 公式数值检查（小目标时比 IoU 宽松，大目标时退化为 IoU）。
5. 频带掩膜之和恒为 1（B=1…4），初始边界精确为 {1/8, 1/4}，hard 模式为严格二值。
6. 频带边界参数能收到梯度。
7. B=1 时 FFT 实现的 DFCA 与逐位移暴力计算的循环互相关结果一致（误差 < 1e-4）。
8. FREF 的引导上采样在核全部集中于中心抽头时等价于最近邻上采样。

另外在真实 RGBTDronePerson 验证集（200 张）上验证了坐标链路：GT → 数据增强/缩放 → 归一化 cxcywh → 后处理 → 原图坐标 → 官方协议，AP25/50/75 均为 100。

端到端运行（详见 RESULTS.md §3）：
- 冒烟测试（32 张图，320 px）：训练、EMA、评测、保存 checkpoint 全流程跑通。
- 短程学习检查（640 px，batch 2，1,200 次迭代）：验证集 300 张图上 AP50 0 → 9.9 → 11.7。`tools/test.py` 重新加载 EMA 权重后得到同样的 11.673。
- 多进程 DDP（torchrun 起 2 个进程，CPU/gloo）：训练、跨卡归约、评测结果汇总、保存 checkpoint 均正常。
- 20 个配置文件（3 个数据集 × 主模型/基线、11 个消融、冒烟配置）都能构建模型，模块开关与输出目录正确。

本次会话中遇到并已解决的问题：
| 问题 | 处理 |
|---|---|
| 本机缺少 poppler，无法渲染 PDF | 改用 pdfplumber 提取文字、pypdfium2 渲染图 2/3 |
| 图 2(a) 的文字提取顺序错乱，看起来顶层 Self-Attn 有跨模态输入 | 放大渲染图片核对，确认是 (F_ir, F_ir, F_ir) 的模态内自注意力 |
| pycocotools 的 `_prepare` 会把 GT 的 `ignore` 属性覆盖为 `iscrowd` | 重写 `_prepare`：保留原始属性，QFDet 协议下 ignore = ignore ∨ iscrowd ∨ uncertain |
| RGBTDronePerson 标注 id 从 0 开始（pycocotools 会把 id 0 当成未匹配） | 转换时重新从 1 编号 |
| 3 张训练图的 json 尺寸与文件不符 | 按图像文件修正 |
| MPS 上 640 px、batch 4 时卡住（16 GB 统一内存换页） | 本机改用 batch 2、关闭多尺度；batch 1 单步 0.70 s、batch 2 单步 1.47 s |
| DDP 下某张卡的 batch 没有 GT 时，去噪嵌入未参与计算，DDP 会报错 | 无 GT 时加一个 0 × 参数的项，使其保留在计算图中 |
| macOS 上 torchrun 默认 rendezvous 超时 | 本机用 static rendezvous + 127.0.0.1；DDP 代码同时支持 NCCL 与 gloo |
| checkpoint 太大（每个 1.35 GB） | best.pth 只存 EMA 权重（322 MB），last.pth 保留优化器状态用于续训 |

## 6. 尚未复现 / 已知风险

- 论文三个主表与三个消融表都需要在 GPU 服务器上完整训练，本机未训练到论文设置（本机为 M5 MacBook Air，16 GB 统一内存，无 CUDA；实测 640 分辨率 batch 2 约 1.5 s/iter，batch 4 即开始换页）。
- Fig. 1(a) 频率能量分析、Fig. 4/5 可视化未复现（非数值结果）。
- 影响最大的推断项（建议优先做敏感性实验）：频带掩膜软/硬（`dfca.mask_mode`）、DFCA 是否有残差、基线定义（`baseline` vs `t4_none`）、batch size 与 warm-up、SAFit 评测细节。
- RGBTDronePerson 的可见光与红外不对齐。DFCA 的互相关在频域计算（模态间整体平移只改变相关图的峰值位置），但 SSE 与跨模态融合是逐像素拼接的，错位对它们的影响论文没有讨论，本实现也未做专门处理。
- IBS 平滑分支两层卷积是否带 BN/激活论文未说明，本实现两层都是 Conv-BN-SiLU [I]。
