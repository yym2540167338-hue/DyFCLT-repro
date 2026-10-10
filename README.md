# DyFCLT 复现（CVPR 2026）

*DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection* 的 PyTorch 独立复现。
论文没有公开代码，本仓库根据论文正文、公式与图 2/3 实现，检测框架沿用论文所述基线 RT-DETR。
每个设计决定的出处（论文给出 / RT-DETR 默认 / 推断）以及已做的验证见 [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)。

## 当前状态

| 项目 | 状态 |
|---|---|
| 模型（双 ResNet-50 + AIFI + DFCA + SSE(IBS/FREF) + 跨模态 CSP 融合 + RT-DETR 解码器） | 完成，参数量 84.08M（论文 85.5M） |
| 基线与 Table 4/5/6 全部消融配置 | 完成（`configs/`） |
| 三个数据集的评测协议 | 完成；RGBTDronePerson 协议与官方 QFDet 评测代码逐项一致 |
| 环境 | 本机 conda 环境 `dyfclt`（torch 2.14.1，MPS）已配置；服务器脚本 `environment/setup_server_cuda.sh` |
| RGBTDronePerson | 已下载并转换（4,900 / 1,225 对） |
| RGBT-Tiny | 需本人填写申请表单 |
| FLIR-aligned | 未下载（脚本已备好） |
| 论文设置的完整训练 | 需在 GPU 服务器上运行（本机只做了冒烟测试与短程学习检查，见文末） |

## 目录

```
configs/            _base_.yaml（公共超参，逐项标注出处）+ 各数据集 dyfclt/baseline + rgbt_tiny/ablations/*
dyfclt/models/      backbone, dfca, sse, encoder, decoder(RT-DETR), criterion, detector
dyfclt/data/        成对 RGB-T COCO 数据集 + 两模态同步的几何增强
dyfclt/evaluation/  coco / safit(RGBT-Tiny) / tinyperson(RGBTDronePerson) 三种协议
dyfclt/engine/      训练循环、AdamW 分组、EMA、warm-up、DDP
tools/              train.py test.py prepare_*.py count_params.py compare_with_paper.py
scripts/            run_droneperson.sh run_rgbt_tiny.sh run_flir.sh（服务器一键运行）
tests/              正确性测试（不需要数据集）
paper_results.yaml  论文 Table 1–6 数值，用于自动对比
```

## 1. 环境配置

GPU 服务器（Linux + NVIDIA）：

```bash
bash environment/setup_server_cuda.sh cu121
```

有 conda 时，脚本创建 conda 环境 `dyfclt`（Python 3.11）。没有 conda 时（报 `conda not found`），脚本会自动用机器上已有的 Python 3.8–3.12 在 `~/venvs/dyfclt` 建一个 venv；以后用 `source ~/venvs/dyfclt/bin/activate` 激活，`PYTHON=/path/to/python3.x` 可以指定用哪个 Python。如果集群用 module 管理软件，也可以先 `module load` 对应的 anaconda 模块。之后脚本安装 torch 2.4.1 和其余依赖，下载 RT-DETR 的 ImageNet 预训练 PResNet50-vd 权重到 `pretrained/`（约 94 MB），运行 `tests/test_pipeline.py`，最后用 `tools/check_env.py` 检查 GPU。驱动较旧时改用 `cu118`。服务器不能联网时，把本机的 `pretrained/*.pth` 拷过去即可。

注意不要用 `pip install --user` 装 PyTorch。装到 `~/.local` 的包会覆盖同版本 Python 环境里的包。如果以前这样装过，运行前先 `export PYTHONNOUSERSITE=1`。

**在集群上（如 `/public/home/...` 这类 Slurm 超算）**：登录节点一般不能用显卡，直接在登录节点上运行会报 `CUDA unknown error`，然后退回 CPU。环境安装在登录节点做（需要联网）；训练要申请 GPU 节点，可以提交作业：

```bash
sbatch -p <GPU分区名> scripts/slurm_train.sbatch scripts/run_droneperson.sh
```

也可以先申请一个交互式 GPU 节点，在里面运行 `python tools/check_env.py`，再运行 `scripts/run_*.sh`：

```bash
srun -p <GPU分区名> --gres=gpu:1 -c 8 --mem=64G -t 48:00:00 --pty bash
```

分区名和 GPU 型号用 `sinfo -o "%P %G %l"` 查看。有的集群申请 GPU 要写 `--gpus=1`，不是 `--gres=gpu:1`。

Mac（Apple Silicon，仅用于调试）：

```bash
bash environment/setup_local_mac.sh
```

## 2. 数据集准备

所有数据放在 `data/` 下，转换脚本生成 `annotations/*_paired.json`（红外为标注模态，`file_name_rgb` 指向可见光）。

**RGBTDronePerson（本机已完成）**

```bash
python -m gdown --folder https://drive.google.com/drive/folders/1Mi3NXQ-YG1iiIWkPbe3GQoDK68dARMN6 -O data/raw/RGBTDronePerson
unzip -q data/raw/RGBTDronePerson/RGBTDronePerson.zip -d data/RGBTDronePerson
cp data/raw/RGBTDronePerson/*.json data/RGBTDronePerson/
python tools/prepare_droneperson.py --root data/RGBTDronePerson
```

Google Drive 下载受限时可用作者的百度网盘（提取码 rr7o）。也可以直接把本机的 `data/RGBTDronePerson` 整个拷到服务器。

**RGBT-Tiny**：在 <https://github.com/XinyiYing/RGBT-Tiny> 填写表单获取数据，解压到 `data/RGBT-Tiny/`（目录形如 `<序列>/00/*.jpg|xml` 为可见光、`<序列>/01/` 为红外），再运行：

```bash
python tools/prepare_rgbt_tiny.py --root data/RGBT-Tiny --train-list <训练序列列表.txt> --test-list <测试序列列表.txt> --dry-run
python tools/prepare_rgbt_tiny.py --root data/RGBT-Tiny --train-list <训练序列列表.txt> --test-list <测试序列列表.txt>
```

如果拿到的发布包里已经有 COCO json，就改用 `--coco-ir-json`。

**FLIR-aligned**：下载 Zhang et al. 的 aligned 版本（作者 Google Drive，或 HuggingFace `jsonhash/FLIR_aligned` 的 `aligned.zip`），解压到 `data/FLIR_aligned/`，然后运行 `python tools/prepare_flir_aligned.py --root data/FLIR_aligned`。如果目录结构不同，按脚本参数调整。

## 3. 训练与测试

先确认 PyTorch 能用上显卡（脚本开头也会自动检查）：

```bash
python tools/check_env.py
```

```bash
NGPU=1 bash scripts/run_droneperson.sh            # Table 2：DyFCLT + 基线，训练→测试→与论文对比
NGPU=1 bash scripts/run_rgbt_tiny.sh              # Table 1
NGPU=1 ABLATIONS=1 bash scripts/run_rgbt_tiny.sh  # 另加 Table 4/5/6（11 组训练）
NGPU=1 bash scripts/run_flir.sh                   # Table 3（50 epoch）
```

单独运行：

```bash
python tools/train.py -c configs/droneperson/dyfclt.yaml
torchrun --nproc_per_node=4 tools/train.py -c configs/rgbt_tiny/dyfclt.yaml        # 多卡；batch_size 为总 batch
python tools/train.py -c configs/droneperson/dyfclt.yaml --resume outputs/droneperson_dyfclt/last.pth
python tools/test.py  -c configs/droneperson/dyfclt.yaml --ckpt outputs/droneperson_dyfclt/best.pth
```

常用覆盖项：`--opts train.batch_size=16 train.amp=bf16 data.train.frame_stride=5`。`amp=bf16` 在 A100 上更快，但不是论文设置；DFCA 的 FFT 始终用 fp32 计算。

每个 epoch 结束会用 EMA 权重评测，日志写在 `outputs/<实验>/log.txt` 与 `metrics.jsonl`，其中也记录学到的频带边界。`best.pth` 按 `eval.key_metric` 选出（RGBT-Tiny 用 AP，另两个数据集用 AP50）；`test_ckpts` 会同时报告 last 和 best。

## 4. 与论文结果对比

```bash
python tools/compare_with_paper.py --dataset droneperson \
  --run dyfclt=outputs/droneperson_dyfclt/test_metrics.json --run baseline=outputs/droneperson_baseline/test_metrics.json
python tools/compare_with_paper.py --dataset rgbt_tiny --table ablation_table4 \
  --run t4_none=outputs/rgbt_tiny_t4_none/test_metrics.json --run dyfclt=outputs/rgbt_tiny_dyfclt/test_metrics.json
```

论文目标值（`paper_results.yaml`）：

| 数据集 | 指标 | 论文 DyFCLT |
|---|---|---|
| RGBT-Tiny（SAFit，红外 GT） | AP / AP50 / AP_t / AP_et / AP_s / AP_m / AP_l / AR | 48.2 / 69.1 / 41.3 / 52.8 / 61.5 / 49.1 / 100.0 / 63.2 |
| RGBTDronePerson（QFDet 协议） | AP50 / AP50_tiny / AP50_small | 61.0 / 62.4 / 35.1 |
| FLIR-aligned（COCO） | AP50 / AP | 84.1 / 45.0 |

对比只作为参考：本实现含若干论文未给出、需要推断的细节（REPRODUCTION_NOTES.md 第 2、3 节），差距的解读需要结合这些推断项。

## 5. 常见错误与排查

| 现象 | 原因与解决 |
|---|---|
| 训练没有用显卡（`log.txt` 第一行显示 `device=cpu`，或 nvidia-smi 里看不到进程） | PyTorch 看不到 CUDA。运行 `python tools/check_env.py`，它会列出具体原因：在集群登录节点上或作业没申请 GPU、CPU 版 PyTorch、用了 `~/.local` 里的 PyTorch、没有 `conda activate dyfclt`、驱动太旧、`CUDA_VISIBLE_DEVICES` 屏蔽了显卡。`scripts/*.sh` 默认 `DEVICE=cuda`，看不到 GPU 时会直接停止，不会在 CPU 上空跑 |
| `UserWarning: CUDA initialization: CUDA unknown error` | 驱动在，但这个进程没有分到显卡，几乎都是因为在登录节点上运行，或者 Slurm 作业没加 `--gres=gpu:1`。按第 1 节在 GPU 节点上运行；如果已经在 GPU 节点上还报这个错，联系管理员检查节点驱动 |
| `log.txt` 显示 `device=cuda`，但 GPU 利用率很低 | 显卡其实在用。利用率低通常是因为正在评测（逐框匹配在 CPU 上）或数据加载跟不上，可以调大 `num_workers`。把 `python tools/summarize_run.py <输出目录>` 和 `nvidia-smi` 的输出发给我 |
| `CUDA out of memory` | 减小 `train.batch_size`；多尺度最大 800，可设 `model.multi_scale=[480,512,544,576,608,640]`；或 `train.amp=bf16` |
| loss 为 NaN / Inf（训练会自动停止并打印是哪一项） | 检查标注里是否有 w/h ≤ 0 的框（转换脚本已过滤）；先关闭 AMP；可降低 `optimizer.lr` 或把 `train.warmup_iters` 调大 |
| cuFFT 报错 `only supports dimensions whose sizes are powers of two` | 半精度 FFT 的限制。DFCA 已强制在 fp32 下做 FFT；如果你改了代码，请保留 `_no_autocast` |
| DDP 报 `Expected to have finished reduction in the prior iteration` | 加 `--opts train.find_unused_parameters=true` |
| 下载 PResNet 权重失败 | 手动下载到 `pretrained/`，设置 `model.backbone.pretrained_path`（`scripts/*.sh` 默认已设置） |
| `pip install pycocotools` 编译失败 | `conda install -c conda-forge pycocotools` |
| gdown 提示 `Too many users have viewed or downloaded this file` | 稍后重试，或用浏览器/百度网盘下载后放到同样的目录 |
| tinyperson / SAFit 评测较慢 | 按官方实现逐框匹配（纯 Python），1,225 张约 1–3 分钟，属正常 |
| Mac 上训练极慢、系统卡顿 | 16 GB 统一内存，640 分辨率下 batch ≥ 4 会触发换页。用 batch 2 并关闭多尺度（见 `configs/smoke/droneperson_learncheck.yaml`） |
| `KeyError: classes [...] not in ...json` | 配置里的 `class_names` 与 json 的类别名不一致（如大小写），改配置或改转换脚本里的 `CLASSES` |
| macOS 上 `torchrun` 卡住 5 分钟后报 `DistNetworkError ... ip6.arpa` | macOS 把 localhost 解析成 IPv6 反向域名。加 `--rdzv-backend=static --master_addr=127.0.0.1 --master_port=29511`。Linux 服务器上一般不会出现 |
| 多机 / 多卡端口冲突 `Address already in use` | 给 torchrun 换一个 `--master_port` |

运行出错时，把 `outputs/<实验>/log.txt` 的末尾和完整的报错信息发给我，我来分析。

## 6. 本机已运行的检查

与论文的对比和本机实测数据见 [RESULTS.md](RESULTS.md)。

- `python tests/test_pipeline.py`：8 项正确性测试全部通过（含与官方 QFDet 评测代码的逐项比对，详见 REPRODUCTION_NOTES.md 第 5 节）。
- `configs/smoke/droneperson_smoke.yaml`：32 张图、320 分辨率，训练→EMA→官方协议评测的全流程跑通。
- `configs/smoke/droneperson_learncheck.yaml`：640 分辨率、batch 2、1,200 次迭代（约 0.5 个 epoch），AP50 从 0 升到 11.7（300 张验证图）。这只说明实现能正常学习，不是论文设置，不能与论文数值比较。权重在 `outputs/learncheck_droneperson_dyfclt/best.pth`（仅 EMA，322 MB）。
- `configs/smoke/ddp_cpu_smoke.yaml`：用 `torchrun` 起 2 个进程（CPU/gloo），跑通多卡代码路径，包括 loss 跨卡归约、分布式采样、评测结果汇总和保存 checkpoint。

`last.pth` 保存优化器状态，用于 `--resume`，约 1.3 GB；`best.pth` 只存 EMA 权重，约 0.3 GB。全部 RGBT-Tiny 实验（13 组）约需 22 GB 磁盘。
