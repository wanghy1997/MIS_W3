# Medical SAM Code Index (2026-04-23)

这份文档整理了当前这轮 `SAM2 / SAM3` 医学 3D 实验相关的主要代码、脚本、参考仓库和 `AMOS` 已产出结果，便于后续直接按路径查看。

## 1. 纯 SAM2 / SAM3 评测入口

| 名字 | 本地路径 | 服务器路径 | 主要用途 |
|---|---|---|---|
| `eval_medical_sam2_3d_repro.py` | [/Users/wanghongyi/codes/SSL4MIS_work3/eval_medical_sam2_3d_repro.py](/Users/wanghongyi/codes/SSL4MIS_work3/eval_medical_sam2_3d_repro.py) | `/home/why/SSL4MIS_work3/eval_medical_sam2_3d_repro.py` | 当前三数据集 `BTCV / MMWHS / AMOS` 的纯 `SAM2` 3D 评测主脚本。负责数据读取、prompt 生成、双向传播、指标汇总和 `summary.json` 输出。 |
| `eval_medical_sam3_3d_repro.py` | [/Users/wanghongyi/codes/SSL4MIS_work3/eval_medical_sam3_3d_repro.py](/Users/wanghongyi/codes/SSL4MIS_work3/eval_medical_sam3_3d_repro.py) | `/home/why/SSL4MIS_work3/eval_medical_sam3_3d_repro.py` | 新增的 `SAM3` 适配版 3D 评测脚本。复用现有医学数据 split / prompt / metrics 逻辑，改成 `SAM3 tracker + 临时 JPEG 帧序列`。 |
| `eval_BTCV_pureSAM2_prompt_sweep.py` | [/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/eval_BTCV_pureSAM2_prompt_sweep.py](/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/eval_BTCV_pureSAM2_prompt_sweep.py) | `/home/why/SSL4MIS_work3/eval_BTCV_pureSAM2_prompt_sweep.py` | 专门做 `BTCV` 纯 `SAM2` 提示数 sweep 的脚本，用来比较 `shared_multiobj` 与 `per_class_loop`。 |

## 2. SAM2 批量运行 / checkpoint 管理脚本

| 名字 | 本地路径 | 服务器路径 | 主要用途 |
|---|---|---|---|
| `run_sam2_prompt_ablation_queue.sh` | [/Users/wanghongyi/codes/SSL4MIS_work3/run_sam2_prompt_ablation_queue.sh](/Users/wanghongyi/codes/SSL4MIS_work3/run_sam2_prompt_ablation_queue.sh) | `/home/why/SSL4MIS_work3/run_sam2_prompt_ablation_queue.sh` | 纯 `SAM2` prompt-type sweep 队列脚本。 |
| `run_official_sam2_family_queue.sh` | [/Users/wanghongyi/codes/SSL4MIS_work3/run_official_sam2_family_queue.sh](/Users/wanghongyi/codes/SSL4MIS_work3/run_official_sam2_family_queue.sh) | `/home/why/SSL4MIS_work3/run_official_sam2_family_queue.sh` | 官方 `SAM2 / SAM2.1` family sweep 队列脚本。 |
| `run_baseplus_prompt_explore_queue.sh` | [/Users/wanghongyi/codes/SSL4MIS_work3/run_baseplus_prompt_explore_queue.sh](/Users/wanghongyi/codes/SSL4MIS_work3/run_baseplus_prompt_explore_queue.sh) | `/home/why/SSL4MIS_work3/run_baseplus_prompt_explore_queue.sh` | 新增的 `SAM2 base_plus` prompt 扩展探索队列。固定 official `sam2-hiera-base-plus`，串行运行 `K=10 mask`、`K=10 mask+box`、`K=10 mask+box+feedback(mask)`。 |
| `download_official_sam2_checkpoints.sh` | [/Users/wanghongyi/codes/SSL4MIS_work3/download_official_sam2_checkpoints.sh](/Users/wanghongyi/codes/SSL4MIS_work3/download_official_sam2_checkpoints.sh) | `/home/why/SSL4MIS_work3/download_official_sam2_checkpoints.sh` | 下载官方 `SAM2 / SAM2.1` checkpoints。 |
| `wait_for_official_sam2_checkpoints.sh` | [/Users/wanghongyi/codes/SSL4MIS_work3/wait_for_official_sam2_checkpoints.sh](/Users/wanghongyi/codes/SSL4MIS_work3/wait_for_official_sam2_checkpoints.sh) | `/home/why/SSL4MIS_work3/wait_for_official_sam2_checkpoints.sh` | 等待官方 `SAM2 / SAM2.1` checkpoints 就绪后再启动队列。 |
| `run_medical_sam3_queue.sh` | [/Users/wanghongyi/codes/SSL4MIS_work3/run_medical_sam3_queue.sh](/Users/wanghongyi/codes/SSL4MIS_work3/run_medical_sam3_queue.sh) | `/home/why/SSL4MIS_work3/run_medical_sam3_queue.sh` | 新增的 `SAM3` 三数据集启动脚本。拿到 `sam3.pt` 后可以直接开 `BTCV / MMWHS / AMOS`。 |

## 3. 当前用过的 SAM2 训练 / 半监督接入脚本

| 名字 | 本地路径 | 服务器路径 | 主要用途 |
|---|---|---|---|
| `work3_BTCV_Baseline_safeSAM2.py` | [/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/work3_BTCV_Baseline_safeSAM2.py](/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/work3_BTCV_Baseline_safeSAM2.py) | `/home/why/SSL4MIS_work3/work3_BTCV_Baseline_safeSAM2.py` | `BTCV` 上“先保住 baseline，再安全接入 SAM2 辅助蒸馏”的版本。 |
| `work3_BTCV_Baseline_sparseObjSAM2.py` | [/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/work3_BTCV_Baseline_sparseObjSAM2.py](/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/work3_BTCV_Baseline_sparseObjSAM2.py) | `/home/why/SSL4MIS_work3/work3_BTCV_Baseline_sparseObjSAM2.py` | `BTCV` 的“稀疏关键帧 + 多对象单次传播”版本。 |
| `work3_BTCV_Baseline_sparseObjSAM2_v2.py` | [/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/work3_BTCV_Baseline_sparseObjSAM2_v2.py](/Users/wanghongyi/Documents/Playground/SSL4MIS_work3_remote/work3_BTCV_Baseline_sparseObjSAM2_v2.py) | `/home/why/SSL4MIS_work3/work3_BTCV_Baseline_sparseObjSAM2_v2.py` | 上一版 sparse-object 的增强版，加入更保守的方向融合和 adaptive blend。 |
| `work3_MMWHS_Baseline_safeSAM2.py` | `/home/why/SSL4MIS_work3/work3_MMWHS_Baseline_safeSAM2.py` | `/home/why/SSL4MIS_work3/work3_MMWHS_Baseline_safeSAM2.py` | `MMWHS` 对齐 `BTCV safeSAM2` 思路的半监督版本。当前本地工作区没有镜像，服务器上是实际版本。 |
| `work3_AMOS_Baseline_safeSAM2.py` | `/home/why/SSL4MIS_work3/work3_AMOS_Baseline_safeSAM2.py` | `/home/why/SSL4MIS_work3/work3_AMOS_Baseline_safeSAM2.py` | 为 `AMOS` 准备的 `safeSAM2` 训练入口，训练逻辑参考 `MagicNet/AMOS`。本地当前没有同步版，服务器上有实际版本。 |

## 4. 配置 / 总表 / 参考文档

| 名字 | 本地路径 | 主要用途 |
|---|---|---|
| `sam2.1_hiera_l512.yaml` | [/Users/wanghongyi/codes/SSL4MIS_work3/sam2/configs/sam2.1_hiera_l512.yaml](/Users/wanghongyi/codes/SSL4MIS_work3/sam2/configs/sam2.1_hiera_l512.yaml) | 我补的 `SAM2.1 large-512` 配置，用来和 `tiny-512` 做公平 scale check。 |
| `SAM2_multiorgan_ablation_20260423.md` | [/Users/wanghongyi/codes/SSL4MIS_work3/SAM2_multiorgan_ablation_20260423.md](/Users/wanghongyi/codes/SSL4MIS_work3/SAM2_multiorgan_ablation_20260423.md) | 当前 `SAM2` 多数据集总表。包含 prompt sweep、K 值、official family、backbone scale check。 |
| `MEDICAL_SAM_CODE_INDEX_20260423.md` | [/Users/wanghongyi/codes/SSL4MIS_work3/MEDICAL_SAM_CODE_INDEX_20260423.md](/Users/wanghongyi/codes/SSL4MIS_work3/MEDICAL_SAM_CODE_INDEX_20260423.md) | 就是这份代码索引文档。 |

补充说明：当前的 `eval_medical_sam2_3d_repro.py` 已经支持两类新实验开关：

- `--prompt_refine {none,box,box_points}`：在基础 prompt 之后，对同一 conditioning slice 做一轮串联 refinement。当前最主要是 `mask -> box`。
- `--feedback_mode {none,mask,box,box_points}`：把已跟踪帧的预测结果二值化后，再作为该帧的新 prompt 写回状态，用于后续相邻帧传播，等价于“逐帧自举式更密提示”。

## 5. 上游参考仓库 / 关键参考文件

| 名字 | 本地路径 | 主要用途 |
|---|---|---|
| `segment-anything2-medical-evaluation` | [/Users/wanghongyi/codes/segment-anything2-medical-evaluation](/Users/wanghongyi/codes/segment-anything2-medical-evaluation) | 论文《Segment anything model 2: an application to 2D and 3D medical images》对应的主要参考仓库。 |
| `eval_sam2_3d.py` | [/Users/wanghongyi/codes/segment-anything2-medical-evaluation/eval_sam2_3d.py](/Users/wanghongyi/codes/segment-anything2-medical-evaluation/eval_sam2_3d.py) | 论文式 `SAM2` 3D 医学评测原始实现参考。 |
| `sam3` official repo | [/Users/wanghongyi/codes/sam3](/Users/wanghongyi/codes/sam3) | 官方 `SAM3` 仓库，本轮已经克隆并同步到服务器。 |
| `sam3_for_sam2_video_task_example.ipynb` | [/Users/wanghongyi/codes/sam3/examples/sam3_for_sam2_video_task_example.ipynb](/Users/wanghongyi/codes/sam3/examples/sam3_for_sam2_video_task_example.ipynb) | 最关键的适配参考。证明 `SAM3` 也能按 `SAM2` 的 tracker 风格来做 points / box / mask + propagate。 |
| `sam3/model_builder.py` | [/Users/wanghongyi/codes/sam3/sam3/model_builder.py](/Users/wanghongyi/codes/sam3/sam3/model_builder.py) | `SAM3` / `SAM3.1` builder、HF 下载逻辑、video model 构建入口。 |
| `sam3/model/sam3_tracking_predictor.py` | [/Users/wanghongyi/codes/sam3/sam3/model/sam3_tracking_predictor.py](/Users/wanghongyi/codes/sam3/sam3/model/sam3_tracking_predictor.py) | `SAM3 tracker` 的核心 API：`init_state / add_new_points_or_box / add_new_mask / propagate_in_video`。 |

## 6. AMOS 当前结果汇总

下面只贴当前已经**实际落盘**的 `AMOS` 结果。

### 6.1 纯 SAM2：基础 prompt / 提示数 / prompt type

| 组别 | Mean Dice | Mean IoU | 结果文件 |
|---|---:|---:|---|
| `paper baseline` (`box, K=3, repeat`) | `0.2329` | `0.2636` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_repeat_eval_20260422_200903/summary.json` |
| `more prompts` (`box, K=5, repeat`) | `0.2566` | `0.2868` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211531/summary.json` |
| `point, K=5, repeat` | `0.1240` | `0.1112` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P1_D1_K5_repeat_eval_20260423_001112/summary.json` |
| `multipoint, K=5, repeat` | `0.1183` | `0.1031` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P2_D1_K5_repeat_eval_20260423_005023/summary.json` |
| `box, K=5, neighbor` | `0.2762` | `0.3007` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_neighbor_eval_20260423_012921/summary.json` |
| `mask, K=5, repeat` | `0.3764` | `0.5599` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_020853/summary.json` |
| `box, K=7, repeat` | `0.2499` | `0.2865` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K7_repeat_eval_20260423_024749/summary.json` |

### 6.2 纯 SAM2：backbone scale check

| 组别 | Mean Dice | Mean IoU | 结果文件 |
|---|---:|---:|---|
| `tiny-512` | `0.2566` | `0.2868` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211531/summary.json` |
| `large-512` | `0.2355` | `0.2760` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260423_032837/summary.json` |

### 6.3 纯 SAM2：官方 family（已完成）

| Family | Model | Mean Dice | Mean IoU | 结果文件 |
|---|---|---:|---:|---|
| `SAM2.1` | `tiny` | `0.4588` | `0.5938` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_103952/summary.json` |
| `SAM2.1` | `small` | `0.4991` | `0.5885` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_120944/summary.json` |
| `SAM2.1` | `base_plus` | `0.6197` | `0.5960` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_134154/summary.json` |
| `SAM2.1` | `large` | `0.5020` | `0.6035` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_154334/summary.json` |
| `SAM2` | `tiny` | `0.6154` | `0.6012` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_190149/summary.json` |
| `SAM2` | `small` | `0.5791` | `0.5943` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_202727/summary.json` |
| `SAM2` | `base_plus` | `0.6685` | `0.5992` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_215849/summary.json` |
| `SAM2` | `large` | `0.4967` | `0.5698` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_235907/summary.json` |

### 6.4 `SAM2 base_plus` prompt 扩展（进行中）

这轮只固定 official `facebook/sam2-hiera-base-plus`，重点试三件事：

- `mask` prompt 从 `K=5` 提到 `K=10`
- `mask -> box` 串联 refinement
- 把已跟踪帧的预测重新作为 prompt 写回 (`feedback`)

当前已完成的完整结果：

| Dataset | Variant | Mean Dice | Mean IoU | 结果文件 |
|---|---|---:|---:|---|
| `MMWHS` | `mask, K=10` | `0.8315` | `0.7786` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rnone_FBnone_eval_20260424_012302/summary.json` |
| `MMWHS` | `mask -> box, K=10` | `0.7707` | `0.6725` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBnone_eval_20260424_012856/summary.json` |
| `MMWHS` | `mask -> box, K=10, feedback=mask` | `0.5418` | `0.5074` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBmask_eval_20260424_013524/summary.json` |
| `BTCV` | `mask, K=10` | `0.5972` | `0.8276` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rnone_FBnone_eval_20260424_020640/summary.json` |
| `BTCV` | `mask -> box, K=10` | `0.5223` | `0.6825` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBnone_eval_20260424_024446/summary.json` |
| `BTCV` | `mask -> box, K=10, feedback=mask` | `0.4385` | `0.6366` | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBmask_eval_20260424_032312/summary.json` |

补充结论：

- `BTCV` 上三条 `base_plus` 扩展结果已经全部完成，并且严格单调变差：`mask, K=10` (`0.5972`) > `mask -> box, K=10` (`0.5223`) > `mask -> box + feedback=mask, K=10` (`0.4385`)。
- 这说明当前这套“更密 prompt + 串联 refinement + 预测回灌”对 `BTCV` 不仅没有带来收益，反而在腹部多器官场景中持续放大误差。
- 先前的 `BTCV ... Rbox_FBnone_eval_20260424_012127` 只有 `1` 个 case，只算 smoke test，不纳入正式比较。

## 7. 当前最值得直接看的文件

如果你现在只想看最关键的几个文件，建议先看：

1. [eval_medical_sam2_3d_repro.py](/Users/wanghongyi/codes/SSL4MIS_work3/eval_medical_sam2_3d_repro.py)
2. [SAM2_multiorgan_ablation_20260423.md](/Users/wanghongyi/codes/SSL4MIS_work3/SAM2_multiorgan_ablation_20260423.md)
3. [eval_medical_sam3_3d_repro.py](/Users/wanghongyi/codes/SSL4MIS_work3/eval_medical_sam3_3d_repro.py)
4. [sam3_for_sam2_video_task_example.ipynb](/Users/wanghongyi/codes/sam3/examples/sam3_for_sam2_video_task_example.ipynb)
5. [eval_sam2_3d.py](/Users/wanghongyi/codes/segment-anything2-medical-evaluation/eval_sam2_3d.py)
