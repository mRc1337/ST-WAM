# LIBERO-Plus 三变体复现实验方案

## 1. 目标与边界

复现论文 [ST-WAM: Semantic-Temporal World Action Model for Robust Manipulation under Visual Distribution Shifts](https://arxiv.org/abs/2607.28993) 表 5 中以下三项 LIBERO-Plus 结果：

| Variant | Future Prediction | Intent Conditioning | 目标成功率 |
| --- | --- | --- | ---: |
| Fast-WAM | VAE | None | 51.5% |
| DINO Future Only | DINO | None | 39.7% |
| Dual-Space w/o CAIR | VAE + DINO | None | 66.4% |

本实验只报告 LIBERO-Plus，但训练必须只使用标准 LIBERO。不得使用 LIBERO-Plus 的训练集、混合 SFT 数据、评测结果进行 checkpoint 选择或超参数搜索。LIBERO-Plus 仅用于训练结束后的零样本评测。

## 2. 固定版本

- ST-WAM：`f20d1de0d334fda76e60118f3da5bb908a0bc1e9`
- LIBERO-Plus：`4976dc30028e805ff8094b55501d532c48fec182`
- Python：参考环境 3.10；本项目要求 `>=3.10`
- CUDA：12.8
- PyTorch：2.7.1+cu128
- torchvision：0.22.1+cu128
- MuJoCo：3.3.2
- 训练精度：BF16
- 随机种子：42

运行前保存 `git rev-parse HEAD`、`pip freeze`、GPU 型号、驱动版本，以及训练数据、模型和评测 assets 的校验和。

## 3. 数据

### 3.1 训练数据

使用 Fast-WAM 发布的标准 LIBERO `no-noops` LeRobot 数据：

- `libero_spatial_no_noops_lerobot`
- `libero_object_no_noops_lerobot`
- `libero_goal_no_noops_lerobot`
- `libero_10_no_noops_lerobot`

数据源：<https://huggingface.co/datasets/yuanty/LIBERO-fastwam>

仓库默认期望目录为：

```text
data/libero_mujoco3.3.2/
├── libero_spatial_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
└── libero_10_no_noops_lerobot/
```

四个 suite 各含 500 条 demonstrations，共 2,000 条。第三人称和 wrist 图像分别缩放至 224×224，再水平拼接为 224×448。

本机已有 LeRobot v3 数据时使用：

```bash
export LIBERO_DATA_ROOT=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero
python scripts/check_libero_v3_dataset.py
```

训练和预计算命令追加 `data=libero_2cam_v3`。该配置直接读取 v3 packed
Parquet/MP4，使用 `front`/`wrist` 相机键，并从 `libero_100` 仅选择 task
index 0-9 作为 LIBERO-Long；验收必须得到四套各 500、总计 2,000 episodes。

四个 suite 的统一归一化统计已经由实际配置使用的
`observation.state` 和 `action` 字段生成，不读取 v3 元数据中的 raw
simulator/debug 字段：

```text
artifacts/libero_v3/dataset_stats.json
SHA256 3c6873f05c35f5b8a2cdd1a14bbe06abb6e234f9636fefd3389093f12b4a0f8b
```

三个变体必须使用同一份统计文件。

### 3.2 评测数据

使用官方 LIBERO-Plus 代码与 assets：<https://github.com/sylvestf/LIBERO-plus>。

本机官方 `assets.zip` 已完整性校验并合并至
`/home/pai/zxw/LIBERO-plus/libero/libero/assets`；当前包含 449,909 个文件，
其中 `new_objects/` 包含 432,482 个文件。四个 suite 已成功枚举为 10,030 个 case。

完整测试集共 10,030 个 case，每个 case 只运行一次：

| Suite | 数量 |
| --- | ---: |
| Spatial | 2,402 |
| Object | 2,518 |
| Goal | 2,591 |
| Long (`libero_10`) | 2,519 |

| 扰动类别 | 数量 |
| --- | ---: |
| Camera Viewpoints | 1,599 |
| Robot Initial States | 1,550 |
| Language Instructions | 1,537 |
| Light Conditions | 1,142 |
| Background Textures | 1,076 |
| Sensor Noise | 1,601 |
| Objects Layout | 1,525 |

必须检查 `completed_tasks == 10030` 且 `missing_classification == []`。

## 4. 共同训练和推理设置

- action horizon `H=32`
- future horizon `K=8`
- 未来图像每 4 个 control steps 采样一次；输入序列共当前帧加 8 个未来帧
- AdamW，weight decay `0.01`
- cosine learning-rate decay
- gradient clipping `1.0`
- shifted flow-matching schedule，shift `5.0`
- 训练 10 epochs
- 推理使用 10 个 flow integration steps
- 每次重规划执行 10 个动作
- classifier-free guidance scale `1.0`
- 不启用 action ensemble
- 测试调用 `infer_action`，不显式生成未来视频
- Wan VAE、T5、DINOv3 ViT-S/16 均冻结

三个变体均禁用 CAIR、Qwen current conditioning 和历史帧输入。

## 5. 实验矩阵与训练命令

推荐 8×80GB GPU；官方完整模型使用 8×H200。先运行 100-step smoke test，再启动完整训练。

在所有训练命令前设置：

```bash
cd /home/pai/zxw/ST-WAM
source scripts/activate_libero_plus_env.sh
STATS="$PWD/artifacts/libero_v3/dataset_stats.json"
```

当前本机使用 4×A800，因此三个正式训练顺序独占四卡。训练脚本默认等待满足显存门槛的空闲 GPU；不要在其他任务仍各占约 60GB 时强行启动。

### 5.0 100-step 训练门禁

正式训练前分别运行下列配置的 100-step 版本：

```bash
# Fast-WAM
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 data=libero_2cam_v3 \
  output_dir=./runs/smoke_fastwam \
  +data.train.pretrained_norm_stats="$STATS" \
  gradient_accumulation_steps=2 max_steps=100 \
  save_every=100 eval_every=0 wandb.enabled=false

# DINO Future Only
bash scripts/train_zero1.sh 4 \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 data=libero_2cam_v3 \
  output_dir=./runs/smoke_dino_future \
  +data.train.pretrained_norm_stats="$STATS" \
  batch_size=6 gradient_accumulation_steps=4 learning_rate=1e-5 \
  model.loss.lambda_video=0.05 model.loss.lambda_action=5.0 \
  max_steps=100 save_every=100 eval_every=0 wandb.enabled=false

# Dual-Space w/o CAIR
bash scripts/train_zero1.sh 4 \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 data=libero_2cam_v3 \
  output_dir=./runs/smoke_dual_space \
  +data.train.pretrained_norm_stats="$STATS" \
  gradient_accumulation_steps=16 max_steps=100 \
  save_every=100 eval_every=0 wandb.enabled=false
```

验收条件：无 OOM/NaN、step 100 权重存在、四个 rank 正常退出。正式耗时应使用第 20～100 step 的平均耗时估算，排除模型和数据初始化：

```text
预计训练小时 = mean_seconds_per_step × total_optimizer_steps / 3600
```

### 5.1 Fast-WAM

配置：`libero_uncond_2cam224_1e-4`

- 5B Wan VAE future expert + 1B action expert
- global batch：`16 × 8 GPUs × 1 accumulation = 128`
- `lr=1e-4`
- 损失：`1.0 L_VAE + 1.0 L_action`

```bash
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  data=libero_2cam_v3 \
  output_dir=./runs/fastwam_seed42 \
  +data.train.pretrained_norm_stats="$STATS" \
  gradient_accumulation_steps=2 \
  seed=42 \
  wandb.enabled=true
```

### 5.2 DINO Future Only

配置：`libero_dino_s_smallvideo_2cam_224_1e-4`

- 1B DINO future expert + 1B action expert
- 不使用 VAE future expert

复现表中 39.7% 时，以官方已发布 DINO-only checkpoint 随附的 resolved config 为优先依据：

- microbatch `6`
- gradient accumulation `2`
- 8 GPUs，global batch `96`
- `lr=1e-5`
- `max_steps=28930`
- `lambda_DINO=0.05`
- `lambda_action=5.0`

```bash
bash scripts/train_zero1.sh 4 \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  data=libero_2cam_v3 \
  output_dir=./runs/dino_future_only_seed42 \
  +data.train.pretrained_norm_stats="$STATS" \
  batch_size=6 \
  gradient_accumulation_steps=4 \
  learning_rate=1e-5 \
  max_steps=28930 \
  model.loss.lambda_video=0.05 \
  model.loss.lambda_action=5.0 \
  seed=42 \
  wandb.enabled=true
```

注意：论文正文写的是 global batch 128、`lr=1e-4`，当前仓库默认任务配置的损失权重也是 `1:1`，与发布 artifact 不一致。目标是复现 39.7% 时使用上述 artifact 配置；若另做统一超参消融，必须单独报告，不能与原表数值混合。

### 5.3 Dual-Space w/o CAIR

配置：`libero_wan5b_dino_s_aux_mot_2cam_224_1e-4`

- 5B VAE future expert + 1B DINO future expert + 1B action expert
- global batch：`2 × 8 GPUs × 8 accumulation = 128`
- `lr=1e-4`
- 损失：`1.0 L_VAE + 0.02 L_DINO + 1.0 L_action`
- 必须确认不存在或禁用 `semantic_history_config`

```bash
bash scripts/train_zero1.sh 4 \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  data=libero_2cam_v3 \
  output_dir=./runs/dual_space_no_cair_seed42 \
  +data.train.pretrained_norm_stats="$STATS" \
  gradient_accumulation_steps=16 \
  seed=42 \
  wandb.enabled=true
```

## 6. 缓存与初始化权重

先生成 Action DiT 与 DINO Video DiT 初始化：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

python scripts/preprocess_dino_video_dit_backbone.py \
  --model-config configs/model/fastwam_dino_s_smallvideo.yaml \
  --output checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
```

预计算文本 embedding：

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 data=libero_2cam_v3
```

DINO-only 发布配置使用 frame-mmap DINO cache。官方 cache 约 83.6GB，可从 ModelScope 的 `libero_dino_cache` 下载。若在线编码或自行生成 cache，必须验证特征维度、帧索引和数值一致性。

当前仓库已经具备三项初始化权重、DINOv3 backbone 和 40 条文本 embedding；尚未配置 VAE/DINO latent cache。cache 只用于消除冻结编码器与视频读取开销，不改变训练目标；没有 cache 也能训练，但耗时更长。

### 6.1 4×A800 调度与保守工期

按当前 global batch，Fast-WAM 和 Dual-Space 的 10 epochs 均约 26,500 optimizer steps；DINO-only 固定为 28,930 steps。未使用 latent cache 时，规划区间为：

| 变体 | 4×A800 预计墙钟时间 |
| --- | ---: |
| Fast-WAM | 5～10 天 |
| DINO Future Only | 3～6 天 |
| Dual-Space w/o CAIR | 8～15 天 |
| 三组顺序完成 | 16～31 天 |

以上只是正式启动前的保守容量规划，不作为实验结果；100-step 实测优先。Fast-WAM 与 Dual-Space 不应并行。若要尝试 Fast-WAM 与 DINO-only 各使用两卡，必须先分别通过两卡 OOM 门禁、保持 global batch，并为两个分布式作业设置不同 `MASTER_PORT`；这种调度不保证缩短总工期。

## 7. LIBERO-Plus 评测

对三个模型使用相同 GPU 数量、worker 划分、任务 manifest、seed 和 dataset statistics。由于随机动作噪声流会受到 worker 分片影响，不要在不同模型间改变 worker 数量。

```bash
export LIBERO_ROOT=/home/pai/zxw/LIBERO-plus

python experiments/libero/run_libero_persistent_pool.py \
  --config <TASK_CONFIG> \
  --ckpt <CHECKPOINT_PT> \
  --output-dir <RESULT_DIR> \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --expected-num-tasks 10030 \
  --num-trials 1 \
  --gpu-ids 0,1,2,3 \
  --workers-per-gpu 1 \
  --plus-summary \
  --classification-path \
    "$LIBERO_ROOT/libero/libero/benchmark/task_classification.json" \
  --extra-override seed=42 \
  --extra-override EVALUATION.sigma_shift=5.0 \
  --extra-override EVALUATION.dataset_stats_path=<DATASET_STATS_JSON> \
  --skip-existing \
  --restart-failed-workers \
  --max-worker-restarts 3
```

三个 `<TASK_CONFIG>` 分别为：

```text
libero_uncond_2cam224_1e-4
libero_dino_s_smallvideo_2cam_224_1e-4
libero_wan5b_dino_s_aux_mot_2cam_224_1e-4
```

正式评测前先使用 `--sample-total 280 --sample-stratified` 做小规模流水线测试；该选项只按 suite 分层，需额外检查七个扰动类别是否都被覆盖。

280-case 的 wall time 可以线性外推单模型全量评测时间：

```text
full_eval_time ≈ sample_280_time × 10030 / 280
               ≈ sample_280_time × 35.82
```

## 8. 指标和验收

主指标是所有 10,030 个 case 的 micro-average success rate，不是七个类别成功率的简单平均。

验收条件：

1. 官方 checkpoint 评测结果应在一位小数上复现目标值。
2. 从零训练结果在目标值 ±1.5 percentage points 内视为成功复现。
3. 排序必须满足 `Dual-Space > Fast-WAM > DINO Future Only`。
4. 预期差值为：Dual-Space 相对 Fast-WAM `+14.9 pp`，相对 DINO-only `+26.7 pp`。
5. 保存逐任务二值结果，并对同一组 case 使用 paired bootstrap 或 McNemar test。

若只运行一个训练 seed，不能把 10,030 个测试 case 当作训练随机性的重复实验。论文级验证建议训练 3 个 seed，并分别完整评测；资源不足时先完成 seed 42 的数值复现。

## 9. 已知复现风险

- DINO-only 的论文正文、仓库默认配置和发布 checkpoint 配置存在超参数差异。
- ModelScope 当前发布了 DINO-only checkpoint，但没有单独发布 Dual-Space w/o CAIR checkpoint，66.4% 需要从零训练验证。
- LIBERO-Plus assets 不包含在 Git 仓库中，必须额外下载并核对路径。
- Python `wand` 还要求系统安装 ImageMagick/MagickWand（Debian/Ubuntu 包
  `libmagickwand-dev`），安装后才能导入 LIBERO-Plus 并做 EGL smoke test。
- 当前执行机器必须能访问 NVIDIA 驱动；无可见 GPU 时只能完成安装和 CPU 导入检查，不能验证 CUDA 训练。
- 完整评测包含 30,090 个 episode（三模型各 10,030），评测前应利用 280-case smoke run 实测吞吐后再估算总时长。

## 10. 最终产物

每个模型保留：

- resolved Hydra config
- `dataset_stats.json`
- checkpoint SHA256
- train log 和 loss 曲线
- LIBERO-Plus worker manifests
- 10,030 个逐任务 JSON
- `plus_summary.json`
- `plus_leaderboard_summary.csv`
- 七扰动类别与四 suite 的明细 CSV

最终汇总表至少包含总体成功率、七类扰动成功率、与目标值的差异、完整任务数、失败/缺失任务数、代码和数据版本。

## 11. 跨机器评测

Git 仓库只传输代码、配置、脚本、文档和小型归一化统计，不包含训练数据、LIBERO-Plus assets、模型权重及 latent/text cache。训练结束后，每个模型向评测机单独传输：

```text
runs/<experiment>/config.yaml
runs/<experiment>/dataset_stats.json
runs/<experiment>/checkpoints/weights/step_XXXXXX.pt
```

如果 run 目录没有统计副本，使用仓库中的
`artifacts/libero_v3/dataset_stats.json`。传输后在两端分别执行
`sha256sum`，确认 checkpoint 和统计文件一致。

评测机准备流程：

1. clone 本实验对应的精确 commit，并按 `scripts/setup_libero_plus_env.sh` 安装环境；
2. 单独 clone LIBERO-Plus，并把完整 assets 解压到
   `$LIBERO_ROOT/libero/libero/assets`；
3. 下载模型运行依赖，包括 Wan2.2、tokenizer、ActionDiT、DINO Video DiT 和 DINOv3 backbone；
4. `source scripts/activate_libero_plus_env.sh`，运行 EGL reset/render smoke test；
5. 用 280-case 固定抽样验证 checkpoint、统计、CUDA/EGL 映射和输出目录；
6. 保持三模型相同的 GPU 数、worker 数、seed、manifest 生成逻辑和统计文件，再运行 10,030-case 全量评测。

跨机器评测不需要复制 LeRobot 训练数据，也不需要训练用 VAE/DINO latent cache；但必须复制训练后的 `.pt` checkpoint，Git 仓库不会包含它。
