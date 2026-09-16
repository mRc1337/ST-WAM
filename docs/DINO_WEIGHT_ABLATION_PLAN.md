# VAE+DINO 辅助权重消融实验方案

## 1. 目的

在当前三任务 LIBERO-Object 实验中，判断以下两个问题：

1. 将 DINO future prediction 的损失权重从 `0.02` 提高后，是否能稳定提升 LIBERO-Plus 视觉扰动鲁棒性；
2. Dual-Space 的收益究竟来自当前帧 DINO 条件，还是来自 DINO future prediction 辅助目标。

本实验不微调冻结的 DINOv3 backbone。`lambda_dino` 影响的是 DINO VideoDiT 的未来特征预测目标；DINO 分支还会通过 MoT 接收 action loss 的梯度。

## 2. 固定实验条件

- 数据：`libero_object_tasks_1_4_7_v3`，即 BBQ sauce、cream cheese、orange juice 三个任务；
- 训练集：120 episodes、17,331 transitions；
- 初始化：同一 Wan2.2、ActionDiT 和 DinoVideoDiT 初始化；
- 训练：4×A800、global batch 128、1,500 optimizer steps、seed 42；
- 损失：`lambda_vae=1.0`、`lambda_action=1.0`；
- Intent/CAIR：关闭；
- normalization statistics：复用原三任务 DINO run 的 `dataset_stats.json`；
- 除 `dino_future_mode/lambda_dino` 外，其余参数保持一致。

## 3. 实验组

| 名称 | `dino_future_mode` | `lambda_dino` | 作用 |
| --- | --- | ---: | --- |
| baseline | `predict` | 0.02 | 已完成的原始 Dual-Space 基线 |
| lambda005 | `predict` | 0.05 | 将 DINO 直接辅助梯度扩大到基线的 2.5 倍 |
| lambda010 | `predict` | 0.10 | 将 DINO 直接辅助梯度扩大到基线的 5 倍 |
| condition_only | `condition_only` | 0 | 保留当前帧 DINO 条件，但不预测未来 DINO 特征 |

不要把 loss 系数解释成表征占比。VAE、DINO 和 action token 在 MoT 中共同参与注意力，动作损失也能更新 DINO VideoDiT。提高 `lambda_dino` 只是增强直接的 DINO future supervision。

## 4. 评价方法

每个 checkpoint 评测：

- Clean：3 个原始任务，每任务 50 次 rollout，共 150 episodes；
- LIBERO-Plus：三个任务对应的全部 745 个 case，每 case 1 次 rollout；
- 主指标：Plus 总成功率；
- 辅助指标：Clean 成功率、各扰动类别成功率、`Plus−Clean` 鲁棒性下降；
- 训练诊断：记录 `loss_video`、`loss_dino`、`loss_action`，但不以 total loss 跨 variant 排名。

判定规则：若 `0.05` 或 `0.10` 的 Plus 成功率提高且 Clean 不下降超过 2 个百分点，视为增强有效。若 `condition_only` 与 baseline 接近，说明收益主要来自当前帧 DINO 条件，而非 future objective。

当前每个 Plus case 只有一次 rollout，结论属于探索性消融。确认最佳权重后，应再用 seed 43 训练，并对最佳组和 baseline 各执行 3 rollouts/case。

## 5. 一键训练

现有 `lambda=0.02` checkpoint 会直接作为 baseline，不会重复训练。脚本依次训练 `0.05`、`0.10` 和 `condition_only`，每次独占四张 GPU，并支持从最新完整状态断点续训。

```bash
cd /home/pai/zxw/ST-WAM
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/train_dino_weight_ablation.sh \
  2>&1 | tee -a runs/dino_weight_ablation/launcher.log
```

tmux 中启动：

```bash
tmux new -s dino-weight
cd /home/pai/zxw/ST-WAM
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_dino_weight_ablation.sh \
  2>&1 | tee -a runs/dino_weight_ablation/launcher.log
```

训练状态：

```bash
cat runs/dino_weight_ablation/status/train.state
tail -f runs/dino_weight_ablation/launcher.log
```

只训练指定组：

```bash
VARIANTS=lambda005 bash scripts/train_dino_weight_ablation.sh
```

checkpoint 实体默认存放于：

```text
/mnt/data/embodied_datasets/ST-WAM/checkpoints/dino_weight_ablation/
```

## 6. 一键评测

训练完成后运行以下命令。脚本会串行评测四个 variant；每个 variant 的 Plus case 在四张 GPU 上并行，并支持跳过已经完成的 case。当前机器上已有的 `0.02` baseline 结果会默认通过软链接复用，不重复 rollout；在其他机器找不到旧结果时会自动重新评测。

```bash
cd /home/pai/zxw/ST-WAM
source .venv/bin/activate

bash scripts/eval_dino_weight_ablation.sh \
  2>&1 | tee -a evaluate_results/dino_weight_ablation/eval_launcher.log
```

评测状态及汇总：

```bash
cat evaluate_results/dino_weight_ablation/step1500_seed42/status/eval.state
tail -f evaluate_results/dino_weight_ablation/eval_launcher.log
cat evaluate_results/dino_weight_ablation/step1500_seed42/comparison.md
```

可先只做 Clean 冒烟与筛查：

```bash
EVAL_SPLITS=clean bash scripts/eval_dino_weight_ablation.sh
```

随后评测 Plus（已完成 case 会被跳过）：

```bash
EVAL_SPLITS=plus bash scripts/eval_dino_weight_ablation.sh
```

只评测指定组：

```bash
VARIANTS=lambda005 EVAL_SPLITS=clean,plus \
  bash scripts/eval_dino_weight_ablation.sh
```

## 7. 输出位置

```text
runs/dino_weight_ablation/                         # 配置、日志及 checkpoint 软链接
/mnt/data/embodied_datasets/ST-WAM/checkpoints/dino_weight_ablation/  # checkpoint 实体
evaluate_results/dino_weight_ablation/step1500_seed42/                # rollout 结果
evaluate_results/dino_weight_ablation/step1500_seed42/comparison.md   # 自动汇总
```

## 8. 解释限制

- `lambda_dino` 变大不代表 DINOv3 backbone 被微调；backbone 始终冻结；
- `loss_video` 在本 Dual-Space 模型中指 VAE latent future loss，`loss_dino` 才是 DINO feature future loss；
- 若 `lambda=0.10` 的 DINO loss 更低但闭环成功率下降，通常表示多任务梯度冲突或辅助目标过强，而不是训练失败；
- 不应根据同一批 Plus case 反复追加权重。完成这组预注册权重后，应固定最佳设置并换 seed 验证。
