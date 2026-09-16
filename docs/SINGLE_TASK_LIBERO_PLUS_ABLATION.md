# LIBERO-Plus 单任务短程消融方案

> 三任务扩展：固定使用 LeRobot task indices `[1, 4, 7]`（BBQ sauce、cream cheese、orange juice），训练和评测的一键入口分别为
> `scripts/train_three_task_ablation.sh` 与 `scripts/eval_three_task_ablation.sh`。默认 1,500 steps、seed 42、三模型串行训练；评测覆盖三个任务的 745 个 LIBERO-Plus case。

## 1. 实验目标

用尽量小的训练和评测成本，验证在同一训练预算下，`VAE + DINO` future prediction 是否比单独使用 `VAE` 或 `DINO` 更稳健。该实验是趋势验证，不等价于论文四套 LIBERO、10,030 个 LIBERO-Plus case 的完整复现，也不应将结果直接与论文的 51.5%、39.7%、66.4% 横向比较。

仅改变 future representation，其他设置保持一致：

| Variant | Future prediction | Intent conditioning | Global batch | Steps | LR |
| --- | --- | --- | ---: | ---: | ---: |
| Fast-WAM | VAE | None | 128 | 500 | 1e-4 |
| DINO Future Only | DINO | None | 128 | 500 | 1e-4 |
| Dual-Space w/o CAIR | VAE + DINO | None | 128 | 500 | 1e-4 |

所有变体关闭 CAIR，固定随机种子 42。损失权重为 Fast-WAM `VAE:action=1:1`、DINO `DINO:action=1:1`、Dual-Space `VAE:DINO:action=1:0.02:1`。

## 2. 单任务与数据划分

- suite：`libero_object`
- task index：`1`
- 指令：`pick up the bbq sauce and place it in the basket`
- 标准 LIBERO 数据：50 episodes、7,348 frames
- 数据配置：`configs/data/libero_object_task1_v3.yaml`
- 固定 seed 42，按 episode 划分为 40 train / 10 validation

禁止按 frame 随机划分，否则同一轨迹的相邻帧会同时出现在训练集和验证集，形成时序泄漏。归一化统计只能由 40 个训练 episodes 生成；三个变体必须复用同一份统计。

500 optimizer steps 对应约 `500 × 128 / (40 × 约150有效窗口)` 的采样预算，约为 10–11 个 episode-equivalent epochs。由于单任务数据很少，超过该预算更容易记忆演示背景，而不一定提升视觉分布外泛化。

## 3. 防止 LIBERO-Plus 泄漏

训练、归一化统计、checkpoint 选择和超参数选择均只使用标准 LIBERO：

- 不把 LIBERO-Plus case、图像或 assets 加入训练集。
- 不采用针对 Plus 扰动类别设计的增强作为本轮主结果。
- 不根据 LIBERO-Plus 成功率选择 step 100/200/300/400/500 checkpoint。
- 主结果固定使用 step 500；中间 checkpoint 仅用于诊断训练崩溃或明显退化。
- 当前 `Trainer.evaluate()` 每个 rank 只采一个验证样本，因此验证 loss 只用于监控，不作为早停依据。

若 500 steps 后三个模型的 clean success 都很低，应统一增加预算并重跑全部变体；不能只延长表现较差的一个模型。

## 4. 训练资源与公平性

使用 4×A800，三种模型依次独占四卡，不建议并行训练。microbatch 和 gradient accumulation 设置为：

| Variant | Per-GPU batch | Accumulation | GPUs | Global batch |
| --- | ---: | ---: | ---: | ---: |
| Fast-WAM | 8 | 4 | 4 | 128 |
| DINO Future Only | 4 | 8 | 4 | 128 |
| Dual-Space w/o CAIR | 2 | 16 | 4 | 128 |

训练前先完成 W&B 登录，并确认三个初始化权重存在：

```bash
wandb login
test -f checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
test -f checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
test -f checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors
```

## 5. 训练命令

先训练 DINO，并让它从 40 个训练 episodes 生成归一化统计：

```bash
cd /home/pai/zxw/ST-WAM
source scripts/activate_libero_plus_env.sh
export LIBERO_DATA_ROOT=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WAIT_FOR_GPUS=1

bash scripts/train_zero1.sh 4 \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  data=libero_object_task1_v3 \
  output_dir=./runs/single_task/dino_future_seed42_500 \
  batch_size=4 gradient_accumulation_steps=8 \
  learning_rate=1e-4 max_steps=500 \
  model.loss.lambda_video=1.0 model.loss.lambda_action=1.0 \
  save_every=100 eval_every=100 seed=42 \
  wandb.enabled=true wandb.mode=online \
  wandb.project=st-wam-libero-plus \
  wandb.group=single-task-500step-ablation \
  wandb.name=dino-future-object-task1-seed42
```

确认统计文件生成后，VAE 和 Dual-Space 必须复用它：

```bash
STATS="$PWD/runs/single_task/dino_future_seed42_500/dataset_stats.json"
test -s "$STATS"

bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  data=libero_object_task1_v3 \
  output_dir=./runs/single_task/fastwam_vae_seed42_500 \
  +data.train.pretrained_norm_stats="$STATS" \
batch_size=8 gradient_accumulation_steps=4 \
  learning_rate=1e-4 max_steps=500 \
  model.loss.lambda_action=1.0 \
  save_every=100 eval_every=100 seed=42 \
  wandb.enabled=true wandb.mode=online \
  wandb.project=st-wam-libero-plus \
  wandb.group=single-task-500step-ablation \
  wandb.name=fastwam-vae-object-task1-seed42

bash scripts/train_zero1.sh 4 \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  data=libero_object_task1_v3 \
  output_dir=./runs/single_task/dual_space_seed42_500 \
  +data.train.pretrained_norm_stats="$STATS" \
  batch_size=2 gradient_accumulation_steps=16 \
  learning_rate=1e-4 max_steps=500 \
  model.loss.lambda_video=1.0 model.loss.lambda_dino=0.02 \
  model.loss.lambda_action=1.0 \
  save_every=100 eval_every=100 seed=42 \
  wandb.enabled=true wandb.mode=online \
  wandb.project=st-wam-libero-plus \
  wandb.group=single-task-500step-ablation \
  wandb.name=dual-space-object-task1-seed42
```

三个命令应串行执行。若某次中断，使用相同命令并追加 `resume=<state checkpoint 目录>`；不要修改 `output_dir`、数据划分或统计文件。例如 DINO 从 step 400 完整恢复 optimizer、scheduler、随机数和 dataloader 进度时，仍使用上面的完整训练命令，并在末尾追加：

```bash
resume=./runs/single_task/dino_future_seed42_500/checkpoints/state/step_000400
```

不要把 `checkpoints/weights/step_000400.pt` 当作严格断点续训；传入 `.pt` 只会恢复模型权重，不会恢复 optimizer/scheduler 状态。

本机 A800 实测 VAE 的每卡 batch 16 会占满约 79.1 GiB 并在首个 forward OOM，因此固定使用 batch 8、累积 4。训练 checkpoint 已迁移到数据盘，原路径为符号链接；命令中的 `output_dir` 保持不变：

```text
runs/single_task/dino_future_seed42_500/checkpoints
  -> /mnt/data/embodied_datasets/ST-WAM/checkpoints/single_task/dino_future_seed42_500
runs/single_task/fastwam_vae_seed42_500/checkpoints
  -> /mnt/data/embodied_datasets/ST-WAM/checkpoints/single_task/fastwam_vae_seed42_500
runs/single_task/dual_space_seed42_500/checkpoints
  -> /mnt/data/embodied_datasets/ST-WAM/checkpoints/single_task/dual_space_seed42_500
```

因此后续 Dual-Space 训练以及任何续训所生成的 checkpoint 都直接写入数据盘；配置、W&B 日志和 dataset stats 仍保留在本地 run 目录。

## 6. 评测设计

先在标准 LIBERO clean task 上评测，确认模型学会基本行为，再评测同一 task 的 LIBERO-Plus 对应 case。该任务在 LIBERO-Plus 中有 292 个 case，覆盖七类扰动。

固定一次性生成并保存 category-balanced manifest：

- 快速 pilot：每类 10 case，共 70 case。
- 主结果：每类 20 case，共 140 case。
- 最终确认：资源允许时运行该任务全部 292 case。

同一 manifest、初始状态、rollout 上限和推理参数用于三个模型。pilot 只能检查执行链路，不能用于选择 checkpoint 或调整某一变体。

每个模型至少报告：

```text
Clean Success = clean successes / clean rollouts
Plus Success  = plus successes / plus rollouts
Robustness Drop = Clean Success - Plus Success
```

同时报告七类扰动的 success rate、case 数、随机种子和 Wilson 95% 置信区间。核心判断为 Dual-Space 的 Plus Success 是否同时高于两个单分支，并结合置信区间判断差异是否足够稳定。单 seed、140 case 只能作为低成本趋势证据；正式论文结论建议至少 3 seeds 或恢复完整评测。

## 7. 继续/停止规则

1. 先完成三个 500-step 训练，不观察 Plus 结果做中途决策。
2. clean success 明显不足时，三者统一扩展到 1,000 steps。
3. clean success 可用且 Dual-Space 在 140-case Plus 集上稳定优于单分支，则完成低成本验证。
4. 差异落在置信区间内时，先增加评测 case 或 seed，而不是单独增加某一模型训练步数。

## 8. 当前状态与评测兼容约定

截至 2026-09-04，DINO Future Only 的 500-step checkpoint 已完成：

```text
runs/single_task/dino_future_seed42_500/checkpoints/weights/step_000500.pt
runs/single_task/dino_future_seed42_500/dataset_stats.json
```

修复 LeRobot v3 评测接口后，DINO 在 clean `libero_object/task_id=3` 上为 `45/50 = 90%`。结果文件位于：

```text
evaluate_results/single_task/dino_clean_v3_fixed/libero_object/gpu0_task3_results.json
```

旧目录 `evaluate_results/single_task/dino_clean` 使用了错误的图像/夹爪约定，不能作为实验结果。LeRobot v3 必须使用 `configs/sim_libero_v3.yaml`，其约定为：

- 图像不旋转 180°；
- 夹爪保持 LIBERO 原生动作语义：`-1=open, +1=close`；
- 仿真相机分辨率为 128，再按训练配置缩放到每路 224×224；
- clean task 必须使用 `/home/pai/zxw/LIBERO-PRO`；
- Plus task 必须使用 `/home/pai/zxw/LIBERO-plus`；
- clean 与 Plus 的 task ID 空间不同。clean 的 `task_id=3` 绝不能直接用于 Plus。

## 9. 实验前检查

以下命令只做检查，不启动训练或 rollout：

```bash
cd /home/pai/zxw/ST-WAM
source .venv/bin/activate

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

test -f configs/data/libero_object_task1_v3.yaml
test -f configs/sim_libero_v3.yaml
test -f checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
test -f checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
test -f checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors
test -f runs/single_task/dino_future_seed42_500/checkpoints/weights/step_000500.pt
test -s runs/single_task/dino_future_seed42_500/dataset_stats.json
test -d /home/pai/zxw/LIBERO-PRO/libero/libero/assets
test -d /home/pai/zxw/LIBERO-plus/libero/libero/assets/new_objects
test -f /home/pai/zxw/LIBERO-plus/libero/libero/benchmark/task_classification.json
test -f /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero/libero_object/meta/info.json

python -m py_compile \
  experiments/libero/eval_libero_single.py \
  experiments/libero/eval_libero_worker.py \
  experiments/libero/libero_utils.py
```

检查 clean task map。输出必须包含任务总数 `10` 和目标指令：

```bash
export LIBERO_ROOT=/home/pai/zxw/LIBERO-PRO
export LIBERO_CONFIG_PATH=/home/pai/zxw/LIBERO-PRO/.libero
export PYTHONPATH="$LIBERO_ROOT:/home/pai/zxw/ST-WAM/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/home/pai/zxw/ST-WAM/.runtime/matplotlib

python - <<'PY'
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(3)
print("n_tasks:", suite.n_tasks)
print("task_id=3:", task.language)
assert suite.n_tasks == 10
assert task.language == "pick up the bbq sauce and place it in the basket"
PY
```

## 10. 标准 LIBERO clean 评测

每次新终端先执行：

```bash
cd /home/pai/zxw/ST-WAM
source .venv/bin/activate

export LIBERO_ROOT=/home/pai/zxw/LIBERO-PRO
export LIBERO_CONFIG_PATH=/home/pai/zxw/LIBERO-PRO/.libero
export LIBERO_DATA_ROOT=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero
export PYTHONPATH="$LIBERO_ROOT:/home/pai/zxw/ST-WAM/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/home/pai/zxw/ST-WAM/.runtime/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0

STATS=/home/pai/zxw/ST-WAM/runs/single_task/dino_future_seed42_500/dataset_stats.json
EXPECTED='pick up the bbq sauce and place it in the basket'
```

先用 5 trials 做 smoke test；通过后改为 50 trials。DINO 的完整命令为：

```bash
python experiments/libero/eval_libero_single.py \
  --config-name sim_libero_v3 \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  data=libero_object_task1_v3 \
  ckpt=/home/pai/zxw/ST-WAM/runs/single_task/dino_future_seed42_500/checkpoints/weights/step_000500.pt \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.task_suite_name=libero_object \
  EVALUATION.task_id=3 \
  EVALUATION.expected_task_description="$EXPECTED" \
  EVALUATION.num_trials=50 \
  EVALUATION.output_dir=/home/pai/zxw/ST-WAM/evaluate_results/single_task/dino_clean_v3_fixed
```

VAE 与 Dual-Space 训练完成后分别执行：

```bash
python experiments/libero/eval_libero_single.py \
  --config-name sim_libero_v3 \
  task=libero_uncond_2cam224_1e-4 \
  data=libero_object_task1_v3 \
  ckpt=/home/pai/zxw/ST-WAM/runs/single_task/fastwam_vae_seed42_500/checkpoints/weights/step_000500.pt \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.task_suite_name=libero_object \
  EVALUATION.task_id=3 \
  EVALUATION.expected_task_description="$EXPECTED" \
  EVALUATION.num_trials=50 \
  EVALUATION.output_dir=/home/pai/zxw/ST-WAM/evaluate_results/single_task/vae_clean_v3_fixed

python experiments/libero/eval_libero_single.py \
  --config-name sim_libero_v3 \
  task=libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  data=libero_object_task1_v3 \
  ckpt=/home/pai/zxw/ST-WAM/runs/single_task/dual_space_seed42_500/checkpoints/weights/step_000500.pt \
  EVALUATION.dataset_stats_path="$STATS" \
  EVALUATION.task_suite_name=libero_object \
  EVALUATION.task_id=3 \
  EVALUATION.expected_task_description="$EXPECTED" \
  EVALUATION.num_trials=50 \
  EVALUATION.output_dir=/home/pai/zxw/ST-WAM/evaluate_results/single_task/dual_clean_v3_fixed
```

检查结果：

```bash
python - <<'PY'
import json
from pathlib import Path
for variant in ("vae", "dino", "dual"):
    path = Path(f"evaluate_results/single_task/{variant}_clean_v3_fixed/libero_object/gpu0_task3_results.json")
    if not path.exists():
        print(f"{variant}: not evaluated")
        continue
    result = json.loads(path.read_text())
    print(f"{variant}: {result['successes']}/{result['total_episodes']} = "
          f"{100 * result['successes'] / result['total_episodes']:.1f}%")
PY
```

## 11. 生成单任务 LIBERO-Plus manifest

Plus 的 `libero_object` 一共有 2,518 个 task，其中与目标 BBQ sauce 任务对应的 case 恰好为 292 个，类别分布为：背景 47、相机 47、语言 26、光照 49、布局 44、传感器噪声 41、机器人初态 38。

下面的命令按类别固定抽样。主实验设置 `PER_CATEGORY=20`，得到 140 cases；smoke test 可改为 `PER_CATEGORY=1`，pilot 可改为 `PER_CATEGORY=10`。一旦生成，三个模型必须复用同一 manifest，不得分别抽样。

```bash
cd /home/pai/zxw/ST-WAM
source scripts/activate_libero_plus_env.sh

export PER_CATEGORY=20
export MANIFEST_DIR=/home/pai/zxw/ST-WAM/evaluate_results/single_task/manifests/bbq_plus_140_seed42
mkdir -p "$MANIFEST_DIR"

python - <<'PY'
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

classification_path = Path(
    "/home/pai/zxw/LIBERO-plus/libero/libero/benchmark/task_classification.json"
)
output_dir = Path(os.environ["MANIFEST_DIR"])
per_category = int(os.environ["PER_CATEGORY"])
seed = 42
prefix = "pick_up_the_bbq_sauce_and_place_it_in_the_basket"

raw = json.loads(classification_path.read_text(encoding="utf-8"))
matches = [
    entry for entry in raw["libero_object"]
    if entry["name"].startswith(prefix)
]
assert len(matches) == 292, f"expected 292 BBQ cases, got {len(matches)}"

by_category = defaultdict(list)
for entry in matches:
    by_category[entry["category"]].append(entry)
assert len(by_category) == 7, by_category.keys()
assert all(len(entries) >= per_category for entries in by_category.values())

rng = random.Random(seed)
selected = []
for category in sorted(by_category):
    entries = sorted(by_category[category], key=lambda item: item["id"])
    selected.extend(rng.sample(entries, per_category))
selected.sort(key=lambda item: item["id"])

tasks = [
    {"suite": "libero_object", "task_id": entry["id"] - 1}
    for entry in selected
]
full_path = output_dir / "full.json"
full_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

for gpu_id in range(4):
    shard = tasks[gpu_id::4]
    (output_dir / f"gpu{gpu_id}.json").write_text(
        json.dumps(shard, indent=2), encoding="utf-8"
    )

print("all matching cases:", len(matches))
print("selected cases:", len(tasks))
print("selected categories:", Counter(entry["category"] for entry in selected))
print("shard sizes:", [len(tasks[gpu_id::4]) for gpu_id in range(4)])
print("manifest:", full_path)
PY
```

主实验应输出 `selected cases: 140` 和 `shard sizes: [35, 35, 35, 35]`。如需评测全部 292 cases，将上面的抽样段替换为 `selected = sorted(matches, key=lambda item: item["id"])`，并使用新的 manifest 目录，防止覆盖主实验清单。

## 12. 四卡 LIBERO-Plus 评测

每次新终端重新激活 Plus 环境并设置公共变量。不要保留上一节 clean 环境的 `LIBERO_ROOT`：

```bash
cd /home/pai/zxw/ST-WAM
source scripts/activate_libero_plus_env.sh

export LIBERO_ROOT=/home/pai/zxw/LIBERO-plus
export LIBERO_PLUS_ROOT=/home/pai/zxw/LIBERO-plus
export LIBERO_CONFIG_PATH=/home/pai/zxw/ST-WAM/.runtime/libero
export PYTHONPATH="$LIBERO_ROOT:/home/pai/zxw/ST-WAM/src:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR=/home/pai/zxw/ST-WAM/.runtime/matplotlib
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export STATS=/home/pai/zxw/ST-WAM/runs/single_task/dino_future_seed42_500/dataset_stats.json
export MANIFEST_DIR=/home/pai/zxw/ST-WAM/evaluate_results/single_task/manifests/bbq_plus_140_seed42
export CLASSIFICATION=/home/pai/zxw/LIBERO-plus/libero/libero/benchmark/task_classification.json

test -s "$MANIFEST_DIR/full.json"
for gpu in 0 1 2 3; do test -s "$MANIFEST_DIR/gpu${gpu}.json"; done
```

定义一次四卡运行函数。每张卡常驻一个模型 worker，避免每个 case 都重新加载约 4–12 GB checkpoint。`MUJOCO_EGL_DEVICE_ID=0` 表示四个进程共用 EGL 设备 0 做渲染，模型仍分别在 `cuda:0..3`；如果本机 EGL/CUDA 编号不同，只调整该变量，不改变 manifest 分片。

```bash
run_plus_variant() {
  local variant="$1"
  local task_config="$2"
  local checkpoint="$3"
  local output_dir="$4"
  local pids=()
  local failed=0

  test -f "$checkpoint" || { echo "missing checkpoint: $checkpoint" >&2; return 1; }
  mkdir -p "$output_dir/worker_logs"

  for gpu in 0 1 2 3; do
    MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0 \
    python experiments/libero/eval_libero_worker.py \
      --config-name sim_libero_v3 \
      task="$task_config" \
      data=libero_object_task1_v3 \
      ckpt="$checkpoint" \
      gpu_id="$gpu" \
      EVALUATION.device="cuda:$gpu" \
      EVALUATION.dataset_stats_path="$STATS" \
      EVALUATION.num_trials=1 \
      EVALUATION.output_dir="$output_dir" \
      +EVALUATION.task_manifest="$MANIFEST_DIR/gpu${gpu}.json" \
      +EVALUATION.skip_existing=true \
      +EVALUATION.continue_on_error=false \
      seed=42 \
      >"$output_dir/worker_logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "$variant failed; inspect $output_dir/worker_logs" >&2
    return 1
  fi

  python experiments/libero/summarize_results.py --output_dir="$output_dir"
  python experiments/libero/summarize_libero_plus_results.py \
    --output_dir "$output_dir" \
    --classification_path "$CLASSIFICATION"
}
```

先运行已完成的 DINO：

```bash
run_plus_variant \
  dino \
  libero_dino_s_smallvideo_2cam_224_1e-4 \
  /home/pai/zxw/ST-WAM/runs/single_task/dino_future_seed42_500/checkpoints/weights/step_000500.pt \
  /home/pai/zxw/ST-WAM/evaluate_results/single_task/dino_plus_140_seed42
```

VAE 和 Dual-Space checkpoint 存在后再运行：

```bash
run_plus_variant \
  vae \
  libero_uncond_2cam224_1e-4 \
  /home/pai/zxw/ST-WAM/runs/single_task/fastwam_vae_seed42_500/checkpoints/weights/step_000500.pt \
  /home/pai/zxw/ST-WAM/evaluate_results/single_task/vae_plus_140_seed42

run_plus_variant \
  dual \
  libero_wan5b_dino_s_aux_mot_2cam_224_1e-4 \
  /home/pai/zxw/ST-WAM/runs/single_task/dual_space_seed42_500/checkpoints/weights/step_000500.pt \
  /home/pai/zxw/ST-WAM/evaluate_results/single_task/dual_plus_140_seed42
```

断点续评时重新执行相同命令即可；`skip_existing=true` 会跳过已有非空结果文件。不要改变 GPU 数、分片文件或 seed。监控命令：

```bash
watch -n 5 nvidia-smi
tail -f evaluate_results/single_task/dino_plus_140_seed42/worker_logs/gpu0.log
find evaluate_results/single_task/dino_plus_140_seed42/libero_object \
  -name 'gpu*_task*_results.json' | wc -l
```

## 13. 汇总与验收

VAE 与 Dual-Space checkpoint 完成后，推荐使用统一脚本一次运行剩余的 clean 和 Plus 评测：

```bash
cd /home/pai/zxw/ST-WAM
bash scripts/eval_single_task_ablation.sh --variants vae,dual
```

脚本默认依次运行 VAE clean、VAE Plus、Dual-Space clean、Dual-Space Plus；clean 使用 50 trials，Plus 使用固定 140-case manifest 和四卡并行。已有完整结果自动跳过，Plus 中断后重跑同一命令会按结果 JSON 断点续评。单独运行某一阶段可使用 `--clean-only` 或 `--plus-only`。

每个 Plus 输出目录会生成 `plus_summary.json`、`plus_leaderboard_summary.csv`、`plus_category_summary.csv` 和 `plus_task_success_rates.csv`。快速汇总三模型：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("evaluate_results/single_task")
for variant in ("vae", "dino", "dual"):
    clean_path = root / f"{variant}_clean_v3_fixed/libero_object/gpu0_task3_results.json"
    plus_path = root / f"{variant}_plus_140_seed42/plus_summary.json"
    if not clean_path.exists() or not plus_path.exists():
        print(f"{variant}: incomplete")
        continue
    clean = json.loads(clean_path.read_text())
    plus = json.loads(plus_path.read_text())
    clean_rate = 100 * clean["successes"] / clean["total_episodes"]
    total = plus["leaderboard"]["Total"]
    plus_rate = float(total["Success Rate (%)"])
    print(
        f"{variant}: clean={clean_rate:.2f}% "
        f"plus={plus_rate:.2f}% drop={clean_rate - plus_rate:.2f}pp "
        f"plus_n={total['Episodes']}"
    )
PY
```

验收顺序：

1. 先确认 clean task 描述正确、输出视频方向正确、夹爪能开合。
2. clean 运行 50 trials，三个模型都使用同一个 stats 文件。
3. 用每类 1 case 的 7-case Plus manifest 做流水线 smoke test。
4. 固定每类 20 case 的 140-case manifest，三个模型各运行一次。
5. 比较 `Plus Success` 与 `Robustness Drop`，并查看七类扰动明细。
6. 只有趋势不稳定时才增加 case 数或 seed；不得根据 Plus 结果挑 checkpoint。

本实验仅验证单任务、短训练预算下的相对趋势。它不能宣称复现论文全量 `LIBERO-Plus = 51.5/39.7/66.4`，因为论文指标来自四套任务和 10,030 cases 的完整评测。

## 14. 跨机器迁移清单

换机器评测时至少携带：

```text
ST-WAM 仓库及当前代码 revision
configs/data/libero_object_task1_v3.yaml
configs/sim_libero_v3.yaml
runs/single_task/*/config.yaml
runs/single_task/*/dataset_stats.json
runs/single_task/*/checkpoints/weights/step_000500.pt
checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
evaluate_results/single_task/manifests/bbq_plus_140_seed42/
```

目标机器还必须安装与当前提交兼容的 ST-WAM 环境，并准备 `/home/pai/zxw/LIBERO-PRO` 与 `/home/pai/zxw/LIBERO-plus` 的等价副本。绝对路径不同可以替换，但 checkpoint、stats、manifest 和代码 revision 必须保持一致。

## 15. 当前结果（2026-09-05）

三种模型均使用 seed 42、500 optimizer steps、global batch 128、相同的 40/10 episode 划分和同一份归一化统计。clean 使用 50 rollouts；Plus 使用固定的 140-case manifest，每类扰动 20 cases、每 case 1 rollout。

### 15.1 总体结果

| Variant | Clean success | Plus success | Robustness drop |
| --- | ---: | ---: | ---: |
| DINO Future Only | 45/50 = 90.00% | 43/140 = 30.71% | 59.29pp |
| Fast-WAM VAE | 45/50 = 90.00% | 108/140 = 77.14% | 12.86pp |
| Dual-Space w/o CAIR | 49/50 = 98.00% | 109/140 = 77.86% | 20.14pp |

结果文件：

```text
evaluate_results/single_task/dino_clean_v3_fixed/
evaluate_results/single_task/vae_clean_v3_fixed/
evaluate_results/single_task/dual_clean_v3_fixed/
evaluate_results/single_task/dino_plus_140_seed42/
evaluate_results/single_task/vae_plus_140_seed42/
evaluate_results/single_task/dual_plus_140_seed42/
```

在当前单任务、单 seed、140-case 设置下，Dual-Space 相比 VAE 只增加 1 个成功 case，即 `+0.71pp`。VAE 的 Wilson 95% 区间为 `[69.52%, 83.32%]`，Dual-Space 为 `[70.29%, 83.94%]`，二者高度重叠。

同一批 case 的配对列联结果为：

```text
两者都成功：96
仅 Dual-Space 成功：13
仅 VAE 成功：12
两者都失败：19
配对 McNemar exact p = 1.0
```

因此当前总体结果不支持“Dual-Space 显著优于 VAE”。它只能说明在该低成本单任务实验中，两者总体成功率相当；不能据此否定论文在全任务、大规模训练下的结论。

### 15.2 按扰动类别

| Category | DINO | VAE | Dual-Space | Dual − VAE |
| --- | ---: | ---: | ---: | ---: |
| Camera Viewpoints | 45% | 65% | 80% | +15pp |
| Robot Initial States | 25% | 40% | 60% | +20pp |
| Language Instructions | 5% | 100% | 100% | 0pp |
| Light Conditions | 0% | 100% | 100% | 0pp |
| Background Textures | 75% | 85% | 45% | −40pp |
| Sensor Noise | 10% | 90% | 100% | +10pp |
| Objects Layout | 55% | 60% | 60% | 0pp |

Dual-Space 并非所有类别都没有收益：它在相机视角、机器人初态和传感器噪声上分别提高 15pp、20pp 和 10pp。但 Background Textures 从 85% 降至 45%，恰好抵消了这些收益。背景类别的配对结果为“仅 VAE 成功 8 cases、仅 Dual 成功 0 cases”，说明需要优先定位 Dual 分支对背景纹理的敏感性，而不是只看总平均值。

### 15.3 过拟合迹象

step 400 到 step 500 期间，训练 loss 继续下降，但当前 trainer 的验证 loss 同时上升：

| Variant | Train loss step 400 → 500 | Val loss step 400 → 500 |
| --- | ---: | ---: |
| DINO | 0.1836 → 0.1389 | 0.0573 → 0.0957 |
| VAE | 0.1983 → 0.1098 | 0.2107 → 0.3411 |
| Dual-Space | 0.2387 → 0.1035 | 0.2493 → 0.4030 |

这与 400→500 阶段发生过拟合相符。Dual-Space 的 clean 成功率达到 98%，但 Plus 与 VAE 基本相同，且 robustness drop 更大，也符合“对标准外观进一步专门化、分布外收益没有同步增加”的现象。

但这仍不是严格证明：当前 `Trainer.evaluate()` 每个 rank 只采少量验证样本，单次 val loss 方差较大；此外每类 Plus 只有 20 cases，当前只有一个训练 seed。结论应表述为“存在较强过拟合迹象，需完整 held-out 评测确认”。

## 16. 下一步实验方案

目标是区分三种解释：后期过拟合、DINO 辅助损失权重不足，以及单任务 VAE 已接近性能上限。整个过程继续禁止使用 Plus 结果选择 checkpoint 或超参数。

### Phase A：完整 held-out open-loop 诊断

使用固定的 10 个 validation episodes，对三个模型的 step 300、400、500 checkpoint 做确定性的完整 open-loop 评测，而不是沿用 trainer 当前的稀疏验证采样。

固定要求：

- 使用 `configs/data/libero_object_task1_v3.yaml` 的相同 episode-level split 和同一 `dataset_stats.json`；
- 三模型评测完全相同的窗口、动作 horizon 和随机种子；
- 对所有 1,559 个窗口报告固定噪声下的 total/action/future diffusion loss，并按 episode 汇总均值和标准差；
- action diffusion inference 成本更高，固定从每个 episode 均匀抽取 10 个窗口（共 100 个）报告反归一化 action L1/L2；资源充足时才扩展到所有窗口；
- DINO 报告 feature prediction loss；Dual 同时报告 VAE 与 DINO 两类 future loss；
- 保存逐窗口 JSONL 和逐 episode 汇总，避免只保留一个全局均值。当前阶段不计算生成视频 PSNR/SSIM，后续若需要单独补充，不能用 diffusion loss 冒充 reconstruction 指标。

判定规则：如果 step 400→500 的 train loss 下降，而完整 validation action/future 指标持续变差，则确认后期过拟合。该阶段需要先补一个遍历完整 validation dataloader 的离线评测脚本；现有 trainer 的单次 val 数值不作为最终依据。

### Phase B：统一 step 400 的 clean 验证

在不重训的情况下，使用现有三个 `step_000400.pt`，在 clean task 上各运行固定 50 trials。三个模型必须统一使用 step 400，不能分别挑各自最优 checkpoint。

```text
runs/single_task/dino_future_seed42_500/checkpoints/weights/step_000400.pt
runs/single_task/fastwam_vae_seed42_500/checkpoints/weights/step_000400.pt
runs/single_task/dual_space_seed42_500/checkpoints/weights/step_000400.pt
```

只有 held-out open-loop 和 clean 结果共同支持 step 400 后，才把 400 steps 定为下一轮统一训练预算。不能根据 LIBERO-Plus 成功率决定使用 step 300、400 或 500。

### Phase C：固定预算后的 Plus 确认

checkpoint 和预算在 Phase A/B 中冻结后，三个 step 400 模型复用当前完全相同的 140-case manifest 进行一次 Plus 评测。报告：

- overall paired success delta；
- McNemar exact test；
- case-level paired bootstrap 95% CI；
- 七类扰动的成功率和 Dual−VAE 差值；
- 特别检查 Background Textures 的 8 个 VAE-only cases 是否仍然失败。

该阶段只用于检验预先冻结的选择，不再调 checkpoint、loss 权重或增强策略。

### Phase D：根据诊断结果分支

1. 若 step 400 的 Dual 明显优于 step 500，且背景退化减轻：将三个模型统一改为 400-step 预算，增加 seed 43、44；每个 seed 仍使用相同数据划分规则和 manifest。
2. 若 step 400 与 step 500 都表现为“视角/初态提升、背景下降”：优先检查融合机制和背景特征依赖，不继续增加训练步数。
3. 若完整 validation 显示 Dual 的 DINO 学习不足：在固定 400 steps 下比较 `lambda_dino ∈ {0.01, 0.02, 0.05}`，仅用 held-out/clean 指标选择权重，再运行一次冻结后的 Plus 测试。当前 step 500 的加权 DINO loss 约为 0.0023，只占总 loss 的约 2.3%，存在辅助信号偏弱的可能。
4. 若各设置仍在单任务上饱和：扩展到 2–3 个预先选定的 `libero_object` tasks，或恢复整个 `libero_object` suite；不能继续用当前 BBQ task 的 Plus 结果选择新任务。

### 推荐执行顺序

```text
1. 实现并验证 full-validation open-loop 脚本
2. 评测三模型 step 300/400/500 的完整 validation 指标
3. 若证实后期过拟合，统一运行三模型 step 400 clean 50 trials
4. 冻结 step 400 后，复用原 140-case manifest 做 Plus 配对评测
5. 趋势成立后再增加 seed 43/44；否则进入融合/权重诊断
```

在完成 Phase A–C 前，不建议增加训练步数、针对 Background Plus case 添加专用增强，或只为 Dual-Space 选择更早 checkpoint。

## 17. tmux 无人值守评测脚本

`scripts/run_single_task_overfit_diagnosis.sh` 将 Phase A–C 固化为可重入流水线：

1. 对 step 300/400/500 的 DINO、VAE、Dual-Space 做 held-out 诊断；同一步的三个模型分别占用 GPU 0/1/2 并行运行。
2. 每个 checkpoint 对全部 1,559 个 validation windows 计算固定噪声/固定 seed 的 diffusion objective；每个 validation episode 均匀取 10 个窗口，共 100 个窗口做较昂贵的 action diffusion inference，并报告反归一化 action L1/L2。
3. 三个模型统一使用 step 400，顺序完成 clean 50 rollouts。
4. 三个模型继续统一使用 step 400，在同一个 140-case manifest 上完成 Plus 评测并生成 VAE/Dual 配对统计。

Phase A 的 `loss_video` 在 DINO-only 模型中表示 DINO feature-space diffusion loss，在 VAE-only 模型中表示 VAE latent diffusion loss；不同 representation 的绝对值不能直接横向比较，只比较同一 variant 的 step 300/400/500 趋势。当前离线脚本不计算生成视频的 PSNR/SSIM，不能把 diffusion loss 写成 reconstruction PSNR/SSIM。

启动前无需手工确认 GPU 是否空闲。脚本默认持续等待 GPU 0–3 同时空闲，之后才开始；已完成的 held-out、clean 和 Plus 项会自动跳过。启动命令：

```bash
cd /home/pai/zxw/ST-WAM
mkdir -p evaluate_results/single_task/overfit_diagnosis_seed42/logs

tmux new-session -d -s stwam-diagnosis \
  'cd /home/pai/zxw/ST-WAM && exec bash scripts/run_single_task_overfit_diagnosis.sh >> evaluate_results/single_task/overfit_diagnosis_seed42/logs/pipeline.log 2>&1'
```

启动后可直接断开 SSH。查看状态：

```bash
tmux ls
tmux attach -t stwam-diagnosis

tail -f /home/pai/zxw/ST-WAM/evaluate_results/single_task/overfit_diagnosis_seed42/logs/pipeline.log
cat /home/pai/zxw/ST-WAM/evaluate_results/single_task/overfit_diagnosis_seed42/status/pipeline.state
```

若 tmux 会话因机器重启消失，重新执行完全相同的 `tmux new-session` 命令即可。Phase A 从 `per_window.jsonl` 续跑，Plus 从已有的 case result JSON 续跑；clean 只有单个 50-trial 结果文件，因此若在中途终止，该模型的 clean 50 trials 会从头重跑。

主要产物：

```text
evaluate_results/single_task/overfit_diagnosis_seed42/open_loop/
evaluate_results/single_task/overfit_diagnosis_seed42/open_loop_comparison.{md,csv}
evaluate_results/single_task/{dino,vae,dual}_clean_step400_seed42/
evaluate_results/single_task/{dino,vae,dual}_plus_140_step400_seed42/
evaluate_results/single_task/overfit_diagnosis_seed42/step400_closed_loop_comparison.{md,json}
```

若只想先跑闭环 step 400、暂时跳过耗时最长的 Phase A：

```bash
bash scripts/run_single_task_overfit_diagnosis.sh --skip-open-loop
```

若只做 held-out 诊断而不启动仿真 rollout：

```bash
bash scripts/run_single_task_overfit_diagnosis.sh --skip-clean --skip-plus
```
