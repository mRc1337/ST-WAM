# ST-WAM 三任务 LIBERO-Plus 消融实验报告

**报告日期：** 2026-09-07  
**实验状态：** 训练与评测全部完成  
**目标论文：** arXiv:2607.28993  
**实验范围：** LIBERO-Object 三任务低成本趋势验证

## 1. 摘要

本实验比较三种 future prediction 表征：DINO、VAE，以及不使用 CAIR 的 VAE+DINO Dual-Space。三个模型采用相同训练数据、随机种子、优化步数和全局 batch size，并在完全相同的 745 个 LIBERO-Plus case 上进行配对评测。

主要结果如下：

- DINO：LIBERO-Plus 成功率 **60.13%**；
- VAE：LIBERO-Plus 成功率 **82.15%**；
- VAE+DINO：LIBERO-Plus 成功率 **87.79%**；
- Dual 相对 VAE 提升 **5.64 个百分点**，case-level 配对 bootstrap 95% CI 为 **[2.95, 8.46]pp**，McNemar 精确检验 **p=8.98×10⁻⁵**；
- Dual 的主要优势集中于 Camera Viewpoints、Background Textures 和 Sensor Noise；在 Robot Initial States 与 Objects Layout 上未优于 VAE。

因此，当前实验支持“VAE+DINO 比单独使用一种 future representation 更稳健”的趋势结论。由于只覆盖三个基础任务、一个训练 seed，且每个 Plus case 只运行一次，本结果不能替代论文全量复现。

## 2. 实验目的

论文报告的 LIBERO-Plus 消融结果为：

| Variant | Future prediction | Intent conditioning | LIBERO | LIBERO-Plus |
| --- | --- | --- | ---: | ---: |
| Fast-WAM | VAE | None | 97.6 | 51.5 |
| DINO Future Only | DINO | None | 96.3 | 39.7 |
| Dual-Space w/o CAIR | VAE+DINO | None | 97.8 | 66.4 |

本实验不追求完整复现上述绝对数值，而是在可控计算预算下回答以下问题：

1. VAE+DINO 是否比 VAE-only 和 DINO-only 获得更高的 LIBERO-Plus 成功率；
2. 提升主要来自哪类分布变化；
3. 单任务实验中 Dual 与 VAE 差异不明显，是否与数据多样性不足或任务专门化有关。

## 3. 实验设置

### 3.1 数据与任务

训练数据来自 LeRobot v3：

```text
/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/libero/libero_object
```

固定使用以下三个 LIBERO-Object 任务，任务在评测之前确定：

| LeRobot task index | Clean LIBERO task id | 任务 | Episodes | Plus cases |
| ---: | ---: | --- | ---: | ---: |
| 1 | 3 | pick up the bbq sauce and place it in the basket | 50 | 292 |
| 4 | 1 | pick up the cream cheese and place it in the basket | 50 | 173 |
| 7 | 9 | pick up the orange juice and place it in the basket | 50 | 280 |
| **合计** | — | — | **150** | **745** |

每个任务按 episode 进行确定性 80/20 划分，共约 120 个训练 episodes 和 30 个验证 episodes。三个模型复用 DINO 训练生成的同一份 `dataset_stats.json`，避免归一化统计差异。

### 3.2 训练配置

| 配置 | DINO | VAE | VAE+DINO |
| --- | ---: | ---: | ---: |
| Optimizer steps | 1,500 | 1,500 | 1,500 |
| GPUs | 4×A800 80GB | 4×A800 80GB | 4×A800 80GB |
| Per-GPU batch | 4 | 8 | 2 |
| Gradient accumulation | 8 | 4 | 16 |
| Global batch | 128 | 128 | 128 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Seed | 42 | 42 | 42 |
| Loss weights | DINO:Action = 1:1 | VAE:Action = 1:1 | VAE:DINO:Action = 1:0.02:1 |
| Intent conditioning / CAIR | None | None | None |

三种模型串行训练，checkpoint 位于：

```text
/mnt/data/embodied_datasets/ST-WAM/checkpoints/three_task/
```

### 3.3 评测协议

- Clean：每个基础任务运行 50 次 rollout，每个模型共 150 次；
- LIBERO-Plus：三个任务的全部 745 个 case，每个 case 运行一次；
- 三个模型使用相同 case、任务定义、归一化统计和 seed；
- Plus case 按四张 GPU 分片并行执行；
- 使用 LeRobot v3 图像及夹爪动作约定：图像不旋转，夹爪为 LIBERO 原生 `-1=open, +1=close`；
- Plus 分类数量为 Camera 93、Robot 124、Language 103、Light 87、Background 92、Noise 126、Layout 120。

## 4. 完整性检查

评测状态为 `COMPLETE`。每个模型均生成：

- 3/3 个 Clean 任务结果；
- 745/745 个 LIBERO-Plus case 结果；
- Clean、Plus 和七类扰动汇总文件。

结果目录中 DINO Clean 的 `worker_gpu0_failed_tasks.jsonl` 是首次运行时 EGL vendor 未注册留下的历史记录。修复 EGL 后该任务及全部 case 已成功重跑，最终结果文件完整，因此该历史记录不影响统计。

## 5. 总体结果

| Variant | Clean | Plus successes | LIBERO-Plus | Plus Wilson 95% CI | Robustness drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINO | 138/150 = 92.00% | 448/745 | 60.13% | [56.58%, 63.59%] | 31.87pp |
| VAE | 150/150 = 100.00% | 612/745 | 82.15% | [79.23%, 84.73%] | 17.85pp |
| **VAE+DINO** | **149/150 = 99.33%** | **654/745** | **87.79%** | **[85.24%, 89.94%]** | **11.54pp** |

Dual 在 Clean 上与 VAE 基本持平，但在 Plus 上多成功 42 个 case，并将相对 Clean 的鲁棒性下降从 VAE 的 17.85pp 降低到 11.54pp。

## 6. 配对统计检验

由于三个模型评测的是同一组 745 个 case，可直接比较每个 case 的成败。下表的置信区间来自 50,000 次 case-level paired bootstrap；p 值来自双侧 McNemar 精确检验。

| 对比 | 成功率差 | 95% bootstrap CI | 前者独占成功 | 后者独占成功 | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dual − VAE | **+5.64pp** | **[+2.95, +8.46]pp** | 77 | 35 | **8.98×10⁻⁵** |
| Dual − DINO | **+27.65pp** | **[+24.16, +31.14]pp** | 220 | 14 | **8.80×10⁻⁴⁹** |
| VAE − DINO | **+22.01pp** | **[+17.99, +26.17]pp** | 221 | 57 | **5.80×10⁻²⁴** |

在当前 case 集合上，Dual 相对 VAE 的总体提升有明确统计证据。但 745 个 case 聚类于三个基础任务，不能等同于 745 个完全独立任务；case-level p 值会高估结论向新任务推广时的确定性。

## 7. 分任务结果

### 7.1 Clean

| 任务 | DINO | VAE | VAE+DINO |
| --- | ---: | ---: | ---: |
| BBQ sauce | 38/50 = 76.00% | 50/50 = 100.00% | 50/50 = 100.00% |
| Cream cheese | 50/50 = 100.00% | 50/50 = 100.00% | 50/50 = 100.00% |
| Orange juice | 50/50 = 100.00% | 50/50 = 100.00% | 49/50 = 98.00% |

DINO 在 BBQ sauce 的 Clean 成功率只有 76%，说明该模型在此任务上首先存在基础策略拟合不足；其较低的 Plus 成功率不能全部解释为分布外泛化失败。

### 7.2 LIBERO-Plus

| 任务 | Cases | DINO | VAE | VAE+DINO | Dual−VAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| BBQ sauce | 292 | 28.42% | 86.30% | **89.38%** | +3.08pp |
| Cream cheese | 173 | 75.14% | 76.88% | **87.86%** | +10.98pp |
| Orange juice | 280 | 83.93% | 81.07% | **86.07%** | +5.00pp |

Dual 的三个任务成功率集中于 86.07%–89.38%，任务间波动明显小于 DINO 和 VAE。逐任务的 Dual−VAE 配对结果为：

| 任务 | 差值 | 95% bootstrap CI | McNemar p | 判断 |
| --- | ---: | ---: | ---: | --- |
| BBQ sauce | +3.08pp | [-1.03, +7.19]pp | 0.188 | 单任务证据不足 |
| Cream cheese | +10.98pp | [+5.20, +16.76]pp | 5.46×10⁻⁴ | 明确提升 |
| Orange juice | +5.00pp | [+0.36, +9.64]pp | 0.054 | 边界性证据 |

## 8. 按扰动类别分析

| LIBERO-Plus 类别 | Cases | DINO | VAE | VAE+DINO | Dual−VAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Camera Viewpoints | 93 | 65.59% | 67.74% | **97.85%** | **+30.11pp** |
| Robot Initial States | 124 | 33.87% | **75.00%** | 66.13% | **−8.87pp** |
| Language Instructions | 103 | 81.55% | 98.06% | **100.00%** | +1.94pp |
| Light Conditions | 87 | 58.62% | 98.85% | **100.00%** | +1.15pp |
| Background Textures | 92 | 57.61% | 77.17% | **91.30%** | **+14.13pp** |
| Sensor Noise | 126 | 72.22% | 88.89% | **100.00%** | **+11.11pp** |
| Objects Layout | 120 | 55.00% | **71.67%** | 67.50% | **−4.17pp** |

### 8.1 Dual 的主要收益

Dual 对 Camera、Background 和 Noise 的提升最大。这些类别主要改变视觉外观、观察视角或像素质量，说明 DINO future feature 与 VAE/RGB latent 结合后，能够减少策略对训练图像表面的依赖。

### 8.2 尚未解决的问题

Dual 在 Robot Initial States 和 Objects Layout 上低于 VAE。这两类变化更直接地改变末端执行器初始位姿、物体几何关系和动作轨迹，表明加入 DINO 特征并不会自动解决控制状态分布变化。后续应优先检查动作条件、状态编码、数据覆盖和空间增强，而不是仅增加视觉表征损失。

Language 与 Light 类别接近饱和，已不适合用于区分 VAE 和 Dual。

## 9. 与论文结果对比

| Variant | 论文 LIBERO-Plus | 当前三任务 | 当前−论文 |
| --- | ---: | ---: | ---: |
| DINO | 39.7% | 60.13% | +20.43pp |
| VAE | 51.5% | 82.15% | +30.65pp |
| VAE+DINO | 66.4% | 87.79% | +21.39pp |

模型排序与论文一致：`VAE+DINO > VAE > DINO`。当前 Dual−DINO 为 +27.65pp，与论文的 +26.7pp 接近；但当前 Dual−VAE 只有 +5.64pp，低于论文的 +14.9pp。

当前绝对成功率明显更高，主要原因是：

1. 仅针对三个 LIBERO-Object 任务训练，而不是学习更宽的任务分布；
2. 评测只覆盖这三个训练任务的 Plus 变体；
3. 当前 DINO-only 使用统一预算配置，并非发布 artifact 中更长训练、不同学习率和损失权重的严格配置；
4. 当前只有一个训练 seed。

因此，本实验验证的是模型排序和局部趋势，不能将当前绝对数值作为论文指标复现值。

## 10. 与先前单任务实验的关系

先前 BBQ sauce 单任务、500-step 结果为 DINO 30.71%、VAE 77.14%、Dual 77.86%，Dual−VAE 只有 +0.72pp。本次三任务、1,500-step 实验中：

- BBQ sauce：Dual−VAE 扩大到 +3.08pp；
- 三任务总体：Dual−VAE 为 +5.64pp；
- Dual 在三个任务上的 Plus 成功率更加一致。

这说明增加训练任务多样性和训练预算后，Dual 的优势更容易显现。但任务数和训练步数同时发生变化，无法仅凭两次实验判断提升究竟来自更多任务还是更长训练；若需要因果区分，应补充三任务 500-step 或单任务 1,500-step 对照。

## 11. 训练指标解释

Step 1,500 的最后一次验证记录为：

| Variant | Validation loss | 其他记录 |
| --- | ---: | --- |
| DINO | 0.1647 | action L2 0.0184，action L1 0.0652 |
| VAE | 0.1315 | inference PSNR 25.2408，SSIM 0.8223 |
| VAE+DINO | 0.4361 | action L2 0.0452，action L1 0.0801 |

不同模型的 total loss 由不同目标组成，数值尺度不可横向比较。Dual 的验证 loss 和 action L1/L2 高于 DINO，但闭环成功率明显更高，进一步说明 fixed-noise validation loss 不能作为唯一 checkpoint 选择依据。

## 12. 关于过拟合的判断

VAE 达到 100% Clean，但在 Plus 上仍有 82.15%，不能据此认定存在严重过拟合。更准确的判断是：

- VAE 存在一定训练分布专门化，尤其在 Camera 和 Background 上弱于 Dual；
- Dual 的 robustness drop 更小，且任务间表现更稳定，说明联合表征具有更好的视觉分布外泛化；
- DINO 的 BBQ sauce Clean 仅 76%，属于明显的基础任务未充分拟合，而不是典型的“Clean 很高、Plus 很低”过拟合模式。

## 13. 局限性

1. 只有三个基础任务，且都属于“拾取物体并放入篮子”的相同技能族；
2. 只有 seed 42，没有训练 seed 方差；
3. 每个 Plus case 只有一次 rollout，无法估计环境随机性下的重复试验方差；
4. case-level 显著性不能替代跨任务、跨 seed 的显著性；
5. DINO-only 超参数与论文发布 artifact 不完全一致；
6. 当前结果不覆盖 LIBERO-Spatial、Goal、Long 等套件。

## 14. 结论与建议

本次低成本消融达到预定目标：在相同数据和预算下，VAE+DINO 的 LIBERO-Plus 成功率显著高于 VAE-only 和 DINO-only，且优势主要来自视觉分布变化下的鲁棒性。

建议按以下优先级继续：

1. 使用另一个固定三任务集合和 seed 43 重复实验，检验 Dual−VAE 是否可重复；
2. 针对 Robot Initial States 与 Objects Layout 增加状态/几何相关数据增强或训练覆盖；
3. 若目标是严格复现论文 DINO baseline，按官方 DINO artifact 的学习率、损失权重和训练长度单独重训；
4. 资源允许时扩展至完整 LIBERO-Object 10 任务，再决定是否扩展到四个 suite。

## 15. 结果与配置位置

```text
训练配置：configs/data/libero_object_tasks_1_4_7_v3.yaml
训练脚本：scripts/train_three_task_ablation.sh
评测脚本：scripts/eval_three_task_ablation.sh
训练输出：runs/three_task/
评测输出：evaluate_results/three_task/step1500_seed42/
总体汇总：evaluate_results/three_task/step1500_seed42/comparison.md
本报告：docs/THREE_TASK_LIBERO_PLUS_RESULTS_REPORT.md
```
