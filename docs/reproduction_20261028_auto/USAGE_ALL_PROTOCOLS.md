# 归档代码使用说明

从 `/home/dais/workspace/FlowDAgger` 执行，使用现有 `.venv-pi05`。本归档仅保留代码、配置和结果总结：原实验模型、日志、图表、29个反演输入及50段诊断数据已删除。以下命令用于重新运行，均需全新输出目录；本次归档没有启动实验。

## 准备

基座模型默认路径为 `/home/dais/workspace/model/pi05-metaworld`，必须包含JAX params和assets。该仓库外模型及现有运行环境未在此次清理范围内；运行前自行确认可用。源码与数据版本见[方法说明](METHODS_AND_AUDIT.md)。GPU最多同时6张，EGL绑定物理卡。

## 三套主表

原默认：

```bash
.venv-pi05/bin/python -u flowdagger_pi05/reproduction_queue.py \
  --output outputs/my_original_main_01 \
  --gpus 0,1,2,3,5,7 --seeds 42,43,44 --min_free_mib 11000
```

双缓冲与过滤：

```bash
.venv-pi05/bin/python -u flowdagger_pi05/reproduction_queue.py \
  --output outputs/my_dual_main_01 \
  --gpus 0,1,2,3,5,7 --seeds 42,43,44 --min_free_mib 11000 \
  --filter_invalid_inversions --dual_buffer 1 \
  --rollout_budget 50 --eval_interval 4000
```

成功纯自主协议：

```bash
.venv-pi05/bin/python -u flowdagger_pi05/reproduction_queue.py \
  --output outputs/my_pure_main_01 \
  --gpus 0,1,2,3,5,7 --seeds 42,43,44 --min_free_mib 11000 \
  --entrypoint reproduction_pure_autonomous.py \
  --filter_invalid_inversions --dual_buffer 1 \
  --rollout_budget 50 --eval_interval 4000
```

纯自主协议10专家warmup＋20纯自主＋20纠正，实际更新可能不足4000。所有方案以完整任务/种子及实际预算终点为准。仅过滤控制可在原默认命令加 `--filter_invalid_inversions --tasks metaworld_box_close,metaworld_door_lock`。

新运行完成后生成主表：

```bash
.venv-pi05/bin/python flowdagger_pi05/reproduction_report.py \
  --runs outputs/my_pure_main_01 --output docs/my_pure_table_01
```

每个运行的complete和budget_matches均应为真。queue_completed只表示队列终止；最终步数看final_training_state.json/per_seed.csv。归档汇总中的complete是历史完成状态，不能用其证明当前模型仍存在。

## 单任务与重新评估

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-pi05/bin/python -u flowdagger_pi05/reproduce_main_table.py \
  --repro_run_dir outputs/my_assembly_s42_new \
  --task_key metaworld_assembly --seed 42 \
  --max_steps 3999 --seed_expert_episodes 10 \
  --eval_episodes 25 --eval_interval 500 \
  --checkpoint_interval 4000 --save_eval_video 0
```

仅Base增加 `--repro_mode base --use_interventions 0`；评估新训练的steering模型增加 `--eval_only_ckpt <包含checkpoint文件的父目录>`，任务、结构和seed一致。旧114个checkpoint已删除，无法直接恢复旧模型。

## 补充反演与迁移

先重新采集输入：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
.venv-pi05/bin/python flowdagger_pi05/reproduction_inversion_benchmark.py collect \
  --output outputs/my_chunks_new --seed 90042 --chunks 29 --chunk_steps 10
```

完成一套新的双缓冲主表后，使用新模型和新输入：

```bash
.venv-pi05/bin/python -u flowdagger_pi05/reproduction_supplementary_queue.py \
  --wait_queue outputs/my_dual_main_01 \
  --output outputs/my_supplementary_01 --docs docs/my_supplementary_01 \
  --chunks outputs/my_chunks_new \
  --gpus 0,1,2,3,5,7 --adam_min_free_mib 12000
```

单独反演可用 `reproduction_inversion_benchmark.py benchmark --chunks_dir outputs/my_chunks_new --method perstep5 --output outputs/my_inversion_new`。暂定迁移任务映射未经作者确认；新采集的输入应重新计算哈希，不能假定与已删除输入逐bit一致。

## 独立Base诊断

噪声诊断可用主队列指定 `--entrypoint reproduction_base_noise_probe.py --tasks metaworld_stick_push,metaworld_assembly`，另选全新输出目录，再使用reproduction_base_noise_report.py汇总。42/43/44只是评估分组。

速度诊断需要先重新准备固定revision的50段stick-push训练数据和索引，再运行reproduction_dataset_velocity_audit.py。该脚本要求samples目录包含stick_push_all_50/audit_v2.json，索引episodes含50条episode/path/sha256，路径对应真实parquet。归档不含这些输入，也不附下载器。生成新的标定报告后，在单任务入口reproduction_reset_velocity_probe.py显式传 `--velocity_audit <新的审计results.json>`、task_key=metaworld_stick_push及seed等参数。队列不转发该路径，速度诊断请用单任务入口。原始样本和历史报告不能用汇总数值伪造。

## 从归档结果重画比较图

```bash
.venv-pi05/bin/python flowdagger_pi05/reproduction_protocol_comparison_plot.py \
  --results docs/reproduction_20261028_auto/results.json \
  --output outputs/my_protocol_comparison_new
```

绘图仅使用保存的任务均值和种子终点，不需旧checkpoint。已有[结果总结](RESULTS.md)和[汇总JSON](results.json)可直接阅读。
