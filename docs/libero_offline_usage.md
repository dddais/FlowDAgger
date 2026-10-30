# FlowDAgger × LIBERO-Recovery：使用教程

这篇教程先带你跑通一个任务：准备配置 → 训练 steering → 对比 Base 和 steering → 查看成绩。随后介绍十任务并行、数据来源对比和手动评测。

整体原理和模块分工见[实现说明](libero_offline_implementation.md)。

## 跑通一个任务

下面使用任务 0“把前方黑碗放到盘子上”，用 Standard+Recovery 混合数据训练。训练完成后，会分别在 Standard 和 Recovery 场景中评测 Base 和 steering，共得到四组结果。

本机已经配好两个环境，运行脚本会自动选择：

| 用途 | Python |
|---|---|
| 训练和策略服务 | `/home/dais/workspace/FlowDAgger/.venv-pi05/bin/python` |
| LIBERO 仿真 | `/home/dais/miniconda3/envs/libero/bin/python` |

### 第一步：给实验起名

在同一个终端里依次执行本节命令。每轮实验用一个新名字，配置、模型和评测结果会放到对应目录。

```bash
cd /home/dais/workspace/FlowDAgger

EXP=libero_task0_demo
TASKS="0"
GPUS="0"
MAX_GPUS=1

RUN="$PWD/outputs/flow_recovery/$EXP"
PREFIX="$PWD/config/flow_recovery/$EXP"
CHECKPOINT="$PWD/outputs/flow_recovery/auto_20261029_2350/verified_model_download/checkpoint"
```

这里复用已下载的官方 JAX `pi05_libero`。`TASKS` 决定跑哪些任务，`GPUS` 是允许使用的物理卡号，`MAX_GPUS` 是同时使用的卡数。想用卡 3，就把 `GPUS` 改成 `"3"`。

### 第二步：生成配置和评测计划

```bash
OPENBLAS_NUM_THREADS=1 .venv-pi05/bin/python -m flowdagger_pi05.flow_recovery.prepare_run \
  --root "$RUN" --config-prefix "$PREFIX" \
  --checkpoint "$CHECKPOINT" --tasks $TASKS
```

这一步生成：

- `config/flow_recovery/libero_task0_demo_task00.json`：任务 0 的训练配置。
- `$RUN/plans/task_00/standard.json`：Standard 评测计划。
- `$RUN/plans/task_00/recovery.json`：Recovery 评测计划。
- `$RUN/libero_config/`：仿真资源路径配置。

默认计划中，Standard 每任务评测 25 次；Recovery 最多选 30 个场景，每个场景评测 10 次。Base 和 steering 会使用相同的计划。

### 第三步：检查数据

```bash
OPENBLAS_NUM_THREADS=1 .venv-pi05/bin/python -m flowdagger_pi05.flow_recovery.audit_data \
  --config "${PREFIX}.json" --output "$RUN/data_audit.json"
```

程序会检查示范的数组格式、任务文本和 state/actions 文件校验值。任务 0 有 90 条示范，正常结束时会打印 `episodes: 90, errors: 0`。

### 第四步：计算噪声标签并训练

```bash
/usr/bin/python3 -u -m flowdagger_pi05.flow_recovery.schedule \
  --root "$RUN" --config-prefix "$PREFIX" --tasks $TASKS \
  --stage offline --gpus $GPUS --max-gpus "$MAX_GPUS" \
  --min-free-mib 16000 --load-free-mib 16000
```

程序先加载 π0.5，把示范动作反演成噪声标签并保存缓存，然后训练小网络。默认训练 4000 步，π0.5 的权重保持固定。

常看的三个文件是：

| 文件 | 看什么 |
|---|---|
| `$RUN/jobs/task_00/offline/precompute.log` | 每条轨迹有多少动作片段通过反演检查 |
| `$RUN/jobs/task_00/offline/train.log` | 当前训练步数、训练误差和验证误差 |
| `$RUN/standard_recovery_seed42/task_00/latest.json` | 最新保存的模型和步数 |

`--min-free-mib 16000` 表示选择至少有约 16 GiB 空闲显存的卡。显存不足时，程序会等待；共享存储上的首次模型导入也可能比较慢。

### 第五步：评测 Base 和 steering

上一步结束后运行：

```bash
/usr/bin/python3 -u -m flowdagger_pi05.flow_recovery.schedule \
  --root "$RUN" --config-prefix "$PREFIX" --tasks $TASKS \
  --stage eval --gpus $GPUS --max-gpus "$MAX_GPUS" \
  --min-free-mib 16000
```

程序自动启动策略服务和仿真客户端，依次完成四组评测：

```text
Base      × Standard
Base      × Recovery
steering  × Standard
steering  × Recovery
```

当前默认每 10 步重新规划一次，开始前静置 20 步，最多执行 520 步策略动作。结果写在 `$RUN/eval_v2/`。

### 第六步：看成绩

```bash
OPENBLAS_NUM_THREADS=1 .venv-pi05/bin/python -m flowdagger_pi05.flow_recovery.summarize \
  --root "$RUN" --tasks $TASKS --output "$RUN/final_analysis"
```

打开 `$RUN/final_analysis/task_results.csv` 查看各任务成绩；`comparison.json` 包含 Base/steering 的差值和统计区间。

| 字段 | 含义 |
|---|---|
| `successes` | 成功完成的次数 |
| `planned` | 计划评测次数 |
| `rate_planned` | 成功次数 ÷ 计划次数，主表使用这个值 |
| `initial_success` | 开始执行策略前，场景就已经满足目标的次数 |
| `reset_errors` | 环境初始化失败的次数 |
| `rate_valid` | 只在实际执行了策略的试验中计算的成功率 |

多个任务的总体成绩按任务等权平均。Standard 表示当前任务的普通初态；这批任务中有 8 个来自 LIBERO-90，其余 2 个来自 Goal/Object。

运行过程中可以另开终端查进度：

```bash
cd /home/dais/workspace/FlowDAgger
/usr/bin/python3 -m flowdagger_pi05.flow_recovery.status \
  --root outputs/flow_recovery/libero_task0_demo
```

## 跑全部十个任务

把第一步中的四个变量替换为下面的值，再按同样的六步运行：

```bash
EXP=libero_all_demo
TASKS="0 1 2 3 4 5 6 7 8 9"
GPUS="0 1 2 3 4 5 6 7"
MAX_GPUS=6
```

随后重新执行第一步里的 RUN、PREFIX、CHECKPOINT 赋值。调度器从可用 GPU 中选择空闲卡，最多同时运行六个任务，每个任务使用一张卡。

## 调整训练参数

第二步生成配置后，编辑 `${PREFIX}_task00.json`，再开始训练。多任务实验中，每个任务都有自己的配置文件。

| 参数 | 默认值 | 怎么理解 |
|---|---:|---|
| `train_steps` | 4000 | 小网络更新多少次 |
| `batch_size` | 256 | 每次更新使用多少个动作片段 |
| `learning_rate` | 0.0001 | 小网络学习率 |
| `recovery_fraction` | 0.5 | 混合训练中 Recovery 样本所占比例 |
| `inversion_batch_size` | 4 | 一次反演多少个动作片段；减小它能降低反演显存占用 |
| `max_normalized_mse` | 0.001 | 反演标签允许的动作重建误差 |
| `validation_fraction` | 0.2 | 留作验证的轨迹比例 |

参数修改好后，按第四步继续运行。批量准备默认使用训练 seed=42，实际每个任务用 42+任务编号；数据划分和评测计划的 seed 固定为 42。

## 只用 Standard 或只用 Recovery 训练

完成第四步后，已有缓存同时包含两类数据，可以直接训练不同来源的小网络。下面仍以任务 0 为例，在原来的 FlowDAgger 终端运行：

```bash
CFG="${PREFIX}_task00.json"

# 只使用 Standard 示范
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-pi05/bin/python -m flowdagger_pi05.libero_offline.train \
  --config "$CFG" --tasks 0 --sources standard \
  --output-dir "$RUN/standard_only_seed42"

# 只使用 Recovery 示范
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-pi05/bin/python -m flowdagger_pi05.libero_offline.train \
  --config "$CFG" --tasks 0 --sources recovery \
  --output-dir "$RUN/recovery_only_seed42"
```

现在有三组模型：

| 训练数据 | 模型目录 |
|---|---|
| Standard+Recovery | `$RUN/standard_recovery_seed42/task_00/` |
| Standard-only | `$RUN/standard_only_seed42/task_00/` |
| Recovery-only | `$RUN/recovery_only_seed42/task_00/` |

要评测后两组模型，使用下面的手动服务方式，指定对应的模型目录。

## 手动启动服务和评测

这里演示评测刚训练好的 Standard-only 模型，需要两个终端。

### 终端 A：启动模型服务

```bash
cd /home/dais/workspace/FlowDAgger
EXP=libero_task0_demo
RUN="$PWD/outputs/flow_recovery/$EXP"
PREFIX="$PWD/config/flow_recovery/$EXP"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
OPENPI_DATA_HOME="$RUN/openpi_cache" JAX_COMPILATION_CACHE_DIR="$RUN/jax_cache" \
.venv-pi05/bin/python -m flowdagger_pi05.libero_offline.serve \
  --config "${PREFIX}_task00.json" --tasks 0 --mode steering \
  --run-root "$RUN/standard_only_seed42" \
  --port 18010 --disable-websocket-pings
```

服务启动后保持这个终端运行。评测 Recovery-only 时，将 `--run-root` 改为 `$RUN/recovery_only_seed42`。评测 Base 时使用 `--mode base`，省略 `--run-root`。

### 终端 B：运行仿真

```bash
cd /home/dais/workspace/LIBERO-Recovery
FLOW_ROOT=/home/dais/workspace/FlowDAgger
RUN="$FLOW_ROOT/outputs/flow_recovery/libero_task0_demo"

export PYTHONPATH="$PWD:$PWD/LIBERO:$FLOW_ROOT/flowdagger_pi05/openpi/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$RUN/libero_config"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NO_PROXY=localhost,127.0.0.1

/home/dais/miniconda3/envs/libero/bin/python -m examples.flowdagger.evaluate \
  --plan "$RUN/plans/task_00/recovery.json" \
  --host 127.0.0.1 --port 18010 --render-gpu-device-id 0 \
  --disable-websocket-pings \
  --output "$RUN/manual/standard_only/task_00/recovery" --video
```

这会在 Recovery 场景中运行 Standard-only 模型，并保存视频和结果。评测 Standard 场景时，把 plan 文件换成 `standard.json`，输出目录最后一级也改成 `standard`。切换模型时，相应地更换输出目录中的 `standard_only`。

模型服务和仿真渲染使用同一张物理卡。例如使用卡 3 时，两边的 `CUDA_VISIBLE_DEVICES`、客户端的 `MUJOCO_EGL_DEVICE_ID` 和 `--render-gpu-device-id` 都设为 3。

## 修改评测设置

评测次数、场景选择和动作执行设置保存在 plan 文件中。修改这些设置时，先生成一份新计划，再用它评测各个模型。

在上面的仿真终端执行。例如，选择任务 0 的 5 个 Recovery 场景，每个场景测 4 次：

```bash
/home/dais/miniconda3/envs/libero/bin/python -m examples.flowdagger.evaluate \
  --tasks 0 --mode recovery --num-scenes 5 --close-trials 2 --open-trials 2 \
  --query-freq 5 --horizon 400 --settle-steps 10 \
  --scene-compatibility --state-restore-reset \
  --output "$RUN/plans/recovery_small" --dry-run
```

`--dry-run` 只生成计划。随后在评测命令中使用 `--plan "$RUN/plans/recovery_small/plan.json"`，执行的就是这份计划中的设置。

| 参数 | 含义 |
|---|---|
| `--num-scenes` | 每个任务选多少个 Recovery 场景 |
| `--close-trials` / `--open-trials` | 每个场景分别以闭合/打开夹爪开始的试验次数 |
| `--trials` | Standard 评测使用多少个初态 |
| `--query-freq` | 每执行多少步动作重新请求策略，范围为 1–10 |
| `--horizon` | 每次试验最多执行多少步策略动作 |
| `--settle-steps` | 恢复场景后先静置多少步 |
| `--noise-xy` | Recovery 物体平面位置的扰动幅度，默认 0.02 米 |

`--scene-compatibility` 和 `--state-restore-reset` 用于加载部分导出的 Recovery 场景；批量运行已经自动启用。`--disable-websocket-pings` 用来容纳首次编译或渲染时较长的等待，手动运行时在服务端和客户端都加上。

## 中断后继续跑

在原来的 FlowDAgger 终端中，给重试日志一个新名字即可继续任务 0 的离线阶段：

```bash
/usr/bin/python3 -u -m flowdagger_pi05.flow_recovery.schedule \
  --root "$RUN" --config-prefix "$PREFIX" --stage offline --tasks 0 \
  --gpus $GPUS --max-gpus 1 --schedule-label offline_retry01
```

程序会复用已完成的反演缓存，并从最近保存的训练状态继续。续跑沿用原配置。

评测重试使用新的结果目录：

```bash
/usr/bin/python3 -u -m flowdagger_pi05.flow_recovery.schedule \
  --root "$RUN" --config-prefix "$PREFIX" --stage eval --tasks 0 \
  --gpus $GPUS --max-gpus 1 \
  --schedule-label eval_retry01 --evaluation-dir eval_retry01

OPENBLAS_NUM_THREADS=1 .venv-pi05/bin/python -m flowdagger_pi05.flow_recovery.summarize \
  --root "$RUN" --tasks 0 --evaluation-dir eval_retry01 \
  --output "$RUN/analysis_retry01"
```

## 结果文件在哪里

| 路径（相对于 RUN） | 内容 |
|---|---|
| `cache/task_00/` | 反演得到的特征和噪声标签 |
| `standard_recovery_seed42/task_00/` | 默认混合训练的小模型、训练配置和误差记录 |
| `jobs/task_00/` | 预计算、训练、服务和仿真的详细日志 |
| `eval_v2/base/task_00/` | Base 在两种场景下的结果 |
| `eval_v2/steering/task_00/` | steering 在两种场景下的结果 |
| `final_analysis/task_results.csv` | 方便查看和制表的成绩明细 |
| `final_analysis/comparison.json` | 总体成绩、提升幅度和统计区间 |

每组评测目录都有 `trials.jsonl` 和 `summary.json`：前者是一条条试验记录，后者是这组试验的汇总。
