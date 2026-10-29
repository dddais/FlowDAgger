"""Additive 12-task registry and execution harness for the official entry point.

All remaining arguments pass through to train_flowdagger.py unchanged.
No original source files or existing results are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'flowdagger_pi05'
sys.path[:0] = [str(ROOT), str(BACKEND), str(BACKEND / 'openpi/src')]


def register_tasks(config):
    from shared.task_configs import TASK_CONFIGS
    from shared.experts.metaworld_expert import _POLICY_BY_ENV
    template = dict(TASK_CONFIGS['metaworld_assembly'])
    for task in config['tasks']:
        TASK_CONFIGS[task['task_key']] = dict(
            template, name=task['env_id'].removesuffix('-v3'),
            env_id=task['env_id'], prompt=task['prompt'])
        _POLICY_BY_ENV[task['env_id']] = task['expert_policy']


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repro_config', default=str(ROOT / 'config/reproduction_20261028_auto/main_table.json'))
    parser.add_argument('--repro_mode', choices=['train', 'base'], default='train')
    parser.add_argument('--repro_run_dir', required=True)
    parser.add_argument('--repro_filter_invalid_inversions', action='store_true')
    parser.add_argument('--repro_rollout_budget', type=int, default=0)
    args, remaining = parser.parse_known_args()
    run_dir = Path(args.repro_run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    os.environ['EXP'] = str(run_dir)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('METAWORLD_CHECKPOINT', str(Path('/home/dais/workspace/model/pi05-metaworld')))
    config = json.loads(Path(args.repro_config).read_text())
    register_tasks(config)
    source_files = list(BACKEND.glob('*.py')) + list((ROOT / 'shared').rglob('*.py'))
    manifest = dict(mode=args.repro_mode, filter_invalid_inversions=args.repro_filter_invalid_inversions,
                    rollout_budget=args.repro_rollout_budget, argv=remaining, started=time.time(),
                    config=config, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'METAWORLD_CHECKPOINT']})
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    # Preserve scalar diagnostics locally even though external logging is off.
    import wandb_logger
    original_log = wandb_logger.WandBLogger.log
    def local_log(self, data, step=None):
        scalars = {}
        for key, value in data.items():
            try:
                scalars[key] = float(value)
            except (TypeError, ValueError):
                pass
        if scalars:
            with (run_dir / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), step=step, metrics=scalars)) + '\n')
        return original_log(self, data, step=step)
    wandb_logger.WandBLogger.log = local_log
    import dagger_loop
    original_collect = dagger_loop.collect_traj
    rollout_count = 0
    class RolloutBudgetReached(Exception):
        pass
    def recorded_collect(*call_args, **call_kwargs):
        nonlocal rollout_count
        if args.repro_rollout_budget and rollout_count >= args.repro_rollout_budget:
            raise RolloutBudgetReached()
        start = time.time()
        traj = original_collect(*call_args, **call_kwargs)
        rollout_count += 1
        keys = ['is_success', 'env_steps', 'initial_metaworld_seed',
                'inversion_7d_mse', 'inversion_w_mean_abs',
                'inversion_w_max_abs', 'inversion_w_p99']
        row = {k: float(traj[k]) if traj.get(k) is not None else None for k in keys}
        row.update(rollout=rollout_count, seconds=time.time()-start,
                   interventions=sum(traj.get('intervention_flags', [])))
        with (run_dir / 'rollouts.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        return traj
    dagger_loop.collect_traj = recorded_collect
    if args.repro_filter_invalid_inversions:
        from reproduction_filters import filtered_add
        original_add = dagger_loop.add_online_data_to_buffer
        def corrected_add(variant, traj, online_replay_buffer, expert_buffer=None,
                          autonomous_buffer=None):
            counts = filtered_add(original_add, variant, traj, online_replay_buffer,
                                  expert_buffer, autonomous_buffer)
            with (run_dir / 'insertion_filter.jsonl').open('a') as f:
                f.write(json.dumps(dict(rollout=rollout_count, **counts)) + '\n')
            print(f'[Reproduction filter] skipped {counts["skipped"]}/{counts["index"]} rejected inversion transitions')
        dagger_loop.add_online_data_to_buffer = corrected_add
    if args.repro_rollout_budget:
        original_loop = dagger_loop.flowdagger_training_loop
        def budgeted_loop(variant, steering_policy, env, eval_env, replay_buffer,
                          wandb_logger, shard_fn=None, agent_dp=None,
                          intervention_handler=None, init_states=None,
                          raw_model=None, autonomous_buffer=None):
            try:
                original_loop(variant, steering_policy, env, eval_env, replay_buffer,
                              wandb_logger, shard_fn, agent_dp, intervention_handler,
                              init_states, raw_model, autonomous_buffer)
            except RolloutBudgetReached:
                print(f'[Reproduction budget] stopped before rollout {rollout_count+1}')
            step = int(steering_policy._actor.step)
            if rollout_count != args.repro_rollout_budget or step <= 0:
                raise RuntimeError(f'Incomplete budgeted training: rollouts={rollout_count}, updates={step}')
            from train_utils import perform_control_eval
            eval_path = Path(variant.outputdir) / 'eval_results.jsonl'
            ev = [json.loads(line) for line in eval_path.read_text().splitlines()] if eval_path.exists() else []
            if not any(r['step'] == step for r in ev):
                perform_control_eval(steering_policy, eval_env, step, variant, wandb_logger,
                                     agent_dp, init_states=init_states, raw_model=raw_model)
            if not (Path(variant.outputdir) / f'checkpoint{step}').exists():
                steering_policy.save_checkpoint(variant.outputdir, step, step)
            (run_dir / 'final_training_state.json').write_text(json.dumps({
                'rollouts':rollout_count, 'final_step':step,
                'checkpoint':str(Path(variant.outputdir) / f'checkpoint{step}')}))
        dagger_loop.flowdagger_training_loop = budgeted_loop
    if args.repro_mode == 'base':
        def base_loop(variant, steering_policy, env, eval_env, replay_buffer,
                      wandb_logger, shard_fn=None, agent_dp=None,
                      intervention_handler=None, init_states=None,
                      raw_model=None, autonomous_buffer=None):
            from train_utils import perform_control_eval
            perform_control_eval(steering_policy, eval_env, 0, variant, wandb_logger,
                                 agent_dp, init_states=init_states, raw_model=raw_model)
        dagger_loop.flowdagger_training_loop = base_loop
    sys.argv = [str(BACKEND / 'train_flowdagger.py'), *remaining]
    try:
        runpy.run_path(str(BACKEND / 'train_flowdagger.py'), run_name='__main__')
    except SystemExit as exc:
        # The official entry point ends with sys.exit() after successful work.
        if exc.code not in (None, 0):
            raise
    (run_dir / 'completed.json').write_text(json.dumps({'finished': time.time()}))


if __name__ == '__main__':
    main()
