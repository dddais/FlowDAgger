"""Bounded, append-only local experiment queue (no automatic failed-run retry)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', default='6,5,7,2,4,0')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--tasks', default='')
    p.add_argument('--timeout_hours', type=float, default=8)
    p.add_argument('--filter_invalid_inversions', action='store_true')
    p.add_argument('--dual_buffer', type=int, default=0)
    p.add_argument('--rollout_budget', type=int, default=0)
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--entrypoint', choices=['reproduce_main_table.py', 'reproduction_pure_autonomous.py',
                                          'reproduction_base_noise_probe.py', 'reproduction_reset_velocity_probe.py'],
                   default='reproduce_main_table.py')
    p.add_argument('--wait_for', default='', help='Comma-separated queues that must be terminal before GPU launch')
    p.add_argument('--first_job_gate', action='store_true')
    p.add_argument('--min_free_mib', type=int, default=0)
    args = p.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    assert len(gpus) <= 6 and len(set(gpus)) == len(gpus)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / 'config/reproduction_20261028_auto/main_table.json').read_text())
    tasks = [x['task_key'] for x in cfg['tasks']]
    if args.tasks:
        tasks = args.tasks.split(',')
        assert set(tasks) <= {x['task_key'] for x in cfg['tasks']}
    pending = [(task, int(seed)) for seed in args.seeds.split(',') for task in tasks]
    active = {}
    with (output / 'events.jsonl').open('x') as events:
        def record(**row):
            row['time'] = time.time()
            events.write(json.dumps(row) + '\n'); events.flush()
            print(json.dumps(row), flush=True)
        dependencies = [Path(x).resolve() for x in args.wait_for.split(',') if x]
        wait_start = time.monotonic()
        if dependencies:
            record(event='waiting_dependencies', paths=[str(x) for x in dependencies])
        while not all((d / 'queue_completed.json').exists() for d in dependencies):
            if time.monotonic() - wait_start > 24 * 3600:
                raise TimeoutError('Dependency queues did not terminate within 24 hours')
            time.sleep(30)
        gate_open = not args.first_job_gate
        gate_launched = False
        idle_start = time.monotonic()
        while pending or active:
            free = {}
            if args.min_free_mib:
                query = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                free = {int(r.split(',')[0]): int(r.split(',')[1]) for r in query.stdout.splitlines()}
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                if not gate_open and gate_launched:
                    continue
                if args.min_free_mib and free[gpu] < args.min_free_mib:
                    continue
                task, seed = pending.pop(0)
                name = f'{task}_s{seed}'
                run_dir = output / name
                log = (output / f'{name}.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
                           PYOPENGL_PLATFORM='egl', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                           TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='4',
                           WANDB_MODE='disabled')
                cmd = [str(ROOT / '.venv-pi05/bin/python'), '-u',
                       str(ROOT / 'flowdagger_pi05' / args.entrypoint),
                       '--repro_run_dir', str(run_dir), '--task_key', task,
                       '--seed', str(seed), '--max_steps', '3999',
                       '--seed_expert_episodes', '10', '--eval_episodes', '25',
                       '--eval_interval', str(args.eval_interval), '--checkpoint_interval', '4000',
                       '--save_eval_video', '0', '--log_interval', '100']
                if args.dual_buffer:
                    cmd.extend(['--dual_buffer','1','--dual_buffer_auto_frac','0.5'])
                if args.rollout_budget:
                    cmd.extend(['--repro_rollout_budget',str(args.rollout_budget)])
                if args.filter_invalid_inversions:
                    cmd.append('--repro_filter_invalid_inversions')
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = dict(proc=proc, log=log, name=name, start=time.time())
                gate_launched = True
                record(event='started', name=name, gpu=gpu, pid=proc.pid, command=cmd)
            for gpu, job in list(active.items()):
                proc = job['proc']
                code = proc.poll()
                if code is None and time.time() - job['start'] > args.timeout_hours * 3600:
                    record(event='hard_timeout', name=job['name'], gpu=gpu)
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
                if code is not None:
                    record(event='finished', name=job['name'], gpu=gpu,
                           returncode=code, elapsed=time.time() - job['start'])
                    job['log'].close()
                    del active[gpu]
                    if args.first_job_gate and not gate_open:
                        if code == 0:
                            gate_open = True
                        else:
                            record(event='stopped_failed_first_job', pending=[f'{t}_s{s}' for t,s in pending])
                            pending.clear()
            if active:
                idle_start = time.monotonic()
            elif pending and time.monotonic() - idle_start > 3600:
                record(event='resource_unavailable', pending=[f'{t}_s{s}' for t,s in pending])
                break
            if active or pending:
                time.sleep(10)
    (output / 'queue_completed.json').write_text(json.dumps({'time': time.time()}))


if __name__ == '__main__':
    main()
