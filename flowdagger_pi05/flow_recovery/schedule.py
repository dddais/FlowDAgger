"""Schedule independent task workers on currently available GPUs (maximum six)."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .run_task import handle_termination, stop_process

ROOT = Path(__file__).resolve().parents[2]


def training_ready(root, task):
    # A successful retry may have a custom job label such as offline_attempt03.
    for path in (root / 'jobs' / f'task_{task:02d}').glob('*/COMPLETED.json'):
        marker = json.loads(path.read_text())
        if marker.get('task') == task and marker.get('stage') in ('offline', 'all'):
            return True
    return False


def main():
    signal.signal(signal.SIGTERM, handle_termination)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config-prefix', type=Path, required=True)
    p.add_argument('--stage', choices=['offline', 'eval'], required=True)
    p.add_argument('--tasks', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--gpus', nargs='+', type=int, choices=range(8), default=list(range(8)))
    p.add_argument('--max-gpus', type=int, choices=range(1, 7), default=3)
    p.add_argument('--min-free-mib', type=int, default=16000)
    p.add_argument('--port-base', type=int, default=18200)
    p.add_argument('--timeout-hours', type=float, default=36)
    p.add_argument('--schedule-label', help='Fresh log directory for a successor schedule')
    p.add_argument('--evaluation-dir', default='eval_v2', help='Use a fresh tree when retrying evaluation')
    p.add_argument('--resource-control', type=Path,
                   help='Optional JSON with gpus and max_gpus, accounting for independently owned workers')
    p.add_argument('--wait-for-training', action='store_true',
                   help='Evaluation only: dispatch each task after its offline completion marker appears')
    p.add_argument('--load-free-mib', type=int, default=16000,
                   help='Offline workers recheck this memory headroom after imports; 0 disables the guard')
    args = p.parse_args()
    if args.wait_for_training and args.stage != 'eval':
        p.error('--wait-for-training requires --stage eval')
    root = args.root.resolve()
    default_control = root / f'schedule_{args.stage}_resources.json'
    if args.resource_control is None and default_control.is_file():
        args.resource_control = default_control
    directory = root / (args.schedule_label or ('schedule_' + args.stage))
    directory.mkdir(parents=True, exist_ok=False)
    pending, active, failures = list(args.tasks), {}, []
    started = time.monotonic()
    with ExitStack() as cleanup, (directory / 'events.jsonl').open('x') as events:
        def emit(**entry):
            entry['time'] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(entry)
            events.write(line + '\n')
            events.flush()
            print(line, flush=True)
        while pending or active:
            if time.monotonic() - started > args.timeout_hours * 3600:
                emit(event='scheduler_timeout', pending=pending, active=list(active))
                raise TimeoutError('scheduler deadline')
            for gpu, (task, child, log) in list(active.items()):
                if child.poll() is not None:
                    emit(event='exit', task=task, gpu=gpu, returncode=child.returncode)
                    log.close()
                    del active[gpu]
                    if child.returncode:
                        failures.append(task)
            if not pending and not active:
                break
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                             '--format=csv,noheader,nounits'], text=True)
            available = [tuple(map(int, row.split(','))) for row in rows.splitlines()]
            available.sort(key=lambda row: (-row[1], row[2]))
            limits = json.loads(args.resource_control.read_text()) if args.resource_control else {}
            allowed_gpus = limits.get('gpus', args.gpus)
            max_gpus = limits.get('max_gpus', args.max_gpus)
            min_free_mib = limits.get('min_free_mib', args.min_free_mib)
            assert 1 <= max_gpus <= 6 and set(allowed_gpus) <= set(range(8))
            for gpu, free, utilization in available:
                if not pending or len(active) >= max_gpus:
                    break
                if gpu not in allowed_gpus or gpu in active or free < min_free_mib:
                    continue
                ready = [task for task in pending if not args.wait_for_training or training_ready(root, task)]
                if not ready:
                    break
                task = ready[0]
                pending.remove(task)
                config = Path(str(args.config_prefix) + f'_task{task:02d}.json').resolve()
                command = ['/usr/bin/python3', '-u', '-m', 'flowdagger_pi05.flow_recovery.run_task',
                           '--config', str(config), '--root', str(root), '--task', str(task),
                           '--gpu', str(gpu), '--port', str(args.port_base + task), '--stage', args.stage]
                if args.schedule_label:
                    command += ['--job-label', args.schedule_label]
                if args.stage == 'eval':
                    command += ['--evaluation-dir', args.evaluation_dir]
                if args.stage == 'offline' and args.load_free_mib:
                    command += ['--load-free-mib', str(args.load_free_mib)]
                log = cleanup.enter_context((directory / f'task_{task:02d}.log').open('x'))
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env=os.environ.copy(), start_new_session=True)
                # Allow the worker to clean up its separately grouped simulator/server.
                cleanup.callback(stop_process, child, timeout=90)
                active[gpu] = task, child, log
                emit(event='start', task=task, gpu=gpu, free_mib=free, utilization=utilization,
                     pid=child.pid, command=command)
            time.sleep(15)
        with (directory / 'summary.json').open('x') as f:
            json.dump({'tasks': args.tasks, 'failed_tasks': failures,
                       'elapsed_seconds': time.monotonic() - started}, f, indent=2)
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
