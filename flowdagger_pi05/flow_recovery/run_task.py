"""Run one task's offline cache, BC training and paired fixed-plan evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from flowdagger_pi05.libero_offline.common import load_config

ROOT = Path(__file__).resolve().parents[2]
CLIENT_ROOT = Path('/home/dais/workspace/LIBERO-Recovery')
TRAIN_PYTHON = ROOT / '.venv-pi05/bin/python'
SIM_PYTHON = Path('/home/dais/miniconda3/envs/libero/bin/python')


def stop_process(child, timeout=30):
    """Stop a process group started by this worker, including on cancellation."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def handle_termination(signum, frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, handle_termination)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--policies', nargs='+', choices=['base', 'steering'], default=['base', 'steering'])
    p.add_argument('--job-label', help='Separate log directory for separately scheduled policy evaluations')
    p.add_argument('--stage', choices=['offline', 'eval', 'all'], default='all')
    p.add_argument('--evaluation-dir', default='eval_v2', help='New result tree; preserve failed eval/ attempt')
    p.add_argument('--load-free-mib', type=int, default=0,
                   help='Offline retry: check free memory again after imports and before checkpoint loading')
    p.add_argument('--wait-marker', type=Path, help='With load guard, wait for a predecessor to release the GPU')
    args = p.parse_args()
    if args.wait_marker and not args.load_free_mib:
        p.error('--wait-marker requires --load-free-mib')
    root = args.root.resolve()
    config = args.config.resolve()
    task = args.task
    logs = root / 'jobs' / f'task_{task:02d}' / (args.job_label or args.stage)
    logs.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_EGL_DEVICE_ID=str(args.gpu),
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               NUMBA_CACHE_DIR=str(root / 'numba_cache'), __GL_SHADER_DISK_CACHE_PATH=str(root / 'opengl_cache'),
               OPENPI_DATA_HOME=str(root / 'openpi_cache'), JAX_COMPILATION_CACHE_DIR=str(root / 'jax_cache'),
               LIBERO_CONFIG_PATH=str(root / 'libero_config'),
               PYTHONPATH=':'.join(map(str, [ROOT, CLIENT_ROOT, CLIENT_ROOT / 'LIBERO',
                    ROOT / 'flowdagger_pi05/openpi/packages/openpi-client/src'])))

    def event(kind, **kwargs):
        record = dict(time=datetime.now(timezone.utc).isoformat(), event=kind, task=task, gpu=args.gpu, **kwargs)
        with (logs / 'events.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def run(name, command, cwd, timeout):
        with (logs / (name + '.log')).open('x') as log:
            child = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            event('start', name=name, pid=child.pid, command=list(map(str, command)), timeout=timeout)
            start = time.monotonic()
            try:
                while child.poll() is None:
                    if time.monotonic() - start > timeout:
                        event('timeout', name=name, pid=child.pid)
                        raise TimeoutError(name)
                    time.sleep(10)
            finally:
                stop_process(child)
            event('exit', name=name, returncode=child.returncode, seconds=time.monotonic() - start)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)

    if args.stage in ('offline', 'all'):
        precompute = [TRAIN_PYTHON, '-u', '-m',
                      'flowdagger_pi05.flow_recovery.guarded_precompute' if args.load_free_mib else
                      'flowdagger_pi05.libero_offline.precompute', '--config', config]
        if args.load_free_mib:
            precompute += ['--min-free-mib', args.load_free_mib]
            if args.wait_marker:
                precompute += ['--wait-marker', args.wait_marker.resolve()]
        run('precompute', precompute, ROOT, 12 * 3600)
        train_command = [TRAIN_PYTHON, '-u', '-m', 'flowdagger_pi05.libero_offline.train',
                         '--config', config, '--tasks', task]
        training_run = Path(load_config(config)['output_dir']) / f'task_{task:02d}' / 'run.json'
        if training_run.is_file():
            # The trainer verifies identical configuration/cache before restoring all state.
            train_command.append('--resume')
        run('train', train_command, ROOT, 3600)
    if args.stage in ('eval', 'all'):
        for policy in args.policies:
            command = [str(TRAIN_PYTHON), '-u', '-m', 'flowdagger_pi05.libero_offline.serve',
                       '--config', str(config), '--tasks', str(task), '--mode', policy, '--port', str(args.port),
                       '--disable-websocket-pings']
            with (logs / f'server_{policy}.log').open('x') as log:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                event('server_start', policy=policy, pid=server.pid, command=command)
                try:
                    started = time.monotonic()
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f'{policy} server exited: {server.returncode}')
                        try:
                            with opener.open(f'http://127.0.0.1:{args.port}/healthz', timeout=2) as response:
                                assert response.status == 200
                            break
                        except OSError:
                            if time.monotonic() - started > 1800:
                                raise TimeoutError('server startup')
                            time.sleep(5)
                    run(f'warmup_{policy}', [TRAIN_PYTHON, '-u', '-m', 'flowdagger_pi05.flow_recovery.warmup',
                        '--config', config, '--task', task, '--port', args.port,
                        '--output', logs / f'warmup_{policy}.json'], ROOT, 1800)
                    for mode in ('standard', 'recovery'):
                        plan = root / 'plans' / f'task_{task:02d}' / f'{mode}.json'
                        output = root / args.evaluation_dir / policy / f'task_{task:02d}' / mode
                        run(f'{policy}_{mode}', [SIM_PYTHON, '-u', '-m', 'examples.flowdagger.evaluate',
                            '--plan', plan, '--host', '127.0.0.1', '--port', args.port,
                            '--render-gpu-device-id', args.gpu, '--scene-compatibility', '--state-restore-reset',
                            '--disable-websocket-pings', '--output', output], CLIENT_ROOT, 12 * 3600)
                finally:
                    stop_process(server)
                    event('server_stop', policy=policy, returncode=server.returncode)
    event('completed', stage=args.stage)
    with (logs / 'COMPLETED.json').open('x') as f:
        json.dump({'task': task, 'stage': args.stage, 'gpu': args.gpu, 'policies': args.policies}, f)


if __name__ == '__main__':
    main()
