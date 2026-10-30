"""Create a fresh, per-task experiment layout without replacing any existing run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CLIENT_ROOT = Path('/home/dais/workspace/LIBERO-Recovery')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config-prefix', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Already downloaded and verified JAX pi05_libero directory')
    p.add_argument('--inversion-batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tasks', nargs='+', type=int, choices=range(10), default=list(range(10)))
    args = p.parse_args()
    root, prefix = args.root.resolve(), args.config_prefix.resolve()
    checkpoint = args.checkpoint.resolve()
    assert (checkpoint / 'params/manifest.ocdbt').is_file() and (checkpoint / 'assets').is_dir()
    root.mkdir(parents=True, exist_ok=False)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'config/libero_offline.json').read_text())
    config.update(checkpoint=str(checkpoint), cache_dir=str(root / 'cache'),
                  output_dir=str(root / f'standard_recovery_seed{args.seed}'), seed=args.seed,
                  inversion_batch_size=args.inversion_batch_size, tasks=args.tasks)
    save(Path(str(prefix) + '.json'), config)
    for task in config['tasks']:
        task_config = dict(config, tasks=[task], cache_dir=str(root / 'cache' / f'task_{task:02d}'))
        save(Path(str(prefix) + f'_task{task:02d}.json'), task_config)
    subprocess.run([sys.executable, '-m', 'examples.flowdagger.setup_paths', '--output', str(root / 'libero_config')],
                   cwd=CLIENT_ROOT, check=True)
    for mode in ('standard', 'recovery'):
        subprocess.run([sys.executable, '-m', 'examples.flowdagger.evaluate', '--mode', mode,
                        '--tasks', *map(str, args.tasks),
                        '--seed', '42', '--scene-compatibility', '--state-restore-reset',
                        '--output', str(root / 'plans' / mode), '--dry-run'], cwd=CLIENT_ROOT, check=True)
        plan = json.loads((root / 'plans' / mode / 'plan.json').read_text())
        for task in config['tasks']:
            directory = root / 'plans' / f'task_{task:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            task_plan = dict(plan, tasks={k: v for k, v in plan['tasks'].items() if v['task_index'] == task},
                             trials=[v for v in plan['trials'] if v['task_index'] == task])
            save(directory / f'{mode}.json', task_plan)
    tokenizer = Path.home() / '.cache/openpi/big_vision/paligemma_tokenizer.model'
    if tokenizer.is_file():
        target = root / 'openpi_cache/big_vision/paligemma_tokenizer.model'
        target.parent.mkdir(parents=True)
        shutil.copyfile(tokenizer, target)
    save(root / 'preparation.json', {'checkpoint': str(checkpoint), 'config_prefix': str(prefix),
                                   'seed': args.seed, 'evaluation_seed': 42,
                                   'max_supported_gpu_concurrency': 6,
                                   'checkpoint_note': 'Directory presence checked; use an independently verified checkpoint.'})
    print('Prepared', root, flush=True)


if __name__ == '__main__':
    main()
