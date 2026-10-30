"""Read-only integrity and prompt/split audit of the selected dataset."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from flowdagger_pi05.libero_offline.common import load_config
from flowdagger_pi05.libero_offline.data import episodes, window_starts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = load_config(args.config)
    root = Path(config['dataset_root'])
    pool, records = episodes(config)
    checksums = {r['path']: r for r in json.loads((root / 'array_checksums.json').read_text())}
    tasks = {r['task_index']: r for r in pool['tasks']}
    groups = defaultdict(lambda: {'episodes': 0, 'windows': 0, 'frames': 0})
    errors = []
    for i, r in enumerate(records):
        directory = Path(pool['cache_root']) / r['source'] / f"{r['episode_index']:06d}"
        try:
            assert r['prompt'] == tasks[r['task_index']]['official_task_language'], 'prompt mismatch'
            for name, shape, dtype in [('images', (r['length'], 2, 256, 256, 3), np.uint8),
                                       ('state', (r['length'], 8), np.float32),
                                       ('actions', (r['length'], 7), np.float32)]:
                rel = directory / f'{name}.npy'
                path = root / rel
                expected = checksums[str(rel)]
                assert path.stat().st_size == expected['size'], f'{name}: size'
                array = np.load(path, mmap_mode='r')
                assert array.shape == shape and array.dtype == dtype, f'{name}: shape/dtype'
                if name != 'images':
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], f'{name}: sha256'
                    assert np.isfinite(array).all(), f'{name}: nonfinite'
            group = groups[f"{r['task_index']}/{r['category']}/{r['split']}"]
            group['episodes'] += 1
            group['frames'] += r['length']
            group['windows'] += len(window_starts(r['length'], config['horizon'], config['stride'], config['tail']))
        except (AssertionError, OSError, ValueError, KeyError) as exc:
            errors.append({'episode': r['id'], 'error': str(exc)})
        if (i + 1) % 50 == 0:
            print(f'audited {i + 1}/{len(records)}; errors={len(errors)}', flush=True)
    result = {'episodes': len(records), 'groups': dict(groups), 'errors': errors,
              'hash_coverage': 'all state/action arrays; images checked for byte size, shape, dtype only',
              'prompt_rule': 'record.prompt == official_task_language; never BDDL language substitution',
              'scene_independence': 'unverified: recovery demos have no original failure scene mapping'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'episodes': len(records), 'errors': len(errors)}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
