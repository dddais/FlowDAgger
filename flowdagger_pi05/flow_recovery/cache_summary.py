"""Report inversion acceptance and training endpoints for each independent task."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', type=int, default=list(range(10)))
    args = p.parse_args()
    rows = []
    for task in args.tasks:
        cache = args.root / 'cache' / f'task_{task:02d}'
        manifest = json.loads((cache / 'manifest.json').read_text())
        groups = defaultdict(list)
        for episode in manifest['episodes']:
            groups[(episode['category'], episode['split'])].append(episode)
        for (source, split), episodes in groups.items():
            errors, accepted, noises, grips = [], [], [], []
            for episode in episodes:
                with np.load(cache / 'shards' / (episode['id'] + '.npz')) as shard:
                    errors.extend(shard['all_mse'].tolist())
                    accepted.extend(shard['accepted'].tolist())
                    noises.append(shard['noise'].ravel())
                    grips.extend(shard['all_grip_match'][shard['accepted']].tolist())
            finite_errors = np.asarray(errors)[np.isfinite(errors)]
            noise = np.concatenate(noises)
            rows.append({'task': task, 'source': source, 'split': split, 'episodes': len(episodes),
                         'candidate_windows': len(errors), 'accepted_windows': sum(accepted),
                         'acceptance_fraction': float(np.mean(accepted)),
                         'finite_mse_median': float(np.median(finite_errors)) if len(finite_errors) else None,
                         'finite_mse_p95': float(np.percentile(finite_errors, 95)) if len(finite_errors) else None,
                         'accepted_noise_abs_p99': float(np.percentile(abs(noise), 99)) if noise.size else None,
                         'accepted_gripper_sign_match': float(np.mean(grips)) if grips else None})
    with args.output.open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', len(rows), 'task/source/split groups')


if __name__ == '__main__':
    main()
