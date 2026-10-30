"""Summarize complete paired evaluations; never mistake partial logs for final results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_trials(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def bootstrap_interval(clusters_by_task):
    rng = np.random.default_rng(20261029)
    draws = []
    for _ in range(5000):
        task_deltas = []
        for clusters in clusters_by_task:
            sampled = [clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]
            task_deltas.append(float(np.concatenate(sampled).mean()))
        draws.append(np.mean(task_deltas))
    return np.percentile(draws, [2.5, 97.5]).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluation-dir', default='eval_v2')
    p.add_argument('--tasks', nargs='+', type=int, default=list(range(10)))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], {}
    for mode in ('standard', 'recovery'):
        clusters_by_task, scene_clusters_by_task = [], []
        for task in args.tasks:
            task_root = f'task_{task:02d}'
            plan = json.loads((args.root / 'plans' / task_root / f'{mode}.json').read_text())
            expected = {v['id']: v for v in plan['trials']}
            trials, settings, servers = {}, {}, {}
            for policy in ('base', 'steering'):
                directory = args.root / args.evaluation_dir / policy / task_root / mode
                assert (directory / 'summary.json').is_file(), f'incomplete evaluation: {directory}'
                actual_plan = json.loads((directory / 'plan.json').read_text())
                assert actual_plan['tasks'] == plan['tasks'] and actual_plan['trials'] == plan['trials'], 'plan mismatch'
                assert all(actual_plan['settings'][k] == v for k, v in plan['settings'].items()), 'protocol drift'
                settings[policy] = actual_plan['settings']
                servers[policy] = json.loads((directory / 'server.json').read_text())
                assert servers[policy]['mode'] == policy
                records = read_trials(directory / 'trials.jsonl')
                assert len(records) == len(expected), f'wrong count: {directory}'
                trials[policy] = {v['id']: v for v in records}
                assert set(trials[policy]) == set(expected)
                for record in records:
                    assert all(record[k] == v for k, v in expected[record['id']].items()), 'plan drift'
                valid = [v for v in records if v['status'] == 'completed']
                successes = sum(v['success'] for v in valid)
                rows.append({'mode': mode, 'task': task, 'policy': policy, 'planned': len(records),
                             'valid': len(valid), 'successes': successes,
                             'rate_planned': successes / len(records),
                             'rate_valid': successes / len(valid) if valid else None,
                             'reset_errors': sum(v['status'] == 'reset_error' for v in records),
                             'initial_success': sum(v['status'] == 'initial_success' for v in records)})
            assert settings['base'] == settings['steering'], 'paired protocols differ'
            for field in ('checkpoint_identity', 'bridge', 'action_horizon', 'action_dim'):
                assert servers['base'][field] == servers['steering'][field], f'paired base mismatch: {field}'
            clusters, scene_clusters = {}, {}
            valid_pairs, improved, worsened, initial_status_disagreements = 0, 0, 0, 0
            for trial_id, trial in expected.items():
                base, steer = trials['base'][trial_id], trials['steering'][trial_id]
                initial_status_disagreements += base['status'] != steer['status']
                if 'object_xy_deltas' in base and 'object_xy_deltas' in steer:
                    assert base['object_xy_deltas'] == steer['object_xy_deltas'], 'paired state perturbations differ'
                a = int(base['status'] == 'completed' and base['success'])
                b = int(steer['status'] == 'completed' and steer['success'])
                scene_key = trial['bddl'] if mode == 'recovery' else str(trial['init_index'])
                if mode == 'recovery':
                    episode = Path(trial['bddl']).parent
                    assert episode.name.startswith('episode_'), 'source episode grouping must be verified'
                    key = str(episode)
                else:
                    key = scene_key
                clusters.setdefault(key, []).append(b - a)
                scene_clusters.setdefault(scene_key, []).append(b - a)
                if base['status'] == steer['status'] == 'completed':
                    valid_pairs += 1
                    improved += b > a
                    worsened += b < a
            clusters_by_task.append([np.asarray(v) for v in clusters.values()])
            scene_clusters_by_task.append([np.asarray(v) for v in scene_clusters.values()])
            comparisons[f'{mode}/{task}'] = {'common_valid_trials': valid_pairs,
                                            'improved_pairs': improved, 'worsened_pairs': worsened,
                                            'initial_status_disagreements': initial_status_disagreements,
                                            'resampling_clusters': len(clusters),
                                            'scene_clusters': len(scene_clusters),
                                            'cluster_unit': 'source_episode' if mode == 'recovery' else 'init_state'}
        selected = [v for v in rows if v['mode'] == mode]
        base = np.mean([v['rate_planned'] for v in selected if v['policy'] == 'base'])
        steering = np.mean([v['rate_planned'] for v in selected if v['policy'] == 'steering'])
        comparisons[mode] = {'macro_base': float(base), 'macro_steering': float(steering),
                             'macro_delta': float(steering - base),
                             'micro_base': sum(v['successes'] for v in selected if v['policy'] == 'base') / sum(v['planned'] for v in selected if v['policy'] == 'base'),
                             'micro_steering': sum(v['successes'] for v in selected if v['policy'] == 'steering') / sum(v['planned'] for v in selected if v['policy'] == 'steering'),
                             'cluster_bootstrap_delta_95_percent_interval': bootstrap_interval(clusters_by_task),
                             'cluster_unit': 'source_episode' if mode == 'recovery' else 'init_state',
                             'scene_bootstrap_delta_95_percent_interval': bootstrap_interval(scene_clusters_by_task),
                             'bootstrap_replicates': 5000,
                             'interpretation': 'Conditional on the selected fixed tasks and one trained model per task. Recovery primary resampling groups every selected scene from one source-episode directory; scene resampling is a sensitivity analysis that ignores shared source episodes. Standard groups by init state. Few source episodes limit interval reliability. Neither interval estimates training-seed or new-task uncertainty; no multiple-comparison significance claim.'}
    with (args.output / 'task_results.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'comparison.json').open('x') as f:
        json.dump({'rows': rows, 'comparisons': comparisons}, f, indent=2)
    print(json.dumps({mode: comparisons[mode] for mode in ('standard', 'recovery')}, indent=2))


if __name__ == '__main__':
    main()
