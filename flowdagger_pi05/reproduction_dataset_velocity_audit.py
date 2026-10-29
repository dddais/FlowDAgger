"""CPU-only replay audit of archived stick-push training demonstrations.

Only renders a scientific plot after simulation. No base-policy inference or
changes to the official environment implementation are involved.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from reproduce_main_table import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    samples, out = Path(args.samples).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import mujoco
    import pyarrow.parquet as pq
    from metaworld import policies
    from metaworld_pi05_adapter import MetaworldPi05Adapter
    # Saved actions and raw states are used; image rendering cannot affect physics.
    MetaworldPi05Adapter._render = lambda self: np.zeros((1, 1, 3), dtype=np.uint8)
    index = json.loads((samples/'stick_push_all_50/audit_v2.json').read_text())
    assert index['complete'] and len(index['episodes']) == 50
    episodes = {}
    for item in index['episodes']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = ROOT/path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        rows = pq.read_table(path).to_pylist()
        assert all(r['task_index'] == 38 for r in rows)
        episodes[item['episode']] = rows
    assert set(episodes) == set(range(1900, 1950))

    def make_env(first, velocity):
        env = MetaworldPi05Adapter('stick-push-v3', seed=42)
        raw = env.raw_env
        raw._last_rand_vec = np.array([first[4], first[5], .0005, *first[36:39]])
        raw._freeze_rand_vec = True
        env.reset(seed=42)
        jid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, 'obj_slidex')
        assert jid >= 0
        dof = int(raw.model.jnt_dofadr[jid])
        assert abs(raw.data.qvel[dof]) < 1e-12
        raw.data.qvel[dof] = velocity
        mujoco.mj_forward(raw.model, raw.data)
        env._last_raw_obs = raw._get_obs().astype(np.float32)
        return env

    calibration = episodes[1900]
    first = calibration[0]['observation.environment_state']
    env = make_env(first, 1.)
    obs, _, _, _ = env.step(calibration[0]['action'])
    unit_dx = float(obs['metaworld_obs'][11] - first[11])
    env.close()
    target_dx = calibration[1]['observation.environment_state'][11] - first[11]
    velocity = target_dx / unit_dx
    summary, traces = [], []
    for episode, rows in sorted(episodes.items()):
        expert = policies.SawyerStickPushV3Policy()
        predicted = np.stack([expert.get_action(np.asarray(r['observation.environment_state'], dtype=np.float32)) for r in rows])
        actions = np.asarray([r['action'] for r in rows])
        results = {}
        for condition, v in [('default_zero', 0.), ('calibrated_velocity', velocity)]:
            env = make_env(rows[0]['observation.environment_state'], v)
            trajectory = []
            for i, row in enumerate(rows):
                raw = env._last_raw_obs.copy()
                trajectory.append(dict(frame=i, local_object_x=float(raw[11]),
                    training_object_x=row['observation.environment_state'][11],
                    hand_max_error=float(np.max(np.abs(raw[:3]-np.asarray(row['observation.environment_state'][:3]))))))
                _, reward, _, _ = env.step(row['action'])
            env.close()
            results[condition] = dict(final_success=bool(reward),
                max_object_x_error=max(abs(r['local_object_x']-r['training_object_x']) for r in trajectory[2:]),
                max_hand_error=max(r['hand_max_error'] for r in trajectory))
            if episode == 1900:
                traces.append(dict(condition=condition, trajectory=trajectory))
        summary.append(dict(episode=episode, frames=len(rows),
            expert_action_max_abs_error=float(np.max(np.abs(predicted-actions))),
            recorded_success=bool(rows[-1]['next.success']), results=results))
    result = dict(calibration_episode=1900, calibration_frames=[0, 1],
        velocity=velocity, joint='obj_slidex', unit_velocity_first_transition_dx=unit_dx,
        source_revision=index['revision'], episodes=summary, example_traces=traces,
        versions={name:importlib.metadata.version(name) for name in ['metaworld','mujoco','gymnasium']},
        note='Velocity inferred from one transition; validated on subsequent frames and 49 other demonstrations. Positions reconstructed from archived observations; original robot joint velocities/reset seeds unavailable. Does not establish historical bug provenance or the paper evaluation environment.')
    with (out/'results.json').open('x') as f:
        json.dump(result, f, indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(len(calibration))/80
    ax.plot(t, [r['observation.environment_state'][11] for r in calibration], 'k-', label='Archived training episode 1900')
    for tr in traces:
        ax.plot(t, [r['local_object_x'] for r in tr['trajectory']],
            '--' if tr['condition']=='calibrated_velocity' else ':',
            label=tr['condition'].replace('_',' '))
    ax.set(xlabel='Time (seconds)', ylabel='Target object x (m)',
        title='Same archived actions; initial object velocity differs')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/'object_motion.png', dpi=180)
    fig.savefig(out/'object_motion.pdf'); plt.close(fig)
    print(json.dumps(dict(velocity=velocity, episodes=len(summary),
        default_replay_successes=sum(r['results']['default_zero']['final_success'] for r in summary),
        velocity_replay_successes=sum(r['results']['calibrated_velocity']['final_success'] for r in summary),
        heldout_max_object_error=max(r['results']['calibrated_velocity']['max_object_x_error'] for r in summary if r['episode'] != 1900))), flush=True)


if __name__ == '__main__':
    main()
