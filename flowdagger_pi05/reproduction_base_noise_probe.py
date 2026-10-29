"""Paired fixed-reset Gaussian/zero-noise Base diagnostic; no adaptation."""
import hashlib
import json
from pathlib import Path

from reproduce_main_table import main as official_wrapper


def reset_seeds(seed, count):
    if seed not in (42, 43, 44) or count != 25:
        raise ValueError('Prespecified diagnostic uses seeds 42/43/44 and 25 pairs each')
    return list(range(620000 + (seed - 42) * 25, 620000 + (seed - 41) * 25))


def main(probe_kind='latent', velocity_audit=None):
    import dagger_loop

    def evaluate(variant, steering_policy, env, eval_env, replay_buffer,
                 wandb_logger, shard_fn=None, agent_dp=None,
                 intervention_handler=None, init_states=None, raw_model=None,
                 autonomous_buffer=None):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from train_utils import obs_to_pi_zero_input

        seeds = reset_seeds(variant.seed, variant.eval_episodes)
        conditions = ('gaussian', 'zero')
        velocity = None
        velocity_source = None
        if probe_kind == 'reset_velocity':
            if variant.task_key != 'metaworld_stick_push':
                raise ValueError('The velocity diagnostic is restricted to stick-push')
            path = Path(velocity_audit).resolve()
            audit = json.loads(path.read_text())
            assert len(audit['episodes']) == 50 and audit['calibration_episode'] == 1900
            velocity = float(audit['velocity'])
            assert 0 < velocity < 1
            velocity_source = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            conditions = ('default_zero_velocity', 'training_velocity')
        elif probe_kind != 'latent':
            raise ValueError(probe_kind)
        query = variant.query_freq or agent_dp.action_horizon
        assert query == 10 and variant.max_timesteps == 300
        assert agent_dp.action_horizon == 10
        out = Path(variant.outputdir)
        with (out / 'noise_probe_protocol.json').open('x') as f:
            json.dump(dict(task=variant.task_key, seed=variant.seed,
                reset_seeds=seeds, horizon=300, query_frequency=query,
                noise_shape=[1, agent_dp.action_horizon, agent_dp.action_dim],
                conditions=conditions, training_rollouts=0, probe_kind=probe_kind,
                calibrated_velocity=velocity, velocity_source=velocity_source,
                note='Independent reset sequence; not the official main-table evaluation. '
                     'Gaussian latent keys are independent of reset RNG and episode length.'), f, indent=2)
        rows = []
        with (out / 'noise_probe_episodes.jsonl').open('x') as f:
            for pair, reset_seed in enumerate(seeds):
                initial = None
                for condition in conditions:
                    obs = eval_env.reset(seed=reset_seed)
                    velocity_dof = None
                    if probe_kind == 'reset_velocity':
                        import mujoco
                        raw = eval_env.raw_env
                        jid = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_JOINT, 'obj_slidex')
                        assert jid >= 0
                        velocity_dof = int(raw.model.jnt_dofadr[jid])
                        assert abs(raw.data.qvel[velocity_dof]) < 1e-12
                        if condition == 'training_velocity':
                            raw.data.qvel[velocity_dof] = velocity
                            mujoco.mj_forward(raw.model, raw.data)
                    snapshot = dict(raw=np.asarray(obs['metaworld_obs']).copy(),
                                    image=hashlib.sha256(obs['image'].tobytes()).hexdigest(),
                                    qpos=eval_env.raw_env.data.qpos.copy(),
                                    qvel=eval_env.raw_env.data.qvel.copy())
                    if initial is None:
                        initial = snapshot
                    else:
                        assert np.allclose(initial['raw'], snapshot['raw'], rtol=0, atol=1e-6), 'Paired initial observations differ'
                        assert initial['image'] == snapshot['image'], 'Paired initial images differ'
                        assert np.allclose(initial['qpos'], snapshot['qpos'], rtol=0, atol=1e-10), 'Paired initial positions differ'
                        delta = snapshot['qvel'] - initial['qvel']
                        if velocity_dof is not None:
                            assert abs(delta[velocity_dof] - velocity) < 1e-10
                            delta[velocity_dof] = 0
                        assert np.max(np.abs(delta)) < 1e-10, 'Unexpected paired initial velocity differences'
                    key = jax.random.fold_in(jax.random.PRNGKey(variant.seed + 456), reset_seed)
                    reward = 0.0
                    for t in range(300):
                        if t % query == 0:
                            key, sample_key = jax.random.split(key)
                            shape = (1, agent_dp.action_horizon, agent_dp.action_dim)
                            noise = (jax.random.normal(sample_key, shape) if condition != 'zero'
                                     else jnp.zeros(shape, dtype=jnp.float32))
                            actions = agent_dp.infer(obs_to_pi_zero_input(obs, variant), noise=noise)['actions']
                            assert np.isfinite(actions).all()
                        obs, reward, done, _ = eval_env.step(np.asarray(actions[t % query], dtype=np.float32)[:4])
                        if done or reward >= variant.env_max_reward:
                            break
                    row = dict(task=variant.task_key, seed=variant.seed, pair=pair,
                        reset_seed=reset_seed, condition=condition,
                        success=bool(reward >= variant.env_max_reward), steps=t+1,
                        initial_raw=snapshot['raw'].tolist(), initial_image_sha256=snapshot['image'],
                        initial_qpos=snapshot['qpos'].tolist(), initial_qvel=snapshot['qvel'].tolist(),
                        velocity_dof=velocity_dof, calibrated_velocity=velocity)
                    f.write(json.dumps(row)+'\n'); f.flush()
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k != 'initial_raw'}), flush=True)
        with (out / 'noise_probe_summary.json').open('x') as f:
            json.dump(dict(task=variant.task_key, seed=variant.seed, pairs=len(seeds),
                success_rate={c:float(np.mean([r['success'] for r in rows if r['condition']==c]))
                              for c in conditions}, paired_initials_verified=True,
                probe_kind=probe_kind, calibrated_velocity=velocity), f, indent=2)

    dagger_loop.flowdagger_training_loop = evaluate
    official_wrapper()


if __name__ == '__main__':
    main()
