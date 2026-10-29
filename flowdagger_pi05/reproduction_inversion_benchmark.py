"""Saved-chunk reconstruction benchmark using the unchanged official inverter."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from reproduce_main_table import ROOT

METHODS = {
    'euler': dict(method='euler_reverse'),
    'adam20': dict(method='adam', refine_steps=20, adam_lr=0.01),
    'trajectory5': dict(method='fixed_point', refine_steps=5),
    'trajectory20': dict(method='fixed_point', refine_steps=20),
    'perstep3': dict(method='perstep_fp', fp_per_step=3),
    'perstep5': dict(method='perstep_fp', fp_per_step=5),
    'perstep10': dict(method='perstep_fp', fp_per_step=10),
    # Explicit diagnostic: official Adam defaults to a five-step gradient decoder.
    'adam20_full10': dict(method='adam', refine_steps=20, adam_lr=0.01,
                          adam_denoise_steps=10),
}


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2)


def collect(args):
    import numpy as np
    from metaworld_pi05_adapter import MetaworldPi05Adapter
    from shared.experts.metaworld_expert import MetaworldScriptedExpert
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = MetaworldPi05Adapter('assembly-v3', seed=args.seed)
    expert = MetaworldScriptedExpert('assembly-v3')
    images, states, actions, rows = [], [], [], []
    episode = 0
    try:
        while len(rows) < args.chunks:
            seed = args.seed + episode
            obs = env.reset(seed=seed)
            expert.reset(env)
            elapsed = 0
            success = False
            while elapsed < 300 and not success and len(rows) < args.chunks:
                image, state = obs['image'].copy(), obs['state'].copy()
                start_step = elapsed
                chunk = []
                for _ in range(min(args.chunk_steps, 300-elapsed)):
                    action = expert.act(env)
                    chunk.append(action)
                    obs, reward, _, _ = env.step(action)
                    elapsed += 1
                    success = reward >= 1
                    if success:
                        break
                valid = len(chunk)
                chunk += [chunk[-1].copy() for _ in range(args.chunk_steps-valid)]
                images.append(image); states.append(state); actions.append(np.stack(chunk))
                row = dict(chunk=len(rows), episode=episode, reset_seed=seed,
                           start_step=start_step, valid_steps=valid,
                           success_at_chunk_end=bool(success))
                rows.append(row)
                print(json.dumps(row), flush=True)
            episode += 1
    finally:
        env.close()
    data = out/'chunks.npz'
    with data.open('xb') as f:
        np.savez_compressed(f, images=np.stack(images), states=np.stack(states),
                            actions=np.stack(actions),
                            valid_steps=np.array([r['valid_steps'] for r in rows]))
    write_json(out/'manifest.json', dict(
        task='assembly-v3', prompt='Pick up a nut and place it onto a peg',
        collection='First consecutive expert chunks; no filtering by success or inversion error',
        chunks=rows, chunk_steps=args.chunk_steps, episode_horizon=300,
        paper_chunks_available=False, sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        camera='corner3', image_resolution=256,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'flowdagger_pi05/metaworld_pi05_adapter.py','shared/experts/metaworld_expert.py',
            'flowdagger_pi05/reproduction_inversion_benchmark.py']}))


def benchmark(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_deterministic_ops=true'
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from openpi.training import config as configs, checkpoints
    from openpi.models import model as model_module
    from openpi.policies import policy as policy_module
    import openpi.transforms as transforms
    from flow_matching_inverter import FlowMatchingInverter
    from action_converter import ActionConverter
    from train_utils import obs_to_pi_zero_input
    source = Path(args.chunks_dir).resolve()
    source_manifest = json.loads((source/'manifest.json').read_text())
    dataset = source/'chunks.npz'
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert digest == source_manifest['sha256'], 'Saved benchmark input changed'
    data = np.load(dataset, allow_pickle=False)
    config = configs.get_config('pi05_metaworld')
    checkpoint = Path(args.checkpoint).resolve()
    assert data['actions'].shape[1] == config.model.action_horizon, 'Chunk length must match pi05_metaworld action horizon'
    raw_model = config.model.load(model_module.restore_params(checkpoint/'params', dtype=jnp.bfloat16))
    dc = config.data.create(config.assets_dirs, config.model)
    norms = checkpoints.load_norm_stats(checkpoint/'assets', dc.asset_id)
    policy = policy_module.Policy(raw_model, transforms=[
        transforms.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
        transforms.Normalize(norms, use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs], output_transforms=[
        *dc.model_transforms.outputs,
        transforms.Unnormalize(norms, use_quantiles=dc.use_quantile_norm),
        *dc.data_transforms.outputs], metadata=config.policy_metadata)
    converter = ActionConverter(norms['actions'], norm_stats_state=norms.get('state'),
        delta_mask=None, action_dim_env=4, action_horizon=raw_model.action_horizon)
    params = dict(num_denoise_steps=10, refine_steps=20, adam_lr=.01,
                  regularization_weight=.01, grad_clip=1., adam_denoise_steps=5,
                  seed=args.seed, solver='euler', fp_per_step=5)
    params.update(METHODS[args.method])
    inverter = FlowMatchingInverter(raw_model, **params)
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                          '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    write_json(out/'manifest.json', dict(method=args.method, inverter=params,
        input_sha256=digest, input_manifest=str(source/'manifest.json'),
        checkpoint=str(checkpoint), jax_devices=[str(d) for d in jax.devices()],
        gpu_snapshot=gpu, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        xla_flags=os.environ.get('XLA_FLAGS'), tensorflow_gpu_devices=tf.config.get_visible_devices('GPU'),
        warmup_calls=2, timing='Complete official invert call, including its reconstruction check; synchronized. Extra metric decode excluded.',
        paper_time_hardware='RTX 5090; local hardware differs; shared load is uncontrolled.',
        source_sha256={p:hashlib.sha256((ROOT/'flowdagger_pi05'/p).read_bytes()).hexdigest()
                       for p in ['flow_matching_inverter.py','action_converter.py',Path(__file__).name]}))
    variant = SimpleNamespace(task_description=source_manifest['prompt'])
    rows = []
    with (out/'chunks.jsonl').open('x') as f:
        for i in range(len(data['images'])):
            state = data['states'][i]
            raw = dict(image=data['images'][i], state=state)
            inputs = policy._input_transform(obs_to_pi_zero_input(raw, variant))
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            obs = model_module.Observation.from_dict(inputs)
            valid = int(data['valid_steps'][i])
            target_np = converter.expert_to_internal(data['actions'][i, :valid], state)
            target = jnp.asarray(target_np)[None, ...]
            jax.block_until_ready((obs, target))
            if i == 0:
                for _ in range(2):
                    jax.block_until_ready(inverter.invert(obs, target))
                # Warmup must not alter random initialization of measured Adam calls.
                inverter._rng = jax.random.PRNGKey(args.seed)
            started = time.perf_counter()
            noise, error = inverter.invert(obs, target)
            jax.block_until_ready((noise, error))
            milliseconds = 1000*(time.perf_counter()-started)
            reconstructed = np.asarray(inverter._denoise(obs, noise)[0])
            env_actions = converter.internal_to_env(reconstructed, state)
            env_target = converter.internal_to_env(target_np, state)
            assert np.isfinite(env_actions).all() and np.isfinite(noise).all()
            squared = np.square(env_actions-env_target)
            row = dict(chunk=i, milliseconds=milliseconds, valid_steps=valid,
                internal_mse=float(np.asarray(error)[0]),
                env4_mse_full_horizon=float(squared.mean()),
                env4_mse_executed=float(squared[:valid].mean()),
                env4_mse_executed_clipped=float(np.square(np.clip(env_actions[:valid],-1,1)-env_target[:valid]).mean()),
                noise_mean_abs=float(np.abs(np.asarray(noise)).mean()))
            rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
            print(json.dumps(row), flush=True)
    summary = dict(method=args.method, chunks=len(rows), input_sha256=digest,
        median_ms=float(np.median([r['milliseconds'] for r in rows])),
        mean_env4_mse_full_horizon=float(np.mean([r['env4_mse_full_horizon'] for r in rows])),
        mean_env4_mse_executed=float(np.mean([r['env4_mse_executed'] for r in rows])),
        mean_env4_mse_executed_clipped=float(np.mean([r['env4_mse_executed_clipped'] for r in rows])),
        horizon=raw_model.action_horizon,
        note='Own fixed chunks, not unpublished paper inputs. Both full padded-horizon and executed-prefix metrics retained. Timing includes validation decode.')
    write_json(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['collect','benchmark'])
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=90042)
    p.add_argument('--chunks',type=int,default=29)
    p.add_argument('--chunk_steps',type=int,default=10)
    p.add_argument('--chunks_dir')
    p.add_argument('--method',choices=list(METHODS),default='perstep5')
    p.add_argument('--checkpoint',default='/home/dais/workspace/model/pi05-metaworld')
    args=p.parse_args()
    if args.mode=='collect':
        collect(args)
    else:
        if not args.chunks_dir:
            p.error('--chunks_dir required for benchmark')
        benchmark(args)


if __name__=='__main__':
    main()
