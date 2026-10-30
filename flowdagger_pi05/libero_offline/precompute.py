"""python -m flowdagger_pi05.libero_offline.precompute --config config/libero_offline.json"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .common import bridge_settings, digest, load_config, read_json, write_json
from .data import episodes, load_episode, observation, window_starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/libero_offline.json")
    parser.add_argument("--dry-run", action="store_true", help="Count windows only; no model imports/downloads")
    parser.add_argument("--max-windows-per-episode", type=int, default=0, help="Small diagnostic cache, use a separate cache_dir")
    args = parser.parse_args()
    config = load_config(args.config)
    pool, records = episodes(config)
    for record in records:
        record["starts"] = window_starts(record["length"], config["horizon"], config["stride"], config["tail"])
        if args.max_windows_per_episode:
            record["starts"] = record["starts"][:args.max_windows_per_episode]
    for task in config["tasks"]:
        rows = [r for r in records if r["task_index"] == task]
        print(f"task={task}: episodes={len(rows)}, windows={sum(len(r['starts']) for r in rows)}", flush=True)
    if args.dry_run:
        return

    from .bridge import LiberoBridge

    bridge = LiberoBridge(config)
    if (bridge.horizon, bridge.action_dim) != (config["horizon"], 32):
        raise ValueError("Cache horizon/dimension must match the loaded pi05_libero model")
    root = Path(config["dataset_root"])
    spec = {
        "version": 1, "bridge": bridge_settings(config), "checkpoint_identity": bridge.identity(),
        "manifest_sha256": hashlib.sha256((root / "data_pool.json").read_bytes()).hexdigest(),
        "arrays_sha256": hashlib.sha256((root / "array_checksums.json").read_bytes()).hexdigest(),
        "settings": {k: config[k] for k in ("tasks", "sources", "split_seed", "validation_fraction", "horizon", "stride", "tail", "fp_steps", "max_normalized_mse")},
        "max_windows_per_episode": args.max_windows_per_episode,
        "features": "vlm_pi0_all_prefix_mean_float32", "noise": "full_horizon_no_basis_projection",
    }
    cache = Path(config["cache_dir"])
    (cache / "shards").mkdir(parents=True, exist_ok=True)
    fingerprint = digest(spec)
    if (cache / "spec.json").exists():
        if read_json(cache / "spec.json")["fingerprint"] != fingerprint:
            raise ValueError("Cache settings/checkpoint changed; choose a new cache_dir")
    else:
        write_json(cache / "spec.json", {"fingerprint": fingerprint, "spec": spec})

    for record in records:
        path = cache / "shards" / (record["id"] + ".npz")
        if not path.exists():
            arrays = load_episode(root, pool["cache_root"], record)
            results = {key: [] for key in ("features", "noise", "mse", "env_mse", "grip_match")}
            starts = record["starts"]
            batch_size = config["inversion_batch_size"]
            for offset in range(0, len(starts), batch_size):
                times = starts[offset:offset + batch_size]
                observations = [observation(arrays, t, record["prompt"]) for t in times]
                actions = np.stack([arrays["actions"][t:t + bridge.horizon] for t in times])
                values = bridge.invert(observations, actions, config["fp_steps"])
                for key, value in zip(results, values):
                    results[key].append(value)
            # plus_featured episodes are all >= H; short episodes are excluded explicitly.
            if not starts:
                raise ValueError(f"Episode {record['id']} is shorter than horizon")
            values = {key: np.concatenate(value) for key, value in results.items()}
            finite = np.isfinite(values["noise"]).all(axis=(1, 2)) & np.isfinite(values["features"]).all(axis=1)
            accepted = finite & np.isfinite(values["mse"]) & (values["mse"] <= config["max_normalized_mse"])
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, features=values["features"][accepted], noise=values["noise"][accepted],
                    state=np.asarray(arrays["state"])[np.asarray(starts)[accepted]], starts=np.asarray(starts)[accepted],
                    accepted=accepted, all_starts=starts, all_mse=values["mse"],
                    all_env_mse=values["env_mse"], all_grip_match=values["grip_match"],
                )
            temporary.replace(path)
        with np.load(path) as shard:
            record["accepted_windows"] = len(shard["noise"])
            record["total_windows"] = len(shard["all_starts"])
            record["noise_abs_p99"] = float(np.percentile(np.abs(shard["noise"]), 99)) if len(shard["noise"]) else None
        print(f"{record['id']}: {record['accepted_windows']}/{record['total_windows']} accepted", flush=True)
    manifest = {"fingerprint": fingerprint, "spec": spec, "tasks": pool["tasks"], "episodes": records}
    write_json(cache / "manifest.json", manifest)
    print(f"Complete cache: {cache}")


if __name__ == "__main__":
    main()
