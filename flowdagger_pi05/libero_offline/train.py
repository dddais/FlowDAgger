"""Train a separate steering network for each selected task, without loading pi05."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import bridge_settings, load_config, read_json, write_json
from .data import LatentDataset


def validation_loss(actor, dataset, batch_size):
    squared = 0.0
    for start in range(0, dataset.size, batch_size):
        batch = dataset.batch(np.arange(start, min(start + batch_size, dataset.size)))
        prediction = actor.sample_actions(batch["observations"]).reshape(batch["actions"].shape)
        squared += float(np.sum((prediction - batch["actions"]) ** 2))
    return squared / dataset.arrays["noise"].size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/libero_offline.json")
    parser.add_argument("--tasks", nargs="+", type=int)
    parser.add_argument("--sources", nargs="+", choices=["standard", "recovery"])
    parser.add_argument("--output-dir", type=Path, help="Separate run directory, e.g. outputs/libero_offline/recovery_seed42")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.resolve())
    sources = args.sources or config["sources"]
    manifest = read_json(Path(config["cache_dir"]) / "manifest.json")
    if bridge_settings(config) != manifest["spec"]["bridge"] or config["horizon"] != manifest["spec"]["settings"]["horizon"]:
        raise ValueError("Training configuration must match the cached base model and horizon")
    cache_keys = ("split_seed", "validation_fraction", "stride", "tail", "fp_steps", "max_normalized_mse")
    if any(config[key] != manifest["spec"]["settings"][key] for key in cache_keys):
        raise ValueError("Split/window/inversion settings differ from the cache; select the matching config or recompute")
    if not set(sources) <= set(manifest["spec"]["settings"]["sources"]):
        raise ValueError("Requested sources are missing from this cache")
    if config["batch_size"] < 2:
        raise ValueError("Legacy feature BC requires batch_size >= 2")

    from .actor import make_actor, restore_actor, save_actor

    for task in args.tasks or config["tasks"]:
        train = LatentDataset(config["cache_dir"], manifest, task, "train", sources)
        val = LatentDataset(config["cache_dir"], manifest, task, "validation", sources)
        if set(train.groups) != set(sources) or set(val.groups) != set(sources):
            raise ValueError(f"task={task}: a requested source has no accepted training/validation windows")
        if config["output_bound"] == "tanh" and np.abs(train.arrays["noise"]).max() > config["action_magnitude"]:
            raise ValueError("Noise targets exceed the tanh range; choose linear or a larger action_magnitude")
        task_config = dict(config, seed=config["seed"] + task, sources=sources, tasks=[task])
        task_info = next(t for t in manifest["tasks"] if t["task_index"] == task)
        run = {"config": task_config, "task": task_info, "cache_fingerprint": manifest["fingerprint"],
               "checkpoint_identity": manifest["spec"]["checkpoint_identity"],
               "feature_dim": train.arrays["features"].shape[1],
               "train_windows": train.size, "validation_windows": val.size}
        directory = Path(config["output_dir"]) / f"task_{task:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if read_json(directory / "run.json") != run:
                raise ValueError("Resume requires the same run configuration and cache")
        else:
            if (directory / "run.json").exists():
                raise FileExistsError(f"Run exists: {directory}; use --resume or a new output_dir")
            write_json(directory / "run.json", run)
        actor = make_actor(task_config, run["feature_dim"])
        rng = np.random.default_rng(task_config["seed"])
        step = restore_actor(actor, rng, directory) if args.resume else 0
        if not args.resume:
            save_actor(actor, rng, directory, 0)
        with (directory / "metrics.jsonl").open("a") as log:
            for step in range(step + 1, config["train_steps"] + 1):
                indices = train.indices(rng, config["batch_size"], config["recovery_fraction"])
                info = actor.update(train.batch(indices))
                if step % config["log_every"] == 0 or step == config["train_steps"]:
                    metrics = {"task": task, "step": step, "train_loss": float(info["bc_loss"]),
                               "validation_loss": validation_loss(actor, val, config["batch_size"])}
                    line = json.dumps(metrics, allow_nan=False)
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                if step % config["save_every"] == 0 or step == config["train_steps"]:
                    save_actor(actor, rng, directory, step)


if __name__ == "__main__":
    main()
