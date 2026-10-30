"""Episode-level splits and full action windows over the existing NumPy cache."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import read_json


def window_starts(length, horizon=10, stride=10, tail="last_full"):
    if not 1 <= stride <= horizon:
        raise ValueError("stride must be in [1, horizon]")
    if tail not in ("drop", "last_full"):
        raise ValueError("tail must be drop or last_full")
    starts = list(range(0, length - horizon + 1, stride))
    if tail == "last_full" and starts and starts[-1] != length - horizon:
        starts.append(length - horizon)
    return starts


def episode_id(record):
    return f"{record['source']}_{record['episode_index']:06d}"


def episodes(config):
    pool = read_json(Path(config["dataset_root"]) / "data_pool.json")
    groups = defaultdict(list)
    for source in config["sources"]:
        for record in pool[source]["records"]:
            if record["task_index"] in config["tasks"]:
                groups[(record["task_index"], source)].append(dict(record, category=source))
    result = []
    for (task, source), records in sorted(groups.items()):
        records.sort(key=episode_id)
        # Independent group RNG keeps a task's split stable when selecting other tasks/sources.
        rng = np.random.default_rng(np.random.SeedSequence([config["split_seed"], task, source == "recovery"]))
        n_val = int(np.ceil(len(records) * config["validation_fraction"]))
        validation = set(rng.permutation(len(records))[:n_val].tolist())
        for index, record in enumerate(records):
            record["split"] = "validation" if index in validation else "train"
            record["id"] = episode_id(record)
            result.append(record)
    return pool, result


def load_episode(root, cache_root, record):
    directory = Path(root) / cache_root / record["source"] / f"{record['episode_index']:06d}"
    return {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ("images", "state", "actions")}


def observation(arrays, time, prompt):
    # Cache images and gripper commands have already been converted to LIBERO conventions.
    return {
        "observation/image": np.asarray(arrays["images"][time, 0]),
        "observation/wrist_image": np.asarray(arrays["images"][time, 1]),
        "observation/state": np.asarray(arrays["state"][time], dtype=np.float32),
        "prompt": prompt,
    }


class LatentDataset:
    """Accepted windows of one task; episode-balanced sampling within each source."""

    def __init__(self, cache, manifest, task, split, sources):
        blocks = defaultdict(list)
        self.metadata = []
        self.groups = defaultdict(list)
        offset = 0
        for record in manifest["episodes"]:
            if record["task_index"] != task or record["split"] != split or record["category"] not in sources:
                continue
            with np.load(Path(cache) / "shards" / (record["id"] + ".npz")) as shard:
                count = len(shard["noise"])
                if not count:
                    continue
                for key in ("features", "state", "noise"):
                    blocks[key].append(shard[key])
            self.groups[record["category"]].append(np.arange(offset, offset + count))
            self.metadata.append(record)
            offset += count
        if not offset:
            raise ValueError(f"No accepted windows: task={task}, split={split}, sources={sources}")
        self.arrays = {key: np.concatenate(value) for key, value in blocks.items()}
        self.size = offset

    def indices(self, rng, batch_size, recovery_fraction):
        sources = list(self.groups)
        if len(sources) == 2:
            chosen = np.where(rng.random(batch_size) < recovery_fraction, "recovery", "standard")
        else:
            chosen = [sources[0]] * batch_size
        return np.asarray([rng.choice(self.groups[s][rng.integers(len(self.groups[s]))]) for s in chosen])

    def batch(self, indices):
        from flax.core import freeze

        return freeze({
            "observations": {
                "pixels": self.arrays["features"][indices, :, None],
                "state": self.arrays["state"][indices, :, None],
            },
            "actions": self.arrays["noise"][indices],
        })
