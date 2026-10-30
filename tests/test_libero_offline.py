import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flowdagger_pi05.libero_offline.common import load_config
from flowdagger_pi05.libero_offline.data import LatentDataset, episodes, observation, window_starts


@pytest.fixture
def dataset(tmp_path):
    pool = {"cache_root": "data/cache", "tasks": [], "standard": {"records": []}, "recovery": {"records": []}}
    for task in range(2):
        prompt = f"task {task}"
        pool["tasks"].append({"task_index": task, "task_key": f"suite/task{task}", "official_task_language": prompt})
        for category, source in [("standard", "original"), ("recovery", "recovery")]:
            for index in range(5):
                record = {"task_index": task, "source": source, "episode_index": task * 5 + index, "length": 23, "prompt": prompt}
                pool[category]["records"].append(record)
    (tmp_path / "data_pool.json").write_text(json.dumps(pool))
    return {"dataset_root": str(tmp_path), "tasks": [0, 1], "sources": ["standard", "recovery"],
            "split_seed": 42, "validation_fraction": 0.2}


def test_episode_split_is_stable_and_disjoint(dataset):
    _, full = episodes(dataset)
    _, selected = episodes(dict(dataset, tasks=[1], sources=["recovery"]))
    expected = [r for r in full if r["task_index"] == 1 and r["category"] == "recovery"]
    assert selected == expected
    assert len({r["id"] for r in full}) == 20
    assert sum(r["split"] == "validation" for r in full) == 4
    assert {r["id"] for r in full if r["split"] == "train"}.isdisjoint(r["id"] for r in full if r["split"] == "validation")


def test_windows_cover_tail_without_padding_or_crossing_episode():
    assert window_starts(23, 10, 10, "drop") == [0, 10]
    assert window_starts(23) == [0, 10, 13]
    assert window_starts(20) == [0, 10]
    assert window_starts(9) == []


def test_cache_observation_does_not_rotate_or_change_state():
    arrays = {"images": np.arange(1 * 2 * 2 * 3 * 3, dtype=np.uint8).reshape(1, 2, 2, 3, 3),
              "state": np.arange(8, dtype=np.float32)[None]}
    obs = observation(arrays, 0, "pick up")
    np.testing.assert_array_equal(obs["observation/image"], arrays["images"][0, 0])
    np.testing.assert_array_equal(obs["observation/wrist_image"], arrays["images"][0, 1])
    np.testing.assert_array_equal(obs["observation/state"], arrays["state"][0])
    assert obs["prompt"] == "pick up"


def test_latent_source_selection_and_episode_balancing(tmp_path):
    (tmp_path / "shards").mkdir()
    records = []
    for index, (category, count) in enumerate([("standard", 1), ("standard", 99), ("recovery", 2)]):
        identity = str(index)
        np.savez(tmp_path / "shards" / f"{identity}.npz", features=np.full((count, 16), index, np.float32),
                 state=np.zeros((count, 8), np.float32), noise=np.zeros((count, 10, 32), np.float32))
        records.append({"id": identity, "category": category, "task_index": 0, "split": "train"})
    manifest = {"episodes": records}
    selected = LatentDataset(tmp_path, manifest, 0, "train", ["recovery"])
    assert selected.size == 2 and set(selected.groups) == {"recovery"}
    mixed = LatentDataset(tmp_path, manifest, 0, "train", ["standard", "recovery"])
    indices = mixed.indices(np.random.default_rng(1), 10000, 0.5)
    labels = mixed.arrays["features"][indices, 0]
    assert 0.45 < np.mean(labels == 2) < 0.55
    assert 0.20 < np.mean(labels == 0) < 0.30  # one short episode has equal weight to the long episode


def test_bridge_normalization_padding_and_distinct_prompts():
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.libero_policy import LiberoInputs
    from flowdagger_pi05.libero_offline.bridge import LiberoBridge

    stats = {key: transforms.NormStats(mean=np.zeros(n), std=np.ones(n), q01=-np.ones(n), q99=np.ones(n))
             for key, n in [("actions", 7), ("state", 8)]}
    def tokens(data):
        data["tokenized_prompt"] = np.array([len(data.pop("prompt"))], np.int32)
        data["tokenized_prompt_mask"] = np.array([True])
        return data
    bridge = object.__new__(LiberoBridge)
    bridge.policy = SimpleNamespace(_input_transform=transforms.compose([
        LiberoInputs(ModelType.PI05), transforms.Normalize(stats, use_quantiles=True), tokens,
        transforms.PadStatesAndActions(32),
    ]))
    obs = {"observation/image": np.zeros((224, 224, 3), np.uint8),
           "observation/wrist_image": np.full((224, 224, 3), 255, np.uint8),
           "observation/state": np.arange(8, dtype=np.float32), "prompt": "a"}
    actions = np.full((2, 10, 7), 0.25, np.float32)
    processed, target = bridge.prepare([obs, dict(obs, prompt="long")], actions)
    np.testing.assert_array_equal(processed.tokenized_prompt[:, 0], [1, 4])
    np.testing.assert_allclose(target[..., :7], 0.25, atol=1e-6)
    np.testing.assert_array_equal(target[..., 7:], 0)
    np.testing.assert_array_equal(processed.images["base_0_rgb"], -1)
    np.testing.assert_array_equal(processed.images["left_wrist_0_rgb"], 1)


def test_real_steering_update_and_exact_resume(tmp_path):
    from flax.core import freeze
    from flowdagger_pi05.libero_offline.actor import make_actor, restore_actor, save_actor

    config = load_config(Path(__file__).parents[1] / "config/libero_offline.json")
    config.update(hidden_dims=[16, 16], latent_dim=8, learning_rate=1e-3)
    actor = make_actor(config, feature_dim=16)
    rng = np.random.default_rng(7)
    obs = freeze({"pixels": rng.normal(size=(8, 16, 1)).astype(np.float32), "state": np.zeros((8, 8, 1), np.float32)})
    target = np.full((8, 10, 32), 0.1, np.float32)
    batch = freeze({"observations": obs, "actions": target})
    before = np.mean((actor.sample_actions(obs).reshape(target.shape) - target) ** 2)
    for _ in range(15):
        actor.update(batch)
    after = np.mean((actor.sample_actions(obs).reshape(target.shape) - target) ** 2)
    assert after < before
    save_actor(actor, rng, tmp_path, 15)
    restored = make_actor(config, feature_dim=16)
    restored_rng = np.random.default_rng(999)
    assert restore_actor(restored, restored_rng, tmp_path) == 15
    np.testing.assert_array_equal(actor.sample_actions(obs), restored.sample_actions(obs))
    np.testing.assert_array_equal(rng.integers(100, size=10), restored_rng.integers(100, size=10))
    actor.update(batch)
    restored.update(batch)
    np.testing.assert_array_equal(actor.sample_actions(obs), restored.sample_actions(obs))


def test_base_service_reproducible_noise_and_prompt_contract(tmp_path):
    from flowdagger_pi05.libero_offline.serve import TaskPolicy

    task = {"task_index": 0, "task_key": "suite/task0", "official_task_language": "pick up"}
    (tmp_path / "data_pool.json").write_text(json.dumps({"tasks": [task]}))
    class Bridge:
        horizon, action_dim = 10, 32
        def prepare(self, observations):
            return observations, None
        def decode(self, processed, noise):
            return noise, noise[..., :7]
    policy = TaskPolicy(Bridge(), {"dataset_root": str(tmp_path), "tasks": [0]}, "base")
    request = {"task_key": task["task_key"], "prompt": "pick up", "noise_seed": 42,
               "observation/image": None, "observation/wrist_image": None, "observation/state": np.zeros(8)}
    first = policy.infer(request)["actions"]
    policy.infer(dict(request, noise_seed=99))
    np.testing.assert_array_equal(first, policy.infer(request)["actions"])
    with pytest.raises(ValueError, match="Prompt"):
        policy.infer(dict(request, prompt="pick"))


def test_offline_commands_filter_cache_and_train_separate_tasks(dataset, tmp_path, monkeypatch):
    """Exercise command orchestration with a tiny substitute for the expensive frozen VLA."""
    from flowdagger_pi05.libero_offline import bridge, precompute, train

    class TinyBridge:
        horizon, action_dim = 10, 32
        def __init__(self, config):
            pass
        def identity(self):
            return {"fixture": "tiny-vla"}
        def prepare(self, observations):
            return observations, None
        def features(self, observations):
            return np.ones((len(observations), 16), np.float32)
        def decode(self, processed, noise):
            return noise, noise[..., :7]
        def invert(self, observations, actions, fp_steps):
            n = len(observations)
            errors = np.zeros(n, np.float32)
            errors[-1] = 1  # rejected windows must not become BC labels
            return (np.ones((n, 16), np.float32), np.full((n, 10, 32), 0.1, np.float32),
                    errors, np.zeros((n, 7), np.float32), np.ones(n, np.float32))
    monkeypatch.setattr(bridge, "LiberoBridge", TinyBridge)
    pool, records = episodes(dataset)
    for record in records:
        directory = tmp_path / pool["cache_root"] / record["source"] / f"{record['episode_index']:06d}"
        directory.mkdir(parents=True)
        np.save(directory / "images.npy", np.zeros((23, 2, 2, 2, 3), np.uint8))
        np.save(directory / "state.npy", np.zeros((23, 8), np.float32))
        np.save(directory / "actions.npy", np.zeros((23, 7), np.float32))
    (tmp_path / "array_checksums.json").write_text("{}")
    config = load_config(Path(__file__).parents[1] / "config/libero_offline.json")
    config.update(dataset, cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "runs"),
                  train_steps=2, log_every=1, save_every=1, batch_size=4, hidden_dims=[16, 16], latent_dim=8)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", ["precompute", "--config", str(config_path)])
    precompute.main()
    manifest = json.loads((tmp_path / "cache/manifest.json").read_text())
    assert all(r["accepted_windows"] == 2 and r["total_windows"] == 3 for r in manifest["episodes"])
    # Re-running precompute should reuse the same shards, not invoke inversion.
    monkeypatch.setattr(TinyBridge, "invert", lambda *a: pytest.fail("completed shard was recomputed"))
    precompute.main()
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config_path), "--sources", "recovery"])
    train.main()
    for index in range(2):
        directory = tmp_path / "runs" / f"task_{index:02d}"
        run = json.loads((directory / "run.json").read_text())
        assert run["task"]["task_index"] == index
        assert run["config"]["sources"] == ["recovery"]
        assert run["train_windows"] == 8 and run["validation_windows"] == 2
        assert json.loads((directory / "latest.json").read_text())["step"] == 2
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config_path), "--sources", "recovery", "--resume"])
    train.main()
    from flowdagger_pi05.libero_offline.serve import TaskPolicy
    service = TaskPolicy(TinyBridge(config), config, "steering")
    assert service.actors["suite/task0"] is not service.actors["suite/task1"]
    for task in pool["tasks"]:
        result = service.infer({"task_key": task["task_key"], "prompt": task["official_task_language"],
                                "observation/image": None, "observation/wrist_image": None,
                                "observation/state": np.zeros(8, np.float32)})
        assert result["actions"].shape == (10, 7) and np.isfinite(result["actions"]).all()
        assert service.checkpoints[task["task_key"]]["sources"] == ["recovery"]
