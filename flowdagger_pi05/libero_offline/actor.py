"""Reuse the existing steering model; store complete training state without altering legacy checkpoints."""

import json
from pathlib import Path

import numpy as np
from flax import serialization
from flax.core import freeze

from steering_policy import SteeringPolicy

from .common import read_json, write_json


def make_actor(config, feature_dim=2048):
    return SteeringPolicy(
        seed=config["seed"],
        observations=freeze({"pixels": np.zeros((2, feature_dim, 1), np.float32), "state": np.zeros((2, 8, 1), np.float32)}),
        actions=np.zeros((2, config["horizon"], 32), np.float32),
        lr=config["learning_rate"], hidden_dims=tuple(config["hidden_dims"]), latent_dim=config["latent_dim"],
        encoder_type="vlm_pi0", num_cameras=2, color_jitter=False,
        output_bound=config["output_bound"], action_magnitude=config["action_magnitude"],
    )


def save_actor(actor, rng, directory, step):
    directory = Path(directory)
    value = {"actor": actor._actor, "rng": actor._rng, "sampler_rng": json.dumps(rng.bit_generator.state)}
    name = f"step_{step:08d}.msgpack"
    temporary = directory / (name + ".tmp")
    temporary.write_bytes(serialization.to_bytes(value))
    temporary.replace(directory / name)
    write_json(directory / "latest.json", {"file": name, "step": step})


def restore_actor(actor, rng, directory):
    directory = Path(directory)
    latest = read_json(directory / "latest.json")
    template = {"actor": actor._actor, "rng": actor._rng, "sampler_rng": json.dumps(rng.bit_generator.state)}
    restored = serialization.from_bytes(template, (directory / latest["file"]).read_bytes())
    actor._actor, actor._rng = restored["actor"], restored["rng"]
    rng.bit_generator.state = json.loads(restored["sampler_rng"])
    return latest["step"]
