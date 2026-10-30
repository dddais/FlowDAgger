"""One frozen JAX policy for observation transforms, VLM features and noise inversion."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from .common import digest


class LiberoBridge:
    def __init__(self, config):
        train_config = openpi_config.get_config(config["openpi_config"])
        if config["openpi_config"] != "pi05_libero":
            raise ValueError("This adapter uses the pi05_libero action and observation contract")
        self.checkpoint = Path(download.maybe_download(config["checkpoint"]))
        if not (self.checkpoint / "params").is_dir() or (self.checkpoint / "model.safetensors").exists():
            raise ValueError("Use a JAX params/ checkpoint, without PyTorch model.safetensors")
        self.policy = policy_config.create_trained_policy(
            train_config, self.checkpoint, sample_kwargs={"num_steps": config["denoise_steps"]}
        )
        self.horizon = self.policy.action_horizon
        self.action_dim = self.policy.action_dim
        self.steps = config["denoise_steps"]
        self.inverter = None

    def identity(self):
        # No multi-GB weight reads: record checkpoint file inventory and hash its normalization assets.
        files = [(str(p.relative_to(self.checkpoint)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted((self.checkpoint / "params").rglob("*")) if p.is_file()]
        stats = {str(p.relative_to(self.checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((self.checkpoint / "assets").rglob("*.json"))}
        return {"path": str(self.checkpoint), "params_inventory": digest(files), "assets": stats}

    def prepare(self, observations, actions=None):
        processed = []
        for index, obs in enumerate(observations):
            value = dict(obs)
            if actions is not None:
                value["actions"] = actions[index]
            processed.append(self.policy._input_transform(value))
        inputs = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *processed)
        target = inputs.pop("actions", None)
        return model_api.Observation.from_dict(inputs), target

    def features(self, processed):
        hidden, _ = self.policy._get_prefix_rep(processed)
        # Match FlowDAgger's vlm_pi0 mean pooling, including its padding-token convention.
        return np.asarray(hidden.mean(axis=1), dtype=np.float32)

    def decode(self, processed, noise):
        internal = self.policy._sample_actions(processed, noise=jnp.asarray(noise), num_steps=self.steps)
        env = self.policy._output_transform({"state": np.asarray(processed.state), "actions": np.asarray(internal)})
        return np.asarray(internal), np.asarray(env["actions"], dtype=np.float32)

    def invert(self, observations, actions, fp_steps):
        if self.inverter is None:
            from flow_matching_inverter import FlowMatchingInverter

            self.inverter = FlowMatchingInverter(self.policy._model, method="perstep_fp", solver="euler",
                                                num_denoise_steps=self.steps, fp_per_step=fp_steps)
        processed, target = self.prepare(observations, actions)
        noise, _ = self.inverter.invert(processed, target)
        # Quality is checked with the actual policy sampler, not only the inverter's forward routine.
        reconstructed, env_actions = self.decode(processed, noise)
        error = np.mean((reconstructed[..., :7] - np.asarray(target)[..., :7]) ** 2, axis=(1, 2))
        env_error = np.mean((env_actions - actions) ** 2, axis=1)
        grip_match = np.mean((env_actions[..., 6] > 0) == (actions[..., 6] > 0), axis=1)
        return self.features(processed), np.asarray(noise, dtype=np.float32), error, env_error, grip_match
