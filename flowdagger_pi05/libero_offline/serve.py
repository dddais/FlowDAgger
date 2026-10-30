"""OpenPI websocket service: select a task's steering model, then run the frozen pi05 sampler."""

import argparse
from pathlib import Path

import numpy as np

from .common import bridge_settings, load_config, read_json


class TaskPolicy:
    def __init__(self, bridge, config, mode, run_root=None, tasks=None):
        self.bridge, self.mode = bridge, mode
        self.actors = {}
        pool = read_json(Path(config["dataset_root"]) / "data_pool.json")
        self.tasks = {t["task_key"]: t for t in pool["tasks"] if t["task_index"] in (tasks or config["tasks"])}
        self.checkpoints = {}
        if mode == "steering":
            from .actor import make_actor, restore_actor

            identity = bridge.identity()
            for key, task in self.tasks.items():
                directory = Path(run_root or config["output_dir"]) / f"task_{task['task_index']:02d}"
                run = read_json(directory / "run.json")
                if (run["task"]["task_key"] != key or run["task"]["official_task_language"] != task["official_task_language"]
                        or bridge_settings(run["config"]) != bridge_settings(config) or run["checkpoint_identity"] != identity):
                    raise ValueError(f"Steering/base checkpoint mismatch for {key}")
                actor = make_actor(run["config"], run["feature_dim"])
                restore_actor(actor, np.random.default_rng(), directory)
                self.actors[key] = actor
                self.checkpoints[key] = dict(read_json(directory / "latest.json"), run_dir=str(directory.resolve()),
                                             sources=run["config"]["sources"], seed=run["config"]["seed"],
                                             cache_fingerprint=run["cache_fingerprint"])

    def infer(self, request):
        task = self.tasks[request["task_key"]]
        if request["prompt"] != task["official_task_language"]:
            raise ValueError("Prompt must equal the dataset's official task.language")
        obs = {key: request[key] for key in ("observation/image", "observation/wrist_image", "observation/state", "prompt")}
        processed, _ = self.bridge.prepare([obs])
        if self.mode == "steering":
            features = self.bridge.features(processed)
            state = np.asarray(obs["observation/state"], dtype=np.float32)[None, :, None]
            noise = self.actors[task["task_key"]].sample_actions({"pixels": features[..., None], "state": state})
            noise = noise.reshape(1, self.bridge.horizon, self.bridge.action_dim)
        else:
            # Client-provided seed makes each trial independent of earlier inference requests.
            noise = np.random.default_rng(int(request["noise_seed"])).standard_normal(
                (1, self.bridge.horizon, self.bridge.action_dim), dtype=np.float32)
        _, actions = self.bridge.decode(processed, noise)
        return {"actions": actions[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/libero_offline.json")
    parser.add_argument("--mode", choices=["base", "steering"], default="steering")
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--run-root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18010)
    parser.add_argument("--disable-websocket-pings", action="store_true",
                        help="Allow long local blocking compilation/rendering; external worker enforces deadlines")
    args = parser.parse_args()
    config = load_config(args.config)
    from .bridge import LiberoBridge
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    bridge = LiberoBridge(config)
    policy = TaskPolicy(bridge, config, args.mode, args.run_root, args.tasks)
    metadata = {"protocol": "libero_offline_v1", "mode": args.mode, "action_horizon": bridge.horizon,
                "action_dim": 7, "task_keys": list(policy.tasks), "bridge": bridge_settings(config),
                "checkpoint_identity": bridge.identity(),
                "steering_checkpoints": policy.checkpoints}
    server_class = WebsocketPolicyServer
    if args.disable_websocket_pings:
        from .transport import NoPingPolicyServer
        server_class = NoPingPolicyServer
    metadata["websocket_pings_disabled"] = args.disable_websocket_pings
    server_class(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
