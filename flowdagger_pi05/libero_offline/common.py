"""Configuration and small on-disk contracts shared by the offline commands."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = read_json(path)
    for key in ("dataset_root", "cache_dir", "output_dir"):
        p = Path(config[key]).expanduser()
        config[key] = str(p if p.is_absolute() else ROOT / p)
    checkpoint = config["checkpoint"]
    if "://" not in checkpoint:
        p = Path(checkpoint).expanduser()
        config["checkpoint"] = str(p if p.is_absolute() else ROOT / p)
    return config


def bridge_settings(config):
    return {key: config[key] for key in ("checkpoint", "openpi_config", "denoise_steps")}
