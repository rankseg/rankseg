"""Shared helpers for the MONAI PR #8908 experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"
RESULT_COLUMNS = [
    "case_id",
    "shape",
    "num_voxels",
    "argmax_dice",
    "rankdice_dice",
    "rankiou_dice",
    "argmax_iou",
    "rankdice_iou",
    "rankiou_iou",
    "argmax_pancreas_dice",
    "rankdice_pancreas_dice",
    "rankiou_pancreas_dice",
    "argmax_tumor_dice",
    "rankdice_tumor_dice",
    "rankiou_tumor_dice",
    "argmax_pancreas_iou",
    "rankdice_pancreas_iou",
    "rankiou_pancreas_iou",
    "argmax_tumor_iou",
    "rankdice_tumor_iou",
    "rankiou_tumor_iou",
    "argmax_time_ms",
    "rankdice_time_ms",
    "rankiou_time_ms",
]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = str(path)
    return config


def root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def output_dir(config: dict[str, Any]) -> Path:
    return root_path(config["outputs"]["directory"])


def probability_dir(config: dict[str, Any]) -> Path:
    return root_path(config["outputs"]["probabilities"])


def case_cache_path(config: dict[str, Any], case_id: str) -> Path:
    return probability_dir(config) / f"{case_id}.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_dtype(name: str) -> torch.dtype:
    options = {"float16": torch.float16, "float32": torch.float32}
    if name not in options:
        raise ValueError(f"Unsupported cache dtype {name!r}; choose one of {sorted(options)}")
    return options[name]


def load_case(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    path = case_cache_path(config, case_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing probability cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported cache schema in {path}")
    if payload.get("case_id") != case_id:
        raise ValueError(f"Cache case ID mismatch in {path}")
    return payload


def validate_probability_payload(payload: dict[str, Any], num_classes: int) -> None:
    probabilities = payload["probabilities"]
    label = payload["label"]
    image = payload["image"]
    if probabilities.ndim != label.ndim + 1:
        raise ValueError(f"Probability/label dimensionality mismatch: {probabilities.shape}, {label.shape}")
    if probabilities.shape[0] != num_classes:
        raise ValueError(f"Expected {num_classes} probability channels, got {probabilities.shape[0]}")
    if tuple(probabilities.shape[1:]) != tuple(label.shape) or tuple(image.shape) != tuple(label.shape):
        raise ValueError(
            f"Spatial shape mismatch: probabilities={probabilities.shape}, image={image.shape}, label={label.shape}"
        )
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Probability cache contains non-finite values")
    if float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise ValueError("Probability cache is outside [0, 1]")
    sums = probabilities.float().sum(dim=0)
    if not torch.allclose(sums, torch.ones_like(sums), atol=2e-4, rtol=2e-4):
        raise ValueError("Probability channels do not sum to one")
    labels = set(int(value) for value in torch.unique(label))
    if not labels.issubset(set(range(num_classes))):
        raise ValueError(f"Illegal labels: {sorted(labels)}")


def resolve_device(requested: str | None, configured: str) -> torch.device:
    name = requested or configured
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Pass --device cpu only for a smoke test.")
    return device


def software_info() -> dict[str, Any]:
    from importlib.metadata import version

    import monai

    import rankseg

    info: dict[str, Any] = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "monai": monai.__version__,
        "monai_path": str(Path(monai.__file__).resolve()),
        "rankseg": version("rankseg"),
        "rankseg_path": str(Path(rankseg.__file__).resolve()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info
