"""Run directories, JSON and tensor files, and the per-model disk cache."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file, save_file

from vectorlab.model import slug

log = logging.getLogger(__name__)


def save_json(path: Path, obj) -> None:
    """Write ``obj`` as indented JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def load_json(path: Path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write named tensors to a safetensors file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    """Read every tensor from a safetensors file."""
    return load_file(str(path))


def run_dir(spec: dict, model_id: str) -> Path:
    """Directory holding the run for a behavior on a model."""
    return Path("runs") / f"{spec['name']}-{slug(model_id)}"


def cache_path(model_id: str, kind: str, *key) -> Path:
    """Cache location (without suffix) for a ``kind`` of result keyed by JSON-serializable ``key`` parts."""
    digest = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]
    return Path("cache") / slug(model_id) / f"{kind}-{digest}"


def cached_tensor(path: Path, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Load a cached tensor, computing and saving it first if it is missing."""
    path = path.with_suffix(".safetensors")
    if not path.exists():
        log.info("cache miss: %s", path)
        save_tensors(path, {"x": compute()})
    return load_tensors(path)["x"]


def cached_json(path: Path, compute: Callable[[], object]):
    """Load a cached JSON value, computing and saving it first if it is missing."""
    path = path.with_suffix(".json")
    if not path.exists():
        log.info("cache miss: %s", path)
        save_json(path, compute())
    return load_json(path)
