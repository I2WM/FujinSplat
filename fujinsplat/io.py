"""Create-only outputs and small provenance helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SCENE_COUNTS = dict(zip(
    ("Akikaze", "Futaba", "Hinoki", "Koharu", "Midori", "Natsume", "Shirohana", "Tsubaki"),
    (25, 23, 22, 25, 26, 25, 26, 23),
))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def create_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def require_source(scene, stem):
    if scene not in SCENE_COUNTS or len(stem) != 4 or not stem.isdigit():
        raise ValueError("unknown scene or non-four-digit source stem")
    if not 1 <= int(stem) <= SCENE_COUNTS[scene]:
        raise ValueError(f"held/non-source stem forbidden: {scene}/{stem}")


def load_raw(path):
    """Read only an explicitly supplied NPZ; no WB, CCM, gamma or auto-gain."""
    with np.load(path, allow_pickle=False) as z:
        raw = np.asarray(z["linear_rgb"])
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("linear_rgb must be H,W,3")
    if raw.dtype == np.uint16:
        raw = raw.astype(np.float32) / 65535.0
    elif np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(np.float32)
    else:
        raise ValueError("unsupported RAW cache dtype")
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite RAW")
    return torch.from_numpy(np.ascontiguousarray(raw)).permute(2, 0, 1)
