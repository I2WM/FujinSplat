"""Small, hash-bound training inputs; identical RAW summary and pixel samples."""

from pathlib import Path
import numpy as np
import torch

from .color import summary
from .io import sha256


def write_cache(arrays, index, path):
    raw = torch.from_numpy(np.asarray(arrays["X_syn"], np.float32)).permute(2, 0, 1)[None]
    low = summary(raw)[0].numpy()
    h, w = arrays["H"].shape[:2]
    ids = np.random.default_rng(index).choice(h * w, 4096, replace=h * w < 4096)
    values = {"raw": low, "base": arrays["H"].reshape(-1, 3)[ids].T[:, None, :],
              "target": arrays["J"].reshape(-1, 3)[ids].T[:, None, :], "p": arrays["p"]}
    if "valid" in arrays:
        values["valid"] = arrays["valid"].reshape(-1)[ids][None, None, :].astype(np.float32)
    path = Path(path)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)
    return {"training_pair": str(path.resolve()), "training_pair_sha256": sha256(path)}
