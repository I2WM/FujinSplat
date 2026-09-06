"""Synthetic-only training datasets. Deployment never imports this module."""

import numpy as np
import torch

from .color import summary
from .io import read_json, sha256
from .mcf import MCFConfig


def validate_metadata(manifest):
    if manifest.get("schema") != "fujinsplat.synthetic.v1":
        raise ValueError("only the external synthetic dataset schema is supported")
    if manifest.get("held_target_reads") != 0:
        raise ValueError("training manifest must declare held_target_reads=0")
    MCFConfig(**manifest["mcf"]).validate()
    rows = manifest["rows"]
    if manifest.get("external_captures") != 1400 or len(rows) != 1400:
        raise ValueError("expected 1400 external captures, one draw each")
    if len({r["capture_id"] for r in rows}) != 1400:
        raise ValueError("synthetic capture duplication")
    if manifest.get("realx_image_reads") != 0:
        raise ValueError("synthetic image inputs must be external")
    if manifest.get("stored_array_cycle_gate") is not True:
        raise ValueError("synthetic labels have not passed the stored-array cycle gate")
    return manifest


def validate_manifest(path, stage="synthetic_pretrain"):
    if stage not in ("synthetic_pretrain", "synthetic_warm_start"):
        raise ValueError("unsupported training stage")
    return validate_metadata(read_json(path))


class SyntheticPairs:
    """Load only hash-bound synthetic NPZs; target images are synthetic endpoints."""

    def __init__(self, manifest):
        self.rows = validate_metadata(manifest)["rows"]
        self.cache, self.hashes = {}, {}

    def load(self, index):
        if index in self.cache:
            return self.cache[index]
        row = self.rows[index]
        if "training_pair" in row:
            digest = sha256(row["training_pair"])
            if digest != row["training_pair_sha256"]:
                raise ValueError("compact training cache hash drift")
            with np.load(row["training_pair"], allow_pickle=False) as z:
                pair = {k: torch.from_numpy(np.asarray(z[k], np.float32).copy()) for k in z.files}
            if set(pair) not in ({"raw", "base", "target", "p"}, {"raw", "base", "target", "p", "valid"}):
                raise ValueError("unexpected compact training cache fields")
            self.hashes[index] = {"training_pair": digest}
        else:
            digest = sha256(row["pair"])
            if digest != row["sha256"]:
                raise ValueError("synthetic pair hash drift")
            with np.load(row["pair"], allow_pickle=False) as z:
                pair = {k: torch.from_numpy(np.asarray(z[src], np.float32).copy())
                        for k, src in (("raw", "X_syn"), ("base", "H"), ("target", "J"))}
                # Summarize RAW only; use exactly matched H/J pixel locations.
                pair["raw"] = summary(pair["raw"].permute(2, 0, 1)[None])[0]
                h, w = pair["base"].shape[:2]
                if pair["target"].shape != pair["base"].shape:
                    raise ValueError("synthetic H/J shape mismatch")
                indices = np.random.default_rng(index).choice(h * w, 4096, replace=h * w < 4096)
                for key in ("base", "target"):
                    pair[key] = pair[key].reshape(-1, 3)[indices].T[:, None, :]
                pair["p"] = torch.from_numpy(np.asarray(z["p"], np.float32).copy())
            self.hashes[index] = {"pair": digest}
        if pair["raw"].shape != (3, 64, 64) or pair["p"].shape != (573,):
            raise ValueError("synthetic training cache shape mismatch")
        if pair["base"].ndim != 3 or pair["base"].shape[0] != 3 or pair["target"].shape != pair["base"].shape:
            raise ValueError("synthetic reconstruction shape mismatch")
        if not all(torch.isfinite(v).all() for v in pair.values()):
            raise ValueError("nonfinite training input")
        self.cache[index] = pair
        return pair

    def batch(self, indices, device):
        values = [self.load(int(i)) for i in indices]
        return {k: torch.stack([v[k] for v in values]).to(device) for k in values[0]}
