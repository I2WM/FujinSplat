"""One scene-agnostic RAW-summary-to-action network; fresh initialization."""

from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn

from .mcf import CONFIG, MCFConfig, pack


SCHEMA = "fujinsplat.controller.v1"


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GELU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.trunk = nn.Sequential(nn.Linear(2048, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU())
        self.curve_heads = nn.ModuleList([nn.Linear(256, 48) for _ in range(8)])
        self.coupling_head = nn.Linear(256, 189)
        for head in [*self.curve_heads, self.coupling_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, raw):
        if raw.ndim != 4 or raw.shape[1:] != (3, 64, 64):
            raise ValueError("controller consumes only B,3,64,64 RAW summaries")
        feature = self.trunk(self.encoder(raw))
        curves = torch.stack([head(feature).reshape(-1, 3, 16) for head in self.curve_heads], 1)
        couplings = self.coupling_head(feature).reshape(-1, 7, 3, 9)
        return pack(curves, couplings)


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
        payload = json.loads(path.with_suffix(".json").read_text())
        from .io import sha256
        if payload.get("weights_sha256") != sha256(path):
            raise ValueError("weights/sidecar hash mismatch")
        payload["state_dict"] = value
    else:
        payload = value  # Backward compatibility for local unit fixtures only.
    if payload.get("schema") != SCHEMA or payload.get("status") != "SEALED":
        raise ValueError("expected a sealed checkpoint with the supported Controller schema")
    MCFConfig(**payload["mcf"]).validate()
    if payload.get("held_target_reads") != 0 or payload.get("scene_id_inputs") != 0:
        raise ValueError("checkpoint contact contract mismatch")
    # Keep training provenance separate from command dispatch.
    if not isinstance(payload.get("training_kind"), str) or not payload["training_kind"]:
        raise ValueError("missing controller training lineage")
    model = Controller().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def load_weights(path, device="cpu", *, expected_sha256=None):
    """Public inference: plain tensor dictionary, no training sidecar required."""
    if expected_sha256 is not None:
        from .io import sha256
        if sha256(path) != expected_sha256:
            raise ValueError("controller SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if type(state) is not dict or not state or not all(type(v) is torch.Tensor for v in state.values()):
        raise ValueError("expected a tensor-only state_dict")
    if hasattr(state, "_metadata") or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("unexpected metadata or nonfinite controller weights")
    model = Controller().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def checkpoint_payload(model, *, training_kind, initial_hash, manifest_hash, schedule):
    return {
        "schema": SCHEMA, "status": "SEALED", "mcf": asdict(CONFIG),
        "training_kind": training_kind,
        "current_training_data": "external_synthetic",
        "uses_realx_calibrated_synthesis_population": True,
        "initial_checkpoint_sha256": initial_hash,
        "training_manifest_sha256": manifest_hash,
        "schedule": schedule, "held_target_reads": 0, "scene_id_inputs": 0,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
