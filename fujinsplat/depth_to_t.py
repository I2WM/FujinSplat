"""Explicit-weight Depth-Anything-V2-Small inference, matching the uploaded code."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .io import read_json, sha256, write_json


def build_model(weights):
    from safetensors.torch import load_file
    from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
    cfg = DepthAnythingConfig(
        backbone_config=dict(model_type="dinov2", hidden_size=384, num_hidden_layers=12,
            num_attention_heads=6, patch_size=14, image_size=518,
            out_features=["stage3", "stage6", "stage9", "stage12"],
            out_indices=[3, 6, 9, 12], reshape_hidden_states=False),
        patch_size=14, neck_hidden_sizes=[48, 96, 192, 384],
        reassemble_hidden_size=384, reassemble_factors=[4, 2, 1, 0.5],
        fusion_hidden_size=64, head_hidden_size=32, head_in_index=-1)
    model = DepthAnythingForDepthEstimation(cfg)
    missing, unexpected = model.load_state_dict(load_file(str(weights)), strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    if missing or unexpected:
        raise ValueError(f"depth checkpoint/architecture mismatch: {missing}, {unexpected}")
    return model.eval().requires_grad_(False)


def preprocess(image):
    x = cv2.resize(np.clip(image, 0, 1), (518, 518), interpolation=cv2.INTER_AREA)
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(x.transpose(2, 0, 1)[None]).float()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--j-cache", type=Path, required=True)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    pop = read_json(a.population)
    if len(pop["t"]) != 195:
        raise ValueError("no fallback to the old 25-pair population")
    model = build_model(a.weights).to(a.device)
    rows = []
    for path in sorted(a.j_cache.glob("*.npy")):
        j = np.load(path, allow_pickle=False).astype(np.float64)
        inverse = model(pixel_values=preprocess(j).to(a.device)).predicted_depth[0].cpu().numpy()
        depth = 1 / np.clip(inverse, 1e-6, None)
        low, high = np.percentile(depth, (10, 90))
        rows.append((path.stem, float(np.log(max(high, 1e-9) / max(low, 1e-9))), float(np.median(depth))))
    if not rows or np.median([r[1] for r in rows]) <= 0:
        raise ValueError("empty or degenerate depth population")
    beta = -np.log(np.median(pop["t"])) / np.median([r[1] for r in rows])
    write_json(a.output, {"beta": float(beta), "statistic": "log(p90/p10 of depth)",
               "depth_model": "Depth-Anything-V2-Small", "weights_sha256": sha256(a.weights),
               "population_sha256": sha256(a.population),
               "captures": {n: {"d_bar": d, "t": float(np.exp(-beta * d)), "median_depth": m} for n, d, m in rows}})


if __name__ == "__main__":
    main()
