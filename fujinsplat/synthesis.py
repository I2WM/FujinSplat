"""Reverse synthesis with exact labels from the MCF parameter compiler.

Labels describe the realized bounded MCF, not necessarily textbook affine haze.
"""
from dataclasses import asdict
import numpy as np
import torch
from . import compiler as C
from .color import decode, encode
from .mcf import CONFIG, apply

COMPILER = {
    "schema": "fujinsplat.compiler.supplement.v1",
    "source_function": "generate_final.py::compile_from_raw",
    "knee_width": 0.02, "minimum_increment": 1e-6,
    "logit_limit": 5.0, "coupling_bound": 0.155,
    "shared_curve_block": 7, "black_floor_ordinate_threshold": 0.001,
    "gain_cap": 10.0, "sampling_seed": 20260902,
    "inverse_domain": "unclipped floating linear and encoded intermediates; summary clamp only",
}


def depth_contrasts(depth_maps, target_median=0.36):
    """Calibrate contrast from log depth ranges using population medians."""
    stats = []
    for depth in depth_maps:
        depth = np.asarray(depth)
        if not np.isfinite(depth).all() or bool((depth <= 0).any()):
            raise ValueError("positive finite depth required")
        low, high = np.percentile(depth, (10, 90))
        stats.append(float(np.log(max(high, 1e-9) / max(low, 1e-9))))
    stats = np.asarray(stats)
    if not 0 < target_median < 1 or np.median(stats) <= 0:
        raise ValueError("invalid median/depth range")
    beta = -np.log(target_median) / float(np.median(stats))
    return np.exp(-beta * stats), beta, stats


def compile_action(c_base, t):
    c = np.asarray(c_base, dtype=np.float64)
    if c.shape != (3,) or not np.isfinite(c).all() or not 0 < t <= 1:
        raise ValueError("finite RGB pivot and 0<t<=1 required")
    act = C.compile_from_raw(c, t)
    p = torch.from_numpy(C.pack_573(act).astype(np.float32))
    return p, C.black_floor(act), {
        "curve_logits_clipped": bool(act["clipped"]),
        "logit_peak": float(act["logit_peak"]), "chroma": act["chroma"].tolist(),
    }


def psnr(a, b):
    mse = (a.double() - b.double()).square().mean()
    return float(-10 * torch.log10(mse.clamp_min(1e-14)))


@torch.inference_mode()
def reverse_capture(clean_raw, matrix, c_base, t, gain=1.0):
    """Generate an analytically inverted RAW observation and verify float32 cycles.

    J is the floor-clamped supervision endpoint; J_unfloored preserves the
    clean output before this clamp.
    H and synthetic RAW retain their analytic floating values, including
    out-of-range values. They are mathematical intermediates, not uint16
    sensor observations. Only the controller RAW summary clips to [0,1].
    """
    x, matrix = np.asarray(clean_raw, np.float64), np.asarray(matrix, np.float64)
    if x.ndim != 3 or x.shape[-1] != 3 or matrix.shape != (3, 3) or gain <= 0:
        raise ValueError("invalid external capture/base")
    inverse = np.linalg.inv(matrix)
    aligned = x * gain
    clean = C.enc(np.clip(aligned @ matrix, 0, 1))
    act = C.compile_from_raw(c_base, float(t))
    hazy = C.action_inverse(clean, act)
    synthetic = (decode(torch.from_numpy(hazy)) @ torch.from_numpy(inverse)).numpy()
    target = np.maximum(clean, C.black_floor(act))
    p, floor, diagnostics = compile_action(c_base, float(t))
    j, h, raw = (torch.from_numpy(a.astype(np.float32)) for a in (target, hazy, synthetic))
    replay = apply(h.permute(2, 0, 1)[None], p[None])[0].permute(1, 2, 0)
    base_replay = encode(raw.double() @ torch.from_numpy(matrix))
    complete = apply(base_replay.float().permute(2, 0, 1)[None], p[None])[0].permute(1, 2, 0)
    diagnostics.update(
        action_cycle_psnr=psnr(replay, j), base_cycle_psnr=psnr(base_replay, h),
        end_to_end_cycle_psnr=psnr(complete, j),
        black_floor=floor, display_oob_fraction=float(np.mean((hazy < 0) | (hazy > 1))),
        raw_negative_fraction=float(np.mean(synthetic < 0)),
        raw_above_one_fraction=float(np.mean(synthetic > 1)),
        raw_min=float(synthetic.min()), raw_max=float(synthetic.max()),
    )
    c_raw = np.asarray(c_base, np.float64) @ inverse
    arrays = {
        "J": j.numpy(), "J_unfloored": clean.astype(np.float32), "H": h.numpy(),
        "X_syn": raw.numpy(), "X_clean": aligned.astype(np.float32), "M": matrix,
        "gain": float(gain), "c_base": np.asarray(c_base), "c_raw": c_raw,
        "t": float(t), "p": p.numpy(),
    }
    return arrays, diagnostics


def abi():
    return asdict(CONFIG)
