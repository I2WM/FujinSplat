"""Explicit camera-linear RAW / encoded RGB boundaries."""

import numpy as np
import torch
from torch.nn import functional as F


def encode(x):
    nonlinear = 1.055 * x.clamp_min(0.0031308).pow(1 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, 12.92 * x, nonlinear)


def decode(x):
    nonlinear = ((x.clamp_min(0.04045) + 0.055) / 1.055).pow(2.4)
    return torch.where(x <= 0.04045, x / 12.92, nonlinear)


def summary(raw):
    """BCHW demosaiced, no-WB RAW: normalize uint16, bilinear 64, clamp.

    Floating input must already be normalized camera-linear RAW. No
    per-image gain, white balance, statistics normalization or gamma is used.
    """
    if isinstance(raw, np.ndarray):
        if raw.dtype == np.uint16:
            raw = torch.from_numpy(raw.astype(np.float32) / 65535.0)
        elif np.issubdtype(raw.dtype, np.floating):
            raw = torch.from_numpy(np.ascontiguousarray(raw))
        else:
            raise ValueError("RAW must be uint16 or normalized floating point")
    if raw.ndim != 4 or raw.shape[1] != 3:
        raise ValueError("RAW must have shape B,3,H,W")
    if raw.dtype == getattr(torch, "uint16", None):
        raw = raw.float() / 65535.0
    elif not raw.is_floating_point():
        raise ValueError("RAW must be uint16 or normalized floating point")
    return F.interpolate(raw, (64, 64), mode="bilinear", align_corners=False).clamp(0, 1)


def q8(encoded):
    return torch.round(encoded.clamp(0, 1) * 255).to(torch.uint8)


def reconstruction_loss(prediction_encoded, target_encoded, valid=None):
    """Linear L1 + .25 encoded L1 + .01 OOB; inputs are ENCODED.

    The old helper expected linear inputs. Passing encoded MCF outputs into
    it would apply the display encoding twice; this boundary is intentional.
    """
    weight = torch.ones_like(prediction_encoded) if valid is None else valid.expand_as(prediction_encoded)
    denom = weight.sum().clamp_min(1)
    linear = (weight * (decode(prediction_encoded.clamp(0, 1)) - decode(target_encoded.clamp(0, 1))).abs()).sum() / denom
    display = (weight * (prediction_encoded - target_encoded).abs()).sum() / denom
    oob = F.relu(-prediction_encoded).square().mean() + F.relu(prediction_encoded - 1).square().mean()
    return linear + 0.25 * display + 0.01 * oob
