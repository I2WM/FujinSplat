"""Paper-snapshot MCF: fixed-endpoint curves and uncentered couplings.

    ABI: 8 x 3 x 16 curve coefficients, then 7 x 3 x 9 coupling coefficients.
    Domain: encoded base RGB. This is NOT the old linear/.1/1.75 checkpoint ABI
    and NOT the experimental offset-plus-15-slopes (unanchored) curve ABI.
"""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class MCFConfig:
    abi: str = "fujinsplat.mcf573.encoded.anchored.v1"
    stages: int = 7
    curves: int = 8
    intervals: int = 16
    knots: int = 9
    curve_bound: float = 5.0
    coupling_bound: float = 0.155
    center_coupling: bool = False
    domain: str = "encoded_srgb"

    def validate(self):
        if asdict(self) != asdict(MCFConfig()):
            raise ValueError("MCF ABI mismatch; legacy/capacity settings require a separate version")


CONFIG = MCFConfig()
DIM = 573


def unpack(p):
    if p.ndim != 2 or p.shape[1] != DIM:
        raise ValueError("action must have shape B,573")
    return p[:, :384].reshape(-1, 8, 3, 16), p[:, 384:].reshape(-1, 7, 3, 9)


def pack(curves, couplings):
    if curves.shape[1:] != (8, 3, 16) or couplings.shape != (len(curves), 7, 3, 9):
        raise ValueError("coefficient block shapes do not match ABI")
    return torch.cat((curves.flatten(1), couplings.flatten(1)), 1)


def curve_nodes(raw):
    g = raw.tanh()
    g = g - g.mean(-1, keepdim=True)
    logits = CONFIG.curve_bound * g / g.abs().amax(-1, keepdim=True).clamp_min(1)
    increments = torch.softmax(logits, -1)
    nodes = torch.cat((increments.new_zeros((*increments.shape[:-1], 1)), increments.cumsum(-1)), -1)
    return increments, nodes


def curve(x, raw, inverse=False):
    inc, nodes = curve_nodes(raw)
    n = inc.shape[-1]
    flat = x.flatten(2)
    if inverse:
        lo = (torch.searchsorted(nodes.contiguous(), flat.contiguous(), right=True) - 1).clamp(0, n - 1)
        a, b = torch.gather(nodes, 2, lo), torch.gather(nodes, 2, lo + 1)
        inside = ((lo.to(x.dtype) + (flat - a) / (b - a).clamp_min(1e-12)) / n).reshape_as(x)
        below = x / (inc[:, :, 0, None, None] * n)
        above = 1 + (x - 1) / (inc[:, :, -1, None, None] * n)
    else:
        scaled = x.clamp(0, 1) * n
        lo = scaled.floor().long().clamp(0, n - 1)
        a = torch.gather(nodes, 2, lo.flatten(2)).reshape_as(x)
        b = torch.gather(nodes, 2, lo.flatten(2) + 1).reshape_as(x)
        inside = a + (scaled - lo) * (b - a)
        below = x * (inc[:, :, 0, None, None] * n)
        above = 1 + (x - 1) * (inc[:, :, -1, None, None] * n)
    return torch.where(x < 0, below, torch.where(x > 1, above, inside))


def shift(x, raw):
    knots = CONFIG.coupling_bound * raw.tanh()  # Deliberately NOT mean-centered.
    scaled = x.clamp(0, 1) * (knots.shape[-1] - 1)
    lo = scaled.floor().long().clamp(0, knots.shape[-1] - 2)
    a = torch.gather(knots, 1, lo.flatten(1)).reshape_as(x)
    b = torch.gather(knots, 1, lo.flatten(1) + 1).reshape_as(x)
    return a + (scaled - lo) * (b - a)


def coupling(x, raw, stage, inverse=False):
    i, j, k = stage % 3, (stage + 1) % 3, (stage + 2) % 3
    channels = list(x.unbind(1))
    first = channels[i]
    if inverse:
        second_out = channels[j]
        channels[j] = second_out - shift(first, raw[:, 0])
        channels[k] = channels[k] - 0.5 * (shift(first, raw[:, 1]) + shift(second_out, raw[:, 2]))
    else:
        channels[j] = channels[j] + shift(first, raw[:, 0])
        channels[k] = channels[k] + 0.5 * (shift(first, raw[:, 1]) + shift(channels[j], raw[:, 2]))
    return torch.stack(channels, 1)


def apply(x, p, inverse=False):
    """Analytic forward/inverse on BCHW encoded RGB; never clamp the result."""
    if x.ndim != 4 or x.shape[1] != 3 or len(x) != len(p):
        raise ValueError("MCF image/action batch mismatch")
    curves, couplings = unpack(p)
    if inverse:
        for stage in reversed(range(7)):
            x = curve(x, curves[:, stage + 1], inverse=True)
            x = coupling(x, couplings[:, stage], stage, inverse=True)
        return curve(x, curves[:, 0], inverse=True)
    x = curve(x, curves[:, 0])
    for stage in range(7):
        x = coupling(x, couplings[:, stage], stage)
        x = curve(x, curves[:, stage + 1])
    return x


def probes(*, dtype=torch.float32, device="cpu"):
    """729 probes, R-major/B-fast Cartesian rows; BCHW shape 1,3,1,729."""
    axis = torch.linspace(0, 1, 9, dtype=dtype, device=device)
    return torch.cartesian_prod(axis, axis, axis).T.reshape(1, 3, 1, -1)
