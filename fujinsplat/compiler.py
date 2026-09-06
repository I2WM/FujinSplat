"""Numerical compiler port from the user-supplied supplementary simulation.

The public entry is compile_from_raw: enc o affine o dec, NOT the older
encoded-affine compile_action helper in compile_final.py. This module is
NumPy-only, has no I/O, random state, refitting, or fallback to 25 source pairs.
"""
import numpy as np

KNOTS = 16
STAGES = 7
CPL_KNOTS = 9
LOGIT_LIMIT = 5.0
CPL_BOUND = 0.155
KNEE_WIDTH = 0.02

def enc(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def dec(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def _smooth_max0(u, w):
    return 0.5 * (u + np.sqrt(u * u + w * w))


def _smooth_min1(u, w):
    return 1.0 - _smooth_max0(1.0 - u, w)


def nodes_to_raw_curve(nodes):
    delta = np.maximum(np.diff(nodes), 1e-6)
    delta = delta / delta.sum()
    z = np.log(delta)
    z = z - z.mean()
    peak = float(np.abs(z).max())
    g = np.clip(z / LOGIT_LIMIT, -0.999999, 0.999999)
    return np.arctanh(g), peak > LOGIT_LIMIT, peak


def _increments(raw_curve):
    c = np.tanh(raw_curve)
    c = c - c.mean(-1, keepdims=True)
    c = LOGIT_LIMIT * c / np.maximum(1.0, np.abs(c).max(-1, keepdims=True))
    e = np.exp(c - c.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def apply_curve(value, raw_curve):
    inc = _increments(raw_curve)
    nodes = np.concatenate([[0.0], np.cumsum(inc)])
    slopes = inc * KNOTS
    v = np.asarray(value)
    scaled = np.clip(v, 0, 1) * KNOTS
    lo = np.clip(np.floor(scaled).astype(int), 0, KNOTS - 1)
    frac = scaled - lo
    inside = nodes[lo] + frac * (nodes[lo + 1] - nodes[lo])
    return np.where(v < 0, v * slopes[0],
                    np.where(v > 1, 1 + (v - 1) * slopes[-1], inside))


def invert_curve(value, raw_curve):
    inc = _increments(raw_curve)
    nodes = np.concatenate([[0.0], np.cumsum(inc)])
    slopes = inc * KNOTS
    v = np.asarray(value)
    idx = np.clip(np.searchsorted(nodes, v, side="right") - 1, 0, KNOTS - 1)
    lo, hi = nodes[idx], nodes[idx + 1]
    frac = (v - lo) / np.maximum(hi - lo, 1e-12)
    inside = (idx + frac) / KNOTS
    return np.where(v < 0, v / max(slopes[0], 1e-12),
                    np.where(v > 1, 1 + (v - 1) / max(slopes[-1], 1e-12), inside))


def _knots(offset):
    return np.full(CPL_KNOTS, np.arctanh(np.clip(offset / CPL_BOUND, -0.999, 0.999)))


def coupling_forward(rgb, stages):
    out = rgb.copy()
    for k in range(STAGES):
        j, l = (k + 1) % 3, (k + 2) % 3
        s1, s2, s3 = stages[k]
        out[..., j] += CPL_BOUND * np.tanh(s1[0])
        out[..., l] += 0.5 * (CPL_BOUND * np.tanh(s2[0]) + CPL_BOUND * np.tanh(s3[0]))
    return out


def coupling_inverse(rgb, stages):
    out = rgb.copy()
    for k in reversed(range(STAGES)):
        j, l = (k + 1) % 3, (k + 2) % 3
        s1, s2, s3 = stages[k]
        out[..., l] -= 0.5 * (CPL_BOUND * np.tanh(s2[0]) + CPL_BOUND * np.tanh(s3[0]))
        out[..., j] -= CPL_BOUND * np.tanh(s1[0])
    return out


def _target_nodes(c_k, t, width=KNEE_WIDTH):
    """Encoded-domain dehaze target of one channel: enc . affine^-1 . dec."""
    grid = np.arange(KNOTS + 1) / KNOTS
    u = (dec(grid) - (1.0 - t) * c_k) / t
    u = _smooth_min1(_smooth_max0(u, width), width)
    f = enc(u)
    return np.clip((f - f[0]) / (f[-1] - f[0]), 0.0, 1.0)


def compile_from_raw(c_base, t, width=KNEE_WIDTH):
    """(airlight in linear base RGB, density) -> one 573-D action.

    Division of labour as stated in the paper: the shared curve takes the
    achromatic veil and contrast, the couplings take the chromatic residual.
    The coupling offsets are obtained by projecting each channel's own target
    onto the shared curve, delta_k = mean (f_k - f_bar) / f_bar', so that
    f_bar(y + delta_k) reproduces f_k(y) over the working range.
    """
    c_base = np.asarray(c_base, float)
    cbar = float(c_base.mean())
    grid = np.arange(KNOTS + 1) / KNOTS
    fbar = _target_nodes(cbar, t, width)
    raw_curve, clipped, peak = nodes_to_raw_curve(fbar)

    slope = np.gradient(fbar, grid)
    w = slope > 1e-3
    chroma = np.array([float(np.mean((_target_nodes(c_base[k], t, width) - fbar)[w]
                                     / slope[w])) for k in range(3)])
    chroma = np.clip(chroma, -0.9 * CPL_BOUND, 0.9 * CPL_BOUND)

    weight = np.zeros(3)
    for k in range(STAGES):
        weight[(k + 1) % 3] += 1.0
        weight[(k + 2) % 3] += 0.5
    stages = []
    for k in range(STAGES):
        j, l = (k + 1) % 3, (k + 2) % 3
        stages.append((_knots(chroma[j] / weight[j]),
                       _knots(chroma[l] / weight[l]),
                       _knots(0.0)))
    return dict(raw_curve=raw_curve, coupling=stages, pivot=c_base, t=t,
                veil=(1.0 - t) * cbar, chroma=chroma, clipped=clipped,
                logit_peak=peak)


def action_forward(y, act):
    """dehaze: pre-correction encoded RGB -> corrected encoded RGB."""
    return apply_curve(coupling_forward(y, act["coupling"]), act["raw_curve"])


def black_floor(act):
    """Smallest clean value the curve can still invert faithfully.

    The curve is pinned at 0 -> 0, but the haze map must send clean 0 to the
    veil level.  Below the first strictly-rising node the inverse therefore
    collapses towards 0, and because the three channels collapse by different
    amounts a saturated dark colour rotates in hue (dark red -> yellow).
    Clamping the clean input to this floor keeps every pixel on the
    well-conditioned part of the curve.
    """
    nodes = np.concatenate([[0.0], np.cumsum(_increments(act["raw_curve"]))])
    return float(nodes[np.argmax(nodes > 1e-3)])


def action_inverse(y, act, floor=True):
    """haze: clean encoded RGB -> pre-correction encoded RGB."""
    if floor:
        y = np.maximum(y, black_floor(act))
    return coupling_inverse(invert_curve(y, act["raw_curve"]), act["coupling"])


def pack_573(act):
    """Layout matching the frozen MCF ABI: 8 curves then 7 coupling stages."""
    curves = np.zeros((8, 3, KNOTS))
    curves[-1] = act["raw_curve"][None, :].repeat(3, axis=0)   # shared curve last
    coupling = np.zeros((STAGES, 3, CPL_KNOTS))
    for k, (s1, s2, s3) in enumerate(act["coupling"]):
        coupling[k] = np.stack([s1, s2, s3])
    return np.concatenate([curves.ravel(), coupling.ravel()])



