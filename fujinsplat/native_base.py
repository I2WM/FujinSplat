"""Native hazy-camera ISP, compatible with the sealed compact G9 checkpoints.

The output is learned encoded camera RGB; do not apply another OETF.
R33 is a dense 33^3 residual lattice, not a rank-33 factorization.
The interpolation preserves the pointwise checkpoint arithmetic.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

class ModelABIError(RuntimeError):
    """A value or payload violates the frozen pointwise model ABI."""


def _require_finite_float(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ModelABIError(f"{label} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ModelABIError(f"{label} contains NaN or Inf")


def _require_rgb(value: Tensor, label: str, *, unit_domain: bool = False) -> None:
    _require_finite_float(value, label)
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ModelABIError(f"{label} must have shape [...,3]")
    if unit_domain and bool(torch.any((value < 0.0) | (value > 1.0))):
        raise ModelABIError(f"{label} lies outside the deployed [0,1] RAW domain")



def raw_lut_coordinate(raw: Tensor, scale: Tensor) -> Tensor:
    """Map RAW to a bounded rational LUT coordinate without dead-branch NaNs."""

    nonnegative = raw >= 0.0
    denominator = torch.where(nonnegative, raw + scale, torch.ones_like(raw))
    positive = raw / denominator
    return torch.where(nonnegative, positive, torch.zeros_like(raw)).clamp(0.0, 1.0)


def _identity_safe_sinh_action(base: Tensor, latent: Tensor, action: Tensor, scale: Tensor) -> Tensor:
    """Apply an identity-rooted latent action without cutting its zero-state gradient."""

    correction = scale * (torch.sinh(latent + action) - torch.sinh(latent))
    return base + correction


def trilinear_action(lut: Tensor, coordinate: Tensor) -> Tensor:
    """Evaluate one explicit ``G x G x G x 3`` LUT at point coordinates."""

    _require_rgb(coordinate, "LUT coordinate")
    if lut.ndim != 4 or lut.shape[-1] != 3 or len(set(lut.shape[:3])) != 1:
        raise ModelABIError("LUT must have shape [G,G,G,3]")
    if lut.shape[0] < 2 or not lut.is_floating_point():
        raise ModelABIError("LUT must be floating with at least two nodes per axis")
    if bool(torch.any((coordinate < 0.0) | (coordinate > 1.0))):
        raise ModelABIError("LUT coordinate lies outside [0,1]")

    size = int(lut.shape[0])
    flat = coordinate.reshape(-1, 3)
    scaled = flat * float(size - 1)
    lower = torch.floor(scaled).to(torch.long).clamp(0, size - 2)
    fraction = scaled - lower.to(dtype=scaled.dtype)
    upper = lower + 1
    output = torch.zeros_like(flat)
    for bit0 in (0, 1):
        index0 = upper[:, 0] if bit0 else lower[:, 0]
        weight0 = fraction[:, 0] if bit0 else 1.0 - fraction[:, 0]
        for bit1 in (0, 1):
            index1 = upper[:, 1] if bit1 else lower[:, 1]
            weight1 = fraction[:, 1] if bit1 else 1.0 - fraction[:, 1]
            for bit2 in (0, 1):
                index2 = upper[:, 2] if bit2 else lower[:, 2]
                weight2 = fraction[:, 2] if bit2 else 1.0 - fraction[:, 2]
                weight = (weight0 * weight1 * weight2).unsqueeze(-1)
                output = output + weight * lut[index0, index1, index2]
    return output.reshape(coordinate.shape)


class MonotonePWL(nn.Module):
    """Channelwise monotone PWL warp with exact identity-rooted residuals."""

    def __init__(
        self,
        knot_count: int,
        *,
        minimum: float,
        maximum: float,
        channels: int = 3,
        identity_tails: bool,
    ) -> None:
        super().__init__()
        if knot_count < 2 or channels < 1 or not minimum < maximum:
            raise ValueError("invalid monotone PWL specification")
        self.knot_count = int(knot_count)
        self.channels = int(channels)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.identity_tails = bool(identity_tails)
        self.log_increment_delta = nn.Parameter(
            torch.zeros(channels, knot_count - 1, dtype=torch.float32)
        )

    @staticmethod
    def _softplus_inverse_one(reference: Tensor) -> Tensor:
        return torch.log(torch.expm1(reference.new_tensor(1.0)))

    def ordinates(self, reference: Tensor, *, identity: bool = False) -> Tensor:
        delta = torch.zeros_like(self.log_increment_delta) if identity else self.log_increment_delta
        raw = self._softplus_inverse_one(reference) + delta.to(
            dtype=reference.dtype, device=reference.device
        )
        increments = F.softplus(raw)
        cumulative = torch.cat(
            (
                torch.zeros(
                    self.channels, 1, dtype=reference.dtype, device=reference.device
                ),
                torch.cumsum(increments, dim=1),
            ),
            dim=1,
        )
        unit = cumulative / cumulative[:, -1:]
        return self.minimum + (self.maximum - self.minimum) * unit

    def _interpolate(self, value: Tensor, ordinates: Tensor) -> Tensor:
        clipped = value.clamp(self.minimum, self.maximum)
        scaled = (clipped - self.minimum) * (
            float(self.knot_count - 1) / (self.maximum - self.minimum)
        )
        lower = torch.floor(scaled).to(torch.long).clamp(0, self.knot_count - 2)
        fraction = scaled - lower.to(dtype=scaled.dtype)
        channel = torch.arange(self.channels, device=value.device)
        low_value = ordinates[channel, lower]
        high_value = ordinates[channel, lower + 1]
        return low_value + fraction * (high_value - low_value)

    def residual(self, value: Tensor) -> Tensor:
        _require_finite_float(value, "PWL input")
        if value.shape[-1] != self.channels:
            raise ModelABIError("PWL input must have shape [...,channels]")
        current = self.ordinates(value, identity=False)
        reference = self.ordinates(value, identity=True)
        delta = self._interpolate(value, current) - self._interpolate(value, reference)
        if self.identity_tails:
            inside = (value >= self.minimum) & (value <= self.maximum)
            delta = torch.where(inside, delta, torch.zeros_like(delta))
        return delta

    def forward(self, value: Tensor) -> Tensor:
        return value + self.residual(value)

    def curvature_energy(self) -> Tensor:
        reference = self.log_increment_delta
        ordinates = self.ordinates(reference, identity=False)
        second = ordinates[:, 2:] - 2.0 * ordinates[:, 1:-1] + ordinates[:, :-2]
        return second.square().mean()



@dataclass(frozen=True)
class BaseConfig:
    fine_grid_size: int = 9
    residual_grid_size: int = 33
    curve_knots: int = 257
    coordinate_scale: float = 0.05
    residual_action_scale: float = 0.1


class NativeBase(nn.Module):
    def __init__(self, config=BaseConfig()):
        super().__init__()
        self.config = config
        self.register_buffer("base_exposure", torch.zeros(()))
        self.register_buffer("base_white_balance", torch.zeros(3))
        self.register_buffer("base_row_matrix", torch.eye(3))
        self.exposure_delta = nn.Parameter(torch.zeros(()))
        self.white_balance_delta = nn.Parameter(torch.zeros(3))
        self.matrix_log_residual = nn.Parameter(torch.zeros(3, 3))
        self.shaper = MonotonePWL(config.curve_knots, minimum=0, maximum=1, identity_tails=False)
        self.fine_action = nn.Parameter(torch.zeros(config.fine_grid_size, config.fine_grid_size, config.fine_grid_size, 3))
        self.residual_action = nn.Parameter(torch.zeros(config.residual_grid_size, config.residual_grid_size, config.residual_grid_size, 3))
        self.tone = MonotonePWL(config.curve_knots, minimum=-8, maximum=8, identity_tails=True)

    def front(self, raw):
        wb = self.base_white_balance + self.white_balance_delta
        wb = wb - wb.mean()
        gain = torch.exp(self.base_exposure + self.exposure_delta + wb).to(raw)
        value = raw * gain
        matrix = (self.base_row_matrix @ torch.matrix_exp(self.matrix_log_residual)).to(raw)
        return torch.stack(tuple(
            value[..., 0] * matrix[0, col] + value[..., 1] * matrix[1, col] + value[..., 2] * matrix[2, col]
            for col in range(3)
        ), -1)

    def forward(self, raw):
        front = self.front(raw)
        scale = front.new_tensor(self.config.coordinate_scale)
        latent = torch.asinh(front / scale)
        shaped = self.shaper(raw_lut_coordinate(front, scale))
        action = trilinear_action(self.fine_action, shaped)
        action = action + self.config.residual_action_scale * torch.tanh(trilinear_action(self.residual_action, shaped))
        restored = _identity_safe_sinh_action(front, latent, action, scale)
        tone_latent = torch.asinh(restored / scale)
        return _identity_safe_sinh_action(restored, tone_latent, self.tone.residual(tone_latent), scale)

    @torch.inference_mode()
    def image(self, raw_chw, chunk=262144):
        flat = raw_chw.permute(1, 2, 0).reshape(-1, 3)
        return torch.cat([self(x) for x in flat.split(chunk)]).reshape(
            *raw_chw.shape[1:], 3
        ).permute(2, 0, 1)


def load_native_base(path, device="cpu"):
    # Only trusted project checkpoints are supported; pickle inputs are not public uploads.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    # Accept the existing compact-Base checkpoint format as well as the release schema.
    if payload.get("schema") not in {
        "PHASE22_COMPACT_NATIVE_HAZE_BASE_CHECKPOINT_V0", "fujinsplat.native_base.v1"
    }:
        raise ValueError("not a supported Native hazy-camera Base checkpoint")
    if payload.get("source_clean_reads") != 0 or payload.get("held_test_reads") != 0:
        raise ValueError("Native Base must have zero clean target contact")
    config = BaseConfig(**payload["config"])
    if config != BaseConfig():
        raise ValueError("Native Base is not the declared G9/R33/Shaper257/Tone257")
    model = NativeBase(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().requires_grad_(False)
