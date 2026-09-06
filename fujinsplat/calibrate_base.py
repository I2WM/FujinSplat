"""Calibrate one scene's frozen hazy-camera Base ISP (paper supplement B).

The optimizer, losses, sampling and parameter projection are ported from the
original scene fitter. All captures and the parent checkpoint are explicit
inputs; controller fitting, clean targets and held views are not read here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import base_lattice as lattice
from . import native_base as native_model
from .native_base import BaseConfig, NativeBase, raw_lut_coordinate, trilinear_action
from .io import SCENE_COUNTS, create_output, load_raw, read_json, sha256, write_json

P32_BLOCKS = 16
P32_WEIGHT = 0.25
OOB_WEIGHT = 0.05
LATTICE_TERMS = 4096


@dataclass(frozen=True)
class CalibrationSettings:
    warmup_steps: int = 100
    joint_steps: int = 400
    pixels_per_update: int = 8192
    jacobian_weight: float = 0.01
    seed: int = 82751
    projection_sigma: float = 1.0


def ste_q8(value: Tensor) -> Tensor:
    quantized = torch.round(value.clamp(0, 1) * 255).to(torch.uint8).to(value.dtype) / 255
    return value + (quantized - value).detach()


@dataclass(frozen=True)
class ParameterBounds:
    exposure_delta_stops: float = 2.0
    wb_log_delta: float = 1.0
    matrix_log_residual: float = 0.5
    shaper_log_increment: float = 4.0
    fine_action_low: float = -32.0
    fine_action_high: float = 16.0
    residual_action: float = 8.0
    tone_log_increment: float = 4.0


class CalibrationBase(NativeBase):
    def __init__(self, config=BaseConfig()):
        super().__init__(config)
        self.bounds = ParameterBounds()

    def back(self, front: Tensor) -> Tensor:
        scale = front.new_tensor(self.config.coordinate_scale)
        latent = torch.asinh(front / scale)
        coordinate = raw_lut_coordinate(front, scale)
        shaped = self.shaper(coordinate)
        fine = trilinear_action(self.fine_action, shaped)
        residual = trilinear_action(self.residual_action, shaped)
        action = fine + self.config.residual_action_scale * torch.tanh(residual)
        restored = native_model._identity_safe_sinh_action(  # noqa: SLF001
            front, latent, action, scale
        )
        tone_latent = torch.asinh(restored / scale)
        tone_action = self.tone.residual(tone_latent)
        return native_model._identity_safe_sinh_action(  # noqa: SLF001
            restored, tone_latent, tone_action, scale
        )

    def forward_with_diagnostics(self, raw: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        front = self.front(raw)
        output = self.back(front)
        return output, {"front": front, "output": output}

    def forward(self, raw: Tensor) -> Tensor:
        return self.forward_with_diagnostics(raw)[0]

    def front_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.exposure_delta,
            self.white_balance_delta,
            self.matrix_log_residual,
        )

    def set_front_trainable(self, enabled: bool) -> None:
        for parameter in self.front_parameters():
            parameter.requires_grad_(enabled)

    @torch.no_grad()
    def project_(self) -> None:
        b = self.bounds
        log2 = math.log(2.0)
        self.exposure_delta.clamp_(
            -b.exposure_delta_stops * log2, b.exposure_delta_stops * log2
        )
        self.white_balance_delta.sub_(self.white_balance_delta.mean())
        self.white_balance_delta.clamp_(-b.wb_log_delta, b.wb_log_delta)
        self.white_balance_delta.sub_(self.white_balance_delta.mean())
        self.matrix_log_residual.clamp_(
            -b.matrix_log_residual, b.matrix_log_residual
        )
        self.shaper.log_increment_delta.clamp_(
            -b.shaper_log_increment, b.shaper_log_increment
        )
        self.fine_action.clamp_(b.fine_action_low, b.fine_action_high)
        self.residual_action.clamp_(-b.residual_action, b.residual_action)
        self.tone.log_increment_delta.clamp_(
            -b.tone_log_increment, b.tone_log_increment
        )

def _coprime_multiplier(count: int, token: str) -> tuple[int, int]:
    digest = hashlib.sha256(token.encode("ascii")).digest()
    multiplier = max(1, int.from_bytes(digest[:8], "little") % count)
    while math.gcd(multiplier, count) != 1:
        multiplier = 1 if multiplier + 1 == count else multiplier + 1
    return multiplier, int.from_bytes(digest[8:16], "little") % count

def unique_pixel_indices(count: int, batch: int, token: str, visit: int) -> np.ndarray:
    if batch > count:
        raise RuntimeError("source frame has fewer pixels than the requested batch")
    epoch, start = divmod(visit * batch, count)
    multiplier, offset = _coprime_multiplier(count, f"{token}|{epoch}")
    position = np.arange(start, start + batch, dtype=np.int64) % count
    return np.ascontiguousarray((multiplier * position + offset) % count)

def fixed_p32_blocks(shape: tuple[int, int, int], token: str) -> np.ndarray:
    height, width, channels = shape
    if channels != 3 or height < 32 or width < 32:
        raise RuntimeError("P32 requires HWC3 with H,W >= 32")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:8], "little"))
    rows = rng.integers(0, height - 31, size=P32_BLOCKS)
    columns = rng.integers(0, width - 31, size=P32_BLOCKS)
    return np.stack((rows, columns), axis=1)

def _patch_tensors(
    raw: np.ndarray,
    clean: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    x = np.stack([raw[row : row + 32, col : col + 32] for row, col in positions])
    y = np.stack([clean[row : row + 32, col : col + 32] for row, col in positions])
    return (
        torch.from_numpy(np.ascontiguousarray(x.reshape(-1, 3))).to(device),
        torch.from_numpy(np.ascontiguousarray(y.astype(np.float32) / 255.0)).to(device),
    )

def _optimizer(model: CalibrationBase) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"name": "front", "params": model.front_parameters(), "lr": 1.0e-4},
            {
                "name": "curves",
                "params": (
                    model.shaper.log_increment_delta,
                    model.tone.log_increment_delta,
                ),
                "lr": 2.0e-4,
            },
            {"name": "L257", "params": (model.fine_action,), "lr": 3.0e-4},
            {"name": "R33", "params": (model.residual_action,), "lr": 3.0e-4},
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )

def _jacobian_loss(model: CalibrationBase, raw: Tensor) -> tuple[Tensor, dict[str, float]]:
    probe = raw[:64]
    epsilon = 1.0e-3
    centre = model(probe)
    columns = []
    for channel in range(3):
        shifted = probe.clone()
        shifted[:, channel] = (shifted[:, channel] + epsilon).clamp(0.0, 1.0)
        columns.append((model(shifted) - centre) / epsilon)
    matrix = torch.stack(columns, dim=-1)
    singular = torch.linalg.svdvals(matrix)
    determinant = torch.linalg.det(matrix)
    trace = matrix.square().sum(dim=(-2, -1))
    # Raw determinants and singular values inherit the LUT coordinate scale and
    # can reach 1e5.  Penalize their dimensionless ratios/log-magnitudes so the
    # safety term cannot replace the registered smoke-to-clean estimand.
    orientation_scale = singular.prod(dim=-1).detach().clamp_min(1.0e-6)
    penalty = (
        F.relu(torch.log1p(trace) - math.log(28.0)).square().mean()
        + F.relu(torch.log1p(singular[..., 0]) - math.log(9.0)).square().mean()
        + F.relu(math.log(0.02) - torch.log(singular[..., -1].clamp_min(1.0e-6))).square().mean()
        + F.relu(-determinant / orientation_scale).square().mean()
    )
    return penalty, {
        "smax_mean": float(singular[..., 0].detach().mean()),
        "smin_mean": float(singular[..., -1].detach().mean()),
        "det_negative_fraction": float((determinant.detach() <= 0).float().mean()),
    }

def _fit(  # noqa: PLR0915
    raw: Sequence[np.ndarray],
    clean: Sequence[np.ndarray],
    stems: Sequence[str],
    base: NativeBase,
    device: torch.device,
    seed: int,
    warmup_steps: int,
    joint_steps: int,
    log_path: Path,
    settings: CalibrationSettings = CalibrationSettings(),
) -> CalibrationBase:
    model = base.to(device).train()
    optimizer = _optimizer(model)
    l257 = lattice.sampler_pair(model.config.fine_grid_size, "FILE46_L257", LATTICE_TERMS)
    r33 = lattice.sampler_pair(model.config.residual_grid_size, "FILE46_R33", LATTICE_TERMS)
    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    frame_order: list[int] = []
    visits = [0 for _ in raw]
    p32_positions = [fixed_p32_blocks(value.shape, f"{seed}|{stem}|P32") for value, stem in zip(raw, stems)]
    generator = np.random.default_rng(seed)
    total = warmup_steps + joint_steps
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.set_front_trainable(False)
    with log_path.open("x", encoding="utf-8") as log:
        for step in range(1, total + 1):
            if step == warmup_steps + 1:
                model.set_front_trainable(True)
            if not frame_order:
                frame_order = list(range(len(raw)))
                generator.shuffle(frame_order)
            frame = frame_order.pop()
            flat_raw = raw[frame].reshape(-1, 3)
            flat_clean = clean[frame].reshape(-1, 3)
            indices = unique_pixel_indices(
                len(flat_raw), settings.pixels_per_update, f"{seed}|{stems[frame]}", visits[frame]
            )
            visits[frame] += 1
            x = torch.from_numpy(np.ascontiguousarray(flat_raw[indices])).to(device)
            target = torch.from_numpy(
                np.ascontiguousarray(flat_clean[indices].astype(np.float32) / 255.0)
            ).to(device)
            patch_x, patch_target = _patch_tensors(
                raw[frame], clean[frame], p32_positions[frame], device
            )

            progress = step / total
            lr_scale = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = initial_lr * lr_scale

            pre, diagnostics = model.forward_with_diagnostics(x)
            prediction = ste_q8(pre)
            photo = (prediction - target).square().mean()
            l1 = (prediction - target).abs().mean()
            patch_prediction = ste_q8(model(patch_x)).reshape(
                P32_BLOCKS, 32, 32, 3
            )
            patch_target = patch_target.reshape(P32_BLOCKS, 32, 32, 3)
            p32 = (patch_prediction.mean((1, 2)) - patch_target.mean((1, 2))).square().mean()
            front = diagnostics["front"]
            oob = (F.relu(-front).square() + F.relu(front - 1.0).square()).mean()
            jacobian, jacobian_row = _jacobian_loss(model, x)
            e1 = lattice.sampled_energy(model.fine_action, l257[0], step - 1)
            e2 = lattice.sampled_energy(model.fine_action, l257[1], step - 1)
            realized_r33 = model.config.residual_action_scale * torch.tanh(model.residual_action)
            r1 = lattice.sampled_energy(realized_r33, r33[0], step - 1)
            r2 = lattice.sampled_energy(realized_r33, r33[1], step - 1)
            curves = model.shaper.curvature_energy() + model.tone.curvature_energy()
            front_reg = sum(value.square().mean() for value in model.front_parameters())
            loss = (
                photo
                + 0.05 * l1
                + P32_WEIGHT * p32
                + OOB_WEIGHT * oob
                + settings.jacobian_weight * jacobian
                + 1.0e-6 * (e1 + r1)
                + 1.0e-7 * (e2 + r2 + curves + front_reg)
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite joint scene ISP loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.project_()
            if step == 1 or step % 50 == 0 or step in (warmup_steps, total):
                row = {
                    "step": step,
                    "stage": "BACK_WARMUP" if step <= warmup_steps else "JOINT",
                    "frame": stems[frame],
                    "loss": float(loss.detach()),
                    "photo": float(photo.detach()),
                    "l1": float(l1.detach()),
                    "p32": float(p32.detach()),
                    "front_oob": float(oob.detach()),
                    "jacobian": float(jacobian.detach()),
                    "lr_scale": lr_scale,
                    **jacobian_row,
                }
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
    return model.eval().requires_grad_(False)


def gaussian_smooth_l257(value: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinate = torch.arange(
        -radius, radius + 1, device=value.device, dtype=value.dtype
    )
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    tensor = value.permute(3, 0, 1, 2).unsqueeze(0)
    channels = int(tensor.shape[1])
    for dimension in range(3):
        padding = [0, 0, 0, 0, 0, 0]
        index = {0: 4, 1: 2, 2: 0}[dimension]
        padding[index] = radius
        padding[index + 1] = radius
        tensor = F.pad(tensor, padding, mode="replicate")
        shape = [1, 1, 1]
        shape[dimension] = int(kernel.numel())
        weight = kernel.reshape(1, 1, *shape).repeat(channels, 1, 1, 1, 1)
        tensor = F.conv3d(tensor, weight, groups=channels)
    return tensor[0].permute(1, 2, 3, 0).contiguous()


@torch.no_grad()
def project_parent_state(state, *, device="cpu", sigma=1.0):
    """The original native-parent -> smoothed G9 initialization, including matrix orientation."""
    model = CalibrationBase().to(device)
    source_fine = state["fine_action"].to(device)
    smooth = gaussian_smooth_l257(source_fine, sigma)
    compact = F.interpolate(smooth.permute(3, 0, 1, 2)[None], size=(9, 9, 9),
                            mode="trilinear", align_corners=True)[0].permute(1, 2, 3, 0).contiguous()
    identity = torch.eye(3, dtype=state["base_row_matrix"].dtype)
    if not torch.allclose(state["base_row_matrix"].cpu(), identity, atol=1e-7, rtol=0):
        raise ValueError("the original parent initialization requires an identity base row matrix")
    model.base_exposure.copy_(state["base_exposure"] + state["exposure_delta"])
    model.base_white_balance.copy_(state["base_white_balance"] + state["white_balance_delta"])
    # The original fitter first folds E into GlobalPointwiseBase.front_matrix(),
    # then initializes JointSceneISP with its transpose. Preserve that convention.
    model.base_row_matrix.copy_(torch.matrix_exp(state["matrix_log_residual"].to(device)).T.contiguous())
    model.shaper.log_increment_delta.copy_(state["shaper.log_increment_delta"])
    model.fine_action.copy_(compact)
    model.residual_action.copy_(state["residual_action"])
    model.tone.log_increment_delta.copy_(state["tone.log_increment_delta"])
    return model


def load_parent(path, scene, device, sigma):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "PHASE21_NATIVE_J0_ZERO_CLEAN_HAZE_BASE_CHECKPOINT_V0":
        raise ValueError("expected the original native parent Base checkpoint")
    if payload.get("scene") != scene or payload.get("source_clean_reads") != 0 or payload.get("held_test_reads") != 0:
        raise ValueError("parent Base scene/supervision mismatch")
    state = payload["state_dict"]
    if tuple(state["fine_action"].shape) != (257, 257, 257, 3):
        raise ValueError("paper initialization requires the parent 257-cubed lattice")
    return project_parent_state(state, device=device, sigma=sigma)


def source_inputs(args):
    poses_path = args.rgb_root / args.scene / "transforms_train.json"
    poses = read_json(poses_path)
    stems = [Path(row["file_path"]).stem for row in poses["frames"]]
    if stems != [f"{i:04d}" for i in range(1, SCENE_COUNTS[args.scene] + 1)]:
        raise ValueError("source camera order/population mismatch")
    raw_paths = [args.raw_root / args.scene / "train" / f"{stem}.npz" for stem in stems]
    hazy_paths = [args.rgb_root / args.scene / "train" / f"{stem}.JPG" for stem in stems]
    for path in [args.initial_parent, *raw_paths, *hazy_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return stems, raw_paths, hazy_paths, poses_path


def save_calibrated_base(output, model, scene, stems, settings, provenance):
    path = output / "complete_model.pt"
    payload = {
        "schema": "fujinsplat.native_base.v1", "status": "CALIBRATED_HAZY_CAMERA_BASE",
        "scene": scene, "config": asdict(BaseConfig()), "source_stems": stems,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "source_clean_reads": 0, "held_test_reads": 0,
        "calibration": asdict(settings), "initialization": provenance,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENE_COUNTS, required=True)
    p.add_argument("--raw-root", type=Path, required=True, help="no-WB RAW root/SCENE/train/*.npz")
    p.add_argument("--rgb-root", type=Path, required=True, help="camera hazy RGB root/SCENE/train/*.JPG")
    p.add_argument("--initial-parent", type=Path, required=True, help="original per-scene native L257 parent checkpoint")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--joint-steps", type=int, default=400)
    p.add_argument("--pixels-per-update", type=int, default=8192)
    p.add_argument("--jacobian-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=82751)
    p.add_argument("--projection-sigma", type=float, default=1.0)
    p.add_argument("--check-only", action="store_true", help="check paths/metadata without reading images or training")
    a = p.parse_args()
    settings = CalibrationSettings(a.warmup_steps, a.joint_steps, a.pixels_per_update,
                                   a.jacobian_weight, a.seed, a.projection_sigma)
    if a.warmup_steps < 0 or a.joint_steps < 1 or a.pixels_per_update < 1:
        raise ValueError("invalid calibration step/pixel count")
    if not math.isfinite(a.jacobian_weight) or a.jacobian_weight < 0 or not math.isfinite(a.projection_sigma) or a.projection_sigma <= 0:
        raise ValueError("invalid calibration weight/projection sigma")
    stems, raw_paths, hazy_paths, poses_path = source_inputs(a)
    if a.check_only:
        print(json.dumps({"scene": a.scene, "source_views": len(stems), "schedule": asdict(settings),
                          "image_reads": 0, "training_started": False}, indent=2))
        return
    if a.output is None:
        p.error("--output is required for calibration")
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    model = load_parent(a.initial_parent, a.scene, device, a.projection_sigma)
    raw, hazy = [], []
    for raw_path, hazy_path in zip(raw_paths, hazy_paths):
        value = load_raw(raw_path).permute(1, 2, 0).numpy()
        with Image.open(hazy_path) as image:
            target = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if value.shape != target.shape or min(value.shape[:2]) < 32:
            raise ValueError("source RAW/camera RGB must have identical dimensions with H,W >= 32")
        raw.append(value)
        hazy.append(target)
    output = create_output(a.output)
    provenance = {"parent_checkpoint_sha256": sha256(a.initial_parent),
                  "projection": "Gaussian smooth L257 then align-corners trilinear sample to G9",
                  "projection_sigma_nodes": a.projection_sigma}
    write_json(output / "CONTRACT.json", {"scene": a.scene, "schedule": asdict(settings),
        "initialization": provenance, "source_pose_sha256": sha256(poses_path),
        "source_clean_reads": 0, "held_test_reads": 0,
        "rows": [{"stem": stem, "raw_sha256": sha256(r), "hazy_rgb_sha256": sha256(h)}
                 for stem, r, h in zip(stems, raw_paths, hazy_paths)]})
    started = time.monotonic()
    model = _fit(raw, hazy, stems, model, device, a.seed, a.warmup_steps,
                 a.joint_steps, output / "TRAIN.jsonl", settings)
    checkpoint = save_calibrated_base(output, model, a.scene, stems, settings, provenance)
    result = {"status": "CALIBRATED_HAZY_CAMERA_BASE", "scene": a.scene,
              "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
              "source_views": len(stems), "source_clean_reads": 0, "held_test_reads": 0,
              "steps": a.warmup_steps + a.joint_steps, "elapsed_seconds": time.monotonic() - started}
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
