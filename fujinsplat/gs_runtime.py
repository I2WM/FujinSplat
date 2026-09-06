"""Graphdeco integration for prepared RAW-domain Gaussian scenes."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .io import SCENE_COUNTS, read_json, require_source, sha256
from .mcf import MCFConfig, apply
from .color import decode, encode


def make_camera(frame, fovx, height, width, index, image=None):
    from scene.cameras import Camera
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    # Resizing rounds dimensions; preserve the ORIGINAL calibrated FoVy.
    native_h, native_w = image.shape[-2:] if image is not None else (height, width)
    fovy = 2 * math.atan(math.tan(fovx / 2) * native_h / native_w)
    stem = Path(frame["file_path"]).stem
    if image is None:
        image = torch.zeros(3, height, width)  # Pose-only, never open a held capture.
    return Camera((width, height), colmap_id=index, R=w2c[:3, :3].T, T=w2c[:3, 3],
                  FoVx=fovx, FoVy=fovy, depth_params=None, image=image, invdepthmap=None,
                  image_name=stem, uid=index, data_device="cuda", train_test_exp=False)


def source_metadata(root):
    root = Path(root)
    m = read_json(root / "MANIFEST.json")
    if m.get("schema") != "fujinsplat.scene.v1" or m.get("source_clean_reads") != 0 or m.get("held_target_reads") != 0:
        raise ValueError("not a clean-free prepared scene")
    MCFConfig(**m["mcf"]).validate()
    poses = read_json(root / "transforms_train.json")
    for file, key in (("points3d.ply", "point_cloud_sha256"), ("transforms_train.json", "poses_sha256"), ("actions.pt", "actions_sha256")):
        if sha256(root / file) != m[key]:
            raise ValueError(f"prepared scene hash drift: {file}")
    stems = [r["stem"] for r in m["rows"]]
    if stems != [Path(f["file_path"]).stem for f in poses["frames"]] or len(stems) != SCENE_COUNTS[m["scene"]]:
        raise ValueError("source image/pose population mismatch")
    if len(set(stems)) != len(stems):
        raise ValueError("duplicate source stem")
    for stem in stems:
        require_source(m["scene"], stem)
    for row in m["rows"]:
        relative = Path(row["relative"])
        if relative != Path("images") / f"{row['stem']}.npz":
            raise ValueError("non-source image path in prepared scene")
    return m, poses


def load_source(root, resolution=-1):
    from scene.dataset_readers import getNerfppNorm
    root = Path(root)
    manifest, poses = source_metadata(root)
    cameras, full_base = [], []
    for index, (row, frame) in enumerate(zip(manifest["rows"], poses["frames"])):
        path = root / row["relative"]
        if sha256(path) != row["sha256"]:
            raise ValueError("source image hash drift")
        with np.load(path, allow_pickle=False) as z:
            linear = torch.from_numpy(np.asarray(z["linear_rgb"], np.float32).copy()).permute(2, 0, 1)
            base = torch.from_numpy(np.asarray(z["base_encoded"], np.float32).copy()).permute(2, 0, 1)
        h, w = linear.shape[1:]
        scale = max(1.0, w / 1600) if resolution == -1 else float(resolution)
        if resolution not in (-1, 1, 2, 4, 8):
            raise ValueError("resolution must be -1 (upstream auto) or 1/2/4/8")
        hh, ww = int(h / scale), int(w / scale)
        cameras.append(make_camera(frame, poses["camera_angle_x"], hh, ww, index, linear))
        full_base.append(base)  # CPU resident; one full-frame MCF evaluation at a time.
    bank = torch.load(root / "actions.pt", map_location="cpu", weights_only=False)
    MCFConfig(**bank["mcf"]).validate()
    if bank["stems"] != [r["stem"] for r in manifest["rows"]]:
        raise ValueError("action bank/source order mismatch")
    return manifest, cameras, full_base, bank["actions"].cuda(), load_point_cloud(root / "points3d.ply"), getNerfppNorm(cameras)["radius"]


def load_point_cloud(path):
    """RealX COLMAP PLYs omit normals; Gaussian initialization uses XYZ/RGB only."""
    from plyfile import PlyData
    from utils.graphics_utils import BasicPointCloud
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.stack([vertex[k] for k in ("x", "y", "z")], -1)
    rgb = np.stack([vertex[k] for k in ("red", "green", "blue")], -1) / 255.0
    names = vertex.data.dtype.names
    normals = np.stack([vertex[k] for k in ("nx", "ny", "nz")], -1) if all(k in names for k in ("nx", "ny", "nz")) else np.zeros_like(xyz)
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError("invalid COLMAP point cloud")
    return BasicPointCloud(points=xyz, colors=rgb, normals=normals)


def developed_target(base_encoded, action, render_shape, *, active):
    """Develop the training target for the joint Gaussian/Delta objective.

    Keep the target differentiable in the active window so alpha receives
    the same image loss as the Gaussians.
    """
    with torch.set_grad_enabled(active):
        target = apply(base_encoded[None], action)[0].clamp(0, 1)
        linear = decode(target)
        if linear.shape[-2:] != tuple(render_shape):
            linear = F.interpolate(linear[None], render_shape, mode="bilinear", align_corners=False)[0]
        return encode(linear)
