"""Materialize SOURCE-only frozen controller outputs and a scene mean bank.

No training manifest, RealX clean path, or held image is accepted here.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np
import torch

from .color import decode, summary
from .controller import load_checkpoint, load_weights
from .io import SCENE_COUNTS, create_output, load_raw, read_json, require_source, sha256, write_json
from .mcf import CONFIG, apply
from .native_base import load_native_base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENE_COUNTS, required=True)
    p.add_argument("--raw-root", type=Path, required=True, help="root containing scene/train/*.npz")
    p.add_argument("--geometry", type=Path, required=True, help="scene directory with transforms_train and point3d.ply")
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--controller", type=Path, required=True)
    p.add_argument("--controller-format", choices=("sealed", "weights"), default="sealed",
                   help="sealed: internal checkpoint plus sidecar; weights: tensor-only delivery, no sidecar reads")
    p.add_argument("--controller-sha256", help="optional expected controller file SHA256")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    poses = read_json(a.geometry / "transforms_train.json")
    stems = [Path(r["file_path"]).stem for r in poses["frames"]]
    if stems != [f"{i:04d}" for i in range(1, SCENE_COUNTS[a.scene] + 1)]:
        raise ValueError("source camera order/population mismatch")
    for stem in stems:
        require_source(a.scene, stem)
    ply = a.geometry / "point3d.ply"
    if not ply.is_file():
        ply = a.geometry / "points3d.ply"
    if not ply.is_file():
        raise FileNotFoundError("COLMAP points required; no random initialization fallback")
    if a.controller_sha256 is not None and sha256(a.controller) != a.controller_sha256:
        raise ValueError("controller SHA256 mismatch")
    if a.controller_format == "weights":
        model = load_weights(a.controller, a.device)
        training_kind = "not_embedded_in_inference_weights"
    else:
        model, checkpoint = load_checkpoint(a.controller, a.device)
        training_kind = checkpoint["training_kind"]
    model.eval().requires_grad_(False)
    base = load_native_base(a.base_checkpoint, a.device)
    raw_paths = [a.raw_root / a.scene / "train" / f"{stem}.npz" for stem in stems]
    with torch.inference_mode():
        low_raw = torch.stack([summary(load_raw(path)[None])[0] for path in raw_paths]).to(a.device)
        actions = model(low_raw)
        mean = actions.mean(0, keepdim=True)
    out = create_output(a.output)
    (out / "images").mkdir()
    rows = []
    with torch.inference_mode():
        for stem, raw_path in zip(stems, raw_paths):
            raw = load_raw(raw_path).to(a.device)
            encoded = base.image(raw).clamp(0, 1)
            corrected = apply(encoded[None], mean)[0].clamp(0, 1)
            linear = decode(corrected)
            path = out / "images" / f"{stem}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, linear_rgb=linear.permute(1, 2, 0).cpu().numpy(),
                                    base_encoded=encoded.permute(1, 2, 0).cpu().numpy())
            rows.append({"stem": stem, "relative": str(path.relative_to(out)), "sha256": sha256(path),
                         "raw_sha256": sha256(raw_path), "shape": list(linear.shape[1:])})
    poses["frames"] = [{**r, "file_path": f"images/{stem}.npz"} for r, stem in zip(poses["frames"], stems)]
    write_json(out / "transforms_train.json", poses)
    # No transforms_test is copied/read: the training dataset is source-only.
    shutil.copyfile(ply, out / "points3d.ply")
    bank = out / "actions.pt"
    with bank.open("xb") as f:
        torch.save({"actions": actions.cpu(), "stems": stems, "mcf": asdict(CONFIG)}, f)
    write_json(out / "MANIFEST.json", {
        "schema": "fujinsplat.scene.v1", "scene": a.scene, "rows": rows,
        "mcf": asdict(CONFIG), "actions_sha256": sha256(bank),
        "point_cloud_sha256": sha256(out / "points3d.ply"),
        "poses_sha256": sha256(out / "transforms_train.json"),
        "controller_sha256": sha256(a.controller), "base_sha256": sha256(a.base_checkpoint),
        "controller_training_kind": training_kind,
        "source_clean_reads": 0, "held_target_reads": 0,
    })


if __name__ == "__main__":
    main()
