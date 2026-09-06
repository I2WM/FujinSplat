"""Pose-only held rendering from a sealed static PLY; never open captures."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from .color import encode, q8
from .gs_runtime import make_camera
from .io import SCENE_COUNTS, create_output, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--poses", type=Path, required=True, help="official transforms_test.json; metadata only")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seal = read_json(a.run / "SEALED.json")
    if seal.get("status") != "SEALED_STATIC_GAUSSIANS" or seal.get("held_target_reads") != 0 or seal.get("source_clean_reads") != 0:
        raise ValueError("run is not frozen/clean-free")
    ply = a.run / seal["point_cloud"]
    if sha256(ply) != seal["point_cloud_sha256"]:
        raise ValueError("PLY hash drift")
    poses = read_json(a.poses)
    frames = poses["frames"]
    stems = [Path(r["file_path"]).stem for r in frames]
    n = SCENE_COUNTS[seal["scene"]]
    if sorted(stems) != [f"{i:04d}" for i in range(n + 1, n + 5)]:
        raise ValueError("not the four official held stems")
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    gs = GaussianModel(3, "default")
    gs.load_ply(str(ply), use_train_test_exp=False)
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, device="cuda")
    out = create_output(a.output)
    h, w = seal["native_shape"]
    rows = []
    with torch.inference_mode():
        for i, frame in enumerate(frames):
            camera = make_camera(frame, poses["camera_angle_x"], h, w, i)
            linear = render(camera, gs, pipe, background, use_trained_exp=False, separate_sh=False)["render"]
            if not torch.isfinite(linear).all():
                raise FloatingPointError("nonfinite static render")
            path = out / f"{stems[i]}.png"
            rgb = q8(encode(linear.clamp(0, 1))).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(rgb).save(path)
            rows.append({"scene": seal["scene"], "stem": stems[i], "relative": path.name, "sha256": sha256(path)})
    write_json(out / "PREDICTIONS_SEALED.json", {"status": "SEALED_BEFORE_TARGET_READS", "rows": rows,
               "run_seal_sha256": sha256(a.run / "SEALED.json"), "poses_sha256": sha256(a.poses),
               "controller_sha256": seal["controller_sha256"],
               "held_target_reads": 0, "query_raw_reads": 0, "query_controller_reads": 0,
               "query_base_reads": 0, "query_delta_reads": 0})


if __name__ == "__main__":
    main()
