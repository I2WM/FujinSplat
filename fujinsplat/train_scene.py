"""18k static linear 3DGS with training-only centered MCF alpha retention.

No clean data, held poses, held targets, controller training manifest, source
score, early stopping or metric-based checkpoint selection is loaded here.
"""

import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch

from .color import encode
from .delta import CenteredDelta
from .gs_runtime import developed_target, load_source
from .io import create_output, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/gs.json")
    p.add_argument("--no-delta", action="store_true")
    a = p.parse_args()
    cfg = read_json(a.config)
    if cfg["iterations"] != 18000 or cfg["sh_degree"] != 3 or cfg["seed"] != 190087:
        raise ValueError("formal 18k/SH3/seed contract mismatch")
    if cfg.get("alpha_objective") != "paper_joint_render_loss" or cfg.get("ssim_backend") != "upstream_torch_both_image_gradients":
        raise ValueError("paper joint-gradient objective required")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    from arguments import OptimizationParams
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import l1_loss, ssim
    op = argparse.ArgumentParser()
    opt_group = OptimizationParams(op)
    defaults = op.parse_args([])
    for k, v in cfg["optimization"].items():
        if not hasattr(defaults, k):
            raise ValueError(f"unknown upstream optimization option: {k}")
        setattr(defaults, k, v)
    defaults.iterations = cfg["iterations"]
    opt = opt_group.extract(defaults)
    manifest, cameras, full_base, actions, point_cloud, extent = load_source(a.dataset, cfg["resolution"])
    out = create_output(a.output)
    write_json(out / "CONTRACT.json", {"config": cfg, "no_delta": a.no_delta,
               "dataset_manifest_sha256": sha256(a.dataset / "MANIFEST.json"),
               "controller_sha256": manifest["controller_sha256"], "base_sha256": manifest["base_sha256"],
               "source_clean_reads": 0, "held_target_reads": 0,
               "code_sha256": {f.name: sha256(f) for f in Path(__file__).parent.glob("*.py")}})
    gs = GaussianModel(3, "default")
    gs.create_from_pcd(point_cloud, cameras, extent)
    gs.training_setup(opt)
    delta = CenteredDelta(actions)
    alpha_optimizer = torch.optim.Adam([delta.alpha], lr=cfg["alpha_lr"])
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, device="cuda")
    stack = list(range(len(cameras)))
    static_targets = {}
    start = time.monotonic()
    import json
    with (out / "TRAIN.jsonl").open("x") as log:
        for iteration in range(1, cfg["iterations"] + 1):
            gs.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gs.oneupSHdegree()
            if not stack:
                stack = list(range(len(cameras)))
            index = stack.pop(random.randrange(len(stack)))
            camera = cameras[index]
            package = render(camera, gs, pipe, background, use_trained_exp=False, separate_sh=False)
            linear = package["render"]
            display = encode(linear.clamp(0, 1))
            action = delta(index)
            active = not a.no_delta and cfg["alpha_start"] <= iteration <= cfg["alpha_stop"]
            if active:
                static_targets.clear()
                target = developed_target(full_base[index].cuda(), action, linear.shape[-2:], active=True)
            else:
                # Alpha is fixed before/after its window. Cache the exact
                # developed target instead of recomputing an identical MCF.
                if index not in static_targets:
                    static_targets[index] = developed_target(full_base[index].cuda(), action, linear.shape[-2:], active=False)
                target = static_targets[index]
            # The pinned fused kernel returns no gradient for its second image.
            # Use upstream differentiable SSIM so BOTH GS and alpha receive it.
            structure = ssim(display, target)
            loss = (1 - opt.lambda_dssim) * l1_loss(display, target) + opt.lambda_dssim * (1 - structure)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite GS loss at {iteration}")
            loss.backward()
            with torch.no_grad():
                radii, visible = package["radii"], package["visibility_filter"]
                if iteration < opt.densify_until_iter:
                    gs.max_radii2D[visible] = torch.maximum(gs.max_radii2D[visible], radii[visible])
                    gs.add_densification_stats(package["viewspace_points"], visible)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gs.densify_and_prune(opt.densify_grad_threshold, 0.005, extent, size_threshold, radii)
                if iteration < cfg["iterations"]:
                    gs.optimizer.step()
                    if active:
                        alpha_optimizer.step()
                        delta.project_()
                gs.optimizer.zero_grad(set_to_none=True)
                alpha_optimizer.zero_grad(set_to_none=True)
            if iteration == 1 or iteration % 100 == 0:
                row = {"iteration": iteration, "loss": float(loss.detach()), "alpha_active": active,
                       "alpha_mean": float(delta.alpha.detach().mean()), "elapsed_seconds": time.monotonic() - start}
                log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
    ply = out / "point_cloud/iteration_18000/point_cloud.ply"
    gs.save_ply(str(ply))
    with (out / "checkpoint.pth").open("xb") as f:
        torch.save((gs.capture(), cfg["iterations"]), f)
    with (out / "delta.pt").open("xb") as f:
        torch.save(delta.state_dict(), f)
    write_json(out / "SEALED.json", {"status": "SEALED_STATIC_GAUSSIANS", "scene": manifest["scene"],
               "point_cloud": str(ply.relative_to(out)), "point_cloud_sha256": sha256(ply),
               "controller_sha256": manifest["controller_sha256"], "base_sha256": manifest["base_sha256"],
               "dataset_manifest_sha256": sha256(a.dataset / "MANIFEST.json"),
               "source_clean_reads": 0, "held_target_reads": 0,
               "native_shape": manifest["rows"][0]["shape"], "sh_degree": 3,
               "elapsed_seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
