"""One final all-eight readout, only after all 32 predictions are hash sealed."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from .io import SCENE_COUNTS, create_output, read_json, sha256, write_json


def image_metrics(prediction, target, perceptual, device="cpu"):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    if prediction.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("scorer requires decoded q8 RGB")
    resized = prediction.shape != target.shape
    if resized:
        prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)
    left, right = prediction.astype(np.float32) / 255, target.astype(np.float32) / 255
    def tensor(x):
        return torch.from_numpy(x).permute(2, 0, 1)[None].to(device) * 2 - 1
    with torch.inference_mode():
        lp = float(perceptual(tensor(left), tensor(right)).item())
    return {"psnr": float(peak_signal_noise_ratio(right, left, data_range=1.0)),
            "ssim": float(structural_similarity(right, left, channel_axis=2, data_range=1.0,
                         win_size=7, gaussian_weights=False, use_sample_covariance=True)),
            "lpips": lp, "render_resized_by_area": resized}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renders", type=Path, required=True, help="root/SCENE/PREDICTIONS_SEALED.json")
    p.add_argument("--rgb-root", type=Path, required=True, help="scorer's official root/SCENE/test/STEM.JPG")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", default="FujinSplat-new")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--purpose", choices=("final_readout", "development"), default="final_readout")
    a = p.parse_args()
    sealed = []
    checkpoint_hashes = set()
    # Verify ALL outputs before opening even one target. No alignment or selection.
    for scene, n in SCENE_COUNTS.items():
        root = a.renders / scene
        m = read_json(root / "PREDICTIONS_SEALED.json")
        if m.get("status") != "SEALED_BEFORE_TARGET_READS" or m.get("held_target_reads") != 0:
            raise ValueError("missing prediction seal")
        if sorted(r["stem"] for r in m["rows"]) != [f"{i:04d}" for i in range(n + 1, n + 5)]:
            raise ValueError("held population drift")
        checkpoint_hashes.add(m["controller_sha256"])
        for r in m["rows"]:
            path = root / r["relative"]
            if r["scene"] != scene or sha256(path) != r["sha256"]:
                raise ValueError("prediction hash/scene drift")
            sealed.append({**r, "path": str(path)})
    if len(checkpoint_hashes) != 1 or Counter(r["scene"] for r in sealed) != Counter({s: 4 for s in SCENE_COUNTS}):
        raise ValueError("single shared checkpoint/all-8 protocol violated")
    out = create_output(a.output)
    write_json(out / "PRE_READOUT_SEAL.json", {"rows": sealed, "controller_sha256": next(iter(checkpoint_hashes))})
    import lpips
    perceptual = lpips.LPIPS(net="vgg", version="0.1").eval().to(a.device)
    results = []
    for r in sealed:
        # Match score_coupling_epsilon_ablation_all8_v1.py literally.
        target_path = a.rgb_root / r["scene"] / "test" / f"{r['stem']}.JPG"
        with Image.open(r["path"]) as im:
            prediction = np.asarray(im.convert("RGB")).copy()
        with Image.open(target_path) as im:
            target = np.asarray(im.convert("RGB")).copy()
        metrics = image_metrics(prediction, target, perceptual, a.device)
        results.append({"scene": r["scene"], "view_stem": r["stem"], "variant": a.variant,
                        **metrics, "prediction_sha256": r["sha256"], "target_sha256": sha256(target_path)})
    with (out / "held_view_metrics.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("scene", "view_stem", "variant", "psnr", "ssim", "lpips"))
        for r in results:
            writer.writerow((r["scene"], r["view_stem"], r["variant"], *(f"{r[k]:.10f}" for k in ("psnr", "ssim", "lpips"))))
    means = {s: {k: float(np.mean([r[k] for r in results if r["scene"] == s]))
                 for k in ("psnr", "ssim", "lpips")} for s in SCENE_COUNTS}
    equal = {k: float(np.mean([v[k] for v in means.values()])) for k in ("psnr", "ssim", "lpips")}
    import importlib.metadata
    result = {"per_scene": means, "equal_scene": equal, "rows": results,
              "evaluation_purpose": a.purpose,
              "independent_test_claim": a.purpose == "final_readout",
              "versions": {k: importlib.metadata.version(k) for k in ("scikit-image", "lpips", "torch", "numpy")},
              "protocol": "q8 RGB; skimage data_range=1 SSIM uniform7/sample covariance; LPIPS-VGG 0.1; area resize render only; no alignment"}
    write_json(out / "RESULT.json", result)
    print(equal)


if __name__ == "__main__":
    main()
