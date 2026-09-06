"""Produce and seal 1400 full-resolution analytic syntheses plus compact caches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import shutil

import numpy as np
import torch

from . import compiler as C
from .io import read_json, sha256, write_json
from .synthesis_inputs import input_metadata
from .synthesis import COMPILER, abi, reverse_capture
from .training_cache import write_cache


def one(job):
    i, name, pivot_index, pivot_sensor, t, ref_med, over, wb, j_cache, output, scratch = job
    output, j_cache, scratch = Path(output), Path(j_cache), Path(scratch)
    receipt = output / "receipts" / f"{i:04d}.json"
    if receipt.exists():
        row = read_json(receipt)
        if sha256(row["pair"]) != row["sha256"] or sha256(row["training_pair"]) != row["training_pair_sha256"]:
            raise ValueError("existing synthesis artifact hash drift")
        return row
    torch.set_num_threads(2)
    meta = read_json(j_cache / f"{name}.json")
    matrix = np.asarray(meta["M"], np.float64)
    cached = np.load(j_cache / f"{name}.npy", allow_pickle=False).astype(np.float64)
    raw = np.clip(C.dec(cached) @ np.linalg.inv(matrix), 0, None)
    y = np.array([.2126, .7152, .0722])
    uncapped = ref_med / max(float(np.median((raw @ matrix) @ y)), 1e-9)
    gain = min(uncapped, 10.0)
    direction = np.asarray(pivot_sensor) * wb
    direction /= direction.mean()
    level = float(np.median(np.clip((raw * gain) @ matrix, 0, None) @ y))
    pivot = np.clip(direction * level * over, 1e-6, None)
    arrays, checks = reverse_capture(raw, matrix, pivot, float(t), gain)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError(f"nonfinite synthesis: {name}")
    if min(checks[k] for k in ("action_cycle_psnr", "base_cycle_psnr", "end_to_end_cycle_psnr")) < 100:
        write_json(output / "receipts" / f"{i:04d}.FAILED.json", {"capture": name, **checks})
        raise ValueError(f"100dB stored-array gate failed: {name} {checks}")
    local = scratch / f"{i:04d}.npz"
    with local.open("wb") as f:
        # Original clean X/J can be regenerated from the hash-bound J cache,
        # M and gain; keep the actual training triplet plus exact action.
        np.savez_compressed(f, **{k: v for k, v in arrays.items() if k not in ("X_clean", "J_unfloored")})
    digest = sha256(local)
    path = output / "pairs" / local.name
    staged = path.with_suffix(".staging")
    shutil.copyfile(local, staged)
    if sha256(staged) != digest:
        raise ValueError("synthesis publication mismatch")
    staged.replace(path)
    local.unlink()
    cache = output / "training" / f"{i:04d}.npz"
    compact = write_cache(arrays, i, cache)
    row = {"index": i, "capture_id": name, "pair": str(path), "sha256": digest, **compact,
           "J_cache_sha256": sha256(j_cache / f"{name}.npy"), "base_metadata_sha256": sha256(j_cache / f"{name}.json"),
           "pivot_index": pivot_index, "t": float(t), "gain": gain, "gain_uncapped": uncapped, **checks}
    write_json(receipt, row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--j-cache", type=Path, required=True)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--depth-t", type=Path, required=True)
    p.add_argument("--camera-wb", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    pop, depths, wb, names = input_metadata(a.j_cache, a.population, a.depth_t, a.camera_wb)
    a.output.mkdir(parents=True, exist_ok=True)
    for child in ("pairs", "training", "receipts"):
        (a.output / child).mkdir(exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    contract = {"mcf": abi(), "compiler": COMPILER, "exposure_coordinate": "base_linear",
                "bindings": {k: sha256(v) for k, v in {"population": a.population, "depth_t": a.depth_t, "camera_wb": a.camera_wb}.items()}}
    if (a.output / "CONTRACT.json").exists():
        if read_json(a.output / "CONTRACT.json") != contract:
            raise ValueError("cannot resume a changed synthesis contract")
    else:
        write_json(a.output / "CONTRACT.json", contract)
    rng = np.random.default_rng(20260902)
    jobs = []
    for i, name in enumerate(names):
        pi = int(rng.integers(195))
        jobs.append((i, name, pi, pop["pivot"][pi], depths["captures"][name]["t"], pop["clean_median"],
                     pop["airlight_over_median"], wb, str(a.j_cache), str(a.output), str(a.scratch)))
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result(); results.append(row)
            if i % 10 == 0 or i == 1:
                print(json.dumps({"synthesized": i, "total": 1400, "last_cycle_db": row["end_to_end_cycle_psnr"]}), flush=True)
    results.sort(key=lambda r: r["index"])
    if not (a.output / "MANIFEST.json").exists():
        write_json(a.output / "MANIFEST.json", {"schema": "fujinsplat.synthetic.v1", "mcf": abi(),
            "compiler": COMPILER, "external_captures": 1400, "rows": results, "augmentations": 8,
            "realx_image_reads": 0, "held_target_reads": 0, "stored_array_cycle_gate": True,
            "contract_sha256": sha256(a.output / "CONTRACT.json")})
    print("COMPLETE_1400_EXACT_ANALYTIC_SYNTHETIC_OBSERVATIONS", flush=True)


if __name__ == "__main__":
    main()
