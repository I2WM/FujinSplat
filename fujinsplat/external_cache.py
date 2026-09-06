"""External Sony Base calibration and clean-image cache generation.

Two explicit stages: fit ONE shared CCM on a declared calibration manifest;
then develop external captures with their recorded WB and that frozen CCM.
Inputs are the external captures listed in the supplied manifest.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import compiler as C
from .io import create_output, read_json, sha256, write_json


def capture_rows(path):
    data = read_json(path)
    if data.get("schema") != "fujinsplat.external_raw.v1":
        raise ValueError("explicit external capture manifest required")
    rows = data["rows"]
    names = [r["capture_id"] for r in rows]
    if len(names) != len(set(names)) or not names:
        raise ValueError("empty or duplicate capture population")
    for row in rows:
        if Path(row["capture_id"]).name != row["capture_id"] or "/" in row["capture_id"] or "\\" in row["capture_id"]:
            raise ValueError("capture_id must be a filename component")
        if Path(row["raw"]).suffix.upper() != ".ARW":
            raise ValueError("Sony ARW input required")
    return sorted(rows, key=lambda r: r["capture_id"])


def read_raw(path, reference=False):
    import rawpy
    with rawpy.imread(str(path)) as r:
        lin = r.postprocess(user_wb=[1, 1, 1, 1], no_auto_bright=True,
                            output_bps=16, gamma=(1, 1), output_color=rawpy.ColorSpace.raw,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
        wb = np.asarray(r.camera_whitebalance, float)[:3]
    wb /= max(wb[1], 1e-9)
    target = None
    if reference:
        with rawpy.imread(str(path)) as r:
            cam = r.postprocess(use_camera_wb=True, no_auto_bright=True,
                                output_bps=8, output_color=rawpy.ColorSpace.sRGB,
                                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
        target = C.dec(cam.astype(np.float64) / 255)
    return lin, wb, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("fit_matrix", "build_cache"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--matrix", type=Path, help="sealed shared matrix JSON for build_cache")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = capture_rows(a.manifest)
    if a.stage == "fit_matrix":
        # Accumulate the declared sampled rows without retaining full captures.
        # Fit one shared matrix by pooled least squares.
        rng = np.random.default_rng(0)
        aa, bb, hashes = [], [], []
        for row in rows:
            lin, wb, target = read_raw(row["raw"], reference=True)
            flat = (lin.astype(np.float64) / 65535 * wb).reshape(-1, 3)
            idx = rng.choice(len(flat), min(60000, len(flat)), replace=False)
            aa.append(flat[idx]); bb.append(target.reshape(-1, 3)[idx])
            hashes.append({"capture_id": row["capture_id"], "raw_sha256": sha256(row["raw"])})
        matrix, *_ = np.linalg.lstsq(np.concatenate(aa), np.concatenate(bb), rcond=None)
        write_json(a.output, {"schema": "fujinsplat.external_matrix.v1", "shared_matrix": matrix.tolist(),
                   "calibration_rows": hashes, "seed": 0, "pixels_per_capture": 60000,
                   "realx_image_reads": 0})
        return
    if a.matrix is None:
        raise ValueError("--matrix required; no guessing a CCM from other cameras")
    frozen = read_json(a.matrix)
    if frozen.get("schema") != "fujinsplat.external_matrix.v1" or frozen.get("realx_image_reads") != 0:
        raise ValueError("invalid shared external Base matrix")
    matrix = np.asarray(frozen["shared_matrix"], float)
    out = create_output(a.output)
    for row in rows:
        lin, wb, _ = read_raw(row["raw"])
        effective = np.diag(wb) @ matrix
        h, w = lin.shape[:2]
        j = np.empty((h, w, 3), np.float32)
        for first in range(0, h, 512):
            strip = lin[first:first + 512].astype(np.float64) / 65535
            j[first:first + 512] = C.enc(np.clip(strip @ effective, 0, 1)).astype(np.float32)
        small = cv2.resize(j, (1506, 1006), interpolation=cv2.INTER_AREA)
        with (out / f"{row['capture_id']}.npy").open("xb") as f:
            np.save(f, small.astype(np.float16), allow_pickle=False)
        write_json(out / f"{row['capture_id']}.json", {
            "name": row["capture_id"], "h": h, "w": w, "wb": wb.tolist(), "M": effective.tolist(),
            "shared_matrix": matrix.tolist(), "raw_sha256": sha256(row["raw"]), "matrix_sha256": sha256(a.matrix)})


if __name__ == "__main__":
    main()
