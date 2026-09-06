"""Source-only scene materialization, paper 3D optimization, sealed held renders."""

import argparse
from pathlib import Path
import subprocess
import sys

from .io import read_json, sha256


def verify_existing_controller(path, expected):
    if path.exists() and read_json(path).get("controller_sha256") != expected:
        raise ValueError(f"existing output belongs to a different controller: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument("--controller", type=Path, required=True)
    p.add_argument("--controller-format", choices=("sealed", "weights"), default="sealed")
    p.add_argument("--controller-sha256")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--rgb-root", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--renders", type=Path, required=True)
    a = p.parse_args()
    digest = sha256(a.controller)
    if a.controller_sha256 is not None and a.controller_sha256 != digest:
        raise ValueError("controller SHA256 mismatch")
    for marker in (a.prepared / "MANIFEST.json", a.run / "SEALED.json", a.renders / "PREDICTIONS_SEALED.json"):
        verify_existing_controller(marker, digest)
    def invoke(module, args):
        subprocess.run([sys.executable, "-B", "-m", module, *map(str, args)], check=True)
    if not (a.prepared / "MANIFEST.json").exists():
        verify = ["--controller-sha256", a.controller_sha256] if a.controller_sha256 else []
        invoke("fujinsplat.prepare_scene", ["--scene", a.scene, "--raw-root", a.raw_root,
            "--geometry", a.rgb_root / a.scene, "--base-checkpoint", a.base_root / a.scene / "complete_model.pt",
            "--controller", a.controller, "--controller-format", a.controller_format, "--output", a.prepared, *verify])
    if not (a.run / "SEALED.json").exists():
        invoke("fujinsplat.train_scene", ["--dataset", a.prepared, "--output", a.run])
    if not (a.renders / "PREDICTIONS_SEALED.json").exists():
        invoke("fujinsplat.render_scene", ["--run", a.run,
            "--poses", a.rgb_root / a.scene / "transforms_test.json", "--output", a.renders])


if __name__ == "__main__":
    main()
