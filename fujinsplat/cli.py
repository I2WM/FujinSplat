"""Explicit portable entry points; importing this module starts no work."""

import argparse
from importlib import import_module
import sys

from . import __version__


COMMANDS = {
    "calibrate-base": ("calibrate_base", "Fit a per-scene hazy-camera Base ISP from source RAW/RGB"),
    "prepare-scene": ("prepare_scene", "Freeze source actions and prepare reconstruction inputs"),
    "train-scene": ("train_scene", "Train one 18k Gaussian scene with Delta-ISP"),
    "scene": ("scene_pipeline", "Prepare, train and render one scene"),
    "render": ("render_scene", "Render a sealed static Gaussian model using camera poses"),
    "evaluate": ("evaluate", "Score all eight sealed scenes using the reference metrics"),
    "export": ("export_weights", "Export tensor-only controller weights and a separate record"),
    "train-controller": ("train_controller", "Train on synthetic data from fresh or initial weights"),
    "synthesize": ("synthesize_parallel", "Build the 1400-capture synthesis and compact training caches"),
    "external-cache": ("external_cache", "Calibrate external CCM or build external clean caches"),
    "depth": ("depth_to_t", "Infer depth-derived transmission from explicit local weights"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="fujinsplat", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(f"  {k:18s} {v[1]}" for k, v in COMMANDS.items())
               + "\n\nUse fujinsplat COMMAND --help for arguments. No paths or jobs are assumed.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    if not argv:
        parser.print_help()
        return
    if argv[0].startswith("-"):
        parser.parse_args(argv)
        return
    command = argv.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    module = import_module(f"fujinsplat.{COMMANDS[command][0]}")
    original = sys.argv
    try:
        sys.argv = [f"fujinsplat {command}", *argv]
        if command == "train-controller":
            module.run(module.parser().parse_args())
        else:
            module.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
