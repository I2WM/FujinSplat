"""Explicit portable entry points; importing this module starts no work."""

import argparse
from importlib import import_module
import json
import math
from pathlib import Path
import re
import sys

from . import __version__


COMMANDS = {
    "calibrate-base": ("calibrate_base", "Fit a per-scene hazy-camera Base ISP from source RAW/RGB"),
    "acquire": ("acquire_mit", "Download the external MIT RAW captures for synthesis"),
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


def config_arguments(path, overrides):
    """Turn one flat JSON object into the existing CLI arguments; no shell expansion."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("command"), str) or data["command"] not in COMMANDS:
        raise ValueError("config must be a JSON object with a supported 'command'")
    command = data.pop("command")
    explicit = {arg.split("=", 1)[0] for arg in overrides if arg.startswith("--")}
    args, seen = [], set()
    for key, value in data.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", key):
            raise ValueError(f"invalid config option: {key}")
        option = "--" + key.replace("_", "-")
        if option in seen:
            raise ValueError(f"duplicate config option: {option}")
        seen.add(option)
        if option in explicit or value is None or value is False:
            continue
        if value is True:
            args.append(option)
        elif isinstance(value, list):
            if not value:
                continue
            if not all(type(v) in (str, int, float) and (not isinstance(v, float) or math.isfinite(v)) for v in value):
                raise ValueError(f"config list must contain finite numbers or strings: {key}")
            args.extend([option, *map(str, value)])
        elif type(value) in (str, int, float) and (not isinstance(value, float) or math.isfinite(value)):
            # The equals form also preserves spaces and values beginning with '-'.
            args.append(f"{option}={value}")
        else:
            raise ValueError(f"config option must be a scalar or a list: {key}")
    return [command, *args, *overrides]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="fujinsplat", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(f"  {k:18s} {v[1]}" for k, v in COMMANDS.items())
               + "\n\nUse fujinsplat COMMAND --help for arguments. No paths or jobs are assumed.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", metavar="FILE.json", help="run the command and options specified in a flat JSON file")
    if not argv:
        parser.print_help()
        return
    if argv[0] == "--config" or argv[0].startswith("--config="):
        if argv[0] == "--config":
            if len(argv) < 2 or argv[1].startswith("--"):
                parser.error("--config requires a JSON file")
            path, overrides = argv[1], argv[2:]
        else:
            path, overrides = argv[0].split("=", 1)[1], argv[1:]
        try:
            argv = config_arguments(path, overrides)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"cannot load config: {error}")
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
