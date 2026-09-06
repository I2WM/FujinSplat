"""Validate the fitted action population and external synthesis inputs."""

from collections import Counter
import numpy as np

from .io import SCENE_COUNTS, read_json, require_source


def input_metadata(j_cache, population, depth_t, camera_wb, expected=1400):
    pop, depths, wb = read_json(population), read_json(depth_t), read_json(camera_wb)
    if len(pop["t"]) != 195 or np.asarray(pop["pivot"]).shape != (195, 3):
        raise ValueError("exactly the 195-pair fitted population is required")
    if Counter(pop["scene"]) != Counter(SCENE_COUNTS) or len(pop["stem"]) != 195:
        raise ValueError("fitted population scene counts mismatch")
    pairs = list(zip(pop["scene"], pop["stem"]))
    if len(set(pairs)) != 195:
        raise ValueError("fitted population contains duplicate pairs")
    for scene, stem in pairs:
        require_source(scene, str(stem))
    names = sorted(p.stem for p in j_cache.glob("*.npy"))
    if len(names) != expected:
        raise ValueError(f"expected {expected} external captures, found {len(names)}; no fallback")
    if set(names) != set(depths["captures"]):
        raise ValueError("J cache / depth capture population mismatch")
    for name in names:
        if not (j_cache / f"{name}.json").is_file():
            raise FileNotFoundError(j_cache / f"{name}.json")
    global_wb = wb["global"]
    if not all(global_wb.get(k) for k in ("constant_across_scenes", "constant_across_views", "constant_haze_vs_clean")):
        raise ValueError("expected the declared global RealX WB")
    return pop, depths, np.asarray(global_wb["camera_wb"]), names
