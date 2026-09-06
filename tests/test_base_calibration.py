"""Portable Base fitting and bundled-input checks using artificial CPU data."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fujinsplat.calibrate_base import (
    CalibrationBase, CalibrationSettings, _fit, main, project_parent_state,
    save_calibrated_base,
)
from fujinsplat.io import SCENE_COUNTS, read_json, sha256, write_json
from fujinsplat.native_base import NativeBase, load_native_base
from fujinsplat.synthesis_inputs import input_metadata


BUNDLE = Path(__file__).resolve().parents[1] / "fujinsplat/data/synthesis_prerequisites"


class BaseCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_calibration_forward_matches_inference_model(self):
        torch.manual_seed(72)
        calibration = CalibrationBase()
        with torch.no_grad():
            for value in calibration.parameters():
                value.uniform_(-.01, .01)
        inference = NativeBase()
        inference.load_state_dict(calibration.state_dict(), strict=True)
        x = torch.rand(65, 3)
        torch.testing.assert_close(calibration(x), inference(x), atol=0, rtol=0)

    def test_two_artificial_optimizer_steps_and_loadable_output(self):
        rng = np.random.default_rng(74)
        raw = [rng.uniform(.08, .7, (40, 48, 3)).astype(np.float32) for _ in range(2)]
        targets = [np.round((x * .8 + .03) * 255).astype(np.uint8) for x in raw]
        settings = CalibrationSettings(warmup_steps=1, joint_steps=1, pixels_per_update=128)
        model = CalibrationBase()
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            fitted = _fit(raw, targets, ["0001", "0002"], model, torch.device("cpu"),
                          settings.seed, 1, 1, output / "TRAIN.jsonl", settings)
            self.assertTrue(all(torch.isfinite(v).all() for v in fitted.state_dict().values()))
            self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in fitted.state_dict().items()))
            rows = [json.loads(line) for line in (output / "TRAIN.jsonl").read_text().splitlines()]
            self.assertEqual([r["stage"] for r in rows], ["BACK_WARMUP", "JOINT"])
            checkpoint = save_calibrated_base(output, fitted, "Hinoki", ["0001", "0002"], settings,
                                               {"parent_checkpoint_sha256": "artificial-fixture"})
            restored = load_native_base(checkpoint)
            x = torch.rand(17, 3)
            torch.testing.assert_close(restored(x), fitted(x), atol=0, rtol=0)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual((payload["source_clean_reads"], payload["held_test_reads"]), (0, 0))

    def test_parent_projection_preserves_constant_lattice_and_folds_front(self):
        state = NativeBase().state_dict()
        state["fine_action"] = torch.full((17, 17, 17, 3), .02)
        state["base_exposure"] = torch.tensor(.1)
        state["exposure_delta"] = torch.tensor(.2)
        state["matrix_log_residual"] = torch.tensor([[0, .02, 0], [0, 0, 0], [0, 0, 0.]])
        projected = project_parent_state(state)
        self.assertEqual(tuple(projected.fine_action.shape), (9, 9, 9, 3))
        torch.testing.assert_close(projected.fine_action, torch.full((9, 9, 9, 3), .02))
        torch.testing.assert_close(projected.base_exposure, torch.tensor(.3))
        torch.testing.assert_close(projected.base_row_matrix, torch.matrix_exp(state["matrix_log_residual"]).T)
        torch.testing.assert_close(projected.matrix_log_residual, torch.zeros(3, 3))

    def test_check_only_does_not_read_images_or_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw/Hinoki/train"
            rgb = root / "rgb/Hinoki/train"
            raw.mkdir(parents=True)
            rgb.mkdir(parents=True)
            stems = [f"{i:04d}" for i in range(1, 23)]
            for stem in stems:
                (raw / f"{stem}.npz").write_bytes(b"artificial RAW placeholder")
                (rgb / f"{stem}.JPG").write_bytes(b"artificial hazy RGB placeholder")
            write_json(rgb.parent / "transforms_train.json", {"frames": [{"file_path": s} for s in stems]})
            parent = root / "parent.pt"
            parent.write_bytes(b"artificial parent placeholder")
            argv = ["calibrate-base", "--scene", "Hinoki", "--raw-root", str(root / "raw"),
                    "--rgb-root", str(root / "rgb"), "--initial-parent", str(parent), "--check-only"]
            buffer = io.StringIO()
            with patch("sys.argv", argv), patch("torch.load", side_effect=AssertionError("weights read")), \
                 patch("fujinsplat.calibrate_base.load_raw", side_effect=AssertionError("RAW read")), \
                 patch("fujinsplat.calibrate_base.Image.open", side_effect=AssertionError("RGB read")), \
                 patch("fujinsplat.calibrate_base._fit", side_effect=AssertionError("training")), redirect_stdout(buffer):
                main()
            report = json.loads(buffer.getvalue())
            self.assertEqual((report["source_views"], report["image_reads"]), (22, 0))
            self.assertFalse(report["training_started"])
            self.assertEqual(report["schedule"], asdict(CalibrationSettings()))

    def test_bundled_prerequisites_and_consumer_compatibility(self):
        manifest = read_json(BUNDLE / "MANIFEST.json")
        for name, row in manifest["files"].items():
            self.assertEqual(sha256(BUNDLE / name), row["sha256"])
        pop = read_json(BUNDLE / "raw_population_195.json")
        self.assertEqual(len(pop["t"]), 195)
        self.assertEqual({s: pop["scene"].count(s) for s in SCENE_COUNTS}, SCENE_COUNTS)
        self.assertTrue(np.isfinite(pop["pivot"]).all())
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            for name in ("capture_a", "capture_b"):
                (cache / f"{name}.npy").write_bytes(b"metadata-only placeholder")
                write_json(cache / f"{name}.json", {})
            depth = cache / "depth.json"
            write_json(depth, {"captures": {"capture_a": {"t": .3}, "capture_b": {"t": .4}}})
            result, _, wb, names = input_metadata(cache, BUNDLE / "raw_population_195.json", depth,
                BUNDLE / "realx_camera_wb.json", expected=2)
            self.assertEqual(result, pop)
            np.testing.assert_array_equal(wb, [2.371094, 1.0, 1.660156])
            self.assertEqual(names, ["capture_a", "capture_b"])


if __name__ == "__main__":
    unittest.main()
