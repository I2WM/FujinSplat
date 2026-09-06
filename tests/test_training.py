"""Synthetic warm-start tests, including one CPU step on artificial arrays only."""

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

from fujinsplat.cli import COMMANDS
from fujinsplat.controller import Controller
from fujinsplat.io import sha256, write_json
from fujinsplat.mcf import CONFIG
from fujinsplat.train_controller import initialize_model, parser, run
from fujinsplat.training_data import SyntheticPairs, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (ROOT / "tests/.tmp").mkdir(exist_ok=True)

    def initial(self, root):
        model = Controller()
        with torch.no_grad():
            for head in [*model.curve_heads, model.coupling_head]:
                head.bias.fill_(0.02)
        path = root / "initial.pt"
        torch.save(dict(model.state_dict()), path)
        return model, path

    def manifest(self, root):
        # One tiny artificial identity pair; distinct fixture IDs exercise the
        # production population gate without any actual external capture reads.
        rng = np.random.default_rng(61)
        base = rng.uniform(0.1, 0.8, (3, 1, 64)).astype(np.float32)
        pair = root / "fixture.npz"
        np.savez_compressed(pair, raw=rng.random((3, 64, 64), dtype=np.float32),
                            base=base, target=base, p=np.zeros(573, np.float32))
        digest = sha256(pair)
        data = {"schema": "fujinsplat.synthetic.v1", "mcf": asdict(CONFIG),
                "held_target_reads": 0, "realx_image_reads": 0, "external_captures": 1400,
                "stored_array_cycle_gate": True,
                "rows": [{"capture_id": f"fixture_{i:04d}", "training_pair": str(pair),
                          "training_pair_sha256": digest} for i in range(1400)]}
        path = root / "MANIFEST.json"
        write_json(path, data)
        return path

    def test_warm_start_preserves_all_parameters_and_enables_gradients(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            original, path = self.initial(Path(tmp))
            with patch.object(Path, "read_text", side_effect=AssertionError("sidecar read")):
                model = initialize_model(path, "cpu", sha256(path))
            self.assertTrue(model.training)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            model(torch.rand(1, 3, 64, 64)).square().mean().backward()
            for head in [*model.curve_heads, model.coupling_head]:
                self.assertIsNotNone(head.bias.grad)
                self.assertTrue(torch.isfinite(head.bias.grad).all())
                self.assertGreater(float(head.bias.grad.abs().sum()), 0)

    def test_one_artificial_cpu_step_and_tensor_only_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            root = Path(tmp)
            original, initial = self.initial(root)
            initial_hash = sha256(initial)
            manifest = self.manifest(root)
            out = root / "warm"
            args = parser().parse_args(["--manifest", str(manifest), "--initial", str(initial),
                "--initial-sha256", initial_hash, "--output", str(out), "--device", "cpu",
                "--steps", "1", "--batch-size", "1", "--lr", "0.00001"])
            with redirect_stdout(io.StringIO()):
                run(args)
            self.assertEqual(sha256(initial), initial_hash)
            state = torch.load(out / "checkpoint.pt", weights_only=True)
            self.assertIs(type(state), dict)
            self.assertEqual(len(state), 30)
            self.assertTrue(all(type(v) is torch.Tensor for v in state.values()))
            self.assertFalse(hasattr(state, "_metadata"))
            self.assertTrue(any(not torch.equal(state[k], v) for k, v in original.state_dict().items()))
            meta = json.loads((out / "checkpoint.json").read_text())
            self.assertEqual(meta["training_kind"], "synthetic_warm_start")
            self.assertEqual(meta["current_training_data"], "external_synthetic")
            self.assertEqual(meta["initial_checkpoint_sha256"], initial_hash)
            self.assertEqual(meta["schedule"]["steps"], 1)
            self.assertTrue((out / "optimizer_state.pt").exists())

    def test_check_only_does_not_load_images_or_initialize_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            root = Path(tmp)
            _, initial = self.initial(root)
            manifest = self.manifest(root)
            args = parser().parse_args(["--manifest", str(manifest), "--initial", str(initial), "--check-only"])
            buffer = io.StringIO()
            with patch("fujinsplat.train_controller.initialize_model", side_effect=AssertionError("model initialized")), \
                 patch("numpy.load", side_effect=AssertionError("image opened")), redirect_stdout(buffer):
                run(args)
            report = json.loads(buffer.getvalue())
            self.assertEqual(report["stage"], "synthetic_warm_start")
            self.assertEqual(report["image_reads"], 0)
            self.assertFalse(report["training_started"])

    def test_non_synthetic_manifest_is_rejected_before_image_reads(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            path = Path(tmp) / "unsupported.json"
            write_json(path, {"schema": "unsupported_source_training", "rows": []})
            with patch("numpy.load", side_effect=AssertionError("image opened")), self.assertRaises(ValueError):
                validate_manifest(path)
            with self.assertRaises(ValueError):
                SyntheticPairs({"schema": "unsupported_source_training"})

    def test_no_removed_training_entry_points(self):
        self.assertNotIn("source-manifest", COMMANDS)
        for name in ("prepare_source_manifest", "compare_controllers", "controller_pipeline",
                     "continue_controller_pipeline", "batch_experiments"):
            self.assertFalse((ROOT / "fujinsplat" / f"{name}.py").exists())
            self.assertFalse((ROOT / "fujinsplat/experiments" / f"{name}.py").exists())

    def test_initial_digest_requires_initial_weights(self):
        with self.assertRaises(ValueError):
            initialize_model(None, "cpu", "0" * 64)


if __name__ == "__main__":
    unittest.main()
