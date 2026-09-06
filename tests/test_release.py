"""CPU-only release entry-point/serialization tests; no captures or training."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fujinsplat.cli import COMMANDS, main
from fujinsplat.controller import Controller, load_weights
from fujinsplat.io import sha256


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        (ROOT / "tests/.tmp").mkdir(exist_ok=True)

    def test_packaged_gs_config_matches_reference(self):
        paths = [ROOT / "configs/gs.json", ROOT / "fujinsplat/configs/gs.json"]
        self.assertEqual(sha256(paths[0]), sha256(paths[1]))
        config = json.loads(paths[0].read_text())
        self.assertEqual((config["iterations"], config["sh_degree"]), (18000, 3))

    def test_record_averages_and_population(self):
        record = json.loads((ROOT / "configs/paper_results.json").read_text())
        self.assertEqual(record["result_type"], "paper_reported")
        self.assertEqual(len(record["per_scene"]), 8)
        from fujinsplat.io import SCENE_COUNTS
        self.assertEqual(set(record["per_scene"]), set(SCENE_COUNTS))
        self.assertEqual((record["source_pairs"], record["held_views"]), (195, 32))
        for metric, value in record["equal_scene"].items():
            mean = sum(s[metric] for s in record["per_scene"].values()) / 8
            # Both the scene cells and overall average are rounded in the PDF.
            precision = record["printed_decimal_places"][metric]
            self.assertLessEqual(abs(mean - value), 10 ** -precision)
        self.assertEqual(record["paper_controller_training"]["steps"], 1500)

    def test_top_level_help_and_unknown_command(self):
        with redirect_stdout(io.StringIO()):
            main([])
            with self.assertRaises(SystemExit) as caught:
                main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["not-a-command"])
        self.assertEqual(caught.exception.code, 2)

    def test_all_portable_command_help_is_side_effect_free(self):
        with patch("fujinsplat.io.load_raw", side_effect=AssertionError("capture read")), \
             patch("subprocess.run", side_effect=AssertionError("job launched")):
            for command in COMMANDS:
                with self.subTest(command=command), redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main([command, "--help"])
                    self.assertEqual(caught.exception.code, 0)

    def test_weights_need_no_sidecar_and_preserve_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            path = Path(tmp) / "controller.pt"
            original = Controller()
            torch.save(dict(original.state_dict()), path)
            with patch.object(Path, "read_text", side_effect=AssertionError("sidecar read")):
                restored = load_weights(path, expected_sha256=sha256(path))
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(restored.training)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))

    def test_hash_mismatch_fails_before_deserialization(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            path = Path(tmp) / "controller.pt"
            torch.save(dict(Controller().state_dict()), path)
            with patch("torch.load") as load, self.assertRaises(ValueError):
                load_weights(path, expected_sha256="0" * 64)
            load.assert_not_called()

    def test_metadata_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            path = Path(tmp) / "controller.pt"
            torch.save({"state_dict": Controller().state_dict(), "metadata": {}}, path)
            with self.assertRaises(ValueError):
                load_weights(path)
            state = dict(Controller().state_dict())
            state["trunk.0.weight"][0, 0] = float("nan")
            torch.save(state, path)
            with self.assertRaises(ValueError):
                load_weights(path)

    def test_cli_and_package_version_agree(self):
        from fujinsplat import __version__
        self.assertIn(f'version = "{__version__}"', (ROOT / "pyproject.toml").read_text())

    def test_pipeline_rejects_different_checkpoint_outputs(self):
        from fujinsplat.io import write_json
        from fujinsplat.scene_pipeline import verify_existing_controller
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/.tmp") as tmp:
            path = Path(tmp) / "SEALED.json"
            verify_existing_controller(path, "expected")
            write_json(path, {"controller_sha256": "expected"})
            verify_existing_controller(path, "expected")
            with self.assertRaises(ValueError):
                verify_existing_controller(path, "different")

    def test_upstream_license_unchanged(self):
        actual = hashlib.sha256((ROOT / "LICENSE.md").read_bytes()).hexdigest()
        self.assertEqual(actual, "c5ba70a2194af2aefe85dfe3da68608dcb3abd21a3aa53b55aa297c2f0b60eb3")


if __name__ == "__main__":
    unittest.main()
