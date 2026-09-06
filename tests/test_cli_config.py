"""JSON configuration dispatch checks; no data reads, training or subprocesses."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fujinsplat.cli import config_arguments, main
from fujinsplat.train_controller import parser as training_parser


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        root = ROOT / "tests/.tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "options.json"

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_scalars_lists_flags_and_cli_override(self):
        self.write({"command": "train-controller", "manifest": "folder with spaces/manifest.json",
                    "steps": 1500, "batch_size": 16, "lr": 1e-5, "save_steps": [100, 200],
                    "check_only": True, "initial": None})
        args = config_arguments(self.path, ["--steps", "300", "--batch-size=4", "--save-steps", "50"])
        self.assertEqual(args[0], "train-controller")
        parsed = training_parser().parse_args(args[1:])
        self.assertEqual(parsed.manifest, Path("folder with spaces/manifest.json"))
        self.assertEqual((parsed.steps, parsed.batch_size, parsed.save_steps), (300, 4, [50]))
        self.assertTrue(parsed.check_only)
        self.assertEqual(parsed.lr, 1e-5)
        self.assertIsNone(parsed.initial)

    def test_false_null_and_empty_list_omit_options(self):
        self.write({"command": "train-controller", "manifest": "manifest.json",
                    "check-only": False, "initial": None, "save_steps": []})
        args = config_arguments(self.path, [])
        self.assertEqual(args, ["train-controller", "--manifest=manifest.json"])

    def test_dispatches_existing_trainer_and_restores_argv(self):
        self.write({"command": "train-controller", "manifest": "manifest.json", "steps": 1500})
        original = sys.argv
        with patch("fujinsplat.train_controller.run") as run:
            main(["--config", str(self.path), "--steps", "20", "--check-only"])
        run.assert_called_once()
        args = run.call_args.args[0]
        self.assertEqual(args.steps, 20)
        self.assertTrue(args.check_only)
        self.assertIs(sys.argv, original)

    def test_example_files_help_without_work(self):
        with patch("subprocess.run", side_effect=AssertionError("job launched")):
            for path in sorted((ROOT / "configs").glob("*.example.json")):
                with self.subTest(name=path.name), redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        main([f"--config={path}", "--help"])
                    self.assertEqual(result.exception.code, 0)

    def test_json_check_only_stays_enabled_without_cli_flag(self):
        self.write({"command": "train-controller", "manifest": "manifest.json", "check_only": True})
        with patch("fujinsplat.train_controller.run") as run:
            main(["--config", str(self.path)])
        self.assertTrue(run.call_args.args[0].check_only)

    def test_existing_subcommand_config_not_intercepted(self):
        captured = []
        with patch("fujinsplat.train_scene.main", side_effect=lambda: captured.append(list(sys.argv))):
            main(["train-scene", "--dataset", "data", "--output", "out", "--config", "gs.json"])
        self.assertEqual(captured[0][1:], ["--dataset", "data", "--output", "out", "--config", "gs.json"])

    def test_unknown_option_still_rejected_by_existing_parser(self):
        self.write({"command": "train-controller", "manifest": "manifest.json", "unknown_option": 1})
        with patch("sys.stderr", io.StringIO()), patch("fujinsplat.train_controller.run") as run:
            with self.assertRaises(SystemExit) as result:
                main(["--config", str(self.path)])
            self.assertEqual(result.exception.code, 2)
            run.assert_not_called()

    def test_bad_config_shapes_fail_clearly(self):
        invalid = [[], {}, {"command": []}, {"command": "missing"},
                   {"command": "render", "nested": {"key": "value"}},
                   {"command": "render", "--run": "value"},
                   {"command": "render", "run": [True]},
                   {"command": "render", "value": float("nan")},
                   {"command": "render", "file_path": 1, "file-path": 2}]
        for data in invalid:
            with self.subTest(data=data):
                self.write(data)
                with self.assertRaises(ValueError):
                    config_arguments(self.path, [])
        self.path.write_text("{invalid", encoding="utf-8")
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as result:
            main(["--config", str(self.path)])
        self.assertEqual(result.exception.code, 2)

    def test_missing_config_argument_and_file(self):
        for args in (["--config"], ["--config", str(self.path)]):
            with self.subTest(args=args), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(args)
                self.assertEqual(result.exception.code, 2)

    def test_utf8_bom_and_literal_paths(self):
        self.path.write_text(json.dumps({"command": "render", "output": "F:/项目/输出", "run": "relative/run"}),
                             encoding="utf-8-sig")
        self.assertEqual(config_arguments(self.path, []),
                         ["render", "--output=F:/项目/输出", "--run=relative/run"])


if __name__ == "__main__":
    unittest.main()
