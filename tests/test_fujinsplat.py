"""CPU-only contract tests. No optimizer step, real capture read or CUDA use."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fujinsplat import compiler as C
from fujinsplat.color import decode, encode, reconstruction_loss, summary
from fujinsplat.controller import Controller, checkpoint_payload, load_checkpoint
from fujinsplat.export_weights import export
from fujinsplat.io import sha256, write_json
from fujinsplat.delta import CenteredDelta
from fujinsplat.io import SCENE_COUNTS, create_output, require_source
from fujinsplat.gs_runtime import developed_target
from fujinsplat.mcf import MCFConfig, apply, coupling, pack, probes, unpack
from fujinsplat.native_base import NativeBase
from fujinsplat.synthesis import depth_contrasts, reverse_capture
from fujinsplat.train_controller import parameter_loss, save_checkpoint


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)

    def test_controller_count(self):
        self.assertEqual(sum(p.numel() for p in Controller().parameters()), 1514717)

    def test_continuation_snapshot_separates_optimizer(self):
        root = Path(__file__).parent / ".tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            out = Path(tmp)
            model = Controller()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            path = save_checkpoint(out, model, optimizer, np.random.default_rng(90202),
                stage="synthetic_warm_start", initial_hash="fixture", manifest_hash="fixture",
                schedule={"steps": 4000, "initialization": "tensor_only_warm_start"})
            state = torch.load(path, weights_only=True)
            self.assertIs(type(state), dict)
            self.assertEqual(len(state), 30)
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in state.values()))
            self.assertFalse(hasattr(state, "_metadata"))
            restored, payload = load_checkpoint(path)
            self.assertEqual(payload["schedule"]["steps"], 4000)
            self.assertTrue((out / "optimizer_state.pt").exists())
            self.assertTrue((out / "sampler_state.json").exists())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)

    def test_base_count(self):
        self.assertEqual(sum(p.numel() for p in NativeBase().parameters()), 111547)

    def test_zero_heads_identity(self):
        p = Controller()(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(p.shape), (2, 573))
        torch.testing.assert_close(p, torch.zeros_like(p), rtol=0, atol=0)

    def test_reject_controller_image_shape(self):
        with self.assertRaises(ValueError):
            Controller()(torch.rand(2, 3, 32, 32))

    def test_pack_order(self):
        p = torch.arange(573)[None]
        c, k = unpack(p)
        self.assertEqual(c[0, -1, -1, -1], 383)
        self.assertEqual(k[0, 0, 0, 0], 384)
        torch.testing.assert_close(pack(c, k), p)

    def test_no_legacy_abi(self):
        for cfg in (MCFConfig(curve_bound=1.75), MCFConfig(coupling_bound=.1), MCFConfig(center_coupling=True), MCFConfig(domain="linear")):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_identity_including_tails(self):
        x = torch.linspace(-.2, 1.2, 99, dtype=torch.float64).reshape(1, 3, 3, 11)
        p = torch.zeros(1, 573, dtype=torch.float64)
        torch.testing.assert_close(apply(x, p), x, rtol=0, atol=1e-14)
        torch.testing.assert_close(apply(x, p, inverse=True), x, rtol=0, atol=1e-14)

    def test_full_roundtrip(self):
        x = torch.rand(3, 3, 8, 11, dtype=torch.float64) * 1.4 - .2
        p = torch.randn(3, 573, dtype=torch.float64) * .15
        torch.testing.assert_close(apply(apply(x, p), p, inverse=True), x, rtol=1e-9, atol=1e-9)

    def test_coupling_jacobian_unit_det(self):
        x = torch.tensor([.2, .4, .7], dtype=torch.float64, requires_grad=True)
        knots = torch.randn(1, 3, 9, dtype=torch.float64) * .1
        for stage in range(7):
            jac = torch.autograd.functional.jacobian(lambda v: coupling(v.reshape(1, 3, 1, 1), knots, stage).flatten(), x)
            self.assertAlmostEqual(float(torch.det(jac)), 1.0, places=10)

    def test_dc_coupling_is_not_removed(self):
        x = torch.ones(1, 3, 2, 2, dtype=torch.float64) * .4
        raw = torch.zeros(1, 3, 9, dtype=torch.float64)
        raw[:, 0] = .2
        y = coupling(x, raw, 0)
        self.assertGreater(float((y - x).abs().sum()), 0)

    def test_finite_zero_action_gradients(self):
        p = torch.zeros(1, 573, requires_grad=True)
        loss = reconstruction_loss(apply(torch.rand(1, 3, 7, 7), p), torch.rand(1, 3, 7, 7))
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_raw_summary_uint16(self):
        x = np.arange(3 * 80 * 90, dtype=np.uint16).reshape(1, 3, 80, 90)
        normalized = torch.from_numpy(x.astype(np.float32) / 65535.0)
        expected = torch.nn.functional.interpolate(normalized, (64, 64), mode="bilinear", align_corners=False).clamp(0, 1)
        torch.testing.assert_close(summary(x), expected, rtol=0, atol=0)

    def test_raw_summary_no_wb(self):
        x = torch.tensor([.1, .2, .3])[None, :, None, None].expand(1, 3, 64, 64)
        torch.testing.assert_close(summary(x), x)

    def test_color_roundtrip(self):
        x = torch.linspace(0, 1, 501, dtype=torch.float64)
        torch.testing.assert_close(decode(encode(x)), x, rtol=1e-12, atol=1e-12)

    def test_reconstruction_domain(self):
        a, b = torch.ones(1, 3, 3, 3) * .3, torch.ones(1, 3, 3, 3) * .6
        expected = (decode(a) - decode(b)).abs().mean() + .25 * (a - b).abs().mean()
        torch.testing.assert_close(reconstruction_loss(a, b), expected)

    def test_parameter_loss_blocks(self):
        p, target = torch.rand(2, 573), torch.rand(2, 573)
        c, k = unpack(p); tc, tk = unpack(target)
        expected = torch.stack([torch.nn.functional.smooth_l1_loss(c[:, i].tanh(), tc[:, i].tanh(), beta=.05) for i in range(8)]).mean()
        expected += torch.nn.functional.smooth_l1_loss(k.tanh(), tk.tanh(), beta=.05)
        torch.testing.assert_close(parameter_loss(p, target), expected)

    def test_centered_delta(self):
        state = CenteredDelta(torch.randn(25, 573))
        torch.testing.assert_close(state.displacement(), torch.zeros_like(state.directions))
        with torch.no_grad():
            state.alpha.copy_(torch.linspace(-1, 2, 25))
            state.project_()
        self.assertGreaterEqual(float(state.alpha.min()), 0)
        self.assertLessEqual(float(state.alpha.max()), 1)
        self.assertLess(float(state.displacement().mean(0).abs().max()), 1e-7)
        self.assertFalse(state.directions.requires_grad)

    def test_paper_joint_gradient_path(self):
        bank = CenteredDelta(torch.randn(3, 573) * .05)
        base = torch.rand(3, 7, 9) * .7 + .1
        render = torch.rand(3, 7, 9, requires_grad=True)
        target = developed_target(base, bank(0), (7, 9), active=True)
        (render - target).abs().mean().backward()
        self.assertTrue(torch.isfinite(bank.alpha.grad).all())
        self.assertGreater(float(bank.alpha.grad.abs().sum()), 0)
        self.assertGreater(float(render.grad.abs().sum()), 0)
        self.assertFalse(developed_target(base, bank(0), (7, 9), active=False).requires_grad)

    def test_dssim_reaches_alpha(self):
        from utils.loss_utils import ssim
        bank = CenteredDelta(torch.randn(3, 573) * .05)
        target = developed_target(torch.rand(3, 11, 13) * .7 + .1, bank(0), (11, 13), active=True)
        render = torch.rand(3, 11, 13, requires_grad=True)
        (1 - ssim(render, target)).backward()
        self.assertTrue(torch.isfinite(bank.alpha.grad).all())
        self.assertGreater(float(bank.alpha.grad.abs().sum()), 0)
        self.assertGreater(float(render.grad.abs().sum()), 0)

    def test_probe_order(self):
        q = probes().flatten(2)[0].T
        self.assertEqual(tuple(q.shape), (729, 3))
        torch.testing.assert_close(q[1], torch.tensor([0., 0., .125]))
        torch.testing.assert_close(q[81], torch.tensor([.125, 0., 0.]))

    def test_numpy_torch_compiler_parity(self):
        x = np.random.default_rng(0).uniform(.01, .95, (10, 13, 3))
        for t in (.15, .36, .9):
            act = C.compile_from_raw([.18, .17, .16], t)
            p = torch.from_numpy(C.pack_573(act))[None]
            tensor = torch.from_numpy(x).permute(2, 0, 1)[None]
            actual = apply(tensor, p)[0].permute(1, 2, 0).numpy()
            np.testing.assert_allclose(actual, C.action_forward(x, act), atol=2e-14, rtol=2e-14)

    def test_black_floor_matches_realized_nodes(self):
        act = C.compile_from_raw([.18, .17, .16], .15)
        nodes = np.r_[0., np.cumsum(C._increments(act["raw_curve"]))]
        self.assertEqual(C.black_floor(act), nodes[np.argmax(nodes > 1e-3)])

    def test_stored_array_cycles(self):
        raw = np.random.default_rng(4).uniform(.05, .4, (8, 9, 3))
        arrays, checks = reverse_capture(raw, np.eye(3), [.18, .17, .16], .36)
        self.assertEqual(arrays["p"].shape, (573,))
        self.assertEqual(arrays["X_syn"].dtype, np.float32)
        self.assertGreater(checks["action_cycle_psnr"], 100)
        self.assertGreater(checks["base_cycle_psnr"], 100)
        self.assertTrue((arrays["J"] >= checks["black_floor"] - 1e-7).all())

    def test_depth_scale_free(self):
        d = np.linspace(.1, 3, 100).reshape(10, 10)
        _, _, stats = depth_contrasts([d, d * 7])
        self.assertAlmostEqual(stats[0], stats[1], places=12)

    def test_source_held_firewall(self):
        self.assertEqual(sum(SCENE_COUNTS.values()), 195)
        for scene, n in SCENE_COUNTS.items():
            require_source(scene, f"{n:04d}")
            with self.assertRaises(ValueError):
                require_source(scene, f"{n + 1:04d}")

    def test_create_only_and_checkpoint_roundtrip(self):
        root = Path(__file__).parent / ".tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            out = create_output(Path(tmp) / "run")
            with self.assertRaises(FileExistsError):
                create_output(out)
            model = Controller()
            payload = checkpoint_payload(model, training_kind="synthetic_pretrain", initial_hash=None, manifest_hash="fixture", schedule={})
            path = out / "model.pt"
            torch.save(payload, path)
            restored, _ = load_checkpoint(path)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, restored.state_dict()[k], atol=0, rtol=0)
            payload["mcf"]["curve_bound"] = 1.75
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_tensor_only_delivery_has_no_training_objects(self):
        root = Path(__file__).parent / ".tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / "checkpoint.pt"
            payload = checkpoint_payload(Controller(), training_kind="synthetic_warm_start", initial_hash="fixture", manifest_hash="fixture", schedule={})
            state = payload.pop("state_dict")
            torch.save(dict(state), path)
            payload["weights_sha256"] = sha256(path)
            write_json(path.with_suffix(".json"), payload)
            output = Path(tmp) / "controller.pt"
            export(path, output, Path(tmp) / "separate_record.json")
            value = torch.load(output, weights_only=True)
            self.assertIs(type(value), dict)
            self.assertTrue(all(type(x) is torch.Tensor for x in value.values()))
            self.assertFalse(hasattr(value, "_metadata"))
            self.assertEqual(set(value), set(Controller().state_dict()))


if __name__ == "__main__":
    unittest.main()
