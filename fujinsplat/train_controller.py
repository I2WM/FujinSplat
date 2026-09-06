"""Synthetic-only controller training, from fresh or tensor-only initial weights."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .color import reconstruction_loss
from .controller import Controller, checkpoint_payload, load_weights
from .io import create_output, sha256, write_json
from .mcf import CONFIG, apply, unpack
from .training_data import SyntheticPairs, validate_manifest


def parameter_loss(prediction, target):
    pc, pk = unpack(prediction)
    tc, tk = unpack(target)
    return F.smooth_l1_loss(pc.tanh(), tc.tanh(), beta=0.05) + F.smooth_l1_loss(pk.tanh(), tk.tanh(), beta=0.05)


def augment(x, code):
    x = torch.rot90(x, code % 4, (-2, -1))
    return x.flip(-1) if code & 4 else x


def initialize_model(initial, device, expected_sha256=None):
    """Load every parameter, including action heads, then explicitly enable gradients."""
    if initial is None:
        if expected_sha256 is not None:
            raise ValueError("--initial-sha256 requires --initial")
        model = Controller().to(device)
    else:
        model = load_weights(initial, device, expected_sha256=expected_sha256)
    return model.train().requires_grad_(True)


def save_checkpoint(output, model, optimizer, rng, *, stage, initial_hash, manifest_hash, schedule):
    """Inference tensors and training state are always separate files."""
    checkpoint = output / "checkpoint.pt"
    payload = checkpoint_payload(model, training_kind=stage, initial_hash=initial_hash,
                                 manifest_hash=manifest_hash, schedule=schedule)
    state = payload.pop("state_dict")
    with checkpoint.open("xb") as f:
        torch.save(dict(state), f)
    payload["weights_sha256"] = sha256(checkpoint)
    write_json(output / "checkpoint.json", payload)
    with (output / "optimizer_state.pt").open("xb") as f:
        torch.save({"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}, f)
    write_json(output / "sampler_state.json", rng.bit_generator.state)
    return checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("synthetic_pretrain",), default="synthetic_pretrain",
                   help="compatibility option; only external synthetic training is supported")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--initial", type=Path, help="optional tensor-only weights for a warm start; no sidecar read")
    p.add_argument("--initial-sha256", help="optional expected SHA256 for --initial")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=1500, help="new optimizer steps in this run")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=90202)
    p.add_argument("--save-steps", type=int, nargs="*", default=[], help="intermediate new-step snapshot boundaries")
    p.add_argument("--check-only", action="store_true", help="metadata/hash checks only; no images or training")
    return p


def run(args):
    stage = "synthetic_warm_start" if args.initial is not None else "synthetic_pretrain"
    snapshots = set(args.save_steps)
    schedule = {
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
        "seed": args.seed, "optimizer": "AdamW", "weight_decay": 1e-5, "gradient_clip": 5.0,
        "parameter_loss_weight": 1.0, "reconstruction_weight": 0.2,
        "augmentations": 8, "synthetic_reconstruction_pixels": 4096,
        "checkpoint_selection": "fixed step boundaries; no validation or score-based selection",
        "initialization": "tensor_only_warm_start" if args.initial is not None else "fresh",
        "optimizer_state": "new AdamW; no optimizer or sampler state loaded",
        "fixed_snapshot_steps": sorted(snapshots),
    }
    if args.steps < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid training schedule")
    if any(s < 1 or s >= args.steps for s in snapshots):
        raise ValueError("invalid intermediate snapshot boundary")
    if args.initial is None and args.initial_sha256 is not None:
        raise ValueError("--initial-sha256 requires --initial")
    initial_hash = sha256(args.initial) if args.initial is not None else None
    if args.initial_sha256 is not None and args.initial_sha256 != initial_hash:
        raise ValueError("controller SHA256 mismatch")
    manifest = validate_manifest(args.manifest, args.stage)
    if args.check_only:
        print(json.dumps({"stage": stage, "rows": len(manifest["rows"]), "schedule": schedule,
                          "initial_checkpoint_sha256": initial_hash, "mcf": asdict(CONFIG),
                          "image_reads": 0, "training_started": False}, indent=2))
        return
    if args.output is None:
        raise ValueError("--output is required for training")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = initialize_model(args.initial, device, args.initial_sha256)
    output = create_output(args.output)
    manifest_hash = sha256(args.manifest)
    write_json(output / "CONTRACT.json", {
        "stage": stage, "schedule": schedule, "mcf": asdict(CONFIG),
        "manifest_sha256": manifest_hash, "initial_checkpoint_sha256": initial_hash,
        "current_training_data": "external_synthetic", "held_target_reads": 0,
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    })
    dataset = SyntheticPairs(manifest)
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    start = time.monotonic()
    with (output / "TRAIN.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            indices = rng.integers(len(dataset.rows), size=args.batch_size)
            batch = dataset.batch(indices, device)
            code = int(rng.integers(8))
            batch = {k: augment(v, code) if k != "p" else v for k, v in batch.items()}
            p = model(batch["raw"])
            corrected = apply(batch["base"], p)
            reconstruction = reconstruction_loss(corrected, batch["target"], batch.get("valid"))
            param = parameter_loss(p, batch["p"])
            loss = param + schedule["reconstruction_weight"] * reconstruction
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step in snapshots:
                destination = create_output(output / f"step_{step:06d}")
                path = save_checkpoint(destination, model, optimizer, rng, stage=stage,
                    initial_hash=initial_hash, manifest_hash=manifest_hash,
                    schedule=dict(schedule, steps=step))
                print(json.dumps({"snapshot": str(path), "step": step}), flush=True)
            if step == 1 or step % 100 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()), "parameter": float(param.detach()),
                          "reconstruction": float(reconstruction.detach()), "elapsed_seconds": time.monotonic() - start}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    hashes = dataset.hashes
    del dataset, batch
    checkpoint = save_checkpoint(output, model, optimizer, rng, stage=stage,
                                 initial_hash=initial_hash, manifest_hash=manifest_hash, schedule=schedule)
    write_json(output / "RESULT.json", {
        "status": "SEALED", "stage": stage, "checkpoint_sha256": sha256(checkpoint),
        "input_hashes": hashes, "held_target_reads": 0,
        "post_training_score_reads": 0, "training_rows": len(manifest["rows"]),
        "unique_loaded_training_rows": len(hashes),
        "elapsed_seconds": time.monotonic() - start,
    })
    print(json.dumps({"status": "SEALED", "checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    run(parser().parse_args())
