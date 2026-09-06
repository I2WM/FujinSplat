"""Export and verify a tensor-only controller, keeping provenance separately."""

import argparse
from pathlib import Path

import torch

from .controller import Controller, load_checkpoint, load_weights
from .io import sha256, write_json


def export(checkpoint, output, record):
    model, metadata = load_checkpoint(checkpoint)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    expected = Controller().state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError("unexpected model keys/shapes")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite weights")
    output = Path(output)
    with output.open("xb") as f:
        torch.save(state, f)
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    if type(loaded) is not dict or not all(type(v) is torch.Tensor for v in loaded.values()):
        raise ValueError("export contains something other than the tensor dictionary")
    if hasattr(loaded, "_metadata") or set(loaded) != set(expected):
        raise ValueError("unexpected export metadata")
    restored = load_weights(output)
    if not all(torch.equal(v, restored.state_dict()[k]) for k, v in state.items()):
        raise ValueError("weight reload drift")
    write_json(record, {"weights_sha256": sha256(output), "tensor_count": len(state),
               "parameter_count": sum(v.numel() for v in state.values()),
               "training_kind": metadata["training_kind"],
               "checkpoint_sha256": sha256(checkpoint),
               "contents": "plain state_dict only; no samples, paths, optimizer, logs or training metadata"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    a = p.parse_args()
    export(a.checkpoint, a.output, a.record)


if __name__ == "__main__":
    main()
