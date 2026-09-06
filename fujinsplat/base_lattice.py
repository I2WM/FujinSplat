"""Uniform deterministic cycles over exact E1/E2 LUT term inventories."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor


TERMS_PER_INVENTORY = 65_536
SEED = 190115


@dataclass(frozen=True)
class DifferenceInventory:
    grid: int
    order: int

    def __post_init__(self) -> None:
        if self.grid < 3 or self.order not in (1, 2) or self.grid <= self.order:
            raise ValueError("invalid lattice difference inventory")

    @property
    def term_count(self) -> int:
        # Three axes x every valid position x three output channels.
        return 9 * (self.grid - self.order) * self.grid * self.grid

    def decode(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("term IDs must be one-dimensional int64")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.term_count):
            raise ValueError("term ID outside inventory")
        length = self.grid - self.order
        per_axis = 3 * length * self.grid * self.grid
        axis = torch.div(ids, per_axis, rounding_mode="floor")
        remainder = torch.remainder(ids, per_axis)
        channel = torch.remainder(remainder, 3)
        position = torch.div(remainder, 3, rounding_mode="floor")
        along = torch.remainder(position, length)
        pair = torch.div(position, length, rounding_mode="floor")
        other1 = torch.remainder(pair, self.grid)
        other0 = torch.div(pair, self.grid, rounding_mode="floor")
        base = torch.empty(ids.numel(), 3, dtype=torch.long, device=ids.device)
        choices = (
            torch.stack((along, other0, other1), 1),
            torch.stack((other0, along, other1), 1),
            torch.stack((other0, other1, along), 1),
        )
        for dimension, values in enumerate(choices):
            mask = axis == dimension
            base[mask] = values[mask]
        return axis, base, channel


def _affine(total: int, label: str, epoch: int) -> tuple[int, int]:
    # This fixed seed prefix preserves the reference sampling sequence.
    digest = hashlib.sha256(
        f"phase19-contract-a-lattice|{label}|{SEED}|{epoch}|{total}".encode("ascii")
    ).digest()
    multiplier = max(1, int.from_bytes(digest[:8], "little") % total)
    while math.gcd(multiplier, total) != 1:
        multiplier = 1 if multiplier + 1 == total else multiplier + 1
    offset = int.from_bytes(digest[8:16], "little") % total
    return multiplier, offset


@dataclass(frozen=True)
class TermSampler:
    inventory: DifferenceInventory
    sample_count: int = TERMS_PER_INVENTORY
    label: str = "L257"

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not 0 < self.sample_count <= self.inventory.term_count:
            raise ValueError("invalid lattice sample count")
        if not self.label:
            raise ValueError("lattice sampler label is empty")

    def ids(self, step: int, epoch: int = 0, device: torch.device = torch.device("cpu")) -> Tensor:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be nonnegative int")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be nonnegative int")
        total = self.inventory.term_count
        multiplier, offset = _affine(total, self.label, epoch)
        start = step * self.sample_count
        position = torch.arange(start, start + self.sample_count, dtype=torch.long, device=device)
        return torch.remainder(multiplier * torch.remainder(position, total) + offset, total)


def sampled_energy(table: Tensor, sampler: TermSampler, step: int, epoch: int = 0) -> Tensor:
    inventory = sampler.inventory
    expected = (inventory.grid, inventory.grid, inventory.grid, 3)
    if tuple(table.shape) != expected or not table.is_floating_point() or not bool(torch.isfinite(table).all()):
        raise ValueError(f"regularized table must be finite floating {expected}")
    axis, base, channel = inventory.decode(sampler.ids(step, epoch, table.device))
    row = torch.arange(base.shape[0], device=base.device)

    def value(offset: int) -> Tensor:
        coordinate = base.clone()
        coordinate[row, axis] += offset
        return table[coordinate[:, 0], coordinate[:, 1], coordinate[:, 2], channel]

    v0 = value(0)
    difference = value(1) - v0 if inventory.order == 1 else value(2) - 2.0 * value(1) + v0
    return difference.square().mean()


def exact_energy(table: Tensor, order: int) -> Tensor:
    if table.ndim != 4 or table.shape[-1] != 3 or len(set(table.shape[:3])) != 1:
        raise ValueError("table must be [G,G,G,3]")
    if order not in (1, 2):
        raise ValueError("order must be one or two")
    return torch.cat([table.diff(n=order, dim=axis).square().reshape(-1) for axis in range(3)]).mean()


def sampler_pair(grid: int, label: str, sample_count: int = TERMS_PER_INVENTORY) -> tuple[TermSampler, TermSampler]:
    return (
        TermSampler(DifferenceInventory(grid, 1), sample_count, label),
        TermSampler(DifferenceInventory(grid, 2), sample_count, label),
    )


__all__ = [
    "DifferenceInventory", "SEED", "TERMS_PER_INVENTORY", "TermSampler",
    "exact_energy", "sampled_energy", "sampler_pair",
]
