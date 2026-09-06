"""Training-only, frozen-direction, exactly centered per-view correction."""

import torch
from torch import nn


class CenteredDelta(nn.Module):
    def __init__(self, actions):
        super().__init__()
        if actions.ndim != 2 or actions.shape[1] != 573 or len(actions) < 2:
            raise ValueError("one 573-vector per source view required")
        self.register_buffer("mean", actions.detach().mean(0, keepdim=True))
        self.register_buffer("directions", actions.detach() - self.mean)
        self.alpha = nn.Parameter(actions.new_zeros(len(actions)))

    def displacement(self):
        residual = self.alpha[:, None] * self.directions
        return residual - residual.mean(0, keepdim=True)

    def forward(self, index):
        return self.mean + self.displacement()[index:index + 1]

    @torch.no_grad()
    def project_(self):
        self.alpha.clamp_(0, 1)
