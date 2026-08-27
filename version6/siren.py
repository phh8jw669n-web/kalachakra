"""SIREN — a Sinusoidal Representation Network (Sitzmann et al. 2020).

Maps the 33-D topocentric sky tensor to a 3-D CIE L*a*b* colour. Every hidden layer is
``sin(omega0 * (W x + b))``; the output layer is linear. Because ``sin`` is infinitely
differentiable, the field it represents is perfectly continuous and smooth across the
globe (and across time) — no grid, no seams, infinite zoom.

The special SIREN weight init (first layer ``U(-1/in, 1/in)``, hidden layers
``U(-sqrt(6/in)/omega0, +...)``) keeps the pre-activations well-scaled so the sines do
not alias. The whole thing is tiny and exports cleanly to a JSON weight payload that the
GLSL shader and the JS HUD re-run verbatim.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SirenLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, omega0: float, is_first: bool,
                 linear: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.linear_out = linear
        self.lin = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_f
            else:
                bound = math.sqrt(6.0 / in_f) / omega0
            self.lin.weight.uniform_(-bound, bound)
            self.lin.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h if self.linear_out else torch.sin(self.omega0 * h)


class Siren(nn.Module):
    """``[N, in_features] -> [N, out_features]`` continuous field."""

    def __init__(self, in_features: int = 33, hidden: int = 48, hidden_layers: int = 2,
                 out_features: int = 3, omega0: float = 30.0):
        super().__init__()
        self.cfg = {"in_features": in_features, "hidden": hidden,
                    "hidden_layers": hidden_layers, "out_features": out_features,
                    "omega0": omega0}
        layers = [SirenLayer(in_features, hidden, omega0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden, hidden, omega0, is_first=False))
        layers.append(SirenLayer(hidden, out_features, omega0, is_first=False, linear=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def export_weights(self) -> dict:
        """Plain nested lists of every layer's weight/bias + metadata, for the shader."""
        layers = []
        for m in self.net:
            layers.append({
                "W": m.lin.weight.detach().cpu().tolist(),   # [out][in]
                "b": m.lin.bias.detach().cpu().tolist(),      # [out]
                "activation": "linear" if m.linear_out else "sin",
            })
        return {"omega0": self.cfg["omega0"], "in_features": self.cfg["in_features"],
                "hidden": self.cfg["hidden"], "hidden_layers": self.cfg["hidden_layers"],
                "out_features": self.cfg["out_features"], "layers": layers}


def build_siren(**kw) -> Siren:
    return Siren(**kw)
