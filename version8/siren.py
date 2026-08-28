"""version8 SIREN — a 4x128 sinusoidal network mapping the 88-D sky to CIE L*a*b*.

Architecture: 88 -> [128 sin] x 4 -> 3 linear, omega0 = 30.
Gamut-bounded output head (eliminates blown-out white / crushed black):
    L* = 5  + 90 * sigmoid(y0)   -> (5, 95)
    a* = 80 * tanh(y1)           -> (-80, 80)
    b* = 80 * tanh(y2)           -> (-80, 80)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SirenLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, omega0: float, is_first: bool, linear: bool = False):
        super().__init__()
        self.omega0 = omega0
        self.linear_out = linear
        self.lin = nn.Linear(in_f, out_f)
        with torch.no_grad():
            bound = (1.0 / in_f) if is_first else (math.sqrt(6.0 / in_f) / omega0)
            self.lin.weight.uniform_(-bound, bound)
            self.lin.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h if self.linear_out else torch.sin(self.omega0 * h)


def bound_lab(z: torch.Tensor, lab_l0: float, lab_lspan: float, lab_ab: float) -> torch.Tensor:
    """Bounded L*a*b* head: L* = l0 + lspan*sigmoid(z0); a*,b* = ab*tanh(z1,z2)."""
    L = lab_l0 + lab_lspan * torch.sigmoid(z[..., 0:1])
    a = lab_ab * torch.tanh(z[..., 1:2])
    b = lab_ab * torch.tanh(z[..., 2:3])
    return torch.cat([L, a, b], dim=-1)


class Siren(nn.Module):
    """``[N, in_features] -> [N, 3]`` bounded L*a*b* field."""

    def __init__(self, in_features: int = 88, hidden: int = 128, hidden_layers: int = 4,
                 out_features: int = 3, omega0: float = 30.0,
                 lab_l0: float = 5.0, lab_lspan: float = 90.0, lab_ab: float = 80.0):
        super().__init__()
        self.cfg = {"in_features": in_features, "hidden": hidden, "hidden_layers": hidden_layers,
                    "out_features": out_features, "omega0": omega0,
                    "lab_l0": lab_l0, "lab_lspan": lab_lspan, "lab_ab": lab_ab}
        layers = [SirenLayer(in_features, hidden, omega0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden, hidden, omega0, is_first=False))
        layers.append(SirenLayer(hidden, out_features, omega0, is_first=False, linear=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return bound_lab(z, self.cfg["lab_l0"], self.cfg["lab_lspan"], self.cfg["lab_ab"])

    def export_weights(self) -> dict:
        layers = []
        for m in self.net:
            layers.append({
                "W": m.lin.weight.detach().cpu().tolist(),   # [out][in]
                "b": m.lin.bias.detach().cpu().tolist(),
                "activation": "linear" if m.linear_out else "sin",
            })
        c = self.cfg
        return {"omega0": c["omega0"], "in_features": c["in_features"], "hidden": c["hidden"],
                "hidden_layers": c["hidden_layers"], "out_features": c["out_features"],
                "output_activation": "v8_gamut", "lab_l0": c["lab_l0"],
                "lab_lspan": c["lab_lspan"], "lab_ab": c["lab_ab"], "layers": layers}


def build_siren(**kw) -> Siren:
    return Siren(**kw)
