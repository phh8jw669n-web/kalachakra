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


def bound_lab(z: torch.Tensor, center: float, lspan: float, ab: float) -> torch.Tensor:
    """Squash raw logits into a displayable L*a*b* box with a slope-1 (near-identity) tanh.

    ``L* = center + lspan*tanh(zL/lspan)`` keeps L* strictly in ``(center-lspan, center+lspan)``
    ( = (0,100) by default) while staying ~linear near the centre; ``a*,b* = ab*tanh(z/ab)``
    bound the chroma to ``(-ab, ab)``. Because the map is slope-1 at the origin, the isometric
    metric is preserved for moderate colours and only the extremes are gently compressed — a
    soft clamp baked into the network, so nothing can leave the gamut before conversion.
    """
    L = center + lspan * torch.tanh(z[..., 0:1] / lspan)
    a = ab * torch.tanh(z[..., 1:2] / ab)
    b = ab * torch.tanh(z[..., 2:3] / ab)
    return torch.cat([L, a, b], dim=-1)


class Siren(nn.Module):
    """``[N, in_features] -> [N, out_features]`` continuous field (bounded L*a*b*)."""

    def __init__(self, in_features: int = 33, hidden: int = 48, hidden_layers: int = 2,
                 out_features: int = 3, omega0: float = 30.0,
                 lab_center: float = 50.0, lab_lspan: float = 50.0, lab_ab: float = 90.0):
        super().__init__()
        self.cfg = {"in_features": in_features, "hidden": hidden,
                    "hidden_layers": hidden_layers, "out_features": out_features,
                    "omega0": omega0, "lab_center": lab_center, "lab_lspan": lab_lspan,
                    "lab_ab": lab_ab}
        layers = [SirenLayer(in_features, hidden, omega0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden, hidden, omega0, is_first=False))
        layers.append(SirenLayer(hidden, out_features, omega0, is_first=False, linear=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)                                       # raw linear logits
        return bound_lab(z, self.cfg["lab_center"], self.cfg["lab_lspan"], self.cfg["lab_ab"])

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
                "out_features": self.cfg["out_features"], "layers": layers,
                "output_activation": "lab_tanh", "lab_center": self.cfg["lab_center"],
                "lab_lspan": self.cfg["lab_lspan"], "lab_ab": self.cfg["lab_ab"]}


def build_siren(**kw) -> Siren:
    return Siren(**kw)
