"""
Fourier Neural Operator layers for the temporal stream (blueprint §4.2).

The temporal axis is learned in the spectral domain: an ``rfft`` lifts the
time-series into frequency space, a learned complex weight multiplies the lowest
``modes`` frequencies (a continuous integral operator, discretization-invariant),
and an inverse transform returns to the time domain. This captures overlapping
periodicities — from the 24-second micro-rotation of the Ascendant up to
multi-millennial precession — without phase lag.

Requires PyTorch. Imported lazily by the training code; not pulled in by the
top-level package.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D spectral convolution (the core FNO operator).

    Input / output shape: ``(batch, channels, length)`` along the time axis.

    The learned modal weights are stored as a **real** parameter of shape
    ``(in, out, modes, 2)`` (last axis = real/imag) and the complex multiply is
    done by hand. Keeping the parameter real means every optimizer — including
    Lion's ``sign`` update (§5.2) — works without special-casing complex tensors.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Real-valued weights: [..., 0] = real part, [..., 1] = imaginary part.
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        in_dtype = x.dtype
        # FFT and complex arithmetic are unsupported in bf16/fp16, so the whole
        # spectral path runs in float32 with autocast disabled; the result is
        # cast back to the caller's dtype. This keeps mixed-precision training
        # (§5.2) intact while respecting the FFT's precision requirements.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            x_ft = torch.fft.rfft(xf, dim=-1)                  # (B, Cin, L//2+1)
            keep = min(self.modes, x_ft.shape[-1])

            xr = x_ft.real[:, :, :keep]
            xi = x_ft.imag[:, :, :keep]
            wr = self.weight[:, :, :keep, 0].float()
            wi = self.weight[:, :, :keep, 1].float()

            # (a+bi)(c+di) = (ac - bd) + (ad + bc)i, contracting input channels.
            out_r = torch.einsum("bix,iox->box", xr, wr) - torch.einsum("bix,iox->box", xi, wi)
            out_i = torch.einsum("bix,iox->box", xr, wi) + torch.einsum("bix,iox->box", xi, wr)

            out_ft = torch.zeros(
                batch, self.out_channels, x_ft.shape[-1],
                dtype=torch.cfloat, device=x.device,
            )
            out_ft[:, :, :keep] = torch.complex(out_r, out_i)
            out = torch.fft.irfft(out_ft, n=length, dim=-1)
        return out.to(in_dtype)


class FourierBlock1d(nn.Module):
    """FNO block: spectral path + pointwise (1x1) residual path + activation."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))
