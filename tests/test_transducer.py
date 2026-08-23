"""Isomorphic transducer tests — the lossless-invertibility validation."""

import numpy as np
import pytest

from kalachakra.transducer import kinematics, photometric
from kalachakra.transducer.spectral import BANDS, SpectralTransducer


# --- scalar channels ------------------------------------------------------
def test_naka_rushton_invertible_infinite_range():
    x = np.array([0.0, 0.01, 1.0, 10.0, 1e3, 1e6])
    back = photometric.naka_rushton_inv(photometric.naka_rushton(x))
    assert np.allclose(back, x, rtol=1e-6, atol=1e-6)
    # never clips: flux strictly inside (0, 1)
    f = photometric.naka_rushton(x)
    assert f.min() >= 0.0 and f.max() < 1.0


def test_rarity_temperature_exact_inverse():
    r = np.linspace(0, 1, 11)
    back = photometric.temperature_to_rarity(photometric.rarity_to_temperature(r))
    assert np.allclose(back, r, atol=1e-12)


def test_planckian_cct_roundtrip_within_mccamy_range():
    # McCamy is the pixel-space inverse used by the shader; valid ~2000-12500 K.
    for T in (2500, 4000, 6500, 10000):
        xy = photometric.planckian_xy(T)
        assert abs(photometric.cct_from_xy(xy) - T) / T < 0.05


# --- spectral channel -----------------------------------------------------
def test_spectral_basis_orthonormal_and_recoverable():
    st = SpectralTransducer(96)
    assert np.allclose(st.gram(), np.eye(len(BANDS)), atol=1e-10)
    energies = {"micro": 0.4, "fast": 1.2, "cyclic": -0.7, "macro": 0.05}
    rec = st.recover(st.emit(energies))
    for b in BANDS:
        assert abs(rec[b] - energies[b]) < 1e-9


# --- vector channel -------------------------------------------------------
def test_helmholtz_hodge_reconstructs_and_separates():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(33, 33)); v = rng.normal(size=(33, 33))   # odd: no Nyquist
    parts = kinematics.helmholtz_hodge(u, v)
    ur, vr = parts.reconstruct()
    assert np.allclose(ur, u, atol=1e-9) and np.allclose(vr, v, atol=1e-9)
    # irrotational is curl-free, solenoidal is divergence-free
    assert np.abs(kinematics.curl(parts.u_irrot, parts.v_irrot)).max() < 1e-8
    assert np.abs(kinematics.divergence(parts.u_solen, parts.v_solen)).max() < 1e-8


def test_vector_channel_recovers_divergence_and_curl():
    rng = np.random.default_rng(1)
    div = rng.normal(size=(25, 25)); div -= div.mean()
    crl = rng.normal(size=(25, 25)); crl -= crl.mean()
    u, v = kinematics.field_from_sources(div, crl)
    assert np.allclose(kinematics.divergence(u, v), div, atol=1e-9)
    assert np.allclose(kinematics.curl(u, v), crl, atol=1e-9)


def test_lic_aligns_with_flow():
    rng = np.random.default_rng(2)
    noise = rng.random((48, 48))
    u = np.ones((48, 48)); v = np.zeros((48, 48))     # uniform horizontal flow
    out = kinematics.line_integral_convolution(u, v, noise, n_steps=16, step=0.7)
    # smoother along x (flow) than along y (across flow)
    var_x = np.var(np.diff(out, axis=1))
    var_y = np.var(np.diff(out, axis=0))
    assert var_x < var_y


# --- composite: lossless latent invertibility (the headline constraint) ---
scipy = pytest.importorskip("scipy")
from kalachakra.transducer.state import IsomorphicTransducer   # noqa: E402


def test_full_transducer_recovers_latent_to_machine_precision():
    tr = IsomorphicTransducer()
    rng = np.random.default_rng(3)
    latent = rng.normal(size=64)
    bands = {"micro": 0.3, "fast": 0.9, "cyclic": 0.2, "macro": 0.6}
    div = rng.normal(size=(25, 25)); div -= div.mean()
    crl = rng.normal(size=(25, 25)); crl -= crl.mean()

    state = tr.transduce(latent, bands, rarity=0.73, potential=4.2,
                         div_field=div, curl_field=crl)
    rec = tr.invert(state)

    assert np.max(np.abs(rec["latent"] - latent)) < 1e-8      # tensor channel
    for b in BANDS:
        assert abs(rec["band_energies"][b] - bands[b]) < 1e-9  # spectral channel
    assert abs(rec["potential"] - 4.2) < 1e-6                  # flux channel
    assert abs(rec["rarity"] - 0.73) < 1e-9                    # temperature channel
    assert np.allclose(rec["divergence"], div, atol=1e-9)      # vector channel
    assert np.allclose(rec["curl"], crl, atol=1e-9)
