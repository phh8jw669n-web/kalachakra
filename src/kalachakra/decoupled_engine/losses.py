"""Self-supervised composite loss for the decoupled engine.

Three terms, all derived from celestial wave mechanics and spherical differential
geometry -- no human categorical labels:

1. Geometric interference contrastive loss (Sky Encoder). A wave-mechanics
   descriptor built from the pairwise harmonic alignments cos(k * delta-longitude)
   of the ten bodies defines which frames are geometrically similar. A soft
   contrastive objective forces the latent tension neighbourhood to match that
   geometric neighbourhood, so constructive-harmonic and destructive-orthogonal
   configurations cannot collapse onto the same latent.

2. Terrestrial physical-consistency loss (Earth Lens). The squared spatial
   gradient of the colour field with respect to geodesic distance is penalised,
   but the penalty is switched off near active planetary culmination boundaries
   (where a sharp Ascendant/culmination edge is physical).

3. Temporal continuity loss. The discrete second time-derivative of the colour at
   fixed coordinates is penalised -- steady fast drift (a rapid lunar transit) has
   zero curvature and passes freely, while erratic frame-to-frame spikes are
   punished.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .features import decode_lon

#: Aspect harmonics: 1=conjunction/opposition axis, 2=square, 3=trine/sextile, ...
ASPECT_HARMONICS: tuple[int, ...] = (1, 2, 3, 4, 6)


def harmonic_interference_descriptor(
        celestial: torch.Tensor,
        harmonics: tuple[int, ...] = ASPECT_HARMONICS) -> torch.Tensor:
    """``(M,10,5)`` -> ``(M, n_pairs*len(harmonics))`` wave-interference signature.

    Each feature is ``cos(k * (lon_i - lon_j))`` for a body pair ``(i, j)`` and
    harmonic ``k`` -- the standing-wave alignment of that aspect. This is the
    geometric ground truth the contrastive loss aligns the latent space to.
    """
    lon = decode_lon(celestial)                                   # (M, 10)
    m, n = lon.shape
    iu, ju = torch.triu_indices(n, n, offset=1, device=lon.device)
    dlon = lon[:, iu] - lon[:, ju]                                # (M, n_pairs)
    feats = [torch.cos(k * dlon) for k in harmonics]
    return torch.cat(feats, dim=-1)                              # (M, n_pairs*K)


def geometric_interference_contrastive_loss(
        z: torch.Tensor, celestial: torch.Tensor,
        temperature: float = 0.1) -> torch.Tensor:
    """Soft contrastive loss aligning latent neighbourhoods to geometric ones.

    ``z`` is the ``(M, 512)`` tension batch; positives/negatives are defined softly
    by the harmonic descriptor's similarity. Collapse (all ``z`` identical) yields a
    uniform predicted distribution against a peaked geometric target -> high loss.
    """
    m = z.shape[0]
    if m < 2:
        return z.new_zeros(())
    zc = F.normalize(z, dim=-1)
    g = F.normalize(harmonic_interference_descriptor(celestial), dim=-1)
    neg_inf = torch.finfo(z.dtype).min
    eye = torch.eye(m, dtype=torch.bool, device=z.device)
    # geometric target distribution over the other frames
    sg = (g @ g.t()) / temperature
    sg = sg.masked_fill(eye, neg_inf)
    target = torch.softmax(sg, dim=1)
    # predicted distribution from latent similarity
    sz = (zc @ zc.t()) / temperature
    sz = sz.masked_fill(eye, neg_inf)
    log_pred = torch.log_softmax(sz, dim=1)
    return -(target * log_pred).sum(dim=1).mean()


def culmination_edge_permission(
        celestial: torch.Tensor, coords: torch.Tensor,
        gmst_rad: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    """``(M, P)`` in ``(0, 1]``: how much a sharp colour edge is permitted per point.

    A body culminates on a point's local meridian when the local sidereal longitude
    ``(gmst + point_lon)`` aligns with the body's ecliptic longitude. Near any such
    alignment the permission -> 1 (sharp edges allowed); far from all bodies it -> 0
    (smoothness enforced). Grounded in meridian-transit geometry, not a heuristic
    mask.
    """
    body_lon = decode_lon(celestial)                             # (M, 10)
    point_lon = coords[..., 1]                                   # (M, P)
    local_sid = point_lon + gmst_rad.unsqueeze(-1)              # (M, P)
    d = local_sid.unsqueeze(-1) - body_lon.unsqueeze(1)         # (M, P, 10)
    sep = torch.atan2(torch.sin(d), torch.cos(d)).abs()         # wrap-safe angular sep
    permit = torch.exp(-(sep / sigma) ** 2)                     # (M, P, 10)
    return permit.max(dim=-1).values                            # (M, P)


def terrestrial_smoothness_loss(
        color: torch.Tensor, color_neighbor: torch.Tensor,
        delta_geodesic: float, edge_permission: torch.Tensor) -> torch.Tensor:
    """Geodesic-gradient smoothness, relaxed at culmination boundaries.

    ``color`` and ``color_neighbor`` are the field at points a geodesic distance
    ``delta_geodesic`` apart. The squared gradient is weighted by ``1 - permission``
    so structure is allowed exactly where a planet culminates.
    """
    grad2 = ((color - color_neighbor) ** 2).sum(dim=-1) / (delta_geodesic ** 2)
    weight = 1.0 - edge_permission
    return (weight * grad2).mean()


def temporal_continuity_loss(color_seq: torch.Tensor) -> torch.Tensor:
    """Penalise erratic colour changes over time via the second time-difference.

    ``color_seq`` is ``(B, T, P, 3)``. The discrete curvature
    ``c[t+1] - 2 c[t] + c[t-1]`` is zero for steady drift (fast but smooth transits)
    and large for spikes. Falls back to the first difference when ``T < 3``.
    """
    t = color_seq.shape[1]
    if t >= 3:
        d = color_seq[:, 2:] - 2.0 * color_seq[:, 1:-1] + color_seq[:, :-2]
    elif t == 2:
        d = color_seq[:, 1:] - color_seq[:, :-1]
    else:
        return color_seq.new_zeros(())
    return (d ** 2).mean()
