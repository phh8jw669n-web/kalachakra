"""Per-frame geometric helpers for the Phase-2 sweep (kept separate for testing).

Connected-component labelling on the geodesic mesh, spherical centroids, and the
subsolar point - the pieces the streaming Domain-2/Domain-3 accumulators need.
"""

from __future__ import annotations

import numpy as np


def connected_components(tokens: np.ndarray, neighbors: np.ndarray):
    """Same-token connected components on the mesh (union-find over shared edges).

    Returns ``(n_comp[token], largest[token])`` dicts keyed by token id present in
    ``tokens``: how many disjoint blobs each token forms and the node count of its
    biggest blob. Two nodes are connected iff they are k-NN neighbours AND carry
    the same token.
    """
    n = tokens.shape[0]
    parent = np.arange(n)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:            # path compression
            parent[a], a = root, parent[a]
        return root

    # union each node with same-token neighbours (dedup i<j via the mask)
    for j in range(neighbors.shape[1]):
        nb = neighbors[:, j]
        same = tokens == tokens[nb]
        for i in np.nonzero(same)[0]:
            ri, rj = find(i), find(int(nb[i]))
            if ri != rj:
                parent[ri] = rj

    roots = np.array([find(i) for i in range(n)])
    n_comp: dict[int, int] = {}
    largest: dict[int, int] = {}
    # group node indices by (token, root)
    order = np.lexsort((roots, tokens))
    st = tokens[order]
    sr = roots[order]
    start = 0
    while start < n:
        tok = st[start]
        end = start
        while end < n and st[end] == tok:
            end += 1
        # components of this token = distinct roots in [start, end)
        seg_roots = sr[start:end]
        uniq, counts = np.unique(seg_roots, return_counts=True)
        n_comp[int(tok)] = int(uniq.size)
        largest[int(tok)] = int(counts.max())
        start = end
    return n_comp, largest


def spherical_centroid(xyz: np.ndarray) -> np.ndarray:
    """Unit-vector mean direction of a set of points on the sphere."""
    m = xyz.mean(axis=0)
    nrm = np.linalg.norm(m)
    return m / nrm if nrm > 1e-12 else m


def latlon_of(vec: np.ndarray):
    """(lat, lon) in radians of a 3-vector (our z=north convention)."""
    lat = np.arcsin(np.clip(vec[2] / (np.linalg.norm(vec) + 1e-12), -1.0, 1.0))
    lon = np.arctan2(vec[1], vec[0])
    return float(lat), float(lon)


def subsolar_point(jd: float):
    """Geographic (lat, lon) in radians where the Sun is at zenith at ``jd``."""
    from .. import geometry as geo
    from ..ephemeris import global_state
    from ..ephemeris.bodies import index_of
    from ..projection.spatial import decode_ecliptic

    g = global_state.global_state_frame(float(jd))
    lon_ecl, lat_ecl = decode_ecliptic(g)
    si = index_of("Sun")
    eps = float(geo.obliquity_of_ecliptic(jd))
    ra, dec = geo.ecliptic_to_equatorial(
        np.array([lon_ecl[si]]), np.array([lat_ecl[si]]), np.array([eps]))
    gmst = np.deg2rad(float(geo.greenwich_mean_sidereal_time_deg(jd)))
    sub_lat = float(dec[0])
    sub_lon = float(geo.wrap_angle(ra[0] - gmst))     # geographic longitude of the Sun
    return sub_lat, sub_lon


def great_circle_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle angle (degrees) between arrays of points and a single point."""
    dlon = lon2 - lon1
    d = np.arccos(np.clip(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon),
        -1.0, 1.0))
    return np.rad2deg(d)
