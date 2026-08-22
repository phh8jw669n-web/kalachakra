"""
Geodesic Earth mesh (blueprint §2.2).

The Earth is represented as a set of near-uniformly distributed observer nodes on
the unit sphere. Two constructions are provided:

* :func:`icosphere` — recursive subdivision of a regular icosahedron. This is the
  literal "geodesic icosahedral mesh"; a subdivision of frequency ``n`` yields
  exactly ``10 * 4**n + 2`` vertices.
* :func:`fibonacci_sphere` — a spherical Fibonacci lattice that yields *exactly*
  ``n`` near-uniform points for any ``n``.

The standard icosphere vertex counts (…, n=6 -> 40,962, n=7 -> 163,842) do not
land on the blueprint's Level-5 target of 122,880 nodes, so :func:`default_grid`
uses the Fibonacci lattice to realize precisely ``N_SPATIAL_NODES`` uniformly
distributed nodes. Both meshes satisfy the design requirement that inter-node
distance is pure angular separation from the planetary core (§2.2).

Every node carries a geographic ``(latitude, longitude)`` so the projection
engine can compute local sidereal time per observer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as C


@dataclass(frozen=True)
class Grid:
    """A spherical observer mesh.

    Attributes
    ----------
    xyz : (N, 3) float64
        Cartesian unit vectors of each node.
    lat : (N,) float64
        Geographic latitude in radians, [-pi/2, pi/2].
    lon : (N,) float64
        Geographic longitude in radians, (-pi, pi].
    """

    xyz: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.xyz.shape[0]


def _latlon_from_xyz(xyz: np.ndarray):
    lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return lat, lon


def fibonacci_sphere(n: int) -> Grid:
    """Return exactly ``n`` near-uniform points via the spherical Fibonacci lattice.

    The golden-angle construction minimizes clustering and gives an even areal
    density — a practical stand-in for a geodesic grid at an arbitrary node count.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(n, dtype=np.float64)
    # z evenly spaced in [-1, 1]; azimuth advances by the golden angle.
    z = 1.0 - (2.0 * i + 1.0) / n
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = i * golden_angle
    xyz = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
    lat, lon = _latlon_from_xyz(xyz)
    return Grid(xyz=xyz, lat=lat, lon=lon)


def _icosahedron():
    """12 vertices / 20 faces of a unit regular icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return verts, faces


def icosphere(subdivisions: int) -> Grid:
    """Recursively subdivide an icosahedron ``subdivisions`` times.

    Vertex count is ``10 * 4**subdivisions + 2``. Each new vertex is the
    normalized midpoint of an edge, so all vertices remain on the unit sphere.
    """
    if subdivisions < 0:
        raise ValueError("subdivisions must be >= 0")
    verts, faces = _icosahedron()
    verts_list = [v for v in verts]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        m = (verts_list[a] + verts_list[b]) / 2.0
        m /= np.linalg.norm(m)
        verts_list.append(m)
        idx = len(verts_list) - 1
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces, dtype=np.int64)

    xyz = np.array(verts_list, dtype=np.float64)
    lat, lon = _latlon_from_xyz(xyz)
    return Grid(xyz=xyz, lat=lat, lon=lon)


def icosphere_vertex_count(subdivisions: int) -> int:
    """Vertices produced by :func:`icosphere` at a given subdivision level."""
    return 10 * (4 ** subdivisions) + 2


def default_grid() -> Grid:
    """The canonical mesh: exactly ``N_SPATIAL_NODES`` (122,880) observer nodes."""
    return fibonacci_sphere(C.N_SPATIAL_NODES)
