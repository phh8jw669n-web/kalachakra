"""
Composite token descriptors and their serialization (blueprint §2).

Each evaluated (frame, node) produces a compact descriptor:
    - macro token id      (uint16)
    - micro token id      (uint16)   -> 4 bytes of discrete state per node
    - rarity index        (float32)
    - 64-d reconstruction latent vector (float32)

The discrete tokens alone are a 4-byte categorical summary of a full ten-body
configuration; the descriptor adds the continuous rarity and latent for
downstream columnar storage (Parquet/DuckDB) and rendering.

Pure numpy; fully tested.
"""

from __future__ import annotations

import numpy as np

from .. import constants as C

#: Compact 4-byte discrete token: (macro, micro) as two uint16.
TOKEN_DTYPE = np.dtype([("macro", "<u2"), ("micro", "<u2")])


def descriptor_dtype(latent_dim: int = C.LATENT_DIM) -> np.dtype:
    """Structured dtype for a full per-node descriptor."""
    return np.dtype([
        ("macro", "<u2"),
        ("micro", "<u2"),
        ("rarity", "<f4"),
        ("latent", "<f4", (latent_dim,)),
    ])


def pack_tokens(macro: np.ndarray, micro: np.ndarray) -> np.ndarray:
    """Pack macro/micro id arrays into a structured ``TOKEN_DTYPE`` array."""
    macro = np.asarray(macro, dtype=np.uint16)
    micro = np.asarray(micro, dtype=np.uint16)
    if macro.shape != micro.shape:
        raise ValueError("macro and micro must share shape")
    out = np.empty(macro.shape, dtype=TOKEN_DTYPE)
    out["macro"] = macro
    out["micro"] = micro
    return out


def leaf_id(macro: np.ndarray, micro: np.ndarray, n_micro: int = 64) -> np.ndarray:
    """Combine (macro, micro) into a single leaf id in ``[0, n_macro*n_micro)``."""
    return np.asarray(macro, dtype=np.int64) * n_micro + np.asarray(micro, np.int64)


def split_leaf(leaf: np.ndarray, n_micro: int = 64):
    """Inverse of :func:`leaf_id`."""
    leaf = np.asarray(leaf, dtype=np.int64)
    return leaf // n_micro, leaf % n_micro


def build_descriptors(macro: np.ndarray, micro: np.ndarray, rarity: np.ndarray,
                      latent: np.ndarray) -> np.ndarray:
    """Assemble a structured descriptor array from component arrays.

    ``latent`` has shape ``(..., latent_dim)``; the id/rarity arrays share the
    leading shape.
    """
    latent = np.asarray(latent, dtype=np.float32)
    dt = descriptor_dtype(latent.shape[-1])
    out = np.empty(latent.shape[:-1], dtype=dt)
    out["macro"] = np.asarray(macro, dtype=np.uint16)
    out["micro"] = np.asarray(micro, dtype=np.uint16)
    out["rarity"] = np.asarray(rarity, dtype=np.float32)
    out["latent"] = latent
    return out


def to_columns(descriptors: np.ndarray) -> dict[str, np.ndarray]:
    """Flatten a descriptor array into columnar arrays for Parquet ingestion."""
    d = descriptors.reshape(-1)
    return {
        "macro": d["macro"].copy(),
        "micro": d["micro"].copy(),
        "leaf": leaf_id(d["macro"], d["micro"]).astype(np.int32),
        "rarity": d["rarity"].copy(),
        "latent": d["latent"].copy(),
    }
