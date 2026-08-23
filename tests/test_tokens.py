import numpy as np

from kalachakra.analysis import tokens as tk


def test_token_dtype_is_4_bytes():
    assert tk.TOKEN_DTYPE.itemsize == 4  # two uint16


def test_pack_tokens_roundtrip():
    macro = np.array([[0, 63], [12, 5]])
    micro = np.array([[1, 63], [0, 9]])
    packed = tk.pack_tokens(macro, micro)
    assert packed.shape == macro.shape
    assert np.array_equal(packed["macro"], macro)
    assert np.array_equal(packed["micro"], micro)


def test_leaf_id_split_inverse():
    macro = np.array([0, 1, 63, 10])
    micro = np.array([0, 63, 63, 7])
    leaf = tk.leaf_id(macro, micro)
    assert leaf.max() < 4096 and leaf.min() >= 0
    m2, mi2 = tk.split_leaf(leaf)
    assert np.array_equal(m2, macro) and np.array_equal(mi2, micro)


def test_build_descriptors_and_columns():
    n = 5
    latent = np.random.default_rng(0).normal(size=(n, 64)).astype(np.float32)
    macro = np.arange(n) % 64
    micro = (np.arange(n) * 3) % 64
    rarity = np.linspace(0, 1, n).astype(np.float32)
    desc = tk.build_descriptors(macro, micro, rarity, latent)
    assert desc.shape == (n,)
    assert desc.dtype["latent"].shape == (64,)

    cols = tk.to_columns(desc)
    assert cols["macro"].shape == (n,) and cols["latent"].shape == (n, 64)
    assert np.array_equal(cols["leaf"], tk.leaf_id(macro, micro))
    assert np.allclose(cols["latent"], latent)


def test_descriptor_dtype_custom_dim():
    dt = tk.descriptor_dtype(32)
    assert dt["latent"].shape == (32,)
