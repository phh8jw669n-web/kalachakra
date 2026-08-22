import numpy as np

from kalachakra.storage import binary_store as bs
from kalachakra.storage.ring_buffer import RingBuffer


def test_bf16_roundtrip_tolerance():
    x = np.array([0.0, 1.0, -1.0, 3.14159, 1e-3, 1e3, -273.15], dtype=np.float32)
    y = bs.bf16_to_float32(bs.float32_to_bf16(x))
    # bf16 keeps 8 bits of mantissa -> ~2-3 significant decimal digits.
    rel = np.abs(y - x) / np.maximum(np.abs(x), 1e-6)
    assert np.all(rel[x != 0] < 1e-2)


def test_bf16_exact_for_representable_values():
    x = np.array([0.0, 1.0, 2.0, 0.5, -4.0], dtype=np.float32)
    y = bs.bf16_to_float32(bs.float32_to_bf16(x))
    assert np.array_equal(x, y)


def test_delta_encode_decode_roundtrip():
    rng = np.random.default_rng(0)
    frames = np.cumsum(rng.normal(size=(64, 10, 7)).astype(np.float32), axis=0)
    encoded = bs.delta_encode(frames)
    decoded = bs.delta_decode(encoded)
    assert np.allclose(decoded, frames, atol=1e-4)


def test_store_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    frames = np.cumsum(rng.normal(scale=0.01, size=(128, 10, 7)).astype(np.float32),
                       axis=0)
    store = bs.EphemerisStore(tmp_path)
    store.write_chunk(0, frames)
    out = store.read_chunk(0)
    assert out.shape == frames.shape
    # BF16 + delta -> lossy but close for smooth trajectories.
    assert np.allclose(out, frames, atol=5e-2)
    assert [c.start_frame for c in store.chunks()] == [0]


def test_ring_buffer_yields_chunks_in_order(tmp_path):
    store = bs.EphemerisStore(tmp_path)
    for start in (0, 8, 16):
        store.write_chunk(start, np.ones((8, 10, 7), dtype=np.float32) * start)
    with RingBuffer(store, [0, 8, 16], max_prefetch=2) as rb:
        seen = [start for start, _chunk in rb]
    assert seen == [0, 8, 16]
