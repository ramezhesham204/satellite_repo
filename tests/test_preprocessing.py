import numpy as np
from src.preprocessing import compute_ndvi, tile_image


def test_ndvi_range():
    stack = np.random.rand(16, 16, 11).astype(np.float32)
    ndvi = compute_ndvi(stack)
    assert ndvi.min() >= -1 and ndvi.max() <= 1


def test_tile_count():
    fs = np.zeros((128, 128, 12), dtype=np.float32)
    assert len(tile_image(fs, tile_size=64)) == 4
