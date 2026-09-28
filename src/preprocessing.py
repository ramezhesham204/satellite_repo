"""
Preprocessing utilities for Sentinel-2 land type classification.

Covers Milestone 1 (preprocessing) and part of Milestone 2 (feature engineering: NDVI).
Works on GeoTIFFs downloaded via download_sentinel2.py (11-band stack: 10 spectral + dataMask).
"""

import numpy as np
import rasterio
from rasterio.windows import Window

BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "dataMask"]
BAND_IDX = {name: i for i, name in enumerate(BAND_NAMES)}


def load_stack(path: str) -> np.ndarray:
    """Load an 11-band GeoTIFF into a (H, W, 11) float32 array."""
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    return np.transpose(arr, (1, 2, 0)).astype(np.float32)


def compute_ndvi(stack: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Distinguishes vegetation (agriculture/trees) from bare/urban."""
    nir = stack[..., BAND_IDX["B08"]]
    red = stack[..., BAND_IDX["B04"]]
    denom = nir + red
    ndvi = np.where(denom == 0, 0, (nir - red) / np.where(denom == 0, 1, denom))
    return ndvi.astype(np.float32)


def compute_ndwi(stack: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). Highlights water bodies (rivers, lakes, canals)."""
    green = stack[..., BAND_IDX["B03"]]
    nir = stack[..., BAND_IDX["B08"]]
    denom = green + nir
    ndwi = np.where(denom == 0, 0, (green - nir) / np.where(denom == 0, 1, denom))
    return ndwi.astype(np.float32)


def normalize_bands(stack: np.ndarray, clip_percentile: float = 2.0) -> np.ndarray:
    """Percentile clip + min-max scale each spectral band to [0, 1] to reduce outlier/haze influence."""
    out = np.zeros_like(stack)
    n_bands = stack.shape[-1]
    for b in range(n_bands):
        band = stack[..., b]
        lo, hi = np.percentile(band, [clip_percentile, 100 - clip_percentile])
        if hi - lo < 1e-6:
            out[..., b] = 0
            continue
        clipped = np.clip(band, lo, hi)
        out[..., b] = (clipped - lo) / (hi - lo)
    return out


def build_feature_stack(raw_stack: np.ndarray) -> np.ndarray:
    """
    Combine normalized spectral bands + NDVI + NDWI into the final feature stack
    used for tiling and model input.
    """
    spectral = raw_stack[..., :10]  # drop dataMask (index 10)
    norm_spectral = normalize_bands(spectral)
    ndvi = compute_ndvi(raw_stack)[..., None]
    ndwi = compute_ndwi(raw_stack)[..., None]
    return np.concatenate([norm_spectral, ndvi, ndwi], axis=-1)  # (H, W, 12)


def tile_image(feature_stack: np.ndarray, tile_size: int = 64, stride: int = None):
    """
    Cut a large AOI feature stack into fixed-size tiles for CNN input.
    Returns a list of (tile, (row, col)) — (row, col) is the top-left pixel offset,
    useful later for manually labeling tiles against a basemap.
    """
    if stride is None:
        stride = tile_size
    h, w, _ = feature_stack.shape
    tiles = []
    for r in range(0, h - tile_size + 1, stride):
        for c in range(0, w - tile_size + 1, stride):
            tile = feature_stack[r : r + tile_size, c : c + tile_size, :]
            tiles.append((tile, (r, c)))
    return tiles


def augment_tile(tile: np.ndarray) -> list:
    """Return the tile plus rotations/flips — cheap augmentation for a small labeled set."""
    variants = [
        tile,
        np.rot90(tile, k=1),
        np.rot90(tile, k=2),
        np.rot90(tile, k=3),
        np.fliplr(tile),
        np.flipud(tile),
    ]
    return variants
