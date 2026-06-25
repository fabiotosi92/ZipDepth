"""Visualization utilities"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Optional 


_LUT_CACHE: Dict[tuple, np.ndarray] = {}


def _build_lut(cmap: str, invert: bool) -> np.ndarray:
    """Build a 256-entry BGR uint8 lookup table from a matplotlib colormap."""
    key = (cmap, invert)
    if key not in _LUT_CACHE:
        colormap = plt.get_cmap(cmap)
        indices = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        if invert:
            indices = 1.0 - indices
        rgba = colormap(indices)
        rgb_u8 = (rgba[:, :3] * 255.0).astype(np.uint8)
        bgr_u8 = rgb_u8[:, ::-1].copy()
        _LUT_CACHE[key] = bgr_u8.reshape(256, 1, 3)
    return _LUT_CACHE[key]


def depth_to_colormap(depth: np.ndarray, cmap: str = 'Spectral',
                      vmin: Optional[float] = None, vmax: Optional[float] = None,
                      invert: bool = True) -> np.ndarray:
    """
    Convert depth map to colormap visualization.

    Uses pre-computed LUT + cv2.LUT for ~10-20x speedup over matplotlib.

    Args:
        depth: Depth map (H, W)
        cmap: Matplotlib colormap name ('Spectral', 'turbo', 'viridis', etc.)
        vmin: Minimum depth for color scaling (auto if None)
        vmax: Maximum depth for color scaling (auto if None)
        invert: Invert colormap (True = far is blue, near is red)

    Returns:
        BGR image (H, W, 3) uint8 for OpenCV
    """
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()

    scale = 255.0 / (vmax - vmin + 1e-8)
    depth_u8 = np.clip((depth - vmin) * scale, 0, 255).astype(np.uint8)

    depth_3ch = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

    lut = _build_lut(cmap, invert)
    return cv2.LUT(depth_3ch, lut)


