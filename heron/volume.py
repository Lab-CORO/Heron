"""Operations on one Scene / Reachability map pair, as 3D arrays indexed [x, y, z]."""

import math

import numpy as np
from scipy.ndimage import affine_transform

from heron.station import Crop


def base_index(attrs: dict, voxel_size: float) -> tuple[float, float, float]:
    """Index of the robot base in the grid.

    The generator places voxel i at origine + i * voxel_size, with the base at the
    world origin, so the base sits at index -origine / voxel_size on each axis.
    """
    return tuple(-float(attrs[f"origine_{a}"]) / voxel_size for a in "xyz")


def rotate_about_base(volume: np.ndarray, base: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    """Rotate around the vertical axis through the base, nearest neighbour, zeros outside.

    Nearest neighbour keeps a binary Scene binary and a Reachability map on its
    discrete levels.
    """
    a = math.radians(angle_deg)
    # affine_transform maps each output index o to the input index M @ o + offset,
    # so M is the inverse rotation; the base is the fixed point.
    m = np.array([[math.cos(a), math.sin(a), 0.0],
                  [-math.sin(a), math.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    c = np.array([base[0], base[1], 0.0])
    return affine_transform(volume, m, offset=c - m @ c, order=0, mode="constant", cval=0.0)


def crop(volume: np.ndarray, ranges: Crop) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = ranges
    return volume[x0:x1, y0:y1, z0:z1]


def cropped_attrs(attrs: dict, ranges: Crop, voxel_size: float) -> dict:
    """Geometry attributes of a cropped volume: origin shifted, sizes updated."""
    out = dict(attrs)
    for axis, (lo, hi) in zip("xyz", ranges):
        if f"origine_{axis}" in out:
            out[f"origine_{axis}"] = float(out[f"origine_{axis}"]) + lo * voxel_size
        out[f"voxel_grid_size_{axis}"] = float(hi - lo)
    return out


def upsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """Split each voxel into factor³ voxels of the same value."""
    for axis in range(3):
        volume = np.repeat(volume, factor, axis=axis)
    return volume


def is_uninformative(label: np.ndarray, min_reachable_frac: float, reachable_thresh: float = 0.05) -> bool:
    """An all-zero Reachability map, or one with too few reachable voxels."""
    if float(label.max()) < 1e-6:
        return True
    return min_reachable_frac > 0 and float(np.mean(label >= reachable_thresh)) < min_reachable_frac
