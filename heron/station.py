"""The Station: where a Robot works, and the region around its base a Scene covers."""

import math
from dataclasses import dataclass
from pathlib import Path

from heron import paths
from heron.config import positive_float, read_config, resolve_file
from heron.errors import HeronConfigError
from heron.robot import Robot

Crop = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class Station:
    name: str
    world: Path | None          # fixed obstacles (curobo world YAML); None = curobo default
    cameras: Path | None        # camera config template for rosbag datasets
    camera_frame: str           # default camera frame for rosbag datasets
    scene_half_extent: float    # m: a Scene spans [-h, h] on each axis around the base
    crop_offset: tuple[float, float, float]  # m: center of the default Crop, from the base

    def scene_voxels(self, voxel_size: float) -> int:
        """Voxels per axis of a Scene.

        Same expression as curobo_ros's get_voxel_grid, so the count matches the
        generated grids exactly (3.04 / 0.02 -> 152).
        """
        return int(math.ceil((self.scene_half_extent - -self.scene_half_extent) / voxel_size))

    def crop(self, voxel_size: float, size: int,
             offset: tuple[float, float, float] | None = None) -> Crop:
        """Index ranges [lo, hi) per axis of a size³ cube cut out of a Scene.

        The cube is centered on the base, shifted by `offset` (meters, defaults to
        the station's crop_offset). Raises if the cube leaves the Scene.
        """
        n = self.scene_voxels(voxel_size)
        if size <= 0:
            raise HeronConfigError(f"crop size must be positive, got {size}")
        offset = self.crop_offset if offset is None else tuple(offset)
        ranges = []
        for axis, shift in zip("xyz", offset):
            center = n // 2 + round(shift / voxel_size)
            lo = center - size // 2
            hi = lo + size
            if lo < 0 or hi > n:
                raise HeronConfigError(
                    f"station {self.name}: a {size}³ crop shifted by {shift} m on {axis} "
                    f"spans [{lo}, {hi}) and leaves the {n}-voxel scene")
            ranges.append((lo, hi))
        return tuple(ranges)


def load_station(name_or_path: str) -> Station:
    path = _station_path(name_or_path)
    data = read_config(
        path,
        required={"name", "scene_half_extent"},
        optional={"world", "cameras", "camera_frame", "crop_offset"},
    )
    crop_offset = data.get("crop_offset", [0.0, 0.0, 0.0])
    if not isinstance(crop_offset, list) or len(crop_offset) != 3:
        raise HeronConfigError(f"{path}: crop_offset must be a list [x, y, z] in meters")
    try:
        crop_offset = tuple(float(v) for v in crop_offset)
    except (TypeError, ValueError):
        raise HeronConfigError(f"{path}: crop_offset values must be numbers") from None

    return Station(
        name=data["name"],
        world=resolve_file(path, "world", data.get("world")),
        cameras=resolve_file(path, "cameras", data.get("cameras")),
        camera_frame=str(data.get("camera_frame") or ""),
        scene_half_extent=positive_float(path, "scene_half_extent", data["scene_half_extent"]),
        crop_offset=crop_offset,
    )


def check_compatible(robot: Robot, station: Station) -> None:
    """A Scene must lie inside the robot's master map, or its edges have no reachability data."""
    if station.scene_half_extent > robot.reach:
        raise HeronConfigError(
            f"station {station.name} covers ±{station.scene_half_extent} m but the master map of "
            f"{robot.name} only covers ±{robot.reach} m")


def list_stations() -> list[str]:
    return sorted(p.stem for p in paths.STATIONS_DIR.glob("*.yaml"))


def _station_path(name_or_path: str) -> Path:
    if name_or_path.endswith((".yaml", ".yml")):
        return Path(name_or_path).resolve()
    path = paths.STATIONS_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise HeronConfigError(
            f"unknown station '{name_or_path}'. Available: {', '.join(list_stations())}")
    return path
