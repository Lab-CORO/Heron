"""The Robot: an arm's kinematics, base frame and reach, independent of any Station."""

from dataclasses import dataclass
from pathlib import Path

from heron import paths
from heron.config import positive_float, read_config
from heron.errors import HeronConfigError

# create_reachability_map.cpp builds the master map in a cube of half side r = 2 m,
# hard-coded. Until the C++ reads the reach as a parameter (next image rebuild),
# a robot that reaches further would get a truncated master map.
MAX_SUPPORTED_REACH = 2.0

# Robot configs that ship with curobo inside the data-generation image.
CUROBO_BUILTIN_DIR = "/usr/local/lib/python3.10/dist-packages/curobo/content/configs/robot"

# Master maps generated before Heron, kept where gen_dataset.sh used to look for
# them (relative to the data root). Removed once they move to master_maps/.
_LEGACY_MASTER_MAP_DIRS = {
    "m1013": "data",
    "ur5e": "data_ur5e",
    "iiwa": "data_iiwa",
    "jaco7": "data_jaco7",
}


@dataclass(frozen=True)
class Robot:
    name: str
    base_link: str
    reach: float
    # "" -> curobo_ros default (m1013); "ur5e.yml" -> curobo builtin;
    # a path -> a config on the host, mounted into the container.
    curobo_config: str
    host_config_file: Path | None

    @property
    def container_config_file(self) -> str:
        """The robot config path as curobo_ros sees it inside the container."""
        if self.host_config_file is not None:
            return str(self.host_config_file)  # mounted at the same path
        if self.curobo_config:
            return f"{CUROBO_BUILTIN_DIR}/{self.curobo_config}"
        return ""

    @property
    def host_config_dir(self) -> Path | None:
        """Directory to mount into the container (the config plus its URDF and meshes)."""
        return self.host_config_file.parent if self.host_config_file else None

    def master_map_dir(self, voxel_size: float) -> Path:
        """Where the master map for this voxel size is written."""
        return paths.data_root() / "master_maps" / self.name / paths.voxel_tag(voxel_size)

    def find_master_map_dir(self, voxel_size: float) -> Path | None:
        """Directory holding an existing master map for this voxel size, or None."""
        filename = f"master_ik_data{paths.voxel_tag(voxel_size)}.npz"
        candidates = [self.master_map_dir(voxel_size)]
        if self.name in _LEGACY_MASTER_MAP_DIRS:
            candidates.append(paths.data_root() / _LEGACY_MASTER_MAP_DIRS[self.name])
        for directory in candidates:
            if (directory / filename).is_file():
                return directory
        return None


def load_robot(name_or_path: str) -> Robot:
    path = _robot_path(name_or_path)
    data = read_config(path, required={"name", "base_link", "reach"}, optional={"curobo_config"})

    if not str(data["base_link"]).strip():
        raise HeronConfigError(f"{path}: base_link must not be empty")
    reach = positive_float(path, "reach", data["reach"])
    if reach > MAX_SUPPORTED_REACH:
        raise HeronConfigError(
            f"{path}: reach {reach} m exceeds {MAX_SUPPORTED_REACH} m, the master map size "
            f"hard-coded in create_reachability_map.cpp. It needs a rebuilt image to go further.")

    curobo_config = str(data.get("curobo_config") or "")
    host_config_file = None
    if "/" in curobo_config:
        host_config_file = (path.parent / curobo_config).resolve()
        if not host_config_file.is_file():
            raise HeronConfigError(f"{path}: curobo_config file not found: {host_config_file}")

    return Robot(name=data["name"], base_link=str(data["base_link"]), reach=reach,
                 curobo_config=curobo_config, host_config_file=host_config_file)


def list_robots() -> list[str]:
    return sorted(p.stem for p in paths.ROBOTS_DIR.glob("*.yaml"))


def _robot_path(name_or_path: str) -> Path:
    if name_or_path.endswith((".yaml", ".yml")):
        return Path(name_or_path).resolve()
    path = paths.ROBOTS_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        hint = ""
        if (paths.STATIONS_DIR / f"{name_or_path}.yaml").is_file():
            hint = f" '{name_or_path}' is a station; pick a robot with --robot and pass --station {name_or_path}."
        raise HeronConfigError(
            f"unknown robot '{name_or_path}'. Available: {', '.join(list_robots())}.{hint}")
    return path
