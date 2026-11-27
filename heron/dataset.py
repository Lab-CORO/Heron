"""A Dataset: the raw output of one generation run, and the manifest that says where it came from."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import yaml

from heron.errors import HeronConfigError
from heron.robot import load_robot
from heron.station import check_compatible, load_station

MANIFEST = "dataset.yaml"
MODES = ("real", "synthetic", "scene")


@dataclass(frozen=True)
class SampleRef:
    """One Scene / Reachability map pair inside a raw generator file."""
    path: Path
    group: str
    source: str   # Scenes sharing a source are near duplicates and stay on one side of a split

    @property
    def name(self) -> str:
        return f"{self.path.stem}_g{self.group}"


@dataclass(frozen=True)
class Dataset:
    path: Path
    robot: str
    station: str
    mode: str
    voxel_size: float
    source: str

    @property
    def name(self) -> str:
        return self.path.name

    def files(self) -> list[Path]:
        return sorted(p for p in self.path.glob("*.h5") if not p.name.startswith("master_ik_data"))

    def source_of(self, path: Path) -> str:
        """Source of the Scenes in one generator file.

        real: every file replays the same rosbag, so the whole Dataset is one source.
        synthetic: each file is its own run of random obstacles.
        scene: files are named scene_<i>_task<n>_job<id>.h5; scene <i> is the source.
        """
        if self.mode == "real":
            return f"{self.name}/{self.source}"
        if self.mode == "scene":
            match = re.match(r"(scene_\d+)", path.stem)
            return f"{self.name}/{match.group(1) if match else path.stem}"
        return f"{self.name}/{path.stem}"

    def samples(self, on_skip=None) -> Iterator[SampleRef]:
        """Every sample of the Dataset, without loading the volumes.

        Unreadable files are reported through on_skip(path, error) and skipped.
        """
        import h5py
        for path in self.files():
            try:
                with h5py.File(path, "r", locking=False) as f:
                    if "group" not in f:
                        raise OSError("no 'group' in file (not generator output)")
                    groups = [g for g in sorted(f["group"].keys(), key=int)
                              if "voxel_grid" in f["group"][g] and "reachability_map" in f["group"][g]]
            except OSError as exc:
                if on_skip:
                    on_skip(path, exc)
                continue
            for group in groups:
                yield SampleRef(path=path, group=group, source=self.source_of(path))


def describe(path: Path, robot: str, station: str, mode: str, voxel_size: float,
             source: str = "", force: bool = False) -> Path:
    """Write the manifest of a Dataset directory. Returns the manifest path."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise HeronConfigError(f"{path}: not a directory")
    manifest = path / MANIFEST
    if manifest.exists() and not force:
        raise HeronConfigError(f"{manifest} already exists (use --force to overwrite)")
    if mode not in MODES:
        raise HeronConfigError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    check_compatible(load_robot(robot), load_station(station))
    if mode == "real" and not source:
        source = path.name
    data = {
        "name": path.name,
        "robot": robot,
        "station": station,
        "mode": mode,
        "voxel_size": float(voxel_size),
        "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest.write_text(yaml.safe_dump(data, sort_keys=False))
    return manifest


def load_dataset(path: Path) -> Dataset:
    path = Path(path).resolve()
    manifest = path / MANIFEST
    if not manifest.is_file():
        raise HeronConfigError(
            f"{path}: no {MANIFEST}. Datasets generated before manifests existed need one: "
            f"heron dataset describe {path} --robot <robot> --station <station> --mode <mode> --voxel <m>")
    data = yaml.safe_load(manifest.read_text()) or {}
    missing = {"robot", "station", "mode", "voxel_size"} - data.keys()
    if missing:
        raise HeronConfigError(f"{manifest}: missing key(s) {', '.join(sorted(missing))}")
    if data["mode"] not in MODES:
        raise HeronConfigError(f"{manifest}: unknown mode {data['mode']!r}")
    return Dataset(path=path, robot=str(data["robot"]), station=str(data["station"]),
                   mode=data["mode"], voxel_size=float(data["voxel_size"]),
                   source=str(data.get("source") or ""))
