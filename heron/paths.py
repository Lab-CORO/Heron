"""Where Heron finds its own files and the site's data."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTS_DIR = REPO_ROOT / "robots"
STATIONS_DIR = REPO_ROOT / "stations"


def data_root() -> Path:
    """Directory holding generated data (master maps, datasets).

    HERON_DATA_ROOT overrides it; by default it is the directory that contains
    the repository, which is where the cluster layout already keeps its data.
    """
    return Path(os.environ.get("HERON_DATA_ROOT", REPO_ROOT.parent)).resolve()


def voxel_tag(voxel_size: float) -> str:
    """Voxel size as it appears in file names, e.g. 0.02 -> '0.02'.

    Matches the C++ generator, which streams the float into
    master_ik_data<voxel>.npz.
    """
    return f"{voxel_size:g}"
