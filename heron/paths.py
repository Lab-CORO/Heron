"""Where Heron finds its own files and the site's data."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTS_DIR = REPO_ROOT / "robots"
STATIONS_DIR = REPO_ROOT / "stations"


def data_root() -> Path:
    """Directory holding generated data (master maps, datasets, corpora, experiments).

    HERON_DATA_ROOT first, then data_root in site.yaml; by default the directory that
    contains the repository, which is where the cluster layout already keeps its data.
    """
    if "HERON_DATA_ROOT" in os.environ:
        return Path(os.environ["HERON_DATA_ROOT"]).resolve()
    from heron.site import configured_data_root
    return (configured_data_root() or REPO_ROOT.parent).resolve()


def voxel_tag(voxel_size: float) -> str:
    """Voxel size as it appears in file names, e.g. 0.02 -> '0.02'.

    Matches the C++ generator, which streams the float into
    master_ik_data<voxel>.npz.
    """
    return f"{voxel_size:g}"
