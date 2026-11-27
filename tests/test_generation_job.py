"""job.sbatch judges a create_rm run by the master map it wrote, not by ros2 launch's exit code.

ros2 launch exits 0 when a required node dies (it shuts the others down): a create_rm killed
for lack of memory used to end with status=0 and no master map.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JOB = REPO / "data_generation" / "pipeline" / "job.sbatch"


def run_create_rm(tmp_path, writes, old=()):
    """Run job.sbatch for create_rm with a fake apptainer that exits 0 and writes `writes`;
    `old` files are already in the master map directory, from a previous run."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    data = tmp_path / "master_maps"
    data.mkdir()
    for name in old:
        (data / name).write_text("old")
        os.utime(data / name, (1, 1))            # written long before this run
    fake = bin_dir / "apptainer"
    fake.write_text("#!/bin/sh\n" + "".join(f'echo x > "$DATA_DIR/{w}"\n' for w in writes) + "exit 0\n")
    fake.chmod(0o755)
    module = bin_dir / "module"                 # `module load apptainer` outside a cluster
    module.write_text("#!/bin/sh\nexit 0\n")
    module.chmod(0o755)
    slurm_tmp = tmp_path / "slurm_tmp"
    slurm_tmp.mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", MODE="create_rm", VOXEL="0.1",
               DATA_DIR=str(data), LAUNCH_DIR=str(tmp_path), SIF=str(tmp_path / "image.sif"),
               WARP_CACHE=str(tmp_path / "warp"), SLURM_TMPDIR=str(slurm_tmp), HERON_IMAGE="jazzy")
    env.pop("SLURM_ARRAY_TASK_ID", None)
    env.pop("BASH_ENV", None)                   # Lmod re-defines `module` in every bash through it
    for k in [k for k in env if k.startswith("BASH_FUNC_")]:   # Lmod's `module` function: use the fake
        del env[k]
    return subprocess.run(["bash", str(JOB)], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_create_rm_that_wrote_its_master_map_succeeds(tmp_path):
    r = run_create_rm(tmp_path, ["master_ik_data0.1.npz", "master_ik_data0.1.h5"])
    assert r.returncode == 0, r.stderr
    assert "status=0" in r.stdout


@pytest.mark.parametrize("writes", [[], ["master_ik_data0.1.h5"], ["master_ik_data0.1.npz"]])
def test_create_rm_without_its_master_map_fails(tmp_path, writes):
    r = run_create_rm(tmp_path, writes)
    assert r.returncode == 1
    assert "create_rm wrote no master_ik_data" in r.stderr


def test_a_previous_runs_master_map_does_not_count(tmp_path):
    r = run_create_rm(tmp_path, [], old=["master_ik_data0.1.npz", "master_ik_data0.1.h5"])
    assert r.returncode == 1
    assert "create_rm wrote no master_ik_data" in r.stderr
