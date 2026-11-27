"""gen_dataset.sh --mode real copies the rosbag to scratch unless a complete copy is there."""
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN_DATASET = REPO / "data_generation" / "pipeline" / "gen_dataset.sh"
JAZZY = {"heron.image": "generation", "heron.ros": "jazzy", "heron.curobo": "v0.8.0"}
BAG = {"metadata.yaml": b"rosbag2_bagfile_information: {}\n", "bag_0.db3": b"x" * 1000}


def run_real(tmp_path):
    """gen_dataset.sh --mode real --dry-run, with a fake image, master map and rosbag."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    payload = json.dumps({"data": {"attributes": {"labels": JAZZY}}})
    (bin_dir / "apptainer").write_text(f"#!/bin/sh\ncat <<'EOF'\n{payload}\nEOF\n")
    (bin_dir / "apptainer").chmod(0o755)
    (tmp_path / "image.sif").write_bytes(b"")
    (tmp_path / "site.yaml").write_text("scheduler: slurm\n")
    master = tmp_path / "master_maps" / "m1013" / "0.02"
    master.mkdir(parents=True, exist_ok=True)
    (master / "master_ik_data0.02.npz").write_bytes(b"")
    bag = tmp_path / "bags" / "my_bag"
    bag.mkdir(parents=True, exist_ok=True)
    for name, data in BAG.items():
        (bag / name).write_bytes(data)
    args = ["bash", str(GEN_DATASET), "--robot", "m1013", "--station", "leeloo", "--mode", "real",
            "--rosbag", str(bag / "bag_0.db3"), "--camera-frame", "cam", "--voxel", "0.02",
            "--sif", str(tmp_path / "image.sif"), "--out", "t", "--dry-run"]
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HERON_DATA_ROOT=str(tmp_path),
               HERON_SITE=str(tmp_path / "site.yaml"), SCRATCH=str(tmp_path / "scratch"))
    return subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)


def staged(tmp_path):
    return tmp_path / "scratch" / "rosbag" / "my_bag"


def test_a_missing_copy_is_made(tmp_path):
    r = run_real(tmp_path)
    assert "Staging rosbag to scratch" in r.stdout, r.stderr


def test_a_purged_or_partial_copy_is_made_again(tmp_path):
    staged(tmp_path).mkdir(parents=True)                       # emptied by the scratch purge
    assert "Staging rosbag to scratch" in run_real(tmp_path).stdout
    (staged(tmp_path) / "metadata.yaml").write_bytes(BAG["metadata.yaml"])
    (staged(tmp_path) / "bag_0.db3").write_bytes(b"x" * 10)    # copy stopped half way
    assert "Staging rosbag to scratch" in run_real(tmp_path).stdout


def test_a_complete_copy_is_kept(tmp_path):
    staged(tmp_path).mkdir(parents=True)
    for name, data in BAG.items():
        (staged(tmp_path) / name).write_bytes(data)
    r = run_real(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Staging rosbag to scratch" not in r.stdout
