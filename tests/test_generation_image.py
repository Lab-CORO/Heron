"""gen_dataset.sh picks the generation image's behaviour from its labels (docs/images.md)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN_DATASET = REPO / "data_generation" / "pipeline" / "gen_dataset.sh"

JAZZY = {"heron.image": "generation", "heron.ros": "jazzy", "heron.curobo": "v0.8.0"}


def run_gen_dataset(tmp_path, labels, mode, inspect_ok=True):
    """Run gen_dataset.sh --dry-run against a fake .sif whose labels a fake apptainer reports."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    payload = json.dumps({"data": {"attributes": {"labels": labels}}})
    fake = bin_dir / "apptainer"
    fake.write_text("#!/bin/sh\n" + (f"cat <<'EOF'\n{payload}\nEOF\n" if inspect_ok else "exit 255\n"))
    fake.chmod(0o755)
    sif = tmp_path / "image.sif"
    sif.write_bytes(b"")
    args = ["bash", str(GEN_DATASET), "--robot", "m1013", "--mode", mode, "--sif", str(sif),
            "--out", "t", "--dry-run"]
    if mode != "create_rm":
        args[4:4] = ["--station", "leeloo"]
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HERON_DATA_ROOT=str(tmp_path),
               HERON_SITE=str(tmp_path / "site.yaml"))
    (tmp_path / "site.yaml").write_text("scheduler: slurm\n")
    return subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)


def test_jazzy_image_refuses_scene_mode(tmp_path):
    r = run_gen_dataset(tmp_path, JAZZY, "scene")
    assert r.returncode != 0
    assert "le mode scene n'est pas encore disponible avec l'image ROS 2 Jazzy + cuRobo v2" in r.stderr
    assert "--sif <data_generation humble.sif>" in r.stderr


@pytest.mark.parametrize("mode", ["synthetic", "create_rm"])
def test_jazzy_image_runs_the_other_modes(tmp_path, mode):
    r = run_gen_dataset(tmp_path, JAZZY, mode)
    assert "Generation image:" in r.stdout and "(jazzy)" in r.stdout, r.stderr
    assert "scene n'est pas encore disponible" not in r.stderr


def test_unlabelled_humble_image_only_runs_scene_mode(tmp_path):
    r = run_gen_dataset(tmp_path, {}, "synthetic")
    assert r.returncode != 0 and "ROS 2 Humble image" in r.stderr
    r = run_gen_dataset(tmp_path, None, "scene")
    assert "(humble)" in r.stdout, r.stderr


def test_unknown_image_or_unreadable_labels_are_refused(tmp_path):
    r = run_gen_dataset(tmp_path, {"heron.image": "generation", "heron.ros": "rolling"}, "synthetic")
    assert r.returncode != 0 and "unknown generation image" in r.stderr
    r = run_gen_dataset(tmp_path, JAZZY, "synthetic", inspect_ok=False)
    assert r.returncode != 0 and "cannot read the labels" in r.stderr
