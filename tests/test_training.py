import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from heron import training
from heron.cli import main
from heron.errors import HeronConfigError

NOW = datetime(2026, 10, 23, 14, 30)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    corpus = tmp_path / "corpora" / "full"
    for part in ("train", "val"):
        (corpus / part).mkdir(parents=True)
        (corpus / part / "a.h5").write_bytes(b"")
    (corpus / "corpus.yaml").write_text(yaml.safe_dump(
        {"name": "full", "crop": [[12, 140], [12, 140], [20, 148]], "inputs_sha256": "abc", "created": "x",
         "parts": {"train": {"samples": 10, "augmented": 20}, "val": {"samples": 4, "augmented": 0}}}))
    (tmp_path / "capa_unet.sif").write_bytes(b"")
    return tmp_path


def recipe(root: Path, **fields) -> training.TrainingRecipe:
    data = {"name": "cbr", "corpus": "full"}
    data.update(fields)
    path = root / "cbr.yaml"
    path.write_text(yaml.safe_dump(data))
    return training.load_training_recipe(path)


def make_experiment(root: Path, name: str, checkpoints=("last",)) -> Path:
    e = root / "experiments" / name
    (e / "checkpoint").mkdir(parents=True)
    for c in checkpoints:
        (e / "checkpoint" / f"{c}_checkpoint.pytorch").write_bytes(b"")
    return e


def test_defaults_are_the_recent_experiments_settings(root):
    p = training.plan(recipe(root), root / "capa_unet.sif", now=NOW)
    c = p.config
    assert p.experiment == root / "experiments" / "cbr_20261023_1430"
    assert c["model"]["layer_order"] == "cbr" and c["model"]["f_maps"][0] == 16
    assert c["loaders"]["dataset"] == "LazyHDF5Dataset" and c["loaders"]["batch_size"] == 4
    assert c["loaders"]["train"]["file_paths"] == [str(root / "corpora" / "full" / "train")]
    assert c["loaders"]["val"]["slice_builder"]["patch_shape"] == [128, 128, 128]
    assert c["trainer"]["checkpoint_dir"] == str(p.experiment / "checkpoint")
    assert c["trainer"]["resume"] is None and c["trainer"]["pre_trained"] is None


def test_recipe_sections_and_raw_overrides_are_merged(root):
    c = training.plan(recipe(root, model={"layer_order": "cbe"}, loaders={"batch_size": 2},
                             pytorch3dunet={"trainer": {"max_num_epochs": 7}, "model": {"num_groups": 4}}),
                      root / "capa_unet.sif", now=NOW).config
    assert c["model"]["layer_order"] == "cbe" and c["model"]["f_maps"][0] == 16
    assert c["loaders"]["batch_size"] == 2 and c["loaders"]["num_workers"] == 4
    assert c["trainer"]["max_num_epochs"] == 7 and c["model"]["num_groups"] == 4


def test_patch_larger_than_the_crop_is_refused(root):
    with pytest.raises(HeronConfigError, match="larger than the corpus crop"):
        training.plan(recipe(root, patch=152), root / "capa_unet.sif", now=NOW)
    assert training.plan(recipe(root, patch=64), root / "capa_unet.sif", now=NOW) \
        .config["loaders"]["train"]["slice_builder"]["stride_shape"] == [64, 64, 64]


@pytest.mark.parametrize("fields, message", [
    ({"loss": {"name": "BCEDiceGradientLoss"}}, "not built into pytorch-3dunet"),
    ({"init": {"resume": "x"}}, "resume is `heron train --resume`"),
    ({"init": {"pretrained": "x", "checkpoint": "middle"}}, "last or best"),
    ({"resources": {"gpu": 2}}, "resources accepts"),
    ({"model": "cbr"}, "model must be a mapping"),
])
def test_invalid_recipes_are_refused(root, fields, message):
    with pytest.raises(HeronConfigError, match=message):
        recipe(root, **fields)


def test_missing_or_incomplete_corpus_is_refused(root):
    with pytest.raises(HeronConfigError, match="not found"):
        training.plan(recipe(root, corpus="nope"), root / "capa_unet.sif", now=NOW)
    (root / "corpora" / "full" / "val" / "a.h5").unlink()
    with pytest.raises(HeronConfigError, match="no val samples"):
        training.plan(recipe(root), root / "capa_unet.sif", now=NOW)


def test_pretrained_points_at_another_experiments_checkpoint(root):
    make_experiment(root, "old_20261001_0900", checkpoints=("last", "best"))
    p = training.plan(recipe(root, init={"pretrained": "old_20261001_0900", "checkpoint": "best"}),
                      root / "capa_unet.sif", now=NOW)
    assert p.config["trainer"]["pre_trained"] == str(root / "experiments/old_20261001_0900/checkpoint/best_checkpoint.pytorch")
    with pytest.raises(HeronConfigError, match="not found"):
        training.plan(recipe(root, init={"pretrained": "missing"}), root / "capa_unet.sif", now=NOW)


def test_pretrained_outside_the_data_root_gets_mounted(root, tmp_path_factory):
    legacy = tmp_path_factory.mktemp("legacy") / "cbr_128_batch4_real"
    (legacy / "checkpoint").mkdir(parents=True)
    (legacy / "checkpoint" / "last_checkpoint.pytorch").write_bytes(b"")
    p = training.plan(recipe(root, init={"pretrained": str(legacy)}), root / "capa_unet.sif", now=NOW)
    assert f"--bind {legacy / 'checkpoint'}" in p.job


def test_job_runs_train3dunet_with_host_paths(root):
    p = training.plan(recipe(root, resources={"time": "2:00:00"}), root / "capa_unet.sif", now=NOW)
    assert "#SBATCH --time=2:00:00" in p.job and "#SBATCH --gpus-per-node=1" in p.job
    assert f"train3dunet --config {p.experiment / 'config.yaml'}" in p.job
    assert f"--bind {root}" in p.job and "/workspace" not in p.job and "format_data" not in p.job


def test_create_writes_the_experiment_and_submit_records_the_job(root, monkeypatch):
    p = training.plan(recipe(root, description="first"), root / "capa_unet.sif", now=NOW)
    e = training.create(p)
    assert {x.name for x in e.iterdir()} == {"checkpoint", "logs", "recipe.yaml", "config.yaml", "job.sh",
                                             "experiment.yaml"}
    monkeypatch.setattr(training.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="Submitted batch job 4242\n"))
    assert training.submit(e, account="def-x") == "4242"
    m = training.read_manifest(e)
    assert m["corpus"] == "full" and m["corpus_inputs_sha256"] == "abc" and m["description"] == "first"
    assert m["jobs"][0]["id"] == "4242"
    with pytest.raises(HeronConfigError, match="already exists"):
        training.plan(recipe(root), root / "capa_unet.sif", now=NOW)


def test_resume_continues_from_the_experiments_own_last_checkpoint(root):
    e = training.create(training.plan(recipe(root), root / "capa_unet.sif", now=NOW))
    with pytest.raises(HeronConfigError, match="last_checkpoint"):
        training.prepare_resume(e)
    (e / "checkpoint" / "last_checkpoint.pytorch").write_bytes(b"")
    training.prepare_resume(e)
    trainer = yaml.safe_load((e / "config.yaml").read_text())["trainer"]
    assert trainer["resume"] == str(e / "checkpoint" / "last_checkpoint.pytorch") and trainer["pre_trained"] is None


def test_summary_reads_the_best_score_from_the_logs(root):
    e = training.create(training.plan(recipe(root), root / "capa_unet.sif", now=NOW))
    (e / "logs" / "slurm_1.out").write_text(
        "INFO UNetTrainer - Validation finished. Loss: 0.7. Evaluation score: 0.79\n"
        "INFO UNetTrainer - Validation finished. Loss: 0.6. Evaluation score: 0.83\n")
    s = training.summary(e)
    assert s["best_score"] == 0.83 and s["validations"] == 2 and s["corpus"] == "full"
    assert [x.name for x in training.list_experiments()] == ["cbr_20261023_1430"]


def test_dry_run_writes_nothing(root, capsys):
    path = root / "cbr.yaml"
    path.write_text("name: cbr\ncorpus: full\n")
    assert main(["train", str(path), "--dry-run"]) == 0
    assert "train3dunet" in capsys.readouterr().out
    assert not (root / "experiments").exists()


def test_a_part_smaller_than_one_batch_is_refused(root):
    # pytorch-3dunet drops incomplete batches: 4 val samples with batch 8 would validate nothing
    with pytest.raises(HeronConfigError, match="val part has 4 patch"):
        training.plan(recipe(root, loaders={"batch_size": 8}), root / "capa_unet.sif", now=NOW)
    # smaller patches give 8 patches per volume, enough for the batch
    training.plan(recipe(root, loaders={"batch_size": 8}, patch=64), root / "capa_unet.sif", now=NOW)
