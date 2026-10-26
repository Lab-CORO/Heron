import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from heron import training
from heron.cli import main
from heron.errors import HeronConfigError
from heron.evaluation import jobs, measure
from heron.evaluation.cache import load_eval_data, save_eval_data
from heron.evaluation.metrics import (N_CLASSES, confusion_matrix, merge_class_metrics, multiclass_metrics,
                                      quadratic_weighted_kappa, to_classes)
from heron.evaluation.report import summarize


# --- metrics (unchanged from evaluate.py) -------------------------------------------------

def test_classes_are_fiftieths():
    assert to_classes(np.array([0.0, 0.02, 0.5, 1.0, 1.0000000004])).tolist() == [0, 1, 25, 50, 50]


def test_perfect_prediction_scores_perfectly():
    labels = to_classes(np.random.RandomState(0).rand(1000))
    m = multiclass_metrics(labels, labels)
    assert m["ExactAccuracy"] == 1.0 and m["QuadraticKappa"] == 1.0 and m["MeanClassError"] == 0.0


def test_metrics_from_the_confusion_matrix_match_the_per_voxel_ones():
    rng = np.random.RandomState(1)
    label = to_classes(rng.rand(5000))
    pred = np.clip(label + rng.randint(-3, 4, size=label.shape), 0, 50).astype(np.int16)
    direct = multiclass_metrics(pred, label)
    merged = merge_class_metrics(confusion_matrix(pred, label), n_voxels=label.size)
    for key in ("ExactAccuracy", "Within1Accuracy", "Within2Accuracy", "MeanClassError", "RMSEClass",
                "QuadraticKappa"):
        assert merged[key] == pytest.approx(direct[key])


def test_kappa_of_an_opposite_prediction_is_negative():
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    cm[0, 50] = cm[50, 0] = 10
    assert quadratic_weighted_kappa(cm) < 0


# --- measurement, with a stand-in model ------------------------------------------------------

class HalfModel:
    """Predicts the label shifted by one class, to get non-trivial metrics."""
    def __init__(self, labels):
        self.labels = iter(labels)

    def predict(self, raw):
        return np.clip(next(self.labels) + 1 / 50, 0, 1).astype(np.float32)


def write_volume(path: Path, label: np.ndarray):
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=(label > 0.5).astype(np.float64))
        f.create_dataset("label", data=label)


def test_measure_part_accumulates_every_file(tmp_path):
    rng = np.random.RandomState(2)
    labels = [np.round(rng.rand(8, 8, 8) * 50) / 50 for _ in range(4)]
    for i, label in enumerate(labels):
        write_volume(tmp_path / f"v{i}.h5", label)
    data = measure.measure_part(HalfModel(labels), sorted(tmp_path.glob("*.h5")), log=lambda *_: None)
    assert data["n_files"] == 4 and data["global_cm"].sum() == 4 * 8 ** 3
    assert len(data["region_samples"]) == 3
    assert data["global_reg"]["MAE"] == pytest.approx(np.mean([np.mean(np.abs(np.clip(l + 0.02, 0, 1) - l))
                                                                for l in labels]))
    s = summarize(data["global_cm"], data["region_cm"], data["global_reg"])
    assert s["voxels"] == 4 * 8 ** 3 and set(s["regions"]) == {"reachable", "graded", "thin"}
    assert s["classification"]["ExactAccuracy"] < 1.0 and s["classification"]["Within1Accuracy"] == 1.0


def test_run_writes_reports_cache_and_results(tmp_path, monkeypatch):
    exp = tmp_path / "exp_20261026_1000"
    exp.mkdir()
    (exp / "config.yaml").write_text("model: {}\nloaders: {}\n")
    (exp / "experiment.yaml").write_text(yaml.safe_dump({"corpus": "c", "recipe": "r.yaml",
                                                         "corpus_inputs_sha256": "abc", "git_commit": "123"}))
    parts = {}
    for part in ("val", "test:reel"):
        d = tmp_path / part.replace(":", "_")
        d.mkdir()
        write_volume(d / "a.h5", np.full((8, 8, 8), 0.5))
        parts[part] = d
    monkeypatch.setattr(measure, "Predictor", lambda config, ckpt: HalfModel([np.full((8, 8, 8), 0.5)] * 2))
    results = tmp_path / "results" / "exp.yaml"
    measure.run(exp, exp / "best_checkpoint.pytorch", parts, results, log=lambda *_: None)

    summary = yaml.safe_load(results.read_text())
    assert summary["corpus"] == "c" and summary["checkpoint"] == "best_checkpoint.pytorch"
    assert set(summary["parts"]) == {"val", "test:reel"}
    assert summary["parts"]["test:reel"]["regression"]["MAE"] == pytest.approx(0.02)
    for part in ("val", "test_reel"):
        assert (exp / "evaluation" / part / "report.txt").read_text().startswith("=== CapaciNet Evaluation Report")
        assert load_eval_data(exp / "evaluation" / part / "eval_data.h5")["n_files"] == 1


# --- host side: parts, job, chaining, figures -------------------------------------------------

@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    corpus = tmp_path / "corpora" / "full"
    for part in ("train", "val", "test/reel", "test/mixte"):
        (corpus / part).mkdir(parents=True)
        (corpus / part / "a.h5").write_bytes(b"")
    (corpus / "corpus.yaml").write_text(yaml.safe_dump(
        {"name": "full", "crop": [[0, 128]] * 3, "inputs_sha256": "abc",
         "parts": {"train": {"samples": 8}, "val": {"samples": 4}}}))
    (tmp_path / "capa_unet.sif").write_bytes(b"")
    recipe = tmp_path / "cbr.yaml"
    recipe.write_text("name: cbr\ncorpus: full\n")
    plan = training.plan(training.load_training_recipe(recipe), tmp_path / "capa_unet.sif",
                         now=datetime(2026, 10, 26, 10, 0))
    return training.create(plan)


def test_an_experiment_is_evaluated_on_val_and_every_named_test(experiment):
    assert list(jobs.corpus_parts(experiment)) == ["val", "test:mixte", "test:reel"]


def test_eval_job_measures_inside_the_training_image(experiment):
    script = jobs.write_eval_job(experiment, checkpoint="last", only=["test:reel"]).read_text()
    assert "python -m heron.evaluation.measure" in script
    assert str(experiment / "checkpoint" / "last_checkpoint.pytorch") in script
    assert "--part test:reel=" in script and "--part val=" not in script
    assert f"--results {jobs.results_path(experiment)}" in script
    assert "capa_unet.sif" in script and "eval_env" not in script
    with pytest.raises(HeronConfigError, match="unknown part"):
        jobs.write_eval_job(experiment, only=["test:nope"])


def test_train_chains_the_evaluation_after_a_successful_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    corpus = tmp_path / "corpora" / "full"
    for part in ("train", "val"):
        (corpus / part).mkdir(parents=True)
        (corpus / part / "a.h5").write_bytes(b"")
    (corpus / "corpus.yaml").write_text(yaml.safe_dump({"name": "full", "crop": [[0, 128]] * 3,
                                                        "parts": {"train": {"samples": 8}, "val": {"samples": 4}}}))
    (tmp_path / "capa_unet.sif").write_bytes(b"")
    recipe = tmp_path / "cbr.yaml"
    recipe.write_text("name: cbr\ncorpus: full\n")
    calls = []

    def fake_sbatch(cmd, **kw):
        if cmd[0] != "sbatch":          # git, for the commit recorded in experiment.yaml
            return subprocess.CompletedProcess(cmd, 0, stdout="")
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"Submitted batch job {100 + len(calls)}\n")
    monkeypatch.setattr(training.subprocess, "run", fake_sbatch)
    assert main(["train", str(recipe)]) == 0
    assert calls[0][-1].endswith("job.sh")
    assert calls[1][-1].endswith("job_eval.sh") and "--dependency=afterok:101" in calls[1]
    experiment = next((tmp_path / "experiments").iterdir())
    assert [j["kind"] for j in training.read_manifest(experiment)["jobs"]] == ["train", "evaluate"]


def test_figures_are_drawn_from_the_cache(experiment):
    rng = np.random.RandomState(3)
    label = np.round(rng.rand(16, 16, 16) * 50) / 50
    pred = np.clip(label + 0.02, 0, 1)
    cm = confusion_matrix(to_classes(pred).ravel(), to_classes(label).ravel())
    out = experiment / "evaluation" / "val"
    out.mkdir(parents=True)
    counts, edges = np.histogram(np.abs(pred - label), bins=10, density=True)
    save_eval_data(out / "eval_data.h5", cm, {r: cm for r in ("reachable", "graded", "thin")}, 1,
                   {"MAE": 0.02, "MedAE": 0.02, "RMSE": 0.02, "Pearson_r": 0.99}, counts, edges,
                   [(pred, label, (label > 0.5).astype(float))])
    jobs.draw_figures(experiment, log=lambda *_: None)
    assert {p.name for p in out.glob("*.png")} == {"confusion_matrix.png", "error_histogram.png",
                                                   "tolerance_curve.png", "regions_0.png"}


def test_figures_before_any_measurement_explain_why(experiment):
    with pytest.raises(HeronConfigError, match="nothing measured yet"):
        jobs.draw_figures(experiment)
