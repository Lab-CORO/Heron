import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from heron import doctor, paths, training
from heron.cli import main
from heron.errors import HeronConfigError
from heron.evaluation import jobs
from heron.execution import Job, JobFailed, LocalExecutor, SlurmExecutor, render
from heron.site import host_arch, load_site

NOW = datetime(2026, 10, 27, 9, 0)
RES = {"time": "1:00:00", "mem": "8G", "cpus": 2, "gpus": 1}


def write_site(isolated_site: Path, **data) -> None:
    isolated_site.write_text(yaml.safe_dump(data))


def fake_venv(path: Path, with_p3d=True) -> Path:
    """A venv as heron sees it: bin/activate, and pytorch3dunet in site-packages if asked."""
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "activate").write_text(f'export PATH="{path / "bin"}:$PATH"\n')
    if with_p3d:
        (path / "lib" / "python3.11" / "site-packages" / "pytorch3dunet").mkdir(parents=True)
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    corpus = tmp_path / "corpora" / "full"
    for part in ("train", "val"):
        (corpus / part).mkdir(parents=True)
        (corpus / part / "a.h5").write_bytes(b"")
    (corpus / "corpus.yaml").write_text(yaml.safe_dump(
        {"name": "full", "crop": [[0, 64]] * 3, "parts": {"train": {"samples": 2}, "val": {"samples": 1}}}))
    (tmp_path / "capa_unet.sif").write_bytes(b"")
    (tmp_path / "cbr.yaml").write_text("name: cbr\ncorpus: full\n")
    return tmp_path


# --- site -----------------------------------------------------------------------------

def test_without_site_file_the_defaults_are_todays_narval(root):
    site = load_site()
    assert (site.scheduler, site.arch, site.training_runtime) == ("slurm", host_arch(), "apptainer")
    assert site.account_gpu is None and site.modules_for("apptainer") == ["apptainer"]
    assert site.image("training") == root / "capa_unet.sif"


def test_site_file_sets_scheduler_accounts_modules_runtime(root, isolated_site):
    write_site(isolated_site, scheduler="local", slurm={"account_gpu": "acc_gpu", "account_cpu": "acc_cpu"},
               modules={"venv": ["python/3.11.5"]}, training={"runtime": "venv", "venv": str(root / "venv")})
    site = load_site()
    assert site.scheduler == "local" and site.account(gpu=True) == "acc_gpu" and site.account(gpu=False) == "acc_cpu"
    assert site.modules_for("venv") == ["python/3.11.5"] and site.modules_for("apptainer") == ["apptainer"]
    assert site.training_runtime == "venv" and site.venv == root / "venv"


def test_arch_can_be_forced_and_a_missing_image_is_explained(root, isolated_site):
    write_site(isolated_site, arch="aarch64")
    with pytest.raises(HeronConfigError, match="no training image for aarch64"):
        load_site().image("training")
    assert load_site(arch="amd64").image("training") == root / "capa_unet.sif"


@pytest.mark.parametrize("data, message", [
    ({"schedular": "slurm"}, "unknown key"),
    ({"scheduler": "pbs"}, "scheduler must be"),
    ({"arch": "riscv"}, "arch must be"),
    ({"training": {"runtime": "conda"}}, "runtime must be"),
    ({"images": {"amd64": {"inference": "x.sif"}}}, "images"),
])
def test_invalid_site_files_are_refused(isolated_site, data, message):
    write_site(isolated_site, **data)
    with pytest.raises(HeronConfigError, match=message):
        load_site()


def test_data_root_comes_from_the_site_unless_the_environment_says_otherwise(tmp_path, isolated_site, monkeypatch):
    write_site(isolated_site, data_root=str(tmp_path / "data"))
    monkeypatch.delenv("HERON_DATA_ROOT", raising=False)
    assert paths.data_root() == tmp_path / "data"
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path / "env"))
    assert paths.data_root() == tmp_path / "env"


# --- rendering ---------------------------------------------------------------------------

def test_apptainer_job_loads_its_modules_and_mounts_paths(root):
    script = render(Job(name="j", log_dir=root, log_stem="train", command="train3dunet --config c.yaml",
                        runtime="apptainer", resources=RES, image=root / "capa_unet.sif",
                        binds=(root,), env={"PYTHONPATH": "/repo"}), load_site())
    assert "module load apptainer" in script and "#SBATCH --gpus-per-node=1" in script
    assert f"apptainer exec -C --nv --bind {root}" in script and "env PYTHONPATH=/repo train3dunet" in script


def test_venv_job_activates_the_venv_without_apptainer(root, isolated_site):
    write_site(isolated_site, modules={"venv": ["python/3.11.5"]}, training={"runtime": "venv", "venv": "/v"})
    script = render(Job(name="j", log_dir=root, log_stem="train", command="train3dunet", runtime="venv",
                        resources=RES), load_site())
    assert "module load python/3.11.5" in script and "source /v/bin/activate" in script
    assert "apptainer" not in script


def test_a_site_without_modules_loads_none(root, isolated_site):
    write_site(isolated_site, modules={"apptainer": [], "venv": []})
    script = render(Job(name="j", log_dir=root, log_stem="x", command="true", runtime="apptainer",
                        resources=RES, image=root / "capa_unet.sif"), load_site())
    assert "module load" not in script


def test_cpu_job_asks_no_gpu(root):
    script = render(Job(name="j", log_dir=root, log_stem="x", command="true", runtime="host",
                        resources=dict(RES, gpus=0)), load_site())
    assert "--gpus" not in script


# --- executors ----------------------------------------------------------------------------

def test_slurm_command_uses_the_right_account_and_dependency(isolated_site):
    write_site(isolated_site, slurm={"account_gpu": "g", "account_cpu": "c"})
    runner = SlurmExecutor(load_site())
    assert runner.command(Path("j.sh"), gpu=True, after="7") == ["sbatch", "--account=g", "--dependency=afterok:7", "j.sh"]
    assert runner.command(Path("j.sh"), gpu=False) == ["sbatch", "--account=c", "j.sh"]
    assert SlurmExecutor(load_site(), account="x").command(Path("j.sh"), gpu=True)[1] == "--account=x"


def test_local_executor_runs_logs_and_reports_failures(tmp_path, capsys):
    ok, ko = tmp_path / "ok.sh", tmp_path / "ko.sh"
    ok.write_text("#SBATCH --time=1:00:00\necho hello\n")
    ko.write_text("echo before\nexit 3\n")
    runner = LocalExecutor(load_site())
    job_id = runner.submit(ok, gpu=True, log_dir=tmp_path / "logs", log_stem="train")
    assert "hello" in capsys.readouterr().out
    assert (tmp_path / "logs" / f"train_{job_id}.out").read_text() == "hello\n"
    with pytest.raises(JobFailed, match="exited with 3"):
        runner.submit(ko, gpu=True, log_dir=tmp_path / "logs", log_stem="eval")


# --- training and evaluation through the seam ---------------------------------------------------

def test_venv_runtime_needs_pytorch3dunet_in_the_venv(root, isolated_site):
    write_site(isolated_site, training={"runtime": "venv", "venv": str(fake_venv(root / "empty", with_p3d=False))})
    with pytest.raises(HeronConfigError, match="has no pytorch3dunet"):
        training.plan(training.load_training_recipe(root / "cbr.yaml"), load_site(), now=NOW)
    write_site(isolated_site, training={"runtime": "venv", "venv": str(fake_venv(root / "venv"))})
    p = training.plan(training.load_training_recipe(root / "cbr.yaml"), load_site(), now=NOW)
    assert f"source {root / 'venv'}/bin/activate" in p.job and "apptainer" not in p.job
    assert p.manifest["runtime"] == "venv" and p.manifest["venv"] == str(root / "venv")


def test_evaluation_runs_in_the_runtime_the_experiment_trained_with(root, isolated_site):
    e = training.create(training.plan(training.load_training_recipe(root / "cbr.yaml"), load_site(), now=NOW))
    write_site(isolated_site, training={"runtime": "venv", "venv": str(fake_venv(root / "venv"))})
    script = jobs.write_eval_job(e, load_site()).read_text()
    assert "apptainer exec" in script and str(root / "capa_unet.sif") in script


def test_local_training_then_evaluation_in_order(root, isolated_site, capsys):
    venv = fake_venv(root / "venv")
    for tool, text in (("train3dunet", "echo TRAINED"), ("python", "echo MEASURED")):
        (venv / "bin" / tool).write_text(f"#!/bin/bash\n{text}\n")
        (venv / "bin" / tool).chmod(0o755)
    write_site(isolated_site, training={"runtime": "venv", "venv": str(venv)}, modules={"venv": []})
    assert main(["train", str(root / "cbr.yaml"), "--local"]) == 0
    out = capsys.readouterr().out
    assert out.index("TRAINED") < out.index("MEASURED")
    e = next((root / "experiments").iterdir())
    assert [(j["kind"], j["runner"]) for j in training.read_manifest(e)["jobs"]] == [("train", "local"), ("evaluate", "local")]


def test_a_failed_local_training_is_not_evaluated(root, isolated_site, capsys):
    venv = fake_venv(root / "venv")
    (venv / "bin" / "train3dunet").write_text("#!/bin/bash\nexit 1\n")
    (venv / "bin" / "train3dunet").chmod(0o755)
    write_site(isolated_site, training={"runtime": "venv", "venv": str(venv)}, modules={"venv": []})
    assert main(["train", str(root / "cbr.yaml"), "--local"]) == 1
    assert "job failed" in capsys.readouterr().err
    e = next((root / "experiments").iterdir())
    assert "jobs" not in training.read_manifest(e) or not training.read_manifest(e)["jobs"]


def test_prepare_submit_is_a_cpu_job_with_the_cpu_account(root, isolated_site, capsys):
    write_site(isolated_site, slurm={"account_gpu": "g", "account_cpu": "c"})
    (root / "c.yaml").write_text("name: c\ndatasets: [datasets/x]\ncrop: {size: 16}\n")
    assert main(["prepare", str(root / "c.yaml"), "--submit", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "--account=c" in out and "--gpus" not in out and "-m heron prepare" in out


# --- doctor ----------------------------------------------------------------------------------

def test_doctor_builds_a_runnable_pipeline(root, isolated_site):
    write_site(isolated_site, training={"runtime": "venv", "venv": str(fake_venv(root / "venv"))})
    work = root / "doctor_run"
    work.mkdir()
    scripts = doctor.build(work, load_site())
    corpus = yaml.safe_load((work / "corpora" / "corpus" / "corpus.yaml").read_text())
    assert corpus["crop"] == [[0, 64]] * 3 and corpus["parts"]["val"]["samples"] > 0
    assert set(corpus["parts"]["train"]["sources"]).isdisjoint(corpus["parts"]["val"]["sources"])
    for step in ("probe", "train", "evaluate"):
        assert scripts[step].is_file()
    assert "max_num_iterations: 10" in (scripts["experiment"] / "config.yaml").read_text()
    assert not (root / "experiments").exists()   # everything stays under the doctor's own root


def test_doctor_report_says_ok_and_cleans_up(root, capsys):
    work = root / "doctor" / "run"
    exp = work / "experiments" / "recipe_x"
    (exp / "checkpoint").mkdir(parents=True)
    (exp / "logs").mkdir()
    (exp / "checkpoint" / "best_checkpoint.pytorch").write_bytes(b"")
    (work / "train.log").write_text("Validation finished. Loss: 0.5. Evaluation score: 0.42\n")
    for step in ("probe", "train", "evaluate"):
        (work / f"{step}.status").write_text("ok\n")
    probe = {"python": "3.11", "torch": "2.12", "cuda": True, "gpu": "A100", "pytorch3dunet": doctor.pinned_pytorch3dunet()}
    (work / "probe.log").write_text("HERON_PROBE " + json.dumps(probe) + "\n")
    results = jobs.results_path(exp)
    results.parent.mkdir(exist_ok=True)
    results.write_text(yaml.safe_dump({"parts": {"val": {}}}))
    assert doctor.report(work) is True
    out = capsys.readouterr().out
    assert "OK  probe" in out and "GPU A100" in out and "best score 0.42" in out and "results written (val)" in out
    assert not work.exists() and not results.exists()


def test_doctor_report_shows_why_a_step_failed(root, capsys):
    work = root / "doctor" / "run"
    work.mkdir(parents=True)
    (work / "probe.status").write_text("ok\n")
    (work / "probe.log").write_text("HERON_PROBE " + json.dumps({"cuda": False, "pytorch3dunet": "1.9.1"}) + "\n")
    (work / "train.status").write_text("ko\n")
    (work / "train.log").write_text("Traceback\nRuntimeError: CUDA out of memory\n")
    assert doctor.report(work, keep=True) is False
    out = capsys.readouterr().out
    assert "KO  probe" in out and "GPU not visible" in out
    assert "requirements-train.txt pins 1.9.6" in out and "CUDA out of memory" in out
    assert work.exists()
