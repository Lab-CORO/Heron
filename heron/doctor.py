"""heron doctor: run a tiny pipeline end to end on this site, then report and clean up.

Here (any node): a fabricated Dataset of 64³ scenes, its Corpus, an Experiment with
10 training iterations. Then ONE job, through the site's executor (SLURM or local),
runs on a GPU: a probe of the training runtime (versions, CUDA, a forward pass), the
training, the evaluation, and finally `heron doctor --report`, which prints OK/KO per
step and deletes everything the run created.

Dataset generation is not part of the run yet: its image contract waits for the
ROS 2 Jazzy + cuRobo v2 image.
"""

import json
import os
import shlex
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from heron import paths
from heron.errors import HeronConfigError
from heron.execution import Job, render, venv_has
from heron.site import Site

VOXEL = 0.02
HALF = 0.64            # 64 voxels a side: pytorch-3dunet refuses patches under 64
FILES, GROUPS = 4, 2
REQUIREMENTS = paths.REPO_ROOT / "requirements-train.txt"
PROBE = (
    "import json, h5py, numpy, scipy, yaml, torch, importlib.metadata as md; "
    "ok = torch.cuda.is_available(); "
    "y = torch.nn.Conv3d(1, 2, 3).to('cuda' if ok else 'cpu')(torch.zeros(1, 1, 8, 8, 8, device='cuda' if ok else 'cpu')); "
    "print('HERON_PROBE ' + json.dumps({'python': __import__('platform').python_version(), "
    "'torch': torch.__version__, 'cuda': ok, 'gpu': torch.cuda.get_device_name(0) if ok else None, "
    "'pytorch3dunet': md.version('pytorch3dunet'), 'forward': list(y.shape)}))"
)
STEPS = ("probe", "train", "evaluate")


@contextmanager
def _data_root(path: Path):
    """Point heron at the doctor's own data root, so nothing it makes touches the real one."""
    before = os.environ.get("HERON_DATA_ROOT")
    os.environ["HERON_DATA_ROOT"] = str(path)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("HERON_DATA_ROOT", None)
        else:
            os.environ["HERON_DATA_ROOT"] = before


def pinned_pytorch3dunet() -> str | None:
    for line in REQUIREMENTS.read_text().splitlines() if REQUIREMENTS.is_file() else []:
        if line.strip().startswith("pytorch3dunet=="):
            return line.split("==", 1)[1].strip()
    return None


def fake_dataset(directory: Path, station: Path, seed: int = 0) -> Path:
    """FILES synthetic generator files of GROUPS scenes: a few boxes, reachability fading with distance."""
    import h5py
    from heron.dataset import describe
    directory.mkdir(parents=True)
    n = int(round(2 * HALF / VOXEL))
    idx = (np.indices((n, n, n)).transpose(1, 2, 3, 0) - n // 2) * VOXEL
    distance = np.linalg.norm(idx, axis=-1)
    rng = np.random.RandomState(seed)
    for f in range(FILES):
        with h5py.File(directory / f"fake_{f}.h5", "w") as h:
            for g in range(GROUPS):
                scene = np.zeros((n, n, n))
                for _ in range(3):
                    lo = rng.randint(0, n - 12, size=3)
                    scene[lo[0]:lo[0] + 12, lo[1]:lo[1] + 12, lo[2]:lo[2] + 12] = 1.0
                label = np.round(np.clip(1 - distance / 0.6, 0, 1) * 50) / 50 * (scene == 0)
                for key, data in (("voxel_grid", scene), ("reachability_map", label)):
                    ds = h.create_dataset(f"group/{g}/{key}", data=data)
                    for a in "xyz":
                        ds.attrs[f"origine_{a}"] = -HALF
                        ds.attrs[f"voxel_grid_size_{a}"] = float(n)
                    ds.attrs["voxel_size"] = VOXEL
    describe(directory, str(paths.ROBOTS_DIR / "m1013.yaml"), str(station), "synthetic", VOXEL)
    return directory


def build(workdir: Path, site: Site) -> dict:
    """Everything the GPU job needs, under workdir. Returns the paths of the scripts it runs."""
    from heron import training
    from heron.corpus import load_recipe, prepare
    from heron.evaluation.jobs import write_eval_job

    with _data_root(workdir):
        station = workdir / "doctor.yaml"
        station.write_text(f"name: doctor\nscene_half_extent: {HALF}\ncrop_offset: [0.0, 0.0, 0.0]\n")
        fake_dataset(workdir / "datasets" / "fake", station)
        (workdir / "corpus.yaml").write_text(yaml.safe_dump({
            "name": "corpus", "datasets": [str(workdir / "datasets" / "fake")], "crop": {"size": 64},
            "val_ratio": 0.5, "augment": {"copies": 1, "angle": 180}, "seed": 0}))
        prepare(load_recipe(workdir / "corpus.yaml"), log=lambda *_: None)
        (workdir / "recipe.yaml").write_text(yaml.safe_dump({
            "name": "recipe", "corpus": "corpus",
            "model": {"f_maps": [8, 16, 32, 64]}, "loaders": {"batch_size": 2, "num_workers": 0},
            "resources": {"time": "0:20:00", "mem": "16G", "cpus": 2, "gpus": 1},
            "pytorch3dunet": {"trainer": {"max_num_iterations": 10, "validate_after_iters": 5,
                                          "log_after_iters": 5}}}))
        experiment = training.create(training.plan(training.load_training_recipe(workdir / "recipe.yaml"), site))
        write_eval_job(experiment, site, resources={"time": "0:20:00", "mem": "16G", "cpus": 2})

    runtime = site.training_runtime
    probe = workdir / "probe.sh"
    probe.write_text(render(Job(
        name="heron_doctor_probe", log_dir=workdir, log_stem="probe", command=f"python -c {shlex.quote(PROBE)}",
        runtime=runtime, resources={"time": "0:05:00", "mem": "4G", "cpus": 1, "gpus": 1},
        image=site.image("training") if runtime == "apptainer" else None,
        binds=(paths.REPO_ROOT, workdir)), site))
    return {"probe": probe, "train": experiment / "job.sh", "evaluate": experiment / "job_eval.sh",
            "experiment": experiment}


def driver_job(workdir: Path, scripts: dict, keep: bool) -> Job:
    """The one GPU job: each step's output and status go to workdir, then the report runs."""
    w = shlex.quote(str(workdir))
    command = "\n".join([
        "set +e",
        f'W={w}',
        'step() { local n=$1; shift; if "$@" > "$W/$n.log" 2>&1; then echo ok > "$W/$n.status"; '
        'else echo ko > "$W/$n.status"; fi; }',
        f"step probe bash {shlex.quote(str(scripts['probe']))}",
        f"step train bash {shlex.quote(str(scripts['train']))}",
        f'if [ "$(cat "$W/train.status")" = ok ]; then step evaluate bash {shlex.quote(str(scripts["evaluate"]))}; fi',
        f"cd {shlex.quote(str(paths.REPO_ROOT))} && {shlex.quote(sys.executable)} -m heron doctor --report {w}"
        + (" --keep" if keep else ""),
    ])
    return Job(name="heron_doctor", log_dir=paths.data_root() / "log", log_stem="heron_doctor",
               command=command, runtime="host", resources={"time": "1:00:00", "mem": "16G", "cpus": 2, "gpus": 1},
               env={"PYTHONPATH": paths.REPO_ROOT})


def checks(site: Site) -> list[tuple[str, bool, str]]:
    """What can be checked here, before any job: the site and the training runtime."""
    out = [("site", True, f"{site.source or 'defaults (no site.yaml)'} · scheduler {site.scheduler} · arch {site.arch}")]
    try:
        if site.training_runtime == "apptainer":
            out.append(("training runtime", True, f"apptainer image {site.image('training')}"))
        else:
            has = venv_has(site.venv, "pytorch3dunet")
            out.append(("training runtime", has, f"venv {site.venv}" + ("" if has else
                        " has no pytorch3dunet: pip install -e '.[train]' in it")))
    except HeronConfigError as exc:
        out.append(("training runtime", False, str(exc)))
    try:
        out.append(("generation image", True, f"{site.image('generation')} (not exercised: contract pending "
                                              f"the ROS 2 Jazzy image)"))
    except HeronConfigError as exc:
        out.append(("generation image", True, f"not set ({exc.args[0].split('.')[0]}); not exercised yet"))
    return out


def start(site: Site, runner, keep: bool = False) -> str:
    """Build the run and hand its GPU job to the executor. Returns the job id."""
    found = checks(site)
    for name, ok, detail in found:
        print(f"{'OK ' if ok else 'KO '} {name:18} {detail}")
    if not all(ok for _, ok, _ in found):
        raise HeronConfigError("fix the KO line(s) above before running the doctor")
    workdir = paths.data_root() / "doctor" / datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir.mkdir(parents=True)
    try:
        scripts = build(workdir, site)
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    job = driver_job(workdir, scripts, keep)
    script = workdir / "doctor.sh"
    script.write_text(render(job, site))
    print(f"{'OK '} {'test data':18} fake dataset, corpus and experiment in {workdir}")
    return runner.submit(script, gpu=True, log_dir=job.log_dir, log_stem=job.log_stem)


def report(workdir: Path, keep: bool = False) -> bool:
    """Print OK/KO per step of a finished run; delete the run unless keep. True if all passed."""
    from heron import training
    from heron.evaluation.jobs import results_path

    status = {s: (workdir / f"{s}.status").read_text().strip() if (workdir / f"{s}.status").is_file()
              else "skipped" for s in STEPS}
    probe = {}
    log = workdir / "probe.log"
    for line in log.read_text(errors="replace").splitlines() if log.is_file() else []:
        if line.startswith("HERON_PROBE "):
            probe = json.loads(line[len("HERON_PROBE "):])
    experiment = next((p for p in (workdir / "experiments").glob("*") if p.is_dir()), None)
    results = results_path(experiment) if experiment else None

    lines, all_ok = [], True

    def line(name, ok, detail):
        nonlocal all_ok
        all_ok &= ok
        lines.append(f"{'OK ' if ok else 'KO '} {name:18} {detail}")

    pinned = pinned_pytorch3dunet()
    line("probe", status["probe"] == "ok" and bool(probe.get("cuda")),
         f"python {probe.get('python')} · torch {probe.get('torch')} · pytorch3dunet {probe.get('pytorch3dunet')}"
         f" · GPU {probe.get('gpu') or 'not visible'}" if probe else f"no probe output ({status['probe']})")
    if probe and pinned and probe.get("pytorch3dunet") != pinned:
        line("version", False, f"pytorch3dunet {probe.get('pytorch3dunet')} but requirements-train.txt pins {pinned}")
    s = training.summary(experiment) if experiment else {}
    # The doctor runs job.sh with bash, its output in train.log rather than the experiment's logs/.
    train_log = workdir / "train.log"
    scores = [float(m.group(1)) for m in training._SCORE.finditer(train_log.read_text(errors="replace"))] \
        if train_log.is_file() else []
    line("train", status["train"] == "ok" and "best" in s.get("checkpoints", {}),
         f"{len(scores)} validation(s), best score {max(scores) if scores else None}" if s else status["train"])
    line("evaluate", status["evaluate"] == "ok" and bool(results and results.is_file()),
         f"results written ({', '.join(yaml.safe_load(results.read_text())['parts'])})"
         if results and results.is_file() else status["evaluate"])

    print("heron doctor")
    print("\n".join(lines))
    for step in STEPS:
        if status[step] == "ko":
            tail = (workdir / f"{step}.log").read_text(errors="replace").splitlines()[-25:]
            print(f"\n--- last lines of {step}.log ---\n" + "\n".join(tail))
    if keep:
        print(f"\nkept {workdir}")
    else:
        if results and results.is_file():
            results.unlink()
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"\nremoved {workdir}")
    return all_ok
