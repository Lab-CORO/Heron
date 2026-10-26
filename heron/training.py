"""Training: an Experiment is one run of the network on a Corpus, described by a recipe.

`create(recipe)` checks the recipe against its Corpus, writes the Experiment
directory (<data root>/experiments/<name>_<timestamp>/ with recipe.yaml,
config.yaml, job.sh, experiment.yaml) and returns it; `submit` hands job.sh to the
site's executor (SLURM or local). The training itself is pytorch-3dunet's train3dunet,
in the site's training runtime: a venv, or the training image.
"""

import copy
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from heron import paths
from heron.config import read_config
from heron.corpus import MANIFEST as CORPUS_MANIFEST, corpus_dir
from heron.errors import HeronConfigError
from heron.execution import Job, render, venv_has
from heron.site import Site

# Losses built into pytorch-3dunet 1.9.6 (requirements-train.txt); the same list as 1.9.1.
BUILTIN_LOSSES = {
    "BCEWithLogitsLoss", "BCEDiceLoss", "CrossEntropyLoss", "WeightedCrossEntropyLoss",
    "GeneralizedDiceLoss", "DiceLoss", "MSELoss", "SmoothL1Loss", "L1Loss", "WeightedSmoothL1Loss",
}

# The settings shared by the recent experiments (cbr/cbe, f_maps 16, batch 4, lazy loading).
DEFAULTS = {
    "manual_seed": 42,
    "model": {"name": "UNet3D", "in_channels": 1, "out_channels": 1, "final_sigmoid": True,
              "f_maps": [16, 32, 64, 128, 256], "layer_order": "cbr"},
    "loss": {"name": "BCEDiceLoss"},
    "eval_metric": {"name": "DiceCoefficient"},
    "optimizer": {"learning_rate": 5.0e-5, "weight_decay": 1.0e-5},
    "lr_scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5, "patience": 50,
                     "cooldown": 10, "min_lr": 1.0e-8},
    "trainer": {"eval_score_higher_is_better": True, "validate_after_iters": 50, "log_after_iters": 50,
                "max_num_epochs": 500, "max_num_iterations": 2000000},
    # global_normalization written out: its default changed from True (1.9.1) to False
    # (1.9.6); Normalize ignores it, but evaluation reads it too and must not guess.
    "loaders": {"dataset": "LazyHDF5Dataset", "batch_size": 4, "num_workers": 4,
                "raw_internal_path": "raw", "label_internal_path": "label", "global_normalization": False},
}
# pytorch-3dunet's SliceBuilder asserts height and width >= 64 (1.9.1 and 1.9.6).
MIN_PATCH = 64
DEFAULT_RESOURCES = {"time": "30:00:00", "mem": "64G", "cpus": 4, "gpus": 1}
RECIPE_SECTIONS = {"model", "loss", "optimizer", "lr_scheduler", "trainer", "loaders"}
TRANSFORMS = {
    "raw": [{"name": "Normalize"}, {"name": "ToTensor", "expand_dims": True}],
    "label": [{"name": "Identity"}, {"name": "ToTensor", "expand_dims": True}],
}
EXPERIMENT_MANIFEST = "experiment.yaml"


@dataclass(frozen=True)
class TrainingRecipe:
    name: str
    source_file: Path
    description: str
    corpus: str
    sections: dict          # model / loss / optimizer / lr_scheduler / trainer / loaders overrides
    patch: int | None
    pretrained: str | None
    pretrained_checkpoint: str
    resources: dict
    overrides: dict         # raw pytorch-3dunet keys, merged last


def experiments_root() -> Path:
    return paths.data_root() / "experiments"


def load_training_recipe(path: str | Path) -> TrainingRecipe:
    path = Path(path).resolve()
    data = read_config(path, required={"name", "corpus"},
                       optional=RECIPE_SECTIONS | {"description", "patch", "init", "resources", "pytorch3dunet"})
    for key in RECIPE_SECTIONS | {"init", "resources", "pytorch3dunet"}:
        if key in data and not isinstance(data[key], dict):
            raise HeronConfigError(f"{path}: {key} must be a mapping")
    init = data.get("init") or {}
    if set(init) - {"pretrained", "checkpoint"}:
        raise HeronConfigError(f"{path}: init accepts pretrained and checkpoint (resume is `heron train --resume`)")
    checkpoint = init.get("checkpoint", "last")
    if checkpoint not in ("last", "best"):
        raise HeronConfigError(f"{path}: init.checkpoint must be last or best")
    resources = dict(DEFAULT_RESOURCES, **(data.get("resources") or {}))
    if set(resources) - set(DEFAULT_RESOURCES):
        raise HeronConfigError(f"{path}: resources accepts {', '.join(DEFAULT_RESOURCES)}")

    sections = {k: data[k] for k in RECIPE_SECTIONS if k in data}
    loss = sections.get("loss", {}).get("name", DEFAULTS["loss"]["name"])
    if loss not in BUILTIN_LOSSES:
        raise HeronConfigError(
            f"{path}: loss {loss!r} is not built into pytorch-3dunet 1.9.1 "
            f"(available: {', '.join(sorted(BUILTIN_LOSSES))})")
    return TrainingRecipe(
        name=data["name"], source_file=path, description=str(data.get("description") or ""),
        corpus=str(data["corpus"]), sections=sections,
        patch=int(data["patch"]) if data.get("patch") is not None else None,
        pretrained=init.get("pretrained"), pretrained_checkpoint=checkpoint,
        resources=resources, overrides=data.get("pytorch3dunet") or {},
    )


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _corpus(name: str) -> tuple[Path, dict]:
    """A prepared Corpus with train and val parts, and its manifest."""
    directory = corpus_dir(name)
    manifest = directory / CORPUS_MANIFEST
    if not manifest.is_file():
        raise HeronConfigError(f"corpus {name!r} not found at {directory} (build it with heron prepare)")
    for part in ("train", "val"):
        if not any((directory / part).glob("*.h5")):
            raise HeronConfigError(f"corpus {name!r} has no {part} samples; pytorch-3dunet needs both train and val")
    return directory, yaml.safe_load(manifest.read_text())


def find_experiment(name_or_path: str) -> Path:
    """An Experiment directory: a name under <data root>/experiments, or a path (older experiments)."""
    candidate = Path(name_or_path)
    if not candidate.is_absolute():
        candidate = experiments_root() / name_or_path
    if not candidate.is_dir():
        raise HeronConfigError(f"experiment {name_or_path!r} not found ({candidate})")
    return candidate.resolve()


def _checkpoint(experiment: Path, which: str) -> Path:
    ckpt = experiment / "checkpoint" / f"{which}_checkpoint.pytorch"
    if not ckpt.is_file():
        raise HeronConfigError(f"{ckpt} not found")
    return ckpt


def build_config(recipe: TrainingRecipe, corpus: Path, corpus_manifest: dict, experiment: Path,
                 pretrained: Path | None) -> dict:
    """The pytorch-3dunet config for this Experiment, with host paths."""
    crop = corpus_manifest["crop"]
    crop_size = min(hi - lo for lo, hi in crop)
    patch = recipe.patch or crop_size
    if patch > crop_size:
        raise HeronConfigError(f"patch {patch} is larger than the corpus crop ({crop_size}³)")
    if patch < MIN_PATCH:
        raise HeronConfigError(
            f"patch {patch} is smaller than {MIN_PATCH}: pytorch-3dunet's SliceBuilder refuses patches under "
            f"{MIN_PATCH} in height and width (use a corpus crop of at least {MIN_PATCH})")

    config = DEFAULTS
    for section, values in recipe.sections.items():
        config = _merge(config, {section: values})
    _check_parts(corpus_manifest)
    slice_builder = {"name": "SliceBuilder", "patch_shape": [patch] * 3, "stride_shape": [patch] * 3}
    config = _merge(config, {
        "trainer": {"checkpoint_dir": str(experiment / "checkpoint"), "resume": None,
                    "pre_trained": str(pretrained) if pretrained else None},
        "loaders": {
            "train": {"file_paths": [str(corpus / "train")], "slice_builder": slice_builder,
                      "transformer": TRANSFORMS},
            "val": {"file_paths": [str(corpus / "val")], "slice_builder": slice_builder,
                    "transformer": TRANSFORMS},
        },
    })
    return _merge(config, recipe.overrides)


def _check_parts(corpus_manifest: dict) -> None:
    """Both parts need samples. (pytorch-3dunet 1.9.1 also dropped incomplete batches,
    so a part smaller than batch_size validated nothing; 1.9.6 keeps them.)"""
    for part in ("train", "val"):
        counts = corpus_manifest.get("parts", {}).get(part, {})
        if counts.get("samples", 0) + counts.get("augmented", 0) < 1:
            raise HeronConfigError(f"the corpus {part} part has no samples")


def training_job(name: str, experiment: Path, resources: dict, site: Site, pretrained: Path | None) -> Job:
    """train3dunet on the Experiment's config, in the site's training runtime."""
    runtime = site.training_runtime
    image = None
    if runtime == "apptainer":
        image = site.image("training")
    elif not venv_has(site.venv, "pytorch3dunet"):
        raise HeronConfigError(
            f"training runtime is venv but {site.venv} has no pytorch3dunet: "
            f"pip install -e '.[train]' in that venv (see requirements-train.txt)")
    return Job(name=name, log_dir=experiment / "logs", log_stem="train",
               command=f"train3dunet --config {experiment / 'config.yaml'}",
               runtime=runtime, resources=resources, image=image, binds=tuple(binds(pretrained)))


def binds(*paths_needed: Path | None) -> list[Path]:
    """Directories to mount: the repository and data root, plus anything outside them."""
    roots = [paths.REPO_ROOT, paths.data_root()]
    for p in paths_needed:
        if p is not None and not any(p.is_relative_to(r) for r in roots):
            roots.append(p.parent)
    return roots


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class Plan:
    """Everything `create` would write, so --dry-run can show it."""
    experiment: Path
    config: dict
    job: str
    manifest: dict
    recipe_text: str


def plan(recipe: TrainingRecipe, site: Site, now: datetime | None = None) -> Plan:
    corpus, corpus_manifest = _corpus(recipe.corpus)
    pretrained = None
    if recipe.pretrained:
        pretrained = _checkpoint(find_experiment(recipe.pretrained), recipe.pretrained_checkpoint)
    now = now or datetime.now()
    name = f"{recipe.name}_{now:%Y%m%d_%H%M}"
    experiment = experiments_root() / name
    if experiment.exists():
        raise HeronConfigError(f"{experiment} already exists (wait a minute or rename the recipe)")
    config = build_config(recipe, corpus, corpus_manifest, experiment, pretrained)
    manifest = {
        "name": name,
        "description": recipe.description,
        "recipe": str(recipe.source_file),
        "corpus": recipe.corpus,
        "corpus_inputs_sha256": corpus_manifest.get("inputs_sha256"),
        "corpus_created": corpus_manifest.get("created"),
        "pretrained": str(pretrained) if pretrained else None,
        "runtime": site.training_runtime,
        "image": str(site.image("training")) if site.training_runtime == "apptainer" else None,
        "venv": str(site.venv) if site.training_runtime == "venv" else None,
        "arch": site.arch,
        "git_commit": _git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jobs": [],
    }
    job = render(training_job(name, experiment, recipe.resources, site, pretrained), site)
    return Plan(experiment, config, job, manifest, recipe.source_file.read_text())


def create(p: Plan) -> Path:
    if p.experiment.exists():
        raise HeronConfigError(f"{p.experiment} already exists")
    for sub in ("checkpoint", "logs"):
        (p.experiment / sub).mkdir(parents=True)
    (p.experiment / "recipe.yaml").write_text(p.recipe_text)
    (p.experiment / "config.yaml").write_text(yaml.safe_dump(p.config, sort_keys=False))
    (p.experiment / "job.sh").write_text(p.job)
    (p.experiment / "job.sh").chmod(0o755)
    _write_manifest(p.experiment, p.manifest)
    return p.experiment


# Each job script of an Experiment, with the stem of its log files.
JOB_LOGS = {"job.sh": "train", "job_eval.sh": "eval"}


def submit(experiment: Path, runner, kind: str = "train", script: str = "job.sh",
           after: str | None = None) -> str:
    """Hand one of the Experiment's job scripts to an executor; record the job in experiment.yaml.

    after: a job this one waits for, and only runs if it succeeded.
    """
    job_id = runner.submit(experiment / script, gpu=True, log_dir=experiment / "logs",
                           log_stem=JOB_LOGS.get(script, "job"), after=after)
    manifest = read_manifest(experiment)
    manifest.setdefault("jobs", []).append(
        {"id": job_id, "kind": kind, "runner": runner.name,
         "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    _write_manifest(experiment, manifest)
    return job_id


def prepare_resume(experiment: Path) -> Path:
    """Point the Experiment's config at its own last checkpoint, so resubmitting continues it."""
    last = _checkpoint(experiment, "last")
    config_path = experiment / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["trainer"]["resume"] = str(last)
    config["trainer"]["pre_trained"] = None
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return last


def read_manifest(experiment: Path) -> dict:
    path = experiment / EXPERIMENT_MANIFEST
    return yaml.safe_load(path.read_text()) if path.is_file() else {}


def _write_manifest(experiment: Path, manifest: dict) -> None:
    (experiment / EXPERIMENT_MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False))


_SCORE = re.compile(r"Evaluation score: ([-+0-9.eE]+)")


def summary(experiment: Path) -> dict:
    """What `heron experiments` shows: corpus, jobs, checkpoints and best validation score from the logs."""
    manifest = read_manifest(experiment)
    scores = []
    for log in sorted((experiment / "logs").glob("*.out")):
        with open(log, errors="replace") as f:
            scores += [float(m.group(1)) for m in _SCORE.finditer(f.read())]
    checkpoints = {}
    for which in ("best", "last"):
        ckpt = experiment / "checkpoint" / f"{which}_checkpoint.pytorch"
        if ckpt.is_file():
            checkpoints[which] = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {
        "name": experiment.name,
        "corpus": manifest.get("corpus", "?"),
        "description": manifest.get("description", ""),
        "jobs": [j["id"] for j in manifest.get("jobs", [])],
        "checkpoints": checkpoints,
        "validations": len(scores),
        "best_score": max(scores) if scores else None,
    }


def list_experiments() -> list[Path]:
    root = experiments_root()
    return sorted(p for p in root.iterdir() if (p / EXPERIMENT_MANIFEST).is_file()) if root.is_dir() else []
