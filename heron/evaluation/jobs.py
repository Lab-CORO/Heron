"""Host side of an evaluation: the measurement job, and the figures afterwards."""

import dataclasses
import shlex
from pathlib import Path

from heron import paths
from heron.corpus import corpus_dir
from heron.errors import HeronConfigError
from heron.execution import Job, render
from heron.site import Site
from heron.training import binds, read_manifest

EVAL_RESOURCES = {"time": "2:00:00", "mem": "32G", "cpus": 4, "gpus": 1}
EVAL_SCRIPT = "job_eval.sh"


def corpus_parts(experiment: Path) -> dict[str, Path]:
    """The parts an Experiment is evaluated on: val, then each named test part of its Corpus."""
    corpus = read_manifest(experiment).get("corpus")
    if not corpus:
        raise HeronConfigError(f"{experiment}: no experiment.yaml naming its corpus (made before heron train?)")
    root = corpus_dir(corpus)
    if not root.is_dir():
        raise HeronConfigError(f"corpus {corpus!r} of {experiment.name} not found at {root}")
    parts = {"val": root / "val"}
    for test in sorted((root / "test").glob("*")) if (root / "test").is_dir() else []:
        parts[f"test:{test.name}"] = test
    return parts


def results_path(experiment: Path) -> Path:
    """Metrics summary kept in the repository, to commit."""
    return paths.REPO_ROOT / "results" / f"{experiment.name}.yaml"


def training_site(experiment: Path, site: Site) -> Site:
    """The site as the Experiment was trained: its runtime, image or venv, from experiment.yaml.

    The model is measured by the same pytorch-3dunet that trained it, even if site.yaml
    changed since. Experiments made before the runtime was recorded use the site's.
    """
    m = read_manifest(experiment)
    if not m.get("runtime"):
        return site
    images = {a: dict(v) for a, v in site.images.items()}
    if m.get("image"):
        images[site.arch] = dict(images.get(site.arch) or {}, training=m["image"])
    return dataclasses.replace(site, training_runtime=m["runtime"], images=images,
                               venv=Path(m["venv"]) if m.get("venv") else site.venv)


def eval_job(experiment: Path, checkpoint: str, parts: dict[str, Path], site: Site,
             resources: dict | None = None, precisions: list[str] = (), samples: int | None = None) -> Job:
    """precisions: also time and measure these TensorRT engines (heron.evaluation.measure)."""
    ckpt = experiment / "checkpoint" / f"{checkpoint}_checkpoint.pytorch"
    part_args = " ".join(f"--part {shlex.quote(f'{name}={d}')}" for name, d in parts.items())
    command = (f"python -m heron.evaluation.measure --experiment {shlex.quote(str(experiment))} "
               f"--checkpoint {shlex.quote(str(ckpt))} {part_args} "
               f"--results {shlex.quote(str(results_path(experiment)))}"
               + (f" --precision {' '.join(precisions)}" if precisions else "")
               + (f" --samples {samples}" if samples else ""))
    runtime = site.training_runtime
    return Job(name=f"eval_{experiment.name}", log_dir=experiment / "logs", log_stem="eval",
               command=command, runtime=runtime, resources=dict(EVAL_RESOURCES, **(resources or {})),
               image=site.image("training") if runtime == "apptainer" else None,
               binds=tuple(binds()), env={"PYTHONPATH": paths.REPO_ROOT})


def eval_job_script(experiment: Path, checkpoint: str, parts: dict[str, Path], site: Site,
                    resources: dict | None = None, precisions: list[str] = (), samples: int | None = None) -> str:
    site = training_site(experiment, site)
    return render(eval_job(experiment, checkpoint, parts, site, resources, precisions, samples), site)


def select_parts(experiment: Path, only: list[str] | None = None) -> dict[str, Path]:
    parts = corpus_parts(experiment)
    if only:
        unknown = set(only) - set(parts)
        if unknown:
            raise HeronConfigError(f"unknown part(s) {', '.join(sorted(unknown))}; available: {', '.join(parts)}")
        parts = {k: v for k, v in parts.items() if k in only}
    return parts


def write_eval_job(experiment: Path, site: Site, checkpoint: str = "best", only: list[str] | None = None,
                   resources: dict | None = None, precisions: list[str] = (), samples: int | None = None) -> Path:
    if checkpoint not in ("best", "last"):
        raise HeronConfigError("checkpoint must be best or last")
    script = experiment / EVAL_SCRIPT
    script.write_text(eval_job_script(experiment, checkpoint, select_parts(experiment, only), site, resources,
                                      precisions, samples))
    script.chmod(0o755)
    return script


def draw_figures(experiment: Path, log=print) -> list[Path]:
    """Figures of every measured part, from its eval_data.h5 (host side, needs matplotlib).
    The PyTorch model's under evaluation/<part>/, each TensorRT engine's under evaluation/tensorrt_*/<part>/."""
    from heron.evaluation import figures
    from heron.evaluation.cache import EVAL_DATA_FILENAME, load_eval_data
    from heron.evaluation.metrics import REGION_NAMES, merge_class_metrics

    drawn = []
    for cache in sorted((experiment / "evaluation").glob(f"**/{EVAL_DATA_FILENAME}")):
        out, d = cache.parent, load_eval_data(cache)
        global_n = int(d["global_cm"].sum())
        region_n = {r: int(d["region_cm"][r].sum()) for r in REGION_NAMES}
        figures.plot_confusion_matrix(merge_class_metrics(d["global_cm"], n_voxels=global_n),
                                      str(out / "confusion_matrix.png"))
        figures.plot_error_histogram(d["hist_counts"], d["hist_edges"], d["global_reg"]["MAE"],
                                     d["global_reg"]["MedAE"], str(out / "error_histogram.png"))
        figures.plot_tolerance_curves(d["region_cm"], region_n, str(out / "tolerance_curve.png"),
                                      global_cm=d["global_cm"], global_n=global_n)
        figures.plot_region_slices(d["region_samples"], str(out))
        log(f"figures: {out}")
        drawn.append(out)
    if not drawn:
        raise HeronConfigError(f"{experiment.name}: nothing measured yet (no evaluation/*/{EVAL_DATA_FILENAME})")
    return drawn
