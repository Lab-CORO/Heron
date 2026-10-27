"""Host side of a benchmark: the job that times a package, in the site's training runtime."""

import shlex
from pathlib import Path

from heron import paths
from heron.execution import Job, render
from heron.inference.package import load
from heron.inference.timing import SAMPLES, WARMUP
from heron.site import Site
from heron.training import binds

BENCHMARK_RESOURCES = {"time": "2:00:00", "mem": "32G", "cpus": 4, "gpus": 1}
BENCHMARK_SCRIPT = "job_benchmark.sh"


def results_dir(experiment_name: str) -> Path:
    """Benchmarks kept in the repository, one file per GPU, to commit."""
    return paths.REPO_ROOT / "results" / experiment_name


def benchmark_job(package: Path, precisions: list[str], site: Site, samples: int = SAMPLES, warmup: int = WARMUP,
                  rebuild: bool = False, resources: dict | None = None) -> Job:
    manifest = load(package)
    command = (f"python -m heron.inference.benchmark --package {shlex.quote(str(package))}"
               + (f" --precision {' '.join(precisions)}" if precisions else "")
               + f" --samples {samples} --warmup {warmup}" + (" --rebuild" if rebuild else "")
               + f" --results-dir {shlex.quote(str(results_dir(manifest['experiment'])))}")
    runtime = site.training_runtime
    return Job(name=f"benchmark_{manifest['experiment']}", log_dir=package / "logs", log_stem="benchmark",
               command=command, runtime=runtime, resources=dict(BENCHMARK_RESOURCES, **(resources or {})),
               image=site.image("training") if runtime == "apptainer" else None,
               binds=tuple(binds(package)), env={"PYTHONPATH": paths.REPO_ROOT, "PYTHONUNBUFFERED": 1})


def write_benchmark_job(package: Path, precisions: list[str], site: Site, **options) -> Path:
    script = package / BENCHMARK_SCRIPT
    script.write_text(render(benchmark_job(package, precisions, site, **options), site))
    script.chmod(0o755)
    return script
