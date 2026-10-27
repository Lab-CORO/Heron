"""Time a package on this machine's GPU: PyTorch fp32, then each TensorRT precision. Runs in the job.

    python -m heron.inference.benchmark --package <dir> --precision fp32 fp16 int8 \
        [--samples 1000] [--warmup 50] [--rebuild] --results-dir <results/<experiment>>

Writes <results-dir>/benchmark_<gpu>.yaml. The speedups are against PyTorch fp32 on the
same machine, median against median. `agreement` compares each engine's prediction with
PyTorch's on the same volume: a check that the engine computes the network, not a quality
measure (that is heron evaluate --precision, on the Corpus).
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from heron.errors import HeronConfigError
from heron.inference.engine import PRECISIONS, TrtNetwork, build, slug
from heron.inference.package import CHECKPOINT, CONFIG, load
from heron.inference.timing import SAMPLES, WARMUP, POOL, DENSITY, machine, time_predictor, volumes

BASELINE = "pytorch_fp32"


def tensorrt_predictor(package: Path, config: dict, precision: str, gpu: str, rebuild: bool = False, log=print):
    """A Predictor running the package's engine for this GPU. Returns (predictor, engine, build seconds)."""
    from heron.evaluation.measure import Predictor
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        raise HeronConfigError("tensorrt is not importable here: install requirements-trt.txt "
                               "(on a Jetson, a venv with --system-site-packages sees JetPack's)") from None
    transform = Predictor(config, None, network=lambda x: x)
    path, seconds = build(package, precision, gpu, prepare=transform.to_input, rebuild=rebuild, log=log)
    return Predictor(config, None, network=TrtNetwork(path)), path, seconds


def speedup(baseline: dict, run: dict) -> dict:
    return {k: round(baseline[k]["median_ms"] / run[k]["median_ms"], 2) for k in ("inference", "end_to_end")}


def agreement(reference, prediction) -> dict:
    import numpy as np
    diff = np.abs(prediction - reference)
    return {"max_abs_diff": round(float(diff.max()), 5), "mean_abs_diff": round(float(diff.mean()), 6)}


def run(package: Path, precisions: list[str], samples: int = SAMPLES, warmup: int = WARMUP,
        rebuild: bool = False, log=print) -> dict:
    import torch
    from heron.evaluation.measure import Predictor

    manifest = load(package)
    config = yaml.safe_load((package / CONFIG).read_text())
    if not torch.cuda.is_available():
        raise HeronConfigError("no GPU visible: the benchmark times GPU inference")
    shape = tuple(manifest["input_shape"])
    pool = volumes(shape)
    info = machine(torch)

    log(f"{BASELINE}: {samples} volumes after {warmup} warmup")
    baseline = Predictor(config, package / CHECKPOINT)
    runs = {BASELINE: time_predictor(baseline, pool, samples, warmup)}
    reference = baseline.predict(pool[0])
    del baseline
    torch.cuda.empty_cache()
    for precision in precisions:
        predictor, engine, seconds = tensorrt_predictor(package, config, precision, info["gpu"], rebuild, log)
        log(f"tensorrt_{precision}: {samples} volumes after {warmup} warmup")
        result = time_predictor(predictor, pool, samples, warmup)
        result["speedup"] = speedup(runs[BASELINE], result)
        result["agreement"] = agreement(reference, predictor.predict(pool[0]))
        result["engine"] = engine.name
        result["build_seconds"] = round(seconds, 1) if seconds is not None else "cached"
        runs[f"tensorrt_{precision}"] = result
        del predictor
        torch.cuda.empty_cache()
    return {
        "experiment": manifest["experiment"],
        "checkpoint": manifest["checkpoint"],
        "package": str(package),
        "input_shape": manifest["input_shape"],
        "machine": info,
        "samples": samples,
        "warmup": warmup,
        "volumes": f"{POOL} random binary volumes ({DENSITY:.0%} occupied), cycled",
        "tf32": False,
        "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
    }


def results_file(results_dir: Path, gpu: str) -> Path:
    return results_dir / f"benchmark_{slug(gpu)}.yaml"


def table(result: dict) -> str:
    """The runs as aligned text lines, for the log."""
    lines = [f"{'':16} {'inference median / p95 (ms)':>30} {'end to end median / p95 (ms)':>30} {'speedup':>8}"]
    for name, r in result["runs"].items():
        i, e = r["inference"], r["end_to_end"]
        s = r.get("speedup", {}).get("inference")
        lines.append(f"{name:16} {i['median_ms']:>17.2f} / {i['p95_ms']:<10.2f} {e['median_ms']:>17.2f} / "
                     f"{e['p95_ms']:<10.2f} {('x' + format(s, '.2f')) if s else '-':>8}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--precision", nargs="*", default=[], choices=PRECISIONS)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--rebuild", action="store_true", help="build the engines again even if cached")
    p.add_argument("--results-dir", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        result = run(args.package, args.precision, args.samples, args.warmup, args.rebuild)
    except HeronConfigError as exc:
        print(f"heron: {exc}", file=sys.stderr)
        return 2
    out = results_file(args.results_dir, result["machine"]["gpu"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(result, sort_keys=False))
    print(table(result))
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
