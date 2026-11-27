"""Measure an Experiment on Corpus parts. Runs inside the training image.

    python -m heron.evaluation.measure --experiment <dir> --checkpoint <file> \
        --part val=<dir> --part test:reel=<dir> --results <results.yaml> [--precision fp16 int8]

For each part: inference on every volume with the Experiment's own val transform
(the config.yaml it was trained with), then eval_data.h5 and report.txt under
<experiment>/evaluation/<part>/. The results file gathers every part's metrics.

With --precision, the same for each TensorRT engine, under evaluation/tensorrt_<precision>/,
plus the timing of PyTorch and of every engine on this GPU (heron.inference.benchmark).
The engines come from the Experiment's package, exported first if needed.
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

from heron.errors import HeronConfigError
from heron.evaluation.cache import EVAL_DATA_FILENAME, save_eval_data
from heron.evaluation.metrics import (N_CLASSES, REGION_NAMES, compute_error_histogram, confusion_matrix,
                                      region_masks, regression_metrics, to_classes)
from heron.evaluation.report import summarize, write_report


def part_dir_name(part: str) -> str:
    """test:reel -> test_reel (a directory name)."""
    return part.replace(":", "_")


class Predictor:
    """The Experiment's val raw transform, as pytorch-3dunet builds it, then a network:
    its PyTorch model (from the checkpoint), or any callable on the same input, such as a
    TensorRT engine (heron.inference.engine.TrtNetwork)."""

    def __init__(self, config: dict, checkpoint: Path | None, network=None):
        import torch

        self.torch = torch
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network or self._pytorch(checkpoint)

    def _pytorch(self, checkpoint: Path):
        from pytorch3dunet.unet3d.model import get_model

        # Strict fp32, as the TensorRT engines are built (TF32 off), so the baseline is comparable.
        self.torch.backends.cudnn.allow_tf32 = False
        self.torch.backends.cuda.matmul.allow_tf32 = False
        model = get_model(self.config["model"])
        state = self.torch.load(checkpoint, map_location=self.device, weights_only=False)  # our own checkpoints
        model.load_state_dict(state["model_state_dict"])
        model.to(self.device).eval()
        return model

    def raw_transform(self, raw: np.ndarray):
        # Same as pytorch3dunet.datasets.hdf5.AbstractHDF5Dataset: stats over the whole
        # volume only if global_normalization is on (heron writes it out; 1.9.6's default is
        # off), then the val 'raw' transformer chain.
        from pytorch3dunet.augment.transforms import Transformer
        from pytorch3dunet.datasets.utils import calculate_stats

        loaders = self.config["loaders"]
        stats = calculate_stats(raw) if loaders.get("global_normalization", False) else calculate_stats(None, True)
        return Transformer(loaders["val"]["transformer"], stats).raw_transform()(raw)

    def to_input(self, raw: np.ndarray):
        """A volume as the network takes it: (1, 1, D, H, W) float32 on the device."""
        x = self.raw_transform(raw)
        x = x if isinstance(x, self.torch.Tensor) else self.torch.from_numpy(np.asarray(x))
        return x.unsqueeze(0).float().to(self.device).contiguous()

    def infer(self, x):
        with self.torch.no_grad():
            return self.network(x)

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.infer(self.to_input(raw)).squeeze().cpu().numpy().astype(np.float32)


def measure_part(predictor: Predictor, files: list[Path], log=print) -> dict:
    """Accumulate everything the report and figures need over one part."""
    all_preds, all_labels = [], []
    global_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    region_cm = {r: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for r in REGION_NAMES}
    region_samples = []
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            raw, label = f["raw"][:], f["label"][:].astype(np.float32)
        pred = predictor.predict(raw)
        all_preds.append(pred.ravel())
        all_labels.append(label.ravel())
        if i < 3:
            region_samples.append((pred, label, raw.astype(np.float32)))
        pred_cls, label_cls = to_classes(pred), to_classes(label)
        global_cm += confusion_matrix(pred_cls.ravel(), label_cls.ravel())
        for r, m in region_masks(label_cls, label).items():
            if m.any():
                region_cm[r] += confusion_matrix(pred_cls[m], label_cls[m])
        log(f"  [{i + 1:4d}/{len(files)}] {path.name}")
    preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)
    hist_counts, hist_edges = compute_error_histogram(preds, labels)
    return {"global_cm": global_cm, "region_cm": region_cm, "n_files": len(files),
            "global_reg": regression_metrics(preds, labels), "hist_counts": hist_counts,
            "hist_edges": hist_edges, "region_samples": region_samples}


def measure_parts(predictor: Predictor, parts: dict[str, Path], out_root: Path, log=print) -> dict:
    """Measure each part; eval_data.h5 and report.txt under out_root/<part>. Returns the metrics per part."""
    measured = {}
    for part, directory in parts.items():
        files = sorted(directory.glob("*.h5"))
        log(f"{part}: {len(files)} files")
        if not files:
            continue
        data = measure_part(predictor, files, log)
        out = out_root / part_dir_name(part)
        out.mkdir(parents=True, exist_ok=True)
        save_eval_data(out / EVAL_DATA_FILENAME, data["global_cm"], data["region_cm"], data["n_files"],
                       data["global_reg"], data["hist_counts"], data["hist_edges"], data["region_samples"])
        write_report(out, data["global_cm"], data["region_cm"], data["n_files"], data["global_reg"])
        measured[part] = dict(files=len(files), **summarize(data["global_cm"], data["region_cm"],
                                                            data["global_reg"]))
    return measured


def delta(run: dict, baseline: dict) -> dict:
    """run - baseline for every metric both have (counts left out)."""
    out = {}
    for key, value in run.items():
        base = baseline.get(key)
        if isinstance(value, dict) and isinstance(base, dict):
            out[key] = delta(value, base)
        elif (isinstance(value, (int, float)) and isinstance(base, (int, float))
              and key not in ("files", "voxels")):
            out[key] = round(float(value) - float(base), 6)
    return out


def tensorrt_package(experiment: Path, checkpoint: Path, log=print) -> Path:
    """The Experiment's package, exported now if it is not yet, for the checkpoint measured."""
    from heron.inference import package
    which = checkpoint.name.split("_")[0]
    path = package.default_package(experiment)
    if not (path / package.MANIFEST).is_file():
        return package.export(experiment, checkpoint=which, log=log)
    exported = package.load(path)["checkpoint"]
    if exported != which:
        raise HeronConfigError(f"{path} holds the {exported} checkpoint, not {which}: "
                               f"heron export {experiment.name} --checkpoint {which} --force")
    return path


def run(experiment: Path, checkpoint: Path, parts: dict[str, Path], results: Path, precisions: list[str] = (),
        samples: int | None = None, log=print) -> dict:
    """Measure the PyTorch model on every part. With precisions, also time it against those
    TensorRT engines on this GPU, and measure each engine on every part, as a delta from PyTorch."""
    config = yaml.safe_load((experiment / "config.yaml").read_text())
    manifest = yaml.safe_load((experiment / "experiment.yaml").read_text()) \
        if (experiment / "experiment.yaml").is_file() else {}
    summary = {
        "experiment": experiment.name,
        "recipe": manifest.get("recipe"),
        "corpus": manifest.get("corpus"),
        "corpus_inputs_sha256": manifest.get("corpus_inputs_sha256"),
        "git_commit": manifest.get("git_commit"),
        "checkpoint": checkpoint.name,
        "evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parts": {},
    }
    predictor = Predictor(config, checkpoint)
    summary["parts"] = measure_parts(predictor, parts, experiment / "evaluation", log)
    del predictor
    if precisions:
        from heron.inference import benchmark
        from heron.inference.timing import SAMPLES
        package = tensorrt_package(experiment, checkpoint, log)
        timing = benchmark.run(package, list(precisions), samples or SAMPLES, log=log)
        log(benchmark.table(timing))
        summary["timing"] = {k: timing[k] for k in ("machine", "samples", "warmup", "volumes", "tf32", "runs")}
        summary["tensorrt"] = {}
        for precision in precisions:
            name = f"tensorrt_{precision}"
            trt, _, _ = benchmark.tensorrt_predictor(package, config, precision, timing["machine"]["gpu"], log=log)
            measured = measure_parts(trt, parts, experiment / "evaluation" / name, log)
            summary["tensorrt"][name] = {
                "parts": measured,
                "delta": {part: delta(m, summary["parts"][part]) for part, m in measured.items()},
            }
            del trt
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


def main(argv=None) -> None:
    from heron.inference.engine import PRECISIONS

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--part", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--precision", nargs="*", default=[], choices=PRECISIONS,
                   help="also time and measure these TensorRT engines")
    p.add_argument("--samples", type=int, help="volumes timed per run (default 1000)")
    args = p.parse_args(argv)
    parts = dict((name, Path(d)) for name, d in (item.split("=", 1) for item in args.part))
    run(args.experiment, args.checkpoint, parts, args.results, args.precision, args.samples)
    print(f"results: {args.results}")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
