"""The portable package of an Experiment: everything another machine needs to run and time it.

    <package>/manifest.yaml         what was exported, from where, with which versions
    <package>/config.yaml           the Experiment's pytorch-3dunet config (model and val transform)
    <package>/checkpoint.pytorch    the exported checkpoint, for the PyTorch baseline
    <package>/model.onnx            fp32, static input shape (the Corpus crop): engines are built from it
    <package>/calibration.h5        `raw` volumes of the train part, for int8 calibration
    <package>/engines/              built on each machine, never copied along (see engine.py)

Export runs where torch, pytorch3dunet and onnx are installed (the cluster venv): a CPU
trace of the network, seconds to a minute.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from heron.errors import HeronConfigError

PACKAGE_DIR = "package"
MANIFEST = "manifest.yaml"
CONFIG = "config.yaml"
CHECKPOINT = "checkpoint.pytorch"
ONNX = "model.onnx"
CALIBRATION = "calibration.h5"
ENGINES = "engines"
CALIBRATION_SAMPLES = 200
OPSET = 18


def input_shape(config: dict) -> tuple[int, int, int]:
    """The network's input: the val patch, which heron sets to the Corpus crop."""
    try:
        shape = config["loaders"]["val"]["slice_builder"]["patch_shape"]
    except (KeyError, TypeError):
        raise HeronConfigError("config.yaml has no loaders.val.slice_builder.patch_shape") from None
    return tuple(int(s) for s in shape)


def calibration_source(experiment: Path, config: dict) -> Path | None:
    """The train part the Experiment learned from: its Corpus's, else the config's first train path."""
    from heron.corpus import corpus_dir
    from heron.training import read_manifest

    corpus = read_manifest(experiment).get("corpus")
    candidates = [corpus_dir(corpus) / "train"] if corpus else []
    candidates += [Path(p) for p in config.get("loaders", {}).get("train", {}).get("file_paths", [])]
    return next((c for c in candidates if any(c.glob("*.h5"))), None)


def default_package(experiment: Path) -> Path:
    return experiment / PACKAGE_DIR


def find_package(name_or_path: str) -> Path:
    """A package directory, or the package of an Experiment (name or path)."""
    candidate = Path(name_or_path)
    if (candidate / MANIFEST).is_file():
        return candidate.resolve()
    from heron.training import find_experiment
    package = default_package(find_experiment(name_or_path))
    if not (package / MANIFEST).is_file():
        raise HeronConfigError(f"{package} is not exported yet: heron export {name_or_path}")
    return package


def load(package: Path) -> dict:
    """The package's manifest, after checking its files are there."""
    manifest = package / MANIFEST
    if not manifest.is_file():
        raise HeronConfigError(f"{package}: not a package (no {MANIFEST}; make one with heron export)")
    for name in (CONFIG, CHECKPOINT, ONNX):
        if not (package / name).is_file():
            raise HeronConfigError(f"{package}: {name} is missing (export again with heron export --force)")
    return yaml.safe_load(manifest.read_text())


def pick(files: list[Path], n: int) -> list[Path]:
    """n files spread evenly over the sorted list (all of them if there are fewer)."""
    files = sorted(files)
    if len(files) <= n:
        return files
    return [files[i] for i in sorted(set(np.linspace(0, len(files) - 1, n).round().astype(int)))]


def write_calibration(path: Path, files: list[Path], shape: tuple) -> None:
    import h5py
    with h5py.File(path, "w") as out:
        ds = out.create_dataset("raw", shape=(len(files), *shape), dtype=np.float32,
                                chunks=(1, *shape), compression="gzip", shuffle=True)
        for i, f in enumerate(files):
            with h5py.File(f, "r") as h:
                raw = h["raw"][:]
            if raw.shape != shape:
                raise HeronConfigError(f"{f}: raw is {raw.shape}, the network takes {shape}")
            ds[i] = raw
        ds.attrs["sources"] = [f.name for f in files]


def export_onnx(config: dict, checkpoint: Path, shape: tuple, path: Path) -> None:
    import onnx
    import torch
    from pytorch3dunet.unet3d.model import get_model

    model = get_model(config["model"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    # The torch.export-based exporter (needs onnxscript >= 0.7 with torch 2.14). Static shape:
    # an engine is built for one input size, the Corpus crop. One file: the weights are small.
    torch.onnx.export(model, (torch.zeros(1, 1, *shape),), str(path), dynamo=True, opset_version=OPSET,
                      external_data=False, input_names=["input"], output_names=["output"])
    onnx.checker.check_model(str(path))


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("torch", "pytorch3dunet", "onnx", "onnxscript"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def export(experiment: Path, out: Path | None = None, checkpoint: str = "best", calibration: Path | None = None,
           samples: int = CALIBRATION_SAMPLES, force: bool = False, log=print) -> Path:
    """Write the Experiment's package (by default <experiment>/package). Returns its directory."""
    from heron.training import _git_commit, read_manifest

    if checkpoint not in ("best", "last"):
        raise HeronConfigError("checkpoint must be best or last")
    out = (out or default_package(experiment)).resolve()
    if out.exists() and not force:
        raise HeronConfigError(f"{out} already exists (--force replaces it, engines included)")
    ckpt = experiment / "checkpoint" / f"{checkpoint}_checkpoint.pytorch"
    if not ckpt.is_file():
        raise HeronConfigError(f"{ckpt} not found")
    config = yaml.safe_load((experiment / "config.yaml").read_text())
    shape = input_shape(config)
    source = calibration or calibration_source(experiment, config)
    if source is None or not any(Path(source).glob("*.h5")):
        raise HeronConfigError(f"{experiment.name}: no train data found for int8 calibration "
                               f"(give a directory of .h5 with a raw dataset: --calibration DIR)")
    chosen = pick(list(Path(source).glob("*.h5")), samples)

    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        shutil.copy2(experiment / "config.yaml", partial / CONFIG)
        shutil.copy2(ckpt, partial / CHECKPOINT)
        log(f"onnx: tracing {experiment.name} at {'x'.join(map(str, shape))}")
        export_onnx(config, ckpt, shape, partial / ONNX)
        log(f"calibration: {len(chosen)} volumes from {source}")
        write_calibration(partial / CALIBRATION, chosen, shape)
        (partial / MANIFEST).write_text(yaml.safe_dump({
            "experiment": experiment.name,
            "experiment_path": str(experiment),
            "corpus": read_manifest(experiment).get("corpus"),
            "checkpoint": checkpoint,
            "input_shape": list(shape),
            "onnx_opset": OPSET,
            "calibration": {"source": str(source), "volumes": len(chosen)},
            "versions": _versions(),
            "git_commit": _git_commit(),
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, sort_keys=False))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    return out
