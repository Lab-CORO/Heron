"""The portable package of an Experiment: everything another machine needs to run and time it.

    <package>/manifest.yaml         what was exported, from where, and how to feed the network
    <package>/config.yaml           the Experiment's pytorch-3dunet config (model and val transform)
    <package>/checkpoint.pytorch    the exported checkpoint, for the PyTorch baseline
    <package>/model.onnx            fp32, static input shape (the Corpus crop): engines are built from it
    <package>/calibration.h5        `input`: train volumes already through the val transform, for int8
    <package>/reference.h5          `scene` occupancy Scenes and the `reachability` PyTorch gives them
    <package>/engines/              built on each machine, never copied along (see engine.py)

A package is self-contained (format 1): a program can run the network from it with
TensorRT alone, without heron or pytorch-3dunet. The manifest spells out, as values, all
it needs:

    input.shape, input.axes       the network's input, axis 0 being x
    input.encoding                the value of a free and of an occupied voxel (what the val
                                  transform makes of an occupancy Scene)
    grid                          where that input lies around the robot base (voxel size,
                                  origin of voxel [0, 0, 0], in meters)
    output                        shape and range of the Reachability map
    files                         the ONNX, the reference Scenes, the calibration volumes

Such a program checks itself against reference.h5: its own encoding and engine must
give the reference Scenes the Reachability maps heron gave them.

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
REFERENCE = "reference.h5"
ENGINES = "engines"
CALIBRATION_SAMPLES = 200
REFERENCE_SCENES = 5
OPSET = 18
FORMAT = 1                      # what manifest.yaml promises; raise it with any change a reader must know
PROBE_OCCUPANCY = (0.0, 0.01, 0.2, 0.6)


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
    """The package's manifest, after checking its files are there and its format is one heron reads.

    A manifest without `format` predates format 1 (no input encoding, no reference Scenes): heron
    still runs it, a robot cannot.
    """
    manifest = package / MANIFEST
    if not manifest.is_file():
        raise HeronConfigError(f"{package}: not a package (no {MANIFEST}; make one with heron export)")
    for name in (CONFIG, CHECKPOINT, ONNX):
        if not (package / name).is_file():
            raise HeronConfigError(f"{package}: {name} is missing (export again with heron export --force)")
    data = yaml.safe_load(manifest.read_text())
    fmt = data.get("format")
    if fmt is not None and fmt > FORMAT:
        raise HeronConfigError(f"{package}: package format {fmt}, this heron reads up to {FORMAT}: update heron")
    return data


def shape_of(manifest: dict) -> tuple[int, int, int]:
    """The network's input shape, from a format 1 manifest or an older one."""
    shape = manifest["input"]["shape"] if "input" in manifest else manifest["input_shape"]
    return tuple(int(s) for s in shape)


def _numpy(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def val_transform(config: dict):
    """The Experiment's val raw transform as a function: volume (D, H, W) -> network input (D, H, W) float32."""
    from heron.inference.predictor import Predictor

    transform = Predictor(config, None, network=lambda x: x)
    return lambda raw: _numpy(transform.raw_transform(raw)).astype(np.float32).reshape(raw.shape)


def encoding(config: dict, shape: tuple) -> dict:
    """What the val transform makes of an occupancy Scene: one value for a free voxel, one for an
    occupied one, which a robot applies without pytorch-3dunet.

    Found by running the transform itself on probe Scenes (empty, sparse, dense) and checking
    that each kind of voxel always gets the same value, whatever the rest of the Scene. A
    transform that depends on the rest of the volume in another way (Standardize: its mean
    follows the occupancy) is refused, since no table can stand for it. A Scene without a single
    free voxel is outside the table (min-max Normalize gives it all -1); no robot Scene is.
    """
    transform = val_transform(config)
    rng = np.random.RandomState(0)
    free, occupied = set(), set()
    for fraction in PROBE_OCCUPANCY:
        scene = rng.rand(*shape) < fraction
        out = transform(scene.astype(np.float64))      # float64, like the generator's volumes
        free.update(np.unique(out[~scene]).tolist())
        occupied.update(np.unique(out[scene]).tolist())
    if len(free) != 1 or len(occupied) != 1:
        raise HeronConfigError(
            "the val transform does not give each kind of voxel one value whatever the Scene "
            f"(free voxels got {sorted(free)[:4]}, occupied {sorted(occupied)[:4]}): a robot could not "
            "apply it without pytorch-3dunet. Train with a fixed encoding (Normalize, or a fixed min/max).")
    return {"free": float(free.pop()), "occupied": float(occupied.pop())}


def pick(files: list[Path], n: int) -> list[Path]:
    """n files spread evenly over the sorted list (all of them if there are fewer)."""
    files = sorted(files)
    if len(files) <= n:
        return files
    return [files[i] for i in sorted(set(np.linspace(0, len(files) - 1, n).round().astype(int)))]


def _read_raw(f: Path, shape: tuple) -> np.ndarray:
    import h5py
    with h5py.File(f, "r") as h:
        raw = h["raw"][:]
    if raw.shape != shape:
        raise HeronConfigError(f"{f}: raw is {raw.shape}, the network takes {shape}")
    return raw


def write_calibration(path: Path, files: list[Path], shape: tuple, transform) -> None:
    """The calibration volumes as the network takes them (`input`), so int8 needs no val transform."""
    import h5py
    with h5py.File(path, "w") as out:
        ds = out.create_dataset("input", shape=(len(files), *shape), dtype=np.float32,
                                chunks=(1, *shape), compression="gzip", shuffle=True)
        for i, f in enumerate(files):
            ds[i] = transform(_read_raw(f, shape))
        ds.attrs["sources"] = [f.name for f in files]


def write_reference(path: Path, files: list[Path], shape: tuple, config: dict, checkpoint: Path) -> dict:
    """Occupancy Scenes (`scene`, the train volumes thresholded at 0.5, as a robot sends them) and the
    Reachability maps PyTorch gives them (`reachability`, fp32, TF32 off). Returns what the manifest says."""
    import h5py
    from heron.inference.predictor import Predictor

    predictor = Predictor(config, checkpoint)
    with h5py.File(path, "w") as out:
        scenes = out.create_dataset("scene", shape=(len(files), *shape), dtype=np.uint8,
                                    chunks=(1, *shape), compression="gzip")
        maps = out.create_dataset("reachability", shape=(len(files), *shape), dtype=np.float32,
                                  chunks=(1, *shape), compression="gzip", shuffle=True)
        for i, f in enumerate(files):
            scene = _read_raw(f, shape) > 0.5
            scenes[i] = scene
            maps[i] = predictor.predict(scene.astype(np.float32))
        out.attrs["sources"] = [f.name for f in files]
        out.attrs["computed_with"] = f"pytorch fp32, TF32 off, on {predictor.device}"
    return {"scenes": len(files), "computed_with": f"pytorch fp32, TF32 off, on {predictor.device}"}


def grid(files: list[Path], shape: tuple, voxel_size: float | None = None,
         origin: tuple[float, float, float] | None = None) -> dict | None:
    """Where the network's input lies: voxel size and origin (m), the position of voxel [0, 0, 0]
    relative to the robot base, as in the generator's origine_* attributes (voxel i at
    origin + i * voxel_size).

    Read from the volumes' attributes when they describe the volume: older formatters cropped
    the volumes but kept the uncropped Scene's attributes, which a voxel_grid_size_* other than
    the shape gives away. Else from voxel_size and origin. None when neither says it.
    """
    if (voxel_size is None) != (origin is None):
        raise HeronConfigError("give the voxel size and the origin together (--voxel-size and --origin)")
    import h5py
    found = set()
    for f in files:
        with h5py.File(f, "r") as h:
            attrs = dict(h["raw"].attrs)
        keys = ["voxel_size"] + [f"origine_{a}" for a in "xyz"] + [f"voxel_grid_size_{a}" for a in "xyz"]
        if any(k not in attrs for k in keys) or \
                tuple(int(round(float(attrs[f"voxel_grid_size_{a}"]))) for a in "xyz") != tuple(shape):
            continue
        found.add((round(float(attrs["voxel_size"]), 6), tuple(round(float(attrs[f"origine_{a}"]), 6) for a in "xyz")))
    if len(found) > 1:
        raise HeronConfigError(f"the calibration volumes disagree on their grid: {sorted(found)}")
    read = found.pop() if found else None
    if voxel_size is not None:
        given = (round(float(voxel_size), 6), tuple(round(float(v), 6) for v in origin))
        if read and not (abs(read[0] - given[0]) < 1e-6 and all(abs(a - b) < 1e-6 for a, b in zip(read[1], given[1]))):
            raise HeronConfigError(f"the volumes say voxel size {read[0]} m and origin {list(read[1])} m, "
                                   f"not the {given[0]} m and {list(given[1])} m given")
        return {"voxel_size": given[0], "origin": list(given[1]),
                "source": "volume attributes" if read else "given at export"}
    if read:
        return {"voxel_size": read[0], "origin": list(read[1]), "source": "volume attributes"}
    return None


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
    for name in ("heron", "torch", "pytorch3dunet", "onnx", "onnxscript"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def export(experiment: Path, out: Path | None = None, checkpoint: str = "best", calibration: Path | None = None,
           samples: int = CALIBRATION_SAMPLES, force: bool = False, voxel_size: float | None = None,
           origin: tuple[float, float, float] | None = None, reference_scenes: int = REFERENCE_SCENES,
           log=print) -> Path:
    """Write the Experiment's package (by default <experiment>/package). Returns its directory.

    voxel_size, origin: the grid, for volumes whose attributes do not describe it (see grid()).
    reference_scenes: how many calibration volumes also become reference Scenes.
    """
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
    input_encoding = encoding(config, shape)
    placement = grid(chosen, shape, voxel_size, origin)
    if placement is None:
        log("grid: the volumes do not say where they lie around the base, and no --voxel-size / --origin "
            "was given: a robot will not be able to check its Scenes against this package")

    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        shutil.copy2(experiment / "config.yaml", partial / CONFIG)
        shutil.copy2(ckpt, partial / CHECKPOINT)
        log(f"onnx: tracing {experiment.name} at {'x'.join(map(str, shape))}")
        export_onnx(config, ckpt, shape, partial / ONNX)
        log(f"calibration: {len(chosen)} volumes from {source}")
        write_calibration(partial / CALIBRATION, chosen, shape, val_transform(config))
        reference = pick(chosen, reference_scenes)
        log(f"reference: {len(reference)} Scenes through PyTorch")
        reference_info = write_reference(partial / REFERENCE, reference, shape, config, ckpt)
        sigmoid = bool(config.get("model", {}).get("final_sigmoid", True))
        (partial / MANIFEST).write_text(yaml.safe_dump({
            "format": FORMAT,
            "experiment": experiment.name,
            "experiment_path": str(experiment),
            "corpus": read_manifest(experiment).get("corpus"),
            "checkpoint": checkpoint,
            "input": {"shape": list(shape), "axes": ["x", "y", "z"], "dtype": "float32",
                      "batch": 1, "channels": 1, "encoding": input_encoding},
            "grid": placement,
            "output": {"shape": list(shape), "dtype": "float32",
                       "activation": "sigmoid" if sigmoid else "none",
                       "range": [0.0, 1.0] if sigmoid else None},
            "onnx": {"file": ONNX, "opset": OPSET, "input": "input", "output": "output"},
            "calibration": {"file": CALIBRATION, "dataset": "input", "source": str(source), "volumes": len(chosen)},
            "reference": {"file": REFERENCE, **reference_info},
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
