"""A Corpus: Datasets made ready for learning, from a recipe file.

`prepare(recipe)` checks the Datasets agree (robot, station, voxel size), splits the
pool by Source, drops uninformative samples, augments the train part by rotating
around the base, crops with the Station's crop, and writes pytorch-3dunet files
(`raw` / `label`) under <data root>/corpora/<name>/{train,val,test/<name>}.
"""

import hashlib
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from heron import paths, volume
from heron.config import read_config
from heron.dataset import Dataset, SampleRef, load_dataset
from heron.errors import HeronConfigError
from heron.station import load_station

MANIFEST = "corpus.yaml"


@dataclass(frozen=True)
class DatasetEntry:
    path: Path
    role: str          # "pool", "val" or "test:<name>"


@dataclass(frozen=True)
class Recipe:
    name: str
    source_file: Path
    datasets: tuple[DatasetEntry, ...]
    crop_size: int
    crop_offset: tuple[float, float, float] | None
    val_ratio: float
    augment_copies: int
    augment_angle: float
    seed: int
    drop_empty: bool
    min_reachable_frac: float
    resample_to: float | None


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path).resolve()
    data = read_config(
        path,
        required={"name", "datasets", "crop"},
        optional={"val_ratio", "augment", "seed", "drop_empty", "min_reachable_frac", "resample_to"},
    )
    if not isinstance(data["datasets"], list) or not data["datasets"]:
        raise HeronConfigError(f"{path}: datasets must be a non-empty list")
    entries = []
    for item in data["datasets"]:
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict) or "path" not in item:
            raise HeronConfigError(f"{path}: each dataset needs a path")
        role = str(item.get("role", "pool"))
        if role not in ("pool", "val") and not (role.startswith("test:") and len(role) > 5):
            raise HeronConfigError(f"{path}: role must be pool, val or test:<name>, got {role!r}")
        dataset_path = Path(str(item["path"]))
        if not dataset_path.is_absolute():
            dataset_path = paths.data_root() / dataset_path
        entries.append(DatasetEntry(path=dataset_path.resolve(), role=role))

    crop = data["crop"]
    if not isinstance(crop, dict) or "size" not in crop or set(crop) - {"size", "offset"}:
        raise HeronConfigError(f"{path}: crop must be {{size: <voxels>}} with an optional offset: [x, y, z]")
    offset = crop.get("offset")
    if offset is not None and (not isinstance(offset, list) or len(offset) != 3):
        raise HeronConfigError(f"{path}: crop.offset must be [x, y, z] in meters")

    augment = data.get("augment") or {}
    if set(augment) - {"copies", "angle"}:
        raise HeronConfigError(f"{path}: augment accepts copies and angle")
    val_ratio = float(data.get("val_ratio", 0.0))
    if not 0.0 <= val_ratio < 1.0:
        raise HeronConfigError(f"{path}: val_ratio must be in [0, 1), got {val_ratio}")

    return Recipe(
        name=data["name"],
        source_file=path,
        datasets=tuple(entries),
        crop_size=int(crop["size"]),
        crop_offset=tuple(float(v) for v in offset) if offset is not None else None,
        val_ratio=val_ratio,
        augment_copies=int(augment.get("copies", 0)),
        augment_angle=float(augment.get("angle", 180.0)),
        seed=int(data.get("seed", 0)),
        drop_empty=bool(data.get("drop_empty", True)),
        min_reachable_frac=float(data.get("min_reachable_frac", 0.0)),
        resample_to=float(data["resample_to"]) if data.get("resample_to") is not None else None,
    )


def corpus_dir(name: str) -> Path:
    return paths.data_root() / "corpora" / name


@dataclass
class _PartStats:
    samples: int = 0
    augmented: int = 0
    dropped: int = 0
    sources: set = field(default_factory=set)


def prepare(recipe: Recipe, force: bool = False, log=print) -> Path:
    """Build the Corpus described by the recipe. Returns its directory."""
    out = corpus_dir(recipe.name)
    if out.exists() and not force:
        raise HeronConfigError(f"{out} already exists (use --force to rebuild it from scratch)")

    datasets = [(load_dataset(e.path), e.role) for e in recipe.datasets]
    voxel_size, station = _check_agreement(recipe, [d for d, _ in datasets])
    ranges = station.crop(voxel_size, recipe.crop_size, recipe.crop_offset)
    scene_n = station.scene_voxels(voxel_size)

    inputs_sha256 = _inputs_hash(datasets)
    skipped = []
    parts = _assign_parts(recipe, datasets, lambda p, e: skipped.append(f"{p}: {e}"))
    if not any(parts.values()):
        raise HeronConfigError(
            f"recipe {recipe.name}: no scenes in " + ", ".join(str(d.path) for d, _ in datasets)
            + " (generator .h5 files expected there; generation writes them there only when its job "
              "finishes)" + "".join(f"\n  skipped {line}" for line in skipped))

    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    stats: dict[str, _PartStats] = defaultdict(_PartStats)
    import numpy as np
    rng = np.random.RandomState(recipe.seed)

    for part in sorted(parts):
        part_dir = partial / (Path("test") / part[5:] if part.startswith("test:") else part)
        part_dir.mkdir(parents=True, exist_ok=True)
        refs = parts[part]
        log(f"{part}: {len(refs)} samples")
        for ref, dataset in refs:
            try:
                scene, label, attrs = _load(ref)
            except OSError as exc:
                skipped.append(f"{ref.path}: {exc}")
                continue
            scene, label, attrs = _to_voxel_size(scene, label, attrs, dataset, voxel_size)
            if scene.shape != (scene_n,) * 3:
                raise HeronConfigError(
                    f"{ref.path} group {ref.group}: scene is {scene.shape}, station {station.name} "
                    f"expects {scene_n}³ at {voxel_size} m")
            s = stats[part]
            if recipe.drop_empty and volume.is_uninformative(label, recipe.min_reachable_frac):
                s.dropped += 1
                continue
            base = volume.base_index(attrs, voxel_size)
            if any(abs(b - scene_n // 2) > 0.5 for b in base[:2]):
                raise HeronConfigError(f"{ref.path}: the robot base is not at the center of the scene")
            out_attrs = volume.cropped_attrs(attrs, ranges, voxel_size)
            out_attrs.update(source=ref.source, corpus=recipe.name)
            stem = f"{dataset.name}__{ref.name}"
            _save(part_dir / f"{stem}.h5", volume.crop(scene, ranges), volume.crop(label, ranges), out_attrs)
            s.samples += 1
            s.sources.add(ref.source)
            if part == "train":
                for k in range(recipe.augment_copies):
                    angle = rng.uniform(-recipe.augment_angle, recipe.augment_angle)
                    rs = volume.rotate_about_base(scene, base, angle)
                    rl = volume.rotate_about_base(label, base, angle)
                    _save(part_dir / f"{stem}_aug{k}.h5", volume.crop(rs, ranges), volume.crop(rl, ranges),
                          dict(out_attrs, augment_angle=float(angle)))
                    s.augmented += 1

    manifest = _manifest(recipe, datasets, voxel_size, station.name, ranges, stats, skipped, inputs_sha256)
    (partial / MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False))
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    for line in skipped:
        log(f"skipped {line}")
    return out


def _check_agreement(recipe: Recipe, datasets: list[Dataset]):
    """All Datasets must share robot and station, and a voxel size unless resampled."""
    for key in ("robot", "station"):
        values = {getattr(d, key) for d in datasets}
        if len(values) > 1:
            raise HeronConfigError(f"recipe {recipe.name}: datasets mix {key}s {sorted(values)}")
    voxel_sizes = {d.voxel_size for d in datasets}
    if recipe.resample_to is None:
        if len(voxel_sizes) > 1:
            raise HeronConfigError(
                f"recipe {recipe.name}: datasets mix voxel sizes {sorted(voxel_sizes)}; "
                f"set resample_to to convert them")
        voxel_size = voxel_sizes.pop()
    else:
        voxel_size = recipe.resample_to
        for d in datasets:
            _upsample_factor(d.voxel_size, voxel_size, d.name)
    return voxel_size, load_station(datasets[0].station)


def _upsample_factor(from_size: float, to_size: float, what: str) -> int:
    factor = round(from_size / to_size)
    if factor < 1 or not math.isclose(from_size, factor * to_size, rel_tol=1e-3):
        raise HeronConfigError(
            f"{what}: voxel size {from_size} cannot be resampled to {to_size} "
            f"(only finer, by a whole factor)")
    return factor


def _assign_parts(recipe: Recipe, datasets, on_skip) -> dict[str, list[tuple[SampleRef, Dataset]]]:
    """Map each part (train, val, test:<name>) to its samples; the pool is split by Source."""
    parts: dict[str, list] = defaultdict(list)
    pool_by_source: dict[str, list] = defaultdict(list)
    for dataset, role in datasets:
        for ref in dataset.samples(on_skip=on_skip):
            if role == "pool":
                pool_by_source[ref.source].append((ref, dataset))
            else:
                parts[role].append((ref, dataset))

    sources = sorted(pool_by_source)
    random.Random(recipe.seed).shuffle(sources)
    n_val = 0
    if recipe.val_ratio > 0 and len(sources) > 1:
        n_val = min(len(sources) - 1, max(1, round(len(sources) * recipe.val_ratio)))
    for i, source in enumerate(sources):
        parts["val" if i < n_val else "train"].extend(pool_by_source[source])
    return parts


def _load(ref: SampleRef):
    import h5py
    with h5py.File(ref.path, "r", locking=False) as f:
        group = f["group"][ref.group]
        return group["voxel_grid"][:], group["reachability_map"][:], dict(group["reachability_map"].attrs)


def _to_voxel_size(scene, label, attrs, dataset: Dataset, voxel_size: float):
    if math.isclose(dataset.voxel_size, voxel_size, rel_tol=1e-3):
        return scene, label, attrs
    factor = _upsample_factor(dataset.voxel_size, voxel_size, dataset.name)
    attrs = dict(attrs, voxel_size=voxel_size)
    for axis in "xyz":
        attrs[f"voxel_grid_size_{axis}"] = float(scene.shape[0] * factor)
    return volume.upsample(scene, factor), volume.upsample(label, factor), attrs


def _save(path: Path, scene, label, attrs: dict) -> None:
    import h5py
    with h5py.File(path, "w") as f:
        raw = f.create_dataset("raw", data=scene, compression="gzip")
        lab = f.create_dataset("label", data=label, compression="gzip")
        for key, value in attrs.items():
            raw.attrs[key] = value
            lab.attrs[key] = value


def _inputs_hash(datasets) -> str:
    """Fingerprint of the input files (names, sizes, mtimes): changes when a Dataset changes."""
    h = hashlib.sha256()
    for dataset, _ in datasets:
        for p in dataset.files():
            try:
                st = p.stat()
                h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}\n".encode())
            except OSError:
                h.update(f"{p}|unreadable\n".encode())
    return h.hexdigest()


def _manifest(recipe, datasets, voxel_size, station, ranges, stats, skipped, inputs_sha256) -> dict:
    return {
        "name": recipe.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recipe": str(recipe.source_file),
        "robot": datasets[0][0].robot,
        "station": station,
        "voxel_size": voxel_size,
        "crop": [list(r) for r in ranges],
        "seed": recipe.seed,
        "val_ratio": recipe.val_ratio,
        "augment": {"copies": recipe.augment_copies, "angle": recipe.augment_angle},
        "datasets": [{"path": str(d.path), "role": role} for d, role in datasets],
        "inputs_sha256": inputs_sha256,
        "parts": {
            part: {"samples": s.samples, "augmented": s.augmented, "dropped_uninformative": s.dropped,
                   "sources": sorted(s.sources)}
            for part, s in sorted(stats.items())
        },
        "skipped": skipped,
    }
