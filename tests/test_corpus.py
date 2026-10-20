from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from heron.cli import main
from heron.corpus import load_recipe, prepare
from heron.dataset import describe, load_dataset
from heron.errors import HeronConfigError
from heron.volume import rotate_about_base

VOXEL = 0.02
HALF = 0.2          # small station: 20³ scenes at 0.02, base at index 10


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    (tmp_path / "tiny.yaml").write_text(
        f"name: tiny\nscene_half_extent: {HALF}\ncrop_offset: [0.0, 0.0, 0.04]\n")
    (tmp_path / "arm.yaml").write_text("name: arm\nbase_link: base\nreach: 1.0\n")
    return tmp_path


def make_dataset(root: Path, name: str, mode: str, files: dict, voxel=VOXEL, half=HALF, source=""):
    """files: {file stem: [label value per group]}; scene = one occupied voxel per group."""
    d = root / "datasets" / name
    d.mkdir(parents=True)
    n = int(round(2 * half / voxel))
    for stem, values in files.items():
        with h5py.File(d / f"{stem}.h5", "w") as f:
            for i, value in enumerate(values):
                g = f.create_group(f"group/{i}")
                scene = np.zeros((n, n, n))
                scene[n // 2 + 3, n // 2, n // 2] = 1.0
                label = np.full((n, n, n), value)
                for key, data in (("voxel_grid", scene), ("reachability_map", label)):
                    ds = g.create_dataset(key, data=data)
                    for a in "xyz":
                        ds.attrs[f"origine_{a}"] = -half
                        ds.attrs[f"voxel_grid_size_{a}"] = float(n)
                    ds.attrs["voxel_size"] = voxel
    describe(d, str(root / "arm.yaml"), str(root / "tiny.yaml"), mode, voxel, source=source)
    return d


def recipe(root: Path, **overrides) -> Path:
    data = {"name": "c", "datasets": [], "crop": {"size": 16}, "val_ratio": 0.5,
            "augment": {"copies": 2, "angle": 180}, "seed": 1}
    data.update(overrides)
    path = root / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def files_in(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("*.h5"))


# --- volume -----------------------------------------------------------------------

def test_rotation_turns_around_the_base_and_keeps_values():
    v = np.zeros((20, 20, 20))
    v[13, 10, 5] = 1.0                       # 3 voxels from the base along +x
    r = rotate_about_base(v, (10.0, 10.0, 10.0), 90.0)
    assert set(np.unique(r)) == {0.0, 1.0}   # nearest neighbour: still binary
    assert np.argwhere(r == 1.0).tolist() in ([[10, 13, 5]], [[10, 7, 5]])


def test_rotation_by_zero_is_identity():
    v = np.random.RandomState(0).rand(12, 12, 12)
    assert np.array_equal(rotate_about_base(v, (6.0, 6.0, 6.0), 0.0), v)


# --- dataset manifest -------------------------------------------------------------------

def test_describe_records_where_a_dataset_comes_from(root):
    d = make_dataset(root, "bag1", "real", {"a": [0.5]}, source="rosbag_x")
    ds = load_dataset(d)
    assert (ds.mode, ds.voxel_size, ds.source) == ("real", VOXEL, "rosbag_x")
    with pytest.raises(HeronConfigError, match="already exists"):
        describe(d, str(root / "arm.yaml"), str(root / "tiny.yaml"), "real", VOXEL)


def test_dataset_without_manifest_explains_how_to_add_one(root):
    (root / "datasets" / "old").mkdir(parents=True)
    with pytest.raises(HeronConfigError, match="heron dataset describe"):
        load_dataset(root / "datasets" / "old")


def test_sources_follow_the_generation_mode(root):
    real = load_dataset(make_dataset(root, "r", "real", {"f1": [1], "f2": [1]}, source="bag"))
    syn = load_dataset(make_dataset(root, "s", "synthetic", {"f1": [1], "f2": [1]}))
    sc = load_dataset(make_dataset(root, "sc", "scene", {"scene_3_task4": [1], "scene_3_task9": [1]}))
    assert {r.source for r in real.samples()} == {"r/bag"}
    assert {r.source for r in syn.samples()} == {"s/f1", "s/f2"}
    assert {r.source for r in sc.samples()} == {"sc/scene_3"}


# --- prepare -------------------------------------------------------------------------

def test_prepare_splits_by_source_and_augments_train_only(root):
    make_dataset(root, "syn", "synthetic", {f"f{i}": [0.5, 0.5] for i in range(4)})
    make_dataset(root, "bag", "real", {"t1": [0.5], "t2": [0.5]}, source="bagT")
    out = prepare(load_recipe(recipe(root, datasets=["datasets/syn", {"path": "datasets/bag", "role": "test:reel"}])),
                  log=lambda *_: None)

    manifest = yaml.safe_load((out / "corpus.yaml").read_text())
    train, val = manifest["parts"]["train"], manifest["parts"]["val"]
    assert not set(train["sources"]) & set(val["sources"])
    assert len(val["sources"]) == 2 and len(train["sources"]) == 2
    assert train["augmented"] == 2 * train["samples"]
    assert manifest["parts"]["val"]["augmented"] == 0
    assert manifest["parts"]["test:reel"] == {"samples": 2, "augmented": 0, "dropped_uninformative": 0,
                                              "sources": ["bag/bagT"]}
    assert len(files_in(out / "test" / "reel")) == 2
    assert all("_aug" not in n for n in files_in(out / "val"))


def test_prepare_crops_with_the_station_crop_and_updates_geometry(root):
    make_dataset(root, "syn", "synthetic", {"f0": [0.5], "f1": [0.5]})
    out = prepare(load_recipe(recipe(root, datasets=["datasets/syn"], val_ratio=0.0, augment={"copies": 0})),
                  log=lambda *_: None)
    with h5py.File(next((out / "train").glob("*.h5"))) as f:
        assert f["raw"].shape == (16, 16, 16) and f["label"].shape == (16, 16, 16)
        # 20³ scene, 16³ crop centered on the base (index 10), z shifted by 0.04 m = 2 voxels
        assert f["raw"].attrs["origine_x"] == pytest.approx(-HALF + 2 * VOXEL)
        assert f["raw"].attrs["origine_z"] == pytest.approx(-HALF + 4 * VOXEL)
        assert f["raw"].attrs["source"].startswith("syn/")
    assert yaml.safe_load((out / "corpus.yaml").read_text())["crop"] == [[2, 18], [2, 18], [4, 20]]


def test_uninformative_maps_are_dropped_and_counted(root):
    make_dataset(root, "syn", "synthetic", {"f0": [0.0, 0.5]})
    out = prepare(load_recipe(recipe(root, datasets=["datasets/syn"], val_ratio=0.0)), log=lambda *_: None)
    assert yaml.safe_load((out / "corpus.yaml").read_text())["parts"]["train"]["dropped_uninformative"] == 1


def test_mixed_voxel_sizes_need_resample_to(root):
    make_dataset(root, "fine", "synthetic", {"f0": [0.5]})
    make_dataset(root, "coarse", "synthetic", {"f0": [0.5]}, voxel=0.04)
    with pytest.raises(HeronConfigError, match="mix voxel sizes"):
        prepare(load_recipe(recipe(root, datasets=["datasets/fine", "datasets/coarse"])), log=lambda *_: None)
    out = prepare(load_recipe(recipe(root, datasets=["datasets/fine", "datasets/coarse"], resample_to=VOXEL,
                                     val_ratio=0.0, augment={"copies": 0})), log=lambda *_: None)
    assert yaml.safe_load((out / "corpus.yaml").read_text())["parts"]["train"]["samples"] == 2


def test_existing_corpus_is_not_overwritten_without_force(root):
    make_dataset(root, "syn", "synthetic", {"f0": [0.5]})
    r = load_recipe(recipe(root, datasets=["datasets/syn"], val_ratio=0.0))
    prepare(r, log=lambda *_: None)
    with pytest.raises(HeronConfigError, match="already exists"):
        prepare(r, log=lambda *_: None)
    prepare(r, force=True, log=lambda *_: None)


def test_same_recipe_gives_the_same_corpus(root):
    make_dataset(root, "syn", "synthetic", {f"f{i}": [0.5] for i in range(6)})
    r = load_recipe(recipe(root, datasets=["datasets/syn"]))
    first = yaml.safe_load((prepare(r, log=lambda *_: None) / "corpus.yaml").read_text())["parts"]
    second = yaml.safe_load((prepare(r, force=True, log=lambda *_: None) / "corpus.yaml").read_text())["parts"]
    assert first == second


@pytest.mark.parametrize("override, message", [
    ({"crop": {"size": 16, "offset": [0, 0]}}, "crop.offset"),
    ({"val_ratio": 1.5}, "val_ratio"),
    ({"datasets": [{"path": "datasets/x", "role": "train"}]}, "role must be"),
    ({"crop": {"size": 16}, "augment": {"copies": 1, "angel": 3}}, "augment accepts"),
])
def test_invalid_recipes_are_refused(root, override, message):
    base = {"datasets": ["datasets/x"]}
    base.update(override)
    with pytest.raises(HeronConfigError, match=message):
        load_recipe(recipe(root, **base))


def test_crop_that_leaves_the_scene_is_refused_before_writing(root):
    make_dataset(root, "syn", "synthetic", {"f0": [0.5]})
    with pytest.raises(HeronConfigError, match="leaves the 20-voxel scene"):
        prepare(load_recipe(recipe(root, datasets=["datasets/syn"], crop={"size": 20})), log=lambda *_: None)
    assert not (root / "corpora" / "c").exists()


def test_submit_dry_run_prints_a_cpu_job(root, capsys):
    make_dataset(root, "syn", "synthetic", {"f0": [0.5]})
    assert main(["prepare", str(recipe(root, datasets=["datasets/syn"])), "--submit", "--dry-run",
                 "--account", "def-test"]) == 0
    out = capsys.readouterr().out
    assert "--account=def-test" in out and "-m heron prepare" in out and "gpu" not in out
