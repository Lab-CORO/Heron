from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from heron import training
from heron.cli import main
from heron.errors import HeronConfigError
from heron.evaluation import jobs as eval_jobs, measure
from heron.inference import benchmark, engine, jobs, package, predictor, timing
from heron.site import load_site

SHAPE = (16, 16, 16)


def config(patch=SHAPE, train_dir=None) -> dict:
    slice_builder = {"name": "SliceBuilder", "patch_shape": list(patch), "stride_shape": list(patch)}
    return {
        "model": {"name": "UNet3D", "in_channels": 1, "out_channels": 1, "final_sigmoid": True,
                  "f_maps": [4, 8], "layer_order": "cbr"},
        "loaders": {"global_normalization": False,
                    "train": {"file_paths": [str(train_dir)] if train_dir else [], "slice_builder": slice_builder,
                              "transformer": training.TRANSFORMS},
                    "val": {"file_paths": [], "slice_builder": slice_builder, "transformer": training.TRANSFORMS}},
    }


def write_raw(path: Path, shape=SHAPE, seed=0, attrs=None):
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("raw", data=(np.random.RandomState(seed).rand(*shape) < 0.2).astype(np.float64))
        ds.attrs.update(attrs or {})


def grid_attrs(voxel_size=0.02, origin=(-0.16, -0.16, -0.16), size=SHAPE) -> dict:
    """The attributes the generator writes, as float32 like it does."""
    attrs = {"voxel_size": np.float32(voxel_size)}
    for axis, o, n in zip("xyz", origin, size):
        attrs[f"origine_{axis}"] = np.float32(o)
        attrs[f"voxel_grid_size_{axis}"] = float(n)
    return attrs


@pytest.fixture
def tiny_experiment(tmp_path):
    """A legacy-style Experiment: config.yaml naming its train data, a checkpoint, no experiment.yaml."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch3dunet")
    from pytorch3dunet.unet3d.model import get_model
    train = tmp_path / "train"
    train.mkdir()
    for i in range(5):
        write_raw(train / f"s{i}.h5", seed=i)
    exp = tmp_path / "tiny_20261026_1000"
    (exp / "checkpoint").mkdir(parents=True)
    cfg = config(train_dir=train)
    (exp / "config.yaml").write_text(yaml.safe_dump(cfg))
    torch.save({"model_state_dict": get_model(cfg["model"]).state_dict()}, exp / "checkpoint" / "best_checkpoint.pytorch")
    return exp


# --- package ----------------------------------------------------------------------------------

def test_the_input_is_the_val_patch():
    assert package.input_shape(config(patch=(128, 128, 128))) == (128, 128, 128)
    with pytest.raises(HeronConfigError, match="patch_shape"):
        package.input_shape({"loaders": {}})


def test_calibration_volumes_are_spread_over_the_train_part():
    files = [Path(f"{i:03d}.h5") for i in range(10)]
    assert package.pick(files, 3) == [Path("000.h5"), Path("004.h5"), Path("009.h5")]
    assert package.pick(files[:2], 3) == files[:2]


def test_export_writes_a_portable_package(tiny_experiment):
    pytest.importorskip("onnx")
    out = package.export(tiny_experiment, samples=3, log=lambda *_: None)
    assert out == tiny_experiment / "package"
    manifest = package.load(out)
    assert manifest["experiment"] == tiny_experiment.name and manifest["checkpoint"] == "best"
    assert manifest["input"]["shape"] == list(SHAPE) and manifest["calibration"]["volumes"] == 3
    with h5py.File(out / package.CALIBRATION) as f:
        assert f["input"].shape == (3, *SHAPE) and f["input"].dtype == np.float32
    import onnx
    dims = [d.dim_value for d in onnx.load(str(out / package.ONNX)).graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 1, *SHAPE]
    with pytest.raises(HeronConfigError, match="already exists"):
        package.export(tiny_experiment, log=lambda *_: None)
    (out / package.ENGINES).mkdir()
    package.export(tiny_experiment, samples=2, force=True, log=lambda *_: None)
    assert not (out / package.ENGINES).exists() and package.load(out)["calibration"]["volumes"] == 2


def test_export_refuses_calibration_volumes_of_another_shape(tiny_experiment, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    write_raw(other / "big.h5", shape=(32, 32, 32))
    with pytest.raises(HeronConfigError, match="the network takes"):
        package.export(tiny_experiment, calibration=other, log=lambda *_: None)
    assert not (tiny_experiment / "package").exists() and not (tiny_experiment / "package.partial").exists()


def test_export_without_train_data_asks_for_it(tiny_experiment, tmp_path):
    for f in (tmp_path / "train").glob("*.h5"):
        f.unlink()
    with pytest.raises(HeronConfigError, match="--calibration"):
        package.export(tiny_experiment, log=lambda *_: None)


def test_the_grid_is_read_from_volumes_that_describe_it(tmp_path):
    write_raw(tmp_path / "a.h5", attrs=grid_attrs())
    write_raw(tmp_path / "b.h5", attrs=grid_attrs())
    assert package.grid([tmp_path / "a.h5", tmp_path / "b.h5"], SHAPE) == \
        {"voxel_size": 0.02, "origin": [-0.16, -0.16, -0.16], "source": "volume attributes"}


def test_attributes_of_the_uncropped_scene_are_not_the_grid(tmp_path):
    # Older formatters cropped 152³ Scenes to 128³ but kept the 152³ attributes.
    write_raw(tmp_path / "a.h5", attrs=grid_attrs(origin=(-1.52,) * 3, size=(152,) * 3))
    files = [tmp_path / "a.h5"]
    assert package.grid(files, SHAPE) is None
    assert package.grid(files, SHAPE, 0.02, (-1.28, -1.28, -1.28)) == \
        {"voxel_size": 0.02, "origin": [-1.28, -1.28, -1.28], "source": "given at export"}


def test_a_given_grid_must_agree_with_the_volumes(tmp_path):
    write_raw(tmp_path / "a.h5", attrs=grid_attrs())
    with pytest.raises(HeronConfigError, match="not the 0.04 m"):
        package.grid([tmp_path / "a.h5"], SHAPE, 0.04, (-0.16, -0.16, -0.16))
    with pytest.raises(HeronConfigError, match="together"):
        package.grid([tmp_path / "a.h5"], SHAPE, voxel_size=0.02)
    write_raw(tmp_path / "b.h5", attrs=grid_attrs(origin=(0.0, 0.0, 0.0)))
    with pytest.raises(HeronConfigError, match="disagree"):
        package.grid([tmp_path / "a.h5", tmp_path / "b.h5"], SHAPE)


def test_export_writes_the_grid_into_the_manifest(tiny_experiment):
    pytest.importorskip("onnx")
    logged = []
    out = package.export(tiny_experiment, samples=2, log=logged.append)
    assert package.load(out)["grid"] is None and any("--voxel-size" in m for m in logged)
    out = package.export(tiny_experiment, samples=2, force=True, voxel_size=0.02, origin=(-0.16, -0.16, -0.16),
                         log=lambda *_: None)
    assert package.load(out)["grid"]["origin"] == [-0.16, -0.16, -0.16]


# --- a self-contained package: everything a robot needs, as values ----------------------------

def test_the_encoding_is_what_normalize_makes_of_occupancy():
    pytest.importorskip("pytorch3dunet")
    assert package.encoding(config(), SHAPE) == {"free": -1.0, "occupied": 1.0}


def test_a_transform_no_table_can_stand_for_is_refused():
    pytest.importorskip("pytorch3dunet")
    cfg = config()
    cfg["loaders"]["val"]["transformer"] = {"raw": [{"name": "Standardize"}, {"name": "ToTensor", "expand_dims": True}]}
    with pytest.raises(HeronConfigError, match="one value whatever the Scene"):
        package.encoding(cfg, SHAPE)


def test_the_manifest_spells_out_how_to_feed_the_network(tiny_package):
    manifest = package.load(tiny_package)
    assert manifest["format"] == package.FORMAT == 1
    assert manifest["input"] == {"shape": list(SHAPE), "axes": ["x", "y", "z"], "dtype": "float32",
                                 "batch": 1, "channels": 1, "encoding": {"free": -1.0, "occupied": 1.0}}
    assert manifest["output"] == {"shape": list(SHAPE), "dtype": "float32", "activation": "sigmoid",
                                  "range": [0.0, 1.0]}
    for key in ("onnx", "calibration", "reference"):
        assert (tiny_package / manifest[key]["file"]).is_file()
    assert package.shape_of(manifest) == SHAPE


def test_calibration_volumes_are_already_through_the_val_transform(tiny_package, tiny_experiment):
    transform = package.val_transform(yaml.safe_load((tiny_experiment / "config.yaml").read_text()))
    sources = sorted((tiny_experiment.parent / "train").glob("*.h5"))
    with h5py.File(tiny_package / package.CALIBRATION) as f:
        names = list(f["input"].attrs["sources"])
        for i, name in enumerate(names):
            raw = h5py.File(next(s for s in sources if s.name == name))["raw"][:]
            np.testing.assert_array_equal(f["input"][i], transform(raw))


def test_reference_scenes_through_the_manifest_encoding_give_the_reference_maps(tiny_package):
    """What a robot does without heron: occupancy -> encoding table -> network."""
    torch = pytest.importorskip("torch")
    from pytorch3dunet.unet3d.model import get_model
    manifest = package.load(tiny_package)
    model = get_model(yaml.safe_load((tiny_package / package.CONFIG).read_text())["model"])
    model.load_state_dict(torch.load(tiny_package / package.CHECKPOINT, weights_only=False)["model_state_dict"])
    model.eval()
    enc = manifest["input"]["encoding"]
    with h5py.File(tiny_package / package.REFERENCE) as f:
        assert len(f["scene"]) == 2 and set(np.unique(f["scene"][:])) <= {0, 1}
        for scene, expected in zip(f["scene"][:], f["reachability"][:]):
            x = torch.from_numpy(np.where(scene > 0, enc["occupied"], enc["free"]).astype(np.float32))[None, None]
            with torch.no_grad():
                out = model(x).squeeze().numpy()
            np.testing.assert_allclose(out, expected, atol=1e-5)


def test_a_package_of_a_newer_format_is_refused(tiny_package):
    manifest = yaml.safe_load((tiny_package / package.MANIFEST).read_text())
    manifest["format"] = package.FORMAT + 1
    (tiny_package / package.MANIFEST).write_text(yaml.safe_dump(manifest))
    with pytest.raises(HeronConfigError, match="this heron reads up to 1: update heron"):
        package.load(tiny_package)


# --- load: the package's network for a program -------------------------------------------------

@pytest.fixture
def tiny_package(tiny_experiment):
    pytest.importorskip("onnx")
    return package.export(tiny_experiment, samples=2, log=lambda *_: None)


def test_load_predicts_through_the_experiments_val_transform(tiny_package, tiny_experiment):
    loaded = predictor.load(tiny_package)
    assert loaded.backend == "pytorch" and loaded.manifest["experiment"] == tiny_experiment.name
    scene = (np.random.RandomState(1).rand(*SHAPE) < 0.2).astype(np.float32)
    direct = predictor.Predictor(yaml.safe_load((tiny_experiment / "config.yaml").read_text()),
                                 tiny_experiment / "checkpoint" / "best_checkpoint.pytorch")
    out = loaded.predict(scene)
    assert out.shape == SHAPE and out.dtype == np.float32
    np.testing.assert_allclose(out, direct.predict(scene), atol=1e-6)


def test_load_refuses_an_unknown_precision(tiny_package):
    with pytest.raises(HeronConfigError, match="precision must be pytorch or one of"):
        predictor.load(tiny_package, "bf16")


def test_a_missing_engine_is_an_error_naming_how_to_build_it(tiny_package):
    pytest.importorskip("tensorrt")
    with pytest.raises(HeronConfigError, match="no fp16 engine for Test GPU .*heron.inference.benchmark --package"):
        predictor.load(tiny_package, "fp16", gpu="Test GPU")


def test_fallback_runs_pytorch_and_says_so(tiny_package):
    pytest.importorskip("tensorrt")
    logged = []
    loaded = predictor.load(tiny_package, "fp16", gpu="Test GPU", fallback=True, log=logged.append)
    assert loaded.backend == "pytorch"
    assert len(logged) == 1 and "no fp16 engine" in logged[0] and "falling back to PyTorch" in logged[0]


def test_load_runs_the_engine_benchmark_built(tiny_package):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    gpu = torch.cuda.get_device_name(0)
    config = yaml.safe_load((tiny_package / package.CONFIG).read_text())
    built, _, _ = benchmark.tensorrt_predictor(tiny_package, config, "fp32", gpu, log=lambda *_: None)
    loaded = predictor.load(tiny_package, "fp32")
    assert built.backend == loaded.backend == "tensorrt_fp32"
    scene = (np.random.RandomState(2).rand(*SHAPE) < 0.2).astype(np.float32)
    np.testing.assert_allclose(loaded.predict(scene), predictor.load(tiny_package).predict(scene), atol=1e-3)


# --- engines ----------------------------------------------------------------------------------

def test_int8_is_refused_where_implicit_calibration_is_gone():
    engine.check_version("10.16.2", "int8")
    engine.check_version("11.4.0", "fp16")
    with pytest.raises(HeronConfigError, match="explicit"):
        engine.check_version("11.4.0", "int8")
    with pytest.raises(HeronConfigError, match="TensorRT 10"):
        engine.check_version("8.6.1", "fp32")


def test_an_engine_is_cached_per_gpu_tensorrt_and_precision(tmp_path):
    assert engine.engine_path(tmp_path, "fp16", "NVIDIA A100-SXM4-40GB", "10.16.1.11") == \
        tmp_path / "engines" / "nvidia-a100-sxm4-40gb_trt10.16.1.11_fp16.engine"


# --- timing -----------------------------------------------------------------------------------

def test_timing_reports_both_measures_over_the_samples():
    pytest.importorskip("pytorch3dunet")
    calls = []

    def network(x):
        calls.append(tuple(x.shape))
        return x
    predictor = measure.Predictor(config(), None, network=network)
    result = timing.time_predictor(predictor, timing.volumes(SHAPE, n=2), samples=5, warmup=2)
    assert len(calls) == 7 and calls[0] == (1, 1, *SHAPE)
    for kind in ("inference", "end_to_end"):
        assert set(result[kind]) == {"median_ms", "p95_ms", "mean_ms", "min_ms", "per_second"}
    assert result["end_to_end"]["median_ms"] >= result["inference"]["median_ms"]


def test_the_machine_description_can_be_written_as_yaml():
    torch = pytest.importorskip("torch")
    info = timing.machine(torch)
    assert yaml.safe_load(yaml.safe_dump(info))["torch"] == str(torch.__version__)


def test_speedup_is_against_the_pytorch_median():
    base = {"inference": {"median_ms": 30.0}, "end_to_end": {"median_ms": 40.0}}
    run = {"inference": {"median_ms": 10.0}, "end_to_end": {"median_ms": 20.0}}
    assert benchmark.speedup(base, run) == {"inference": 3.0, "end_to_end": 2.0}


# --- host side: the benchmark job, the evaluation with precisions -----------------------------

@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERON_DATA_ROOT", str(tmp_path))
    corpus = tmp_path / "corpora" / "full"
    for part in ("train", "val"):
        (corpus / part).mkdir(parents=True)
        (corpus / part / "a.h5").write_bytes(b"")
    (corpus / "corpus.yaml").write_text(yaml.safe_dump({"name": "full", "crop": [[0, 128]] * 3,
                                                        "parts": {"train": {"samples": 8}, "val": {"samples": 4}}}))
    (tmp_path / "capa_unet.sif").write_bytes(b"")
    recipe = tmp_path / "cbr.yaml"
    recipe.write_text("name: cbr\ncorpus: full\n")
    return training.create(training.plan(training.load_training_recipe(recipe), load_site(),
                                         now=datetime(2026, 10, 26, 10, 0)))


def fake_package(experiment: Path) -> Path:
    out = package.default_package(experiment)
    out.mkdir()
    for name in (package.CONFIG, package.CHECKPOINT, package.ONNX):
        (out / name).write_bytes(b"")
    (out / package.MANIFEST).write_text(yaml.safe_dump({"experiment": experiment.name, "checkpoint": "best",
                                                        "input_shape": [128] * 3}))
    return out


def test_benchmark_job_times_the_package_in_the_training_runtime(experiment):
    pkg = fake_package(experiment)
    script = jobs.write_benchmark_job(pkg, ["fp16", "int8"], load_site(), samples=100).read_text()
    assert f"python -m heron.inference.benchmark --package {pkg} --precision fp16 int8 --samples 100" in script
    assert f"--results-dir {jobs.results_dir(experiment.name)}" in script
    assert "capa_unet.sif" in script and "--gpus-per-node=1" in script


def test_benchmark_takes_an_experiment_or_a_package(experiment, capsys):
    with pytest.raises(HeronConfigError, match="heron export"):
        package.find_package(experiment.name)
    pkg = fake_package(experiment)
    assert package.find_package(experiment.name) == pkg == package.find_package(str(pkg))
    assert main(["benchmark", experiment.name, "--dry-run"]) == 0
    assert "--precision fp32 fp16 int8 --samples 1000 --warmup 50" in capsys.readouterr().out


def test_evaluate_passes_the_precisions_to_the_measurement(experiment):
    script = eval_jobs.write_eval_job(experiment, load_site(), precisions=["fp16", "int8"], samples=10).read_text()
    assert "--precision fp16 int8 --samples 10" in script


def test_engine_quality_is_reported_as_a_delta_from_pytorch(tmp_path, monkeypatch):
    exp = tmp_path / "exp_20261026_1000"
    exp.mkdir()
    (exp / "config.yaml").write_text("model: {}\nloaders: {}\n")
    val = tmp_path / "val"
    val.mkdir()
    label = np.full((8, 8, 8), 0.5)
    with h5py.File(val / "a.h5", "w") as f:
        f["raw"], f["label"] = np.zeros_like(label), label

    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, raw):
            return np.full(raw.shape, self.value, dtype=np.float32)
    monkeypatch.setattr(measure, "Predictor", lambda config, ckpt: Constant(0.5))
    monkeypatch.setattr(measure, "tensorrt_package", lambda *a, **k: tmp_path / "package")
    runs = {"pytorch_fp32": {"inference": {"median_ms": 9.0}}}
    monkeypatch.setattr(benchmark, "run", lambda *a, **k: {"machine": {"gpu": "A100"}, "samples": 3, "warmup": 1,
                                                           "volumes": "", "tf32": False, "runs": runs})
    monkeypatch.setattr(benchmark, "table", lambda r: "")
    monkeypatch.setattr(benchmark, "tensorrt_predictor", lambda *a, **k: (Constant(0.6), None, None))
    results = tmp_path / "results.yaml"
    measure.run(exp, exp / "best_checkpoint.pytorch", {"val": val}, results, ["fp16"], log=lambda *_: None)

    summary = yaml.safe_load(results.read_text())
    assert summary["timing"]["runs"] == runs
    trt = summary["tensorrt"]["tensorrt_fp16"]
    assert trt["parts"]["val"]["regression"]["MAE"] == pytest.approx(0.1)
    assert trt["delta"]["val"]["regression"]["MAE"] == pytest.approx(0.1)
    assert "voxels" not in trt["delta"]["val"]
    assert (exp / "evaluation" / "tensorrt_fp16" / "val" / "eval_data.h5").is_file()
