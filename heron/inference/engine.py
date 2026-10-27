"""TensorRT engines of a package, built on the machine that runs them.

Written against TensorRT 10 (10.16 on the cluster venv and on the Orin's JetPack 7.2).
int8 uses implicit calibration (IInt8EntropyCalibrator2), deprecated since TensorRT 10.1
and removed in 11: under TensorRT 11 heron refuses int8 rather than fail mid-build, until
it moves to explicit quantization (Q/DQ in the ONNX, e.g. with NVIDIA ModelOpt).

All precisions build with TF32 off, so fp32 means fp32 (as for the PyTorch baseline).
An engine is cached per GPU, TensorRT version and precision; the int8 calibration cache
per TensorRT version, since it does not depend on the GPU.
"""

import re
import time
from pathlib import Path

from heron.errors import HeronConfigError
from heron.inference.package import CALIBRATION, ENGINES, ONNX

PRECISIONS = ("fp32", "fp16", "int8")


def check_version(version: str, precision: str) -> None:
    major = int(version.split(".")[0])
    if major < 10:
        raise HeronConfigError(f"TensorRT {version}: heron needs TensorRT 10 (requirements-trt.txt)")
    if major >= 11 and precision == "int8":
        raise HeronConfigError(
            f"TensorRT {version} has no implicit int8 calibration (removed in 11): int8 needs explicit "
            f"quantization, not written yet. Use TensorRT 10, or fp32 / fp16.")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def engine_path(package: Path, precision: str, gpu: str, trt_version: str) -> Path:
    return package / ENGINES / f"{slug(gpu)}_trt{trt_version}_{precision}.engine"


def _logger():
    import tensorrt as trt
    return trt.Logger(trt.Logger.WARNING)


def _calibrator(package: Path, prepare, cache: Path):
    """Entropy calibrator over the package's calibration volumes, through the val transform."""
    import h5py
    import tensorrt as trt

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.file = h5py.File(package / CALIBRATION, "r")
            self.raw = self.file["raw"]
            self.i = 0
            self.batch = None           # kept alive until TensorRT asks for the next one

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(self.raw):
                return None
            self.batch = prepare(self.raw[self.i]).contiguous()
            self.i += 1
            return [self.batch.data_ptr()]

        def read_calibration_cache(self):
            return cache.read_bytes() if cache.is_file() else None

        def write_calibration_cache(self, data):
            cache.write_bytes(bytes(data))

    return Calibrator()


def build(package: Path, precision: str, gpu: str, prepare=None, rebuild: bool = False, log=print):
    """The engine for this machine, built unless cached. Returns (path, build seconds or None if cached).

    prepare: raw volume -> network input on the GPU, for int8 calibration.
    """
    import tensorrt as trt

    if precision not in PRECISIONS:
        raise HeronConfigError(f"precision must be one of {', '.join(PRECISIONS)}")
    check_version(trt.__version__, precision)
    path = engine_path(package, precision, gpu, trt.__version__)
    if path.is_file() and not rebuild:
        return path, None
    path.parent.mkdir(exist_ok=True)

    logger = _logger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(package / ONNX)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise HeronConfigError(f"{package / ONNX}: TensorRT cannot parse it: {errors}")
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)       # int8: fp16 for the layers int8 cannot run
    if precision == "int8":
        if prepare is None:
            raise ValueError("int8 needs prepare, to feed the calibration volumes")
        config.set_flag(trt.BuilderFlag.INT8)
        cache = package / ENGINES / f"calibration_trt{trt.__version__}.cache"
        if rebuild and cache.is_file():
            cache.unlink()
        config.int8_calibrator = _calibrator(package, prepare, cache)
    log(f"{precision}: building the TensorRT engine for {gpu} (minutes, longer on a Jetson)")
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise HeronConfigError(f"TensorRT could not build the {precision} engine (see the messages above)")
    seconds = time.perf_counter() - start
    path.write_bytes(bytes(serialized))
    return path, seconds


class TrtNetwork:
    """An engine called like the PyTorch model: a (1, 1, D, H, W) float32 CUDA tensor in, one out.

    The output tensor is reused from call to call: copy it before the next call.
    """

    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.runtime = trt.Runtime(_logger())
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise HeronConfigError(f"{path}: TensorRT cannot load it (built by another version? use --rebuild)")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        if self.engine.get_tensor_dtype(output) != trt.DataType.FLOAT:
            raise HeronConfigError(f"{path}: expected a float32 output")
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(output)), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(output, self.out.data_ptr())
        # Its own stream: on the default one, TensorRT adds synchronizations of its own.
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        current = self.torch.cuda.current_stream()
        self.stream.wait_stream(current)                # x is ready
        x.record_stream(self.stream)
        self.context.set_tensor_address(self.input, x.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        current.wait_stream(self.stream)                # out is ready for whoever reads it next
        return self.out
