"""How long a predictor takes per volume, and on what machine.

Two timings per volume, from the same loop:
    inference    the network alone, its input already on the device
    end_to_end   numpy volume in, numpy prediction out: val transform, copies, network

Content does not change the time of a dense 3D U-Net, so the volumes are random binary
ones, a few distinct ones cycled, at the network's input shape.
"""

import importlib.metadata as md
import platform as _platform
import socket
import time
from pathlib import Path

import numpy as np

SAMPLES = 1000
WARMUP = 50
POOL = 8
DENSITY = 0.2


def volumes(shape: tuple, n: int = POOL, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(rng.random(shape) < DENSITY).astype(np.float32) for _ in range(n)]


def stats(seconds: list[float]) -> dict:
    ms = np.asarray(seconds) * 1000.0
    return {"median_ms": round(float(np.median(ms)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3),
            "mean_ms": round(float(ms.mean()), 3), "min_ms": round(float(ms.min()), 3),
            "per_second": round(float(1000.0 / ms.mean()), 2)}


def time_predictor(predictor, pool: list[np.ndarray], samples: int = SAMPLES, warmup: int = WARMUP) -> dict:
    """predictor: to_input(raw) -> device tensor, infer(x) -> device tensor (heron Predictor)."""
    torch = predictor.torch
    sync = torch.cuda.synchronize if predictor.device.type == "cuda" else (lambda: None)
    inference, end_to_end = [], []
    for i in range(warmup + samples):
        raw = pool[i % len(pool)]
        sync()
        t0 = time.perf_counter()
        x = predictor.to_input(raw)
        sync()
        t1 = time.perf_counter()
        y = predictor.infer(x)
        sync()
        t2 = time.perf_counter()
        y.squeeze().cpu().numpy()
        t3 = time.perf_counter()
        if i >= warmup:
            inference.append(t2 - t1)
            end_to_end.append(t3 - t0)
    return {"inference": stats(inference), "end_to_end": stats(end_to_end)}


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def machine(torch) -> dict:
    """What the times depend on: GPU, its software stack, and the Jetson release if any."""
    try:
        import tensorrt
        trt = tensorrt.__version__
    except ImportError:
        trt = None
    tegra = Path("/etc/nv_tegra_release")
    return {
        "hostname": socket.gethostname(),
        "arch": _platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "tensorrt": trt,
        "torch": str(torch.__version__),        # a TorchVersion, which yaml.safe_dump refuses
        "pytorch3dunet": _version("pytorch3dunet"),
        "python": _platform.python_version(),
        "l4t": tegra.read_text().splitlines()[0].strip("# ") if tegra.is_file() else None,
    }
