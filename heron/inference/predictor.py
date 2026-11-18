"""The network as callers outside training use it: a Scene in, its Reachability map out.

    from heron.inference import load
    predictor = load(package, "fp16")          # or "pytorch", "fp32", "int8"
    reachability = predictor.predict(scene)    # (D, H, W) occupancy -> (D, H, W) float32 in [0, 1]

The input goes through the Experiment's own val transform (its config.yaml), so a caller
never encodes a Scene itself. load() never builds an Engine: building takes minutes (more
on a Jetson) and is heron benchmark's job, on the machine that runs it.
"""

from pathlib import Path

import numpy as np
import yaml

from heron.errors import HeronConfigError

PYTORCH = "pytorch"


class Predictor:
    """The Experiment's val raw transform, as pytorch-3dunet builds it, then a network:
    its PyTorch model (from the checkpoint), or any callable on the same input, such as a
    TensorRT engine (heron.inference.engine.TrtNetwork).

    backend: what runs the network ("pytorch", "tensorrt_fp16", ...).
    manifest: the package's manifest when made by load(), else None.
    """

    def __init__(self, config: dict, checkpoint: Path | None, network=None, backend: str = PYTORCH,
                 manifest: dict | None = None):
        import torch

        self.torch = torch
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network or self._pytorch(checkpoint)
        self.backend = backend
        self.manifest = manifest

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


def load(package: Path, precision: str = PYTORCH, gpu: str | None = None, fallback: bool = False,
         log=print) -> Predictor:
    """The package's network at one precision: "pytorch" (its checkpoint), or the cached Engine
    for this GPU and TensorRT version ("fp32", "fp16", "int8").

    A missing or unloadable Engine is an error naming the command that builds it, unless
    fallback: then the PyTorch model runs instead, said through log and in predictor.backend.
    gpu: the GPU's name as torch reports it (default: this machine's first GPU).
    """
    from heron.inference import package as pkg
    from heron.inference.engine import PRECISIONS, TrtNetwork, check_version, engine_path

    package = Path(package)
    manifest = pkg.load(package)
    config = yaml.safe_load((package / pkg.CONFIG).read_text())
    if precision == PYTORCH:
        return Predictor(config, package / pkg.CHECKPOINT, manifest=manifest)
    if precision not in PRECISIONS:
        raise HeronConfigError(f"precision must be {PYTORCH} or one of {', '.join(PRECISIONS)}, got {precision!r}")
    try:
        try:
            import tensorrt
        except ImportError:
            raise HeronConfigError("tensorrt is not importable here: install requirements-trt.txt "
                                   "(on a Jetson, a venv with --system-site-packages sees JetPack's)") from None
        check_version(tensorrt.__version__, precision)
        gpu = gpu or _gpu_name()
        path = engine_path(package, precision, gpu, tensorrt.__version__)
        if not path.is_file():
            raise HeronConfigError(
                f"{package}: no {precision} engine for {gpu} with TensorRT {tensorrt.__version__}. "
                f"Build it on this machine: python -m heron.inference.benchmark --package {package} "
                f"--precision {precision} --results-dir results/{manifest['experiment']}")
        network = TrtNetwork(path)
    except HeronConfigError as exc:
        if not fallback:
            raise
        log(f"heron: {exc}\nheron: falling back to PyTorch for {package}")
        return Predictor(config, package / pkg.CHECKPOINT, manifest=manifest)
    return Predictor(config, None, network=network, backend=f"tensorrt_{precision}", manifest=manifest)


def _gpu_name() -> str:
    import torch
    if not torch.cuda.is_available():
        raise HeronConfigError("no GPU visible: TensorRT engines run on a GPU (precision pytorch runs anywhere)")
    return torch.cuda.get_device_name(0)
