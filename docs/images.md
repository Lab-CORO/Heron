# What heron needs from its environments

Heron runs on any machine that meets the contracts below: an Alliance cluster, another
SLURM cluster, or a workstation, amd64 or aarch64. You provide the environments; heron
only reads where they are from `site.yaml` (see `site.example.yaml`) and checks them with:

```bash
heron doctor            # through SLURM
heron doctor --local    # here, in the foreground (a node with a GPU)
```

`heron doctor` fabricates a small Dataset, builds its Corpus, trains for 10 iterations and
evaluates, in the configured environment, then reports OK/KO per step and deletes what it made.

Docker is never used to *run* anything (Alliance clusters do not allow it). It is only one way
to *build* an image, converted to Apptainer afterwards:

```bash
docker build -f images/training/Dockerfile -t heron-training:1.9.6 .
apptainer build heron_training.sif docker-daemon://heron-training:1.9.6
```

## Training and evaluation environment

Training and evaluation run in the same environment, chosen in `site.yaml`
(`training.runtime`), and an Experiment is always evaluated in the environment it trained in
(recorded in its `experiment.yaml`).

| Runtime | What it is | When |
|---|---|---|
| `venv` | a Python virtual environment; jobs `module load` the site's `modules.venv`, then activate it | Alliance clusters (their recommendation), workstations |
| `apptainer` | an Apptainer image, `images.<arch>.training` in `site.yaml` | a site without suitable Python wheels, or a frozen environment |

Either one must provide what `requirements-train.txt` lists:

- Python 3.10 or later
- **pytorch3dunet 1.9.6** exactly, with its `train3dunet` command: heron's checks (built-in
  losses, config keys, the minimum patch of 64) are written against it
- torch 2.2 or later, built with CUDA for the GPU at hand
- h5py, numpy, scipy, PyYAML

A venv is set up with `pip install -e '.[train]'` (on Alliance clusters, first
`pip install --no-index torch torchvision tensorboard tqdm scikit-image`: torch comes from
the Alliance wheels, pytorch3dunet only from PyPI). An image is built from
`images/training/Dockerfile`, which installs the same file. Matplotlib is only needed where
`heron evaluate --figures` runs, on the host.

## TensorRT (export, benchmark, evaluate --precision)

On top of the training environment, `requirements-trt.txt`: **TensorRT 10.16**, the version
of JetPack 7.2 on the Jetson Orin, so both machines build engines with the same optimizer.
Not TensorRT 11: it removed the implicit int8 calibration heron uses.

- **Cluster venv**: `pip install -e '.[trt]'` (TensorRT, plus onnx and onnxscript for `heron export`).
  Alliance's pip refuses manylinux wheels, and NVIDIA publishes the TensorRT Python bindings
  only in that form. So install that one wheel by hand first, as `requirements-trt.txt`
  shows. Alliance's `tensorrt/8.6.1.6` module is too old.
- **Jetson (JetPack 7.2)**: TensorRT and torch come with JetPack. Use a venv that sees them:
  `python3 -m venv --system-site-packages ~/envs/heron`, then `pip install -e '.[train]'`
  (pytorch3dunet, for the PyTorch baseline and the val transform). The Jetson needs neither
  onnx nor the `[trt]` extra, since the package already holds `model.onnx`. In `site.yaml`:
  `arch: aarch64`, `scheduler: local`, `training: {runtime: venv, venv: ~/envs/heron}`.
- **Apptainer**: the current training image has no TensorRT. Benchmark from a venv.

## Generation image

Dataset generation needs ROS 2 and cuRobo, so it always runs in an Apptainer image,
`images.<arch>.generation` in `site.yaml`.

**This contract is being rewritten for the ROS 2 Jazzy + cuRobo v2 image.** The image must
also include what heron currently patches into the Humble image (`data_generation/pipeline/curobo_patch/`):

- the scene extent read from the Station (`HERON_SCENE_HALF_EXTENT`) instead of a fixed ±1.52 m;
- the `load_voxel_scene` service used by the `scene` mode.

Generation keeps `rmw_fastrtps_cpp`, with discovery limited to the machine
(`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`). `heron doctor` will exercise generation once
this image is in place.

## Architectures

Images are per architecture (`images.amd64`, `images.aarch64`); heron picks the machine's,
and `arch:` in `site.yaml` or `--arch` forces one. No aarch64 image exists yet. On a Jetson,
GPU access from Apptainer (`--nv`) is not established, since the Jetson driver stack differs
from a discrete GPU's: run `heron doctor --local` there first.
