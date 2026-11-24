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

The image is ROS 2 Jazzy + cuRobo v0.8.0 (v2), amd64, built on a machine with Docker and
converted to Apptainer:

```bash
images/generation/build.sh --out /var/tmp/heron-build   # -> heron_data_generation_jazzy.sif
```

`build.sh` builds curobo_ros's `docker/x86_jazzy.dockerfile` (pinned by
`images/generation/patch_base.py`; curobo_ros's `x86_jazzy_prod.dockerfile` still installs
cuRobo v1), then `images/generation/Dockerfile`, which clones
curobo_ros, curobo_msgs and Heron at fixed commits (Heron's must be pushed) and builds
`curobo_msgs`, `curobo_ros` and `data_generation`. Every pinned version is at the top of
`build.sh`. What the base still leaves unpinned (apt packages, cuRobo's own pip
dependencies) is recorded in the image: `/opt/heron/{pip-freeze,dpkg,git}.txt`.

What a generation job may rely on (`data_generation/pipeline/job.sbatch`):

- the installed package at `/home/ros2_ws/install/data_generation/share/data_generation/`,
  whose `data/` (master map in, `.h5` out) and `launch/` are bind-mounted by the job;
- parameters as `APPTAINERENV_*` variables: `MODE`, `VOXEL`, `DATASET_SIZE`, `BATCH_SIZE`,
  `INIT_BATCH_SIZE`, `OBJ_MAX`, `REACH_MAX`, `ROBOT_CONFIG_FILE`, `BASE_LINK`, `WORLD_FILE`,
  `CAMERA_FRAME`, `SLURM_ARRAY_TASK_ID`, `HERON_SCENE_HALF_EXTENT`;
- a writable `/tmp` (curobo_ros writes resolved robot configs there);
- warp's kernel cache at `$HOME/.cache/warp`, mounted from the host: under `-C` the container's
  `$HOME` is an empty in-memory directory, so without the mount warp recompiles at every start;
- no network: it runs under `apptainer exec -C --net --network=none --nv`. cuRobo v2 compiles
  its CUDA kernels at run time with NVRTC, in memory, for the GPU at hand, so the image needs
  neither network nor a writable directory for them;
- `rmw_fastrtps_cpp`, discovery limited to the machine (`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`);
- an NVIDIA driver that runs the NGC 24.11 image (CUDA 12.6): 560 or later, or the data-center
  branches NGC lists as compatible. cuRobo's kernels are built for the GPU present when a job
  starts, so `TORCH_CUDA_ARCH_LIST` (`heron.cuda_arch`) only concerns code compiled when the
  image is built.

curobo_ros runs as one node, `curobo_trajectory_planner`, named `unified_planner`; heron's nodes
reach its services (`/unified_planner/ik_batch`, `…/get_voxel_grid`, …) through their
`curobo_node` parameter. The Scene cube comes from the Station (`HERON_SCENE_HALF_EXTENT`): the
launch files set curobo_ros's `mapper_extent_xyz` to `2 × scene_half_extent`, and `generate_data`
refuses a grid that is not `ceil(2 × scene_half_extent / voxel)` voxels per side (152 for
leeloo at 0.02).

The image carries labels, which `apptainer inspect --labels <sif>` shows:

| Label | Value |
|---|---|
| `heron.image` | `generation` |
| `heron.ros` | `jazzy` |
| `heron.curobo` | cuRobo tag (`v0.8.0`) |
| `heron.curobo_ros`, `heron.curobo_msgs`, `heron.heron` | commits built |
| `heron.highfive` | HighFive tag |
| `heron.cuda_arch` | `TORCH_CUDA_ARCH_LIST` of the build (`8.0;8.6;9.0`) |
| `heron.base` | NGC base image, by digest |
| `heron.built` | build date (UTC) |

`gen_dataset.sh` reads `heron.ros` to know which image it was given. The `scene` mode is not
ported to this image yet (curobo_ros v2 has no `load_voxel_scene`): it still needs the older
ROS 2 Humble + cuRobo v1 image, which has no `heron.*` label, with `pipeline/curobo_patch/`
mounted over it. With that image, only `--mode scene` is accepted.

## Architectures

Images are per architecture (`images.amd64`, `images.aarch64`); heron picks the machine's,
and `arch:` in `site.yaml` or `--arch` forces one. No aarch64 image exists yet. On a Jetson,
GPU access from Apptainer (`--nv`) is not established, since the Jetson driver stack differs
from a discrete GPU's: run `heron doctor --local` there first.
