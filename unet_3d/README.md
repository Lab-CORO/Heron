# Training and evaluation

Training runs pytorch-3dunet 1.9.6 (`train3dunet`), on a Corpus
built with `heron prepare` (see `data_generation/pipeline/README.md` and `GLOSSARY.md`).

## Where it runs

Jobs go through SLURM, or run here in the foreground with `--local` (on a node with a GPU).
Training and evaluation run in a Python venv or in an Apptainer image, as `site.yaml` says
(copy `site.example.yaml`; personal accounts stay out of git). `docs/images.md` lists what the
environment must provide, and `heron doctor` checks it end to end on a tiny pipeline.

## Train

Describe the Experiment in a recipe (`training/example.yaml` shows every key) and run:

```bash
heron train training/my_run.yaml --dry-run     # print config.yaml and job.sh, write nothing
heron train training/my_run.yaml               # create the experiment and submit it
heron train --resume my_run_20261023_1430      # continue it from its last checkpoint
```

`heron train` checks the recipe before submitting: the corpus exists with samples in its
train and val parts, the loss is built into pytorch-3dunet, and the patch fits in the corpus
crop and is at least 64 (pytorch-3dunet's minimum; by default the patch is the crop size).

Each Experiment gets `<data root>/experiments/<name>_<YYYYmmdd_HHMM>/`:

| File | Content |
|---|---|
| `recipe.yaml` | copy of the recipe |
| `config.yaml` | pytorch-3dunet config, with host paths (the job mounts them at the same path) |
| `job.sh` | the SLURM job |
| `experiment.yaml` | corpus and its fingerprint, git commit, submitted jobs |
| `checkpoint/` | `best_` / `last_checkpoint.pytorch`, TensorBoard logs |
| `logs/` | SLURM output |

Keep recipes in `training/` under git; the experiment directories stay in the data root.

## Evaluate

`heron train` also submits the evaluation, to run once training succeeds. Otherwise
(training stopped at its time limit, a resumed run, another checkpoint):

```bash
heron evaluate my_run_20261023_1430                       # best checkpoint, val and every test part
heron evaluate my_run_20261023_1430 --part test:reel --checkpoint last
heron evaluate my_run_20261023_1430 --figures             # after the job: draw the figures here
```

The measurement runs in the environment the experiment trained in (recorded in its
`experiment.yaml`), so the model is evaluated by the same pytorch-3dunet that trained it,
with the val transform of its own `config.yaml`. For each
part it writes `evaluation/<part>/` (`report.txt`, `eval_data.h5`) in the experiment, and
`results/<experiment>.yaml` in this repository: recipe, corpus fingerprint, checkpoint and
every part's metrics. Commit that file to keep the results. `--figures` draws, on the host
(matplotlib), `confusion_matrix.png`, `error_histogram.png`, `tolerance_curve.png` and
`regions_<n>.png` from `eval_data.h5`.

The metrics are those of the former `script/evaluate.py`, unchanged: on the same checkpoint
and volumes both give identical reports.

## TensorRT: inference time per precision

How fast the network runs once compiled by TensorRT in fp32, fp16 or int8, against PyTorch
fp32 on the same GPU. Inference time does not depend on the content of a volume, so the
timing uses 1000 random binary volumes (after 50 warmup ones) and reports, per run, the
median, p95 and throughput of two measures:
- **inference**: the network alone, its input already on the GPU;
- **end_to_end**: from a numpy volume to a numpy prediction, including the val transform
  and the copies.

The environment needs TensorRT 10.16 (`requirements-trt.txt`; see `docs/images.md`).

**On the cluster, with quality.** `--precision` adds the TensorRT engines to the evaluation:

```bash
heron evaluate my_run_20261023_1430 --precision fp32 fp16 int8
```

For each engine, the evaluation measures every part as for PyTorch, under
`evaluation/tensorrt_<precision>/<part>/`. `results/<experiment>.yaml` gains `timing`
(every run on this GPU) and `tensorrt` (each engine's metrics, and their `delta` from
PyTorch's).

**On any machine, time only.** First, package the Experiment on the cluster:

```bash
heron export my_run_20261023_1430         # -> <experiment>/package/
```

The package holds:
- the config, the checkpoint and `model.onnx` (fp32, at the Corpus crop size);
- 200 train volumes for int8 calibration, already encoded;
- `reference.h5`, with a few Scenes and the maps PyTorch gives them.

Its `manifest.yaml` (format 1) is the contract with the program that runs the network.
It gives the input shape and axes, the value of a free and of an occupied voxel, the
grid around the base, the output range, and the Robot and Station it was trained for
(`setup`). A robot runs a package with TensorRT alone (see `heron_ros`). For an Experiment
made before Corpora, give what the volumes do not say: `--voxel-size` and `--origin` for
the grid, `--robot` and `--station` for the setup.

Copy the package directory to the other machine (a Jetson, a workstation), then:

```bash
heron benchmark /path/to/package --local                   # fp32 fp16 int8, 1000 volumes
heron benchmark my_run_20261023_1430 --precision fp16      # on the cluster: through SLURM
```

Each machine builds its own engines on first use, which takes minutes, longer on a Jetson.
It keeps them in `<package>/engines/`, one per GPU, TensorRT version and precision; never
copy or commit them. The results go to `results/<experiment>/benchmark_<gpu>.yaml`, tagged
with the GPU, CUDA, TensorRT and torch versions (and the L4T release on a Jetson).

Both PyTorch and the engines run with TF32 off, so fp32 really means fp32. int8 uses
TensorRT's implicit calibration, which TensorRT 11 removed: heron refuses int8 under 11.

## Follow experiments

```bash
heron experiments                       # name, corpus, best validation score, last checkpoint
heron experiments my_run_20261023_1430  # details
tensorboard --logdir <data root>/experiments/my_run_20261023_1430/checkpoint
```

The best score is read from the SLURM logs (`Evaluation score:` lines).

## Older tools

`script/launch_training.py` (interactive launcher), `run_training.sh`,
`script/format_data.py`, `script/evaluate.py`, `script/eval_job.sh` and `custom/losses.py`
are superseded by `heron prepare` /
`heron train` and will be removed. The experiments under `experiments/` and `archive/`
were made with them; their `config.yaml` use `/workspace/...` container paths.
