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
