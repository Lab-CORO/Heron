# Training and evaluation

Training runs pytorch-3dunet 1.9.1 (`train3dunet`) inside `capa_unet.sif`, on a Corpus
built with `heron prepare` (see `data_generation/pipeline/README.md` and `GLOSSARY.md`).

## Train

Describe the Experiment in a recipe (`training/example.yaml` shows every key) and run:

```bash
heron train training/my_run.yaml --dry-run     # print config.yaml and job.sh, write nothing
heron train training/my_run.yaml               # create the experiment and submit it
heron train --resume my_run_20261023_1430      # continue it from its last checkpoint
```

`heron train` checks the recipe before submitting: the corpus exists with train and val
parts, each part holds at least one full batch (pytorch-3dunet drops incomplete batches,
so a smaller val part would validate nothing), the loss is built into pytorch-3dunet,
and the patch fits in the corpus crop (by default the patch is the crop size).

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

## Follow experiments

```bash
heron experiments                       # name, corpus, best validation score, last checkpoint
heron experiments my_run_20261023_1430  # details
tensorboard --logdir <data root>/experiments/my_run_20261023_1430/checkpoint
```

The best score is read from the SLURM logs (`Evaluation score:` lines).

## Older tools

`script/launch_training.py` (interactive launcher), `run_training.sh`,
`script/format_data.py` and `custom/losses.py` are superseded by `heron prepare` /
`heron train` and will be removed. The experiments under `experiments/` and `archive/`
were made with them; their `config.yaml` use `/workspace/...` container paths.
