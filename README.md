# Heron

Heron predicts, for a robot in a cluttered workspace, which points it can reach. It generates
reachability datasets for a robot and a station, then trains and evaluates a 3D U-Net on them,
and measures how fast the network runs once compiled with TensorRT.

The repository is still named `CapaciNet` (the project's first name); the Python package and
the command line are `heron`. The terms used everywhere below (Robot, Station, Dataset, Source,
Corpus, Experiment, Package, Engine…) are defined in [GLOSSARY.md](GLOSSARY.md).

## Pipeline

| Step | Command | Runs | Output |
|---|---|---|---|
| Describe the setup | `robots/<name>.yaml`, `stations/<name>.yaml` | – | a Robot, a Station |
| Generate a Dataset | `data_generation/pipeline/gen_dataset.sh` | SLURM GPU jobs, in the generation image | `<data root>/datasets/<name>/` |
| Build a Corpus | `heron prepare <recipe>` | here, or a CPU job (`--submit`) | `<data root>/corpora/<name>/` |
| Train | `heron train <recipe>` | a GPU job (SLURM or `--local`) | `<data root>/experiments/<name>_<date>/` |
| Evaluate | `heron evaluate <experiment>` | a GPU job, chained after training | `results/<experiment>.yaml` |
| Time with TensorRT | `heron export`, `heron benchmark` | any machine with a GPU | `results/<experiment>/benchmark_<gpu>.yaml` |

Each step only reads what the previous one wrote, and records where its input came from
(`dataset.yaml`, `corpus.yaml`, `experiment.yaml`), so a result traces back to its data.

## Layout

```
robots/  stations/       one YAML per Robot and per Station
corpus/  training/       recipes: example.yaml shows every key
heron/                   the Python package and the `heron` command
data_generation/         ROS 2 C++ package (IK over a voxel grid -> HDF5) and its SLURM pipeline
images/                  container build files (docs/images.md: what each environment must provide)
results/                 evaluation and benchmark summaries, committed
tests/                   pytest, no GPU or cluster needed
docs/                    training guide, environments, ADRs, prompts for building the images
site.example.yaml        copy to site.yaml: accounts, modules, images, data root (not versioned)
```

Generated data lives outside the repository, under the data root: `HERON_DATA_ROOT`, else
`data_root` in `site.yaml`, else the directory that contains the repository.

## Install

Heron itself needs Python 3.10+ with PyYAML, numpy, scipy, h5py and matplotlib. Training,
evaluation and TensorRT need more, in a venv or an Apptainer image ([docs/images.md](docs/images.md)).

On an Alliance cluster (Narval, Rorqual…), in a venv:

```bash
module load python/3.11.5
python -m venv ~/envs/heron && source ~/envs/heron/bin/activate
pip install --no-index torch torchvision tensorboard tqdm scikit-image   # Alliance wheels
pip install -e '.[train,test]'                                           # pytorch3dunet 1.9.6 from PyPI
# optional, for heron export / benchmark / evaluate --precision: see requirements-trt.txt
cp site.example.yaml site.yaml    # then fill in your SLURM accounts and paths
heron doctor                      # tiny end-to-end run on a GPU, reports OK/KO per step
```

On a workstation or a Jetson, without SLURM: set `scheduler: local` in `site.yaml`, or pass
`--local` to any command that runs a job.

## Quick start

```bash
heron robots                      # m1013, ur5e, jaco7, iiwa
heron stations leeloo             # its scene grid and 128³ crop at 0.02 m

# 1. Dataset: 4 tasks of 2 scenes (about 1 h 30 per scene on an A100)
bash data_generation/pipeline/gen_dataset.sh --robot m1013 --station leeloo \
    --mode synthetic --array 4 --dataset-size 2 --out my_dataset --dry-run   # then without --dry-run

# 2. Corpus: crop, split by Source, augment the train part (copy corpus/example.yaml)
heron prepare corpus/my_corpus.yaml

# 3. Experiment: training, then evaluation on val and every test part
heron train training/my_run.yaml
heron experiments                  # progress and best validation score
heron evaluate my_run_<date> --figures

# 4. TensorRT (optional)
heron export my_run_<date>
heron benchmark my_run_<date> --precision fp32 fp16 int8
```

Details: [data_generation/pipeline/README.md](data_generation/pipeline/README.md) (generation
modes and options), [docs/training.md](docs/training.md) (training, evaluation, TensorRT),
[results/README.md](results/README.md) (what the result files hold).

## Status

| Part | State |
|---|---|
| Robot, Station, Corpus, training, evaluation | tested on Narval A100 (SLURM and `--local`), 106 pytest tests |
| TensorRT export and benchmark | tested on A100 (TensorRT 10.16); Jetson Orin (JetPack 7.2) not yet measured |
| Generation | runs in the ROS 2 Humble + cuRobo v1 image; moving to ROS 2 Jazzy + cuRobo v2 ([docs/prompts/generation-image.md](docs/prompts/generation-image.md)) |
| Generation, `scene` mode | old image only: not yet ported to the Jazzy image |
| Training image (`images/training/Dockerfile`) | never built; clusters use the venv |

## Contributing

Issues and pull requests are welcome on [Lab-CORO/CapaciNet](https://github.com/Lab-CORO/CapaciNet).
Commits follow [Conventional Commits](https://www.conventionalcommits.org). Run
`python -m pytest -q` before submitting. Never commit `site.yaml` or a personal SLURM account.

## License

MIT, see [LICENSE](LICENSE).
