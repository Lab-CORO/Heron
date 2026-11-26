#!/bin/bash
#SBATCH --job-name=cbe_64patch_20260602
#SBATCH --output=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_64patch_20260602/logs/slurm_%j.out
#SBATCH --error=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_64patch_20260602/logs/slurm_%j.err
#SBATCH --time=30:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL,DISABLE_DCGM=1

set -e

mkdir -p /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_64patch_20260602/logs
mkdir -p /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_64patch_20260602/checkpoint

cd /lustre06/project/6089348/willore

module load apptainer

apptainer exec -C --nv \
    --bind ./CapaciNet/:/workspace/ \
    --bind /lustre06/project/6089348/willore/CapaciNet/unet_3d/custom/losses.py:/opt/conda/lib/python3.10/site-packages/pytorch3dunet/unet3d/losses.py \
    /lustre06/project/6089348/willore/capa_unet.sif \
    train3dunet --config /workspace/unet_3d/experiments/cbe_64patch_20260602/config.yaml

echo "==> Training complete: cbe_64patch_20260602"
