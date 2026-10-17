#!/bin/bash
#SBATCH --account=def-jerobg
#SBATCH --job-name=data_002_scratch_128_20260518_1315
#SBATCH --output=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/data_002_scratch_128_20260518_1315/logs/slurm_%j.out
#SBATCH --error=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/data_002_scratch_128_20260518_1315/logs/slurm_%j.err
#SBATCH --time=30:00:00
#SBATCH --mem=500G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL,DISABLE_DCGM=1

set -e

cd /lustre06/project/6089348/willore

# ── Train ─────────────────────────────────────────────────────────────────────
module load apptainer

apptainer exec -C --nv \
    --bind ./CapaciNet/:/workspace/ \
    /lustre06/project/6089348/willore/capa_unet.sif \
    train3dunet --config /workspace/unet_3d/experiments/data_002_scratch_128_20260518_1315/config.yaml

echo "==> Training complete: data_002_scratch_128_20260518_1315"
