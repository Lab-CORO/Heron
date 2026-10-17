#!/bin/bash
#SBATCH --account=def-jerobg
#SBATCH --job-name=eval_cbe_128_batch4_real
#SBATCH --output=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_real/logs/slurm_eval_%j.out
#SBATCH --error=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_real/logs/slurm_eval_%j.err
#SBATCH --time=0:10:00
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL,DISABLE_DCGM=1

set -e

module load python/3.11.5
source /home/willore/envs/eval_env/bin/activate

python /lustre06/project/6089348/willore/CapaciNet/unet_3d/script/evaluate.py \
    --config     /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_real/config.yaml \
    --checkpoint /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_real/checkpoint/best_checkpoint.pytorch \
    --output_dir /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_real/evaluation \
    --val_dir    /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/128_002_real_data_20260528_1134/data/val

deactivate
echo "==> Evaluation complete: cbe_128_batch4_real"
echo "    Results: /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_real/evaluation/"
