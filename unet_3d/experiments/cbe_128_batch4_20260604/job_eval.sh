#!/bin/bash
#SBATCH --job-name=eval_cbe_128_batch4_20260604
#SBATCH --output=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/logs/slurm_eval_%j.out
#SBATCH --error=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/logs/slurm_eval_%j.err
#SBATCH --time=0:10:00
#SBATCH --mem=10G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL,DISABLE_DCGM=1

set -e

module load python/3.11.5
source /home/willore/envs/eval_env/bin/activate

python /lustre06/project/6089348/willore/CapaciNet/unet_3d/script/evaluate.py \
    --config     /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/config.yaml \
    --checkpoint /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/checkpoint/best_checkpoint.pytorch \
    --output_dir /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/evaluation \
    --val_dir    /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/data_eval
deactivate
echo "==> Evaluation complete: cbe_128_batch4_20260604"
echo "    Results: /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/cbe_128_batch4_20260604/evaluation/"
