#!/bin/bash
#SBATCH --job-name=eval_CBR_real_dataset_20260623_1108
#SBATCH --output=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/CBR_real_dataset_20260623_1108/logs/slurm_eval_%j.out
#SBATCH --error=/lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/CBR_real_dataset_20260623_1108/logs/slurm_eval_%j.err
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL,DISABLE_DCGM=1

set -e

module load python/3.11.5
source /home/willore/envs/eval_env/bin/activate

python /lustre06/project/6089348/willore/CapaciNet/unet_3d/script/evaluate.py \
    --config     /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/CBR_real_dataset_20260623_1108/config.yaml \
    --checkpoint /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/CBR_real_dataset_20260623_1108/checkpoint/best_checkpoint.pytorch \
    --output_dir /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/CBR_real_dataset_20260623_1108/evaluation \
    --val_dir    /lustre06/project/6089348/willore/CapaciNet/unet_3d/data_eval/data_reel_more_diverse/

deactivate
echo "==> Evaluation complete: CBR_real_dataset_20260623_1108"
echo "    Results: /lustre06/project/6089348/willore/CapaciNet/unet_3d/experiments/CBR_real_dataset_20260623_1108/evaluation/"
