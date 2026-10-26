#!/bin/bash
#SBATCH --job-name=verify_voxel_esdf
#SBATCH --output=log/output/verify_voxel_esdf_%j.txt
#SBATCH --error=log/error/verify_voxel_esdf_%j.txt
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#
# One-off check: does curobo.geom.types.VoxelGrid (native ESDF obstacle) get
# enforced by WorldVoxelCollision when populated from an externally-supplied
# array, instead of via depth-camera integration? See
# CapaciNet/data_generation/pipeline/verify_voxel_esdf.py for the actual test.
# Not part of the data-generation pipeline itself -- delete once answered.

module load apptainer

PROJECT_ROOT="/lustre06/project/6089348/willore"
SIF="${PROJECT_ROOT}/docker_images/data_generation.sif"

mkdir -p log/output log/error

apptainer exec -C --net --network=none --nv \
  --bind "${PROJECT_ROOT}/CapaciNet":/mnt/CapaciNet \
  --bind "${PROJECT_ROOT}/.cache/warp":/home/willore/.cache/warp \
  "${SIF}" \
  python3 /mnt/CapaciNet/data_generation/pipeline/verify_voxel_esdf.py
