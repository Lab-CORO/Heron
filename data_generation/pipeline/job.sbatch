#!/bin/bash
#SBATCH --account=def-jerobg
#SBATCH --job-name=capa_gen
#SBATCH --output=log/output/capa_gen_%A_%a.txt
#SBATCH --error=log/error/capa_gen_%A_%a.txt
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL,DISABLE_DCGM=1
#
# Generic SLURM job for the unified data-generation pipeline.
# Submitted by gen_dataset.sh, which overrides --array / --time / --job-name /
# --mem-per-cpu (mode-dependent default, see --mem in gen_dataset.sh) and exports
# all parameters. Do NOT run directly; use gen_dataset.sh.
#
# Expected environment (exported by gen_dataset.sh):
#   MODE, VOXEL, DATASET_SIZE, BATCH_SIZE, INIT_BATCH_SIZE, OBJ_MAX, REACH_MAX,
#   ROBOT_CONFIG_FILE, BASE_LINK, WORLD_FILE, CAMERA_FRAME
#   HERON_SCENE_HALF_EXTENT  station scene half extent (m), read by the patched obstacle_manager.py
#   ROBOT_CONFIG_BIND  host dir of a robot config that is not built into the image (may be empty)
#   DATA_DIR         host dir bound to .../share/data_generation/data (assets + output)
#   LAUNCH_DIR       host dir bound to .../share/data_generation/launch (parametrized launches)
#   SIF              path to data_generation.sif
#   WARP_CACHE       host warp cache dir
#   ROSBAG_DIR       (real only) host rosbag directory to stage to local SSD
#   CUROBO_PATCH_DIR host dir with obstacle_manager.py / ros_service_manager.py, bind-mounted
#                    over curobo_ros's installed copies without rebuilding the .sif.
#                    See pipeline/curobo_patch/README.md.

module load apptainer
module load hdf5/1.14.2

mkdir -p log/output log/error

CONTAINER_DATA=/home/ros2_ws/install/data_generation/share/data_generation/data
CONTAINER_LAUNCH=/home/ros2_ws/install/data_generation/share/data_generation/launch

# generate_data names its output HDF5 purely from the wall-clock second at node
# construction (data_generator.cpp:38-43) and opens it with HighFive::File::Create
# (= O_CREAT|O_EXCL). Every array task used to bind the SAME host DATA_DIR here, so
# tasks starting in the same second all raced for one path: the first won, the rest
# died on EEXIST/SIGABRT (18 tasks collided on a single name in job 66848714, and
# only 8 of 59 tasks produced usable output). Rather than change the name the C++
# picks -- the binary lives inside the .sif and would need an image rebuild -- give
# each task a PRIVATE directory, so the timestamp name can no longer collide.
#
# The shared read-only assets are hardlinked in (same filesystem => no extra disk,
# no copy of the 2.2 GB master_ik_data). The task's output is moved back to DATA_DIR
# under a deterministic name after the container exits, and the directory removed --
# unlinking hardlinks never touches the originals.
#
# This applies to every array mode (scene, synthetic, real): they all run the same
# generate_data node into the same shared dir, so they all had the collision.
# create_rm is exempt -- it is a single job whose OUTPUT is master_ik_data*.npz in
# DATA_DIR itself, so it keeps the direct bind.
DATA_BIND_SRC="${DATA_DIR}"

if [ "${MODE}" != "create_rm" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  TASK_DIR="${DATA_DIR}/.tasks/task_${TASK_ID}"

  rm -rf "${TASK_DIR}"                      # idempotent if SLURM requeues the task
  mkdir -p "${TASK_DIR}"

  for f in run_generation.sh load_scene.py cameras.yaml "${WORLD_FILE}" \
           "master_ik_data${VOXEL}.npz" "master_ik_data${VOXEL}.h5"; do
    [ -n "${f}" ] && [ -f "${DATA_DIR}/${f}" ] && ln -f "${DATA_DIR}/${f}" "${TASK_DIR}/${f}"
  done

  if [ "${MODE}" = "scene" ]; then
    SCENE_INDEX=$(( TASK_ID - 1 ))
    mkdir -p "${TASK_DIR}/scenes"
    SCENE_SRC="${DATA_DIR}/scenes/scene_${SCENE_INDEX}.npz"
    [ -f "${SCENE_SRC}" ] || { echo "ERROR: ${SCENE_SRC} not found (bad --array size vs. number of staged scenes?)" >&2; exit 1; }
    ln -f "${SCENE_SRC}" "${TASK_DIR}/scenes/scene_${SCENE_INDEX}.npz"
  fi

  DATA_BIND_SRC="${TASK_DIR}"
fi

BINDS=(
  --bind "${DATA_BIND_SRC}":"${CONTAINER_DATA}"
  --bind "${LAUNCH_DIR}":"${CONTAINER_LAUNCH}"
  --bind "${SLURM_TMPDIR}":/tmp
  --bind "${WARP_CACHE}":/home/willore/.cache/warp
)

if [ "${MODE}" = "real" ]; then
  echo "Staging rosbag ${ROSBAG_DIR} -> ${SLURM_TMPDIR}/rosbag (local SSD)..."
  cp -r "${ROSBAG_DIR}" "${SLURM_TMPDIR}/rosbag"
  BINDS+=( --bind "${SLURM_TMPDIR}/rosbag":/home/ros2_ws/rosbag )
fi

# curobo_ros files patched without rebuilding the image (see curobo_patch/README.md).
# obstacle_manager.py reads the station's scene extent (HERON_SCENE_HALF_EXTENT), so
# every mode needs it; ros_service_manager.py only adds the scene-mode service.
CUROBO_INSTALL=/home/ros2_ws/install/curobo_ros/lib/python3.10/site-packages/curobo_ros/core
PATCHED=( obstacle_manager.py )
[ "${MODE}" = "scene" ] && PATCHED+=( ros_service_manager.py )
for f in "${PATCHED[@]}"; do
  [ -f "${CUROBO_PATCH_DIR}/${f}" ] || { echo "ERROR: missing ${CUROBO_PATCH_DIR}/${f}" >&2; exit 1; }
  BINDS+=( --bind "${CUROBO_PATCH_DIR}/${f}":"${CUROBO_INSTALL}/${f}" )
done
# A robot whose curobo config lives on the host: mount its directory (config, URDF,
# meshes) at the same path, so absolute paths inside the config keep working.
if [ -n "${ROBOT_CONFIG_BIND:-}" ]; then
  BINDS+=( --bind "${ROBOT_CONFIG_BIND}":"${ROBOT_CONFIG_BIND}":ro )
fi

# apptainer is run with -C (--contain), which ISOLATES the environment: host env
# vars are NOT visible inside the container. Forward the parameters explicitly by
# exporting them with the APPTAINERENV_ prefix (apptainer strips the prefix inside).
for v in MODE VOXEL DATASET_SIZE BATCH_SIZE INIT_BATCH_SIZE OBJ_MAX REACH_MAX \
         ROBOT_CONFIG_FILE BASE_LINK WORLD_FILE CAMERA_FRAME SLURM_ARRAY_TASK_ID \
         HERON_SCENE_HALF_EXTENT; do
  export "APPTAINERENV_${v}=${!v-}"
done

echo "Launching apptainer (MODE=${MODE}, task=${SLURM_ARRAY_TASK_ID:-n/a})..."
apptainer exec -C --net --network=none --nv \
  "${BINDS[@]}" \
  "${SIF}" \
  bash "${CONTAINER_DATA}/run_generation.sh"
STATUS=$?

# Collect the task's output and drop its private directory. Every .h5 here belongs to
# this task (it owned the whole directory), so unlike the old `ls -t | head -1` over a
# shared dir this cannot pick up another task's file. Scene mode gets a name keyed on
# the scene it processed; the other modes keep generate_data's timestamp name with the
# task id appended, which is enough to make it unique.
if [ -n "${TASK_DIR:-}" ]; then
  n_out=0
  for out in "${TASK_DIR}"/*.h5; do
    [ -e "${out}" ] || continue
    case "$(basename "${out}")" in master_ik_data*) continue;; esac
    if [ "${MODE}" = "scene" ]; then
      dest="${DATA_DIR}/scene_${SCENE_INDEX}_task${TASK_ID}.h5"
    else
      dest="${DATA_DIR}/$(basename "${out}" .h5)_task${TASK_ID}.h5"
    fi
    [ "${n_out}" -gt 0 ] && dest="${dest%.h5}_${n_out}.h5"
    mv "${out}" "${dest}"
    echo "Output: ${dest}"
    n_out=$((n_out + 1))
  done
  if [ "${n_out}" -eq 0 ]; then
    echo "WARNING: task ${TASK_ID} produced no output .h5" >&2
    [ "${STATUS}" -eq 0 ] && STATUS=1
  fi
  rm -rf "${TASK_DIR}"                      # hardlinks only; originals in DATA_DIR untouched
  rmdir "${DATA_DIR}/.tasks" 2>/dev/null || true
fi

echo "Job task ${SLURM_ARRAY_TASK_ID:-n/a} completed (MODE=${MODE}, status=${STATUS})."
exit "${STATUS}"
