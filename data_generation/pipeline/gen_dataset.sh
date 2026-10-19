#!/bin/bash
#
# gen_dataset.sh — single entry point for the unified CapaciNet data-generation
# pipeline. Generates REAL (from a rosbag), SYNTHETIC, or SCENE (from a
# user-supplied occupancy-grid HDF5) datasets, for any robot, and can also
# (re)build a robot's master reachability map (create_rm).
#
# Usage:
#   gen_dataset.sh --robot <name> --station <name> --mode <real|synthetic|scene> [options]
#   gen_dataset.sh --robot <name> --mode create_rm [--voxel <m>]
#
# Robots are CapaciNet/robots/<name>.yaml, stations CapaciNet/stations/<name>.yaml
# (list them with: PYTHONPATH=CapaciNet python3 -m heron robots|stations).
#
# Examples:
#   # Real dataset from a rosbag (camera extrinsics resolved from /tf):
#   gen_dataset.sh --robot m1013 --station leeloo --mode real \
#       --rosbag /path/rosbag2_2026_06_02-20_58_42_0.db3 \
#       --camera-frame depth_camera_link --array 100 --dataset-size 10 --out real_long
#
#   # Synthetic dataset for another robot:
#   gen_dataset.sh --robot ur5e --station bare --mode synthetic --array 500 --obj-max 20 --out ur5e_syn
#
#   # (Re)generate a robot's master_ik_data<voxel>.npz:
#   gen_dataset.sh --robot jaco7 --mode create_rm --voxel 0.02
#
#   # Dataset from a scene HDF5 (group/<i>/voxel_grid per scene), one array task per scene:
#   gen_dataset.sh --robot m1013 --station leeloo --mode scene \
#       --scene-file /path/scenes.h5 --array 100 --out scene_batch1
#
#   # Same, with a different HDF5 layout (default is group/{i}/voxel_grid):
#   gen_dataset.sh --robot m1013 --station leeloo --mode scene \
#       --scene-file /path/scenes.h5 --scene-h5-path 'voxel_grid/{i}' --out scene_batch1
#
#   # Override the per-task memory request (GB) if the mode default under/over-shoots:
#   gen_dataset.sh --robot m1013 --station leeloo --mode scene --scene-file /path/scenes.h5 --mem 24 ...
#
set -euo pipefail

# ---- Locate package / project ---------------------------------------------------
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${PIPELINE_DIR}/.." && pwd)"                 # data_generation/
PROJECT_ROOT="$(cd "${PKG_DIR}/../.." && pwd)"             # repo root (contains data/, datasets/, ...)
REPO_ROOT="$(cd "${PKG_DIR}/.." && pwd)"                   # CapaciNet/ (robots/, stations/, heron/)
LAUNCH_DIR="${PKG_DIR}/launch"

# ---- Defaults -------------------------------------------------------------------
ROBOT=""
STATION=""
MODE=""
ROSBAG=""
CAMERA_FRAME=""
SCENE_FILE=""
SCENE_H5_PATH=""   # scene mode: dataset path template, e.g. 'voxel_grid/{i}' (default: group/{i}/voxel_grid)
SCENE_OCCUPIED_VALUE=""   # scene mode: raw value meaning 'occupied' (default in split_scenes.py: -1)
OUT=""
ARRAY="100"
ARRAY_EXPLICIT=0   # scene mode defaults --array to the number of staged scenes
VOXEL="0.02"
DATASET_SIZE="10"
BATCH_SIZE="10000"
INIT_BATCH_SIZE="10000"
OBJ_MAX="20"
REACH_MAX="1.3"
TIME=""
MEM=""    # per-task --mem-per-cpu in GB; empty = mode-dependent default (see below)
SIF="${PROJECT_ROOT}/docker_images/data_generation.sif"
WARP_CACHE="${PROJECT_ROOT}/.cache/warp"
SCRATCH_BASE="${SCRATCH:-/scratch/${USER}}/rosbag"
DRY_RUN=0

usage() { awk 'NR>=2 && /^#/ {sub(/^# ?/,""); print; next} NR>=2 {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- Parse args -----------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --robot)         ROBOT="$2"; shift 2;;
    --station)       STATION="$2"; shift 2;;
    --mode)          MODE="$2"; shift 2;;
    --rosbag)        ROSBAG="$2"; shift 2;;
    --camera-frame)  CAMERA_FRAME="$2"; shift 2;;
    --scene-file)    SCENE_FILE="$2"; shift 2;;
    --scene-h5-path) SCENE_H5_PATH="$2"; shift 2;;
    --scene-occupied-value) SCENE_OCCUPIED_VALUE="$2"; shift 2;;
    --out)           OUT="$2"; shift 2;;
    --array)         ARRAY="$2"; ARRAY_EXPLICIT=1; shift 2;;
    --voxel)         VOXEL="$2"; shift 2;;
    --dataset-size)  DATASET_SIZE="$2"; shift 2;;
    --batch-size)    BATCH_SIZE="$2"; shift 2;;
    --init-batch-size) INIT_BATCH_SIZE="$2"; shift 2;;
    --obj-max)       OBJ_MAX="$2"; shift 2;;
    --reach-max)     REACH_MAX="$2"; shift 2;;
    --time)          TIME="$2"; shift 2;;
    --mem)           MEM="$2"; shift 2;;
    --sif)           SIF="$2"; shift 2;;
    --dry-run)       DRY_RUN=1; shift;;
    -h|--help)       usage 0;;
    *) die "unknown argument: $1 (see --help)";;
  esac
done

# ---- Validate -------------------------------------------------------------------
[ -n "$ROBOT" ] || die "--robot is required"
[ -n "$MODE" ]  || die "--mode is required (real|synthetic|scene|create_rm)"
if [ "$MODE" != "create_rm" ]; then
  [ -n "$STATION" ] || die "--station is required for mode ${MODE} (available: $(ls "${REPO_ROOT}"/stations/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' '))"
fi
[ -f "$SIF" ] || die "container image not found: $SIF (override with --sif)"

# Robot and station facts come from heron (CapaciNet/robots, CapaciNet/stations).
# It validates both files and prints KEY='value' lines; it needs PyYAML, which the
# cluster's python3 has.
heron_args=( shell-env --robot "$ROBOT" --voxel "$VOXEL" )
[ -n "$STATION" ] && heron_args+=( --station "$STATION" )
HERON_ENV="$(PYTHONPATH="${REPO_ROOT}" python3 -m heron "${heron_args[@]}")" || die "invalid robot/station (see heron message above)"
eval "$HERON_ENV"
WORLD_FILE=""
[ -n "$WORLD_PATH" ] && WORLD_FILE="$(basename "$WORLD_PATH")"

cd "$PROJECT_ROOT"
mkdir -p log/output log/error

# Normalize array spec: "100" -> "1-100"; "a-b" kept as-is.
if [[ "$ARRAY" == *-* ]]; then ARRAY_SPEC="$ARRAY"; else ARRAY_SPEC="1-${ARRAY}"; fi

# Per-task memory (GB). job.sbatch used a flat 60G for every mode, which measured
# ~10x actual usage on mode=scene (6.3 GB observed). Default per mode instead, with
# --mem as an escape hatch; synthetic/scene are curobo_ik+generate_data (measured
# low), real additionally buffers rosbag/camera data, create_rm builds the full
# IK database over the whole voxel grid (heaviest, unmeasured -> stays conservative).
if [ -z "$MEM" ]; then
  case "$MODE" in
    scene|synthetic) MEM=16 ;;
    real)             MEM=32 ;;
    create_rm)        MEM=40 ;;
    *)                MEM=16 ;;
  esac
fi

# Common environment forwarded to the job.
COMMON_EXPORT="ALL,DISABLE_DCGM=1,MODE=${MODE},VOXEL=${VOXEL},DATASET_SIZE=${DATASET_SIZE},BATCH_SIZE=${BATCH_SIZE},INIT_BATCH_SIZE=${INIT_BATCH_SIZE},OBJ_MAX=${OBJ_MAX},REACH_MAX=${REACH_MAX},ROBOT_CONFIG_FILE=${ROBOT_CONFIG_FILE},ROBOT_CONFIG_BIND=${ROBOT_CONFIG_BIND},BASE_LINK=${BASE_LINK},WORLD_FILE=${WORLD_FILE},HERON_SCENE_HALF_EXTENT=${SCENE_HALF_EXTENT},CUROBO_PATCH_DIR=${PIPELINE_DIR}/curobo_patch,LAUNCH_DIR=${LAUNCH_DIR},SIF=${SIF},WARP_CACHE=${WARP_CACHE}"

submit() {  # submit <data_dir> <extra_export> <sbatch_opts...>
  local data_dir="$1"; shift
  local extra="$1"; shift
  local export_str="${COMMON_EXPORT},DATA_DIR=${data_dir}${extra:+,${extra}}"
  local cmd=( sbatch "$@" --export="${export_str}" "${PIPELINE_DIR}/job.sbatch" )
  echo "+ ${cmd[*]}"
  if [ "$DRY_RUN" -eq 1 ]; then echo "(dry-run: not submitted)"; else "${cmd[@]}"; fi
}

# Run a filesystem-mutating command, or just describe it under --dry-run.
fs() { if [ "$DRY_RUN" -eq 1 ]; then echo "(dry-run) would: $*"; else "$@"; fi; }

case "$MODE" in
  create_rm)
    DATA_DIR="${MASTER_MAP_NEW_DIR}"
    fs mkdir -p "$DATA_DIR"
    fs cp "${PIPELINE_DIR}/run_generation.sh" "${DATA_DIR}/run_generation.sh"
    submit "$DATA_DIR" "" \
      --job-name="capa_rm_${ROBOT}" \
      --output="log/output/capa_rm_${ROBOT}_%j.txt" \
      --error="log/error/capa_rm_${ROBOT}_%j.txt" \
      --time="${TIME:-24:00:00}" \
      --mem-per-cpu="${MEM}G"
    ;;

  synthetic|real)
    # ---- Validate everything BEFORE touching the filesystem ----
    [ -n "$OUT" ] || die "--out <name> is required for mode ${MODE}"
    DATA_DIR="${PROJECT_ROOT}/datasets/${OUT}"
    local_npz="${MASTER_MAP_DIR}/master_ik_data${VOXEL}.npz"
    [ -f "$local_npz" ] || die "missing ${local_npz}. Run: gen_dataset.sh --robot ${ROBOT} --mode create_rm --voxel ${VOXEL}"
    if [ "$MODE" = "real" ]; then
      [ -n "$CAMERAS_TEMPLATE" ] || die "station ${STATION} has no cameras template (needed for mode real)"
      [ -n "$ROSBAG" ] || die "--rosbag is required for mode real"
      [ -z "$CAMERA_FRAME" ] && CAMERA_FRAME="$DEFAULT_CAMERA_FRAME"
      [ -n "$CAMERA_FRAME" ] || die "--camera-frame is required for mode real (station ${STATION} has no camera_frame)"
      if [ -d "$ROSBAG" ]; then bag_src="$ROSBAG"; else bag_src="$(dirname "$ROSBAG")"; fi
      bag_src="$(cd "$bag_src" && pwd)"
      [ -f "${bag_src}/metadata.yaml" ] || die "no metadata.yaml in ${bag_src} (ros2 bag play needs the bag directory)"
      bag_name="$(basename "$bag_src")"
      ROSBAG_DIR="${SCRATCH_BASE}/${bag_name}"
    fi

    # ---- Stage assets ----
    fs mkdir -p "$DATA_DIR"
    fs cp "${PIPELINE_DIR}/run_generation.sh" "${DATA_DIR}/run_generation.sh"
    # Hardlink the robot's master reachability map (instant, no extra disk).
    fs ln -f "$local_npz" "${DATA_DIR}/master_ik_data${VOXEL}.npz"
    [ -f "${MASTER_MAP_DIR}/master_ik_data${VOXEL}.h5" ] && \
      fs ln -f "${MASTER_MAP_DIR}/master_ik_data${VOXEL}.h5" "${DATA_DIR}/master_ik_data${VOXEL}.h5"
    [ -n "$WORLD_FILE" ] && fs cp "${WORLD_PATH}" "${DATA_DIR}/${WORLD_FILE}"

    if [ "$MODE" = "real" ]; then
      # Render the TF cameras config now (inject the chosen frame), so the staged
      # file is explicit and the same for every array task.
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "(dry-run) would: render ${DATA_DIR}/cameras.yaml with frame_id=${CAMERA_FRAME}"
      else
        sed "s|__CAMERA_FRAME__|${CAMERA_FRAME}|g" "${CAMERAS_TEMPLATE}" > "${DATA_DIR}/cameras.yaml"
      fi
      # Stage rosbag to scratch (fast) once; array tasks copy from there to local SSD.
      if [ "$bag_src" != "$ROSBAG_DIR" ] && [ ! -d "$ROSBAG_DIR" ]; then
        echo "Staging rosbag to scratch: ${bag_src} -> ${ROSBAG_DIR}"
        fs mkdir -p "$(dirname "$ROSBAG_DIR")"
        fs cp -r "$bag_src" "$ROSBAG_DIR"
      fi

      submit "$DATA_DIR" "CAMERA_FRAME=${CAMERA_FRAME},ROSBAG_DIR=${ROSBAG_DIR}" \
        --array="${ARRAY_SPEC}" \
        --job-name="capa_real_${OUT}" \
        --output="log/output/capa_real_${OUT}_%A_%a.txt" \
        --error="log/error/capa_real_${OUT}_%A_%a.txt" \
        --time="${TIME:-16:00:00}" \
        --mem-per-cpu="${MEM}G"
    else
      submit "$DATA_DIR" "" \
        --array="${ARRAY_SPEC}" \
        --job-name="capa_syn_${OUT}" \
        --output="log/output/capa_syn_${OUT}_%A_%a.txt" \
        --error="log/error/capa_syn_${OUT}_%A_%a.txt" \
        --time="${TIME:-2:00:00}" \
        --mem-per-cpu="${MEM}G"
    fi
    ;;

  scene)
    # ---- Validate everything BEFORE touching the filesystem ----
    [ -n "$OUT" ] || die "--out <name> is required for mode scene"
    [ -n "$SCENE_FILE" ] || die "--scene-file <path> is required for mode scene"
    [ -f "$SCENE_FILE" ] || die "scene file not found: $SCENE_FILE"
    DATA_DIR="${PROJECT_ROOT}/datasets/${OUT}"
    local_npz="${MASTER_MAP_DIR}/master_ik_data${VOXEL}.npz"
    [ -f "$local_npz" ] || die "missing ${local_npz}. Run: gen_dataset.sh --robot ${ROBOT} --mode create_rm --voxel ${VOXEL}"

    # ---- Stage assets ----
    fs mkdir -p "$DATA_DIR"
    fs cp "${PIPELINE_DIR}/run_generation.sh" "${DATA_DIR}/run_generation.sh"
    fs cp "${PIPELINE_DIR}/load_scene.py" "${DATA_DIR}/load_scene.py"
    # Hardlink the robot's master reachability map and the scene file (instant, no
    # extra disk -- scenes.h5 can be large and is read-only for the whole array).
    fs ln -f "$local_npz" "${DATA_DIR}/master_ik_data${VOXEL}.npz"
    [ -f "${MASTER_MAP_DIR}/master_ik_data${VOXEL}.h5" ] && \
      fs ln -f "${MASTER_MAP_DIR}/master_ik_data${VOXEL}.h5" "${DATA_DIR}/master_ik_data${VOXEL}.h5"
    [ -n "$WORLD_FILE" ] && fs cp "${WORLD_PATH}" "${DATA_DIR}/${WORLD_FILE}"

    if [ ! -d "${PIPELINE_DIR}/curobo_patch" ]; then
      die "missing ${PIPELINE_DIR}/curobo_patch (native ESDF curobo_ros patch) -- required for mode scene"
    fi

    # Split the shared scene HDF5 into one .npz per scene, HERE on the host: the
    # container has no h5py (and runs with --network=none, so nothing can be
    # installed at job time). Each array task then reads only its own small file,
    # so no two tasks ever open the same file concurrently.
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "(dry-run) would: split ${SCENE_FILE} into ${DATA_DIR}/scenes/scene_<i>.npz"
      n_scenes="<n>"
    else
      mkdir -p "${DATA_DIR}/scenes"
      split_args=( --scene-file "$SCENE_FILE" --out-dir "${DATA_DIR}/scenes" )
      [ -n "$SCENE_H5_PATH" ] && split_args+=( --h5-path "$SCENE_H5_PATH" )
      [ -n "$SCENE_OCCUPIED_VALUE" ] && split_args+=( --occupied-value "$SCENE_OCCUPIED_VALUE" )
      n_scenes="$(python3 "${PIPELINE_DIR}/split_scenes.py" "${split_args[@]}")" \
        || die "split_scenes.py failed (needs h5py: activate the capacinet conda env)"
      echo "Staged ${n_scenes} scene(s) into ${DATA_DIR}/scenes/"
      # One array task per scene, unless the caller asked for a specific --array
      # (e.g. --array 1 to smoke-test a single scene before the full run).
      [ "$ARRAY_EXPLICIT" -eq 1 ] || ARRAY_SPEC="1-${n_scenes}"
    fi

    submit "$DATA_DIR" "" \
      --array="${ARRAY_SPEC}" \
      --job-name="capa_scene_${OUT}" \
      --output="log/output/capa_scene_${OUT}_%A_%a.txt" \
      --error="log/error/capa_scene_${OUT}_%A_%a.txt" \
      `# One scene = one RM = ~4000 IK batches ~= 1.5 h measured at voxel 0.02 /` \
      `# batch 10000, plus ~2 min of curobo startup. Override with --time.` \
      --time="${TIME:-4:00:00}" \
      --mem-per-cpu="${MEM}G"
    ;;

  *) die "invalid --mode '$MODE' (expected real|synthetic|scene|create_rm)";;
esac

echo "Done. Output dir: ${DATA_DIR}"
