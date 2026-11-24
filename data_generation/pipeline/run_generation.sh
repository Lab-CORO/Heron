#!/bin/bash
#
# run_generation.sh — in-container launcher for the unified data-generation pipeline.
#
# Runs INSIDE the data_generation.sif container (invoked by job.sbatch). All inputs
# are passed as environment variables (via `sbatch --export`). It selects the right
# ros2 launch based on $MODE and forwards the robot / dataset parameters.
#
# A copy of this script (plus cameras.tf.yaml, the world file and a hardlink of the
# robot master_ik_data*.npz) lives in the "data" directory that gets bind-mounted to
# /home/ros2_ws/install/data_generation/share/data_generation/data — which is also
# where data_generator.cpp writes the output HDF5 files.

set -e

# The ROS 2 Jazzy + cuRobo v2 image runs every mode but scene; the Humble image is only
# used for scene (gen_dataset.sh picks, from the image's heron.ros label).
if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   # ROS_LOCALHOST_ONLY is deprecated in Jazzy
else
  source /opt/ros/humble/setup.bash
  export ROS_LOCALHOST_ONLY=1
fi
source /home/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

DATA_SHARE=/home/ros2_ws/install/data_generation/share/data_generation/data

# ---- Parameters (with defaults) -------------------------------------------------
MODE="${MODE:-synthetic}"
VOXEL="${VOXEL:-0.02}"
DATASET_SIZE="${DATASET_SIZE:-10}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
INIT_BATCH_SIZE="${INIT_BATCH_SIZE:-10000}"
OBJ_MAX="${OBJ_MAX:-20}"
REACH_MAX="${REACH_MAX:-1.3}"
ROBOT_CONFIG_FILE="${ROBOT_CONFIG_FILE:-}"
BASE_LINK="${BASE_LINK:-base_0}"
WORLD_FILE="${WORLD_FILE:-}"      # basename in DATA_SHARE, or empty for curobo default
CAMERA_FRAME="${CAMERA_FRAME:-}"

# Optional launch args are only forwarded when set, so empty values fall back to the
# launch / curobo defaults instead of being passed as an empty string.
ROBOT_ARG=""
[ -n "$ROBOT_CONFIG_FILE" ] && ROBOT_ARG="robot_config_file:=$ROBOT_CONFIG_FILE"
WORLD_ARG=""
[ -n "$WORLD_FILE" ] && WORLD_ARG="world_file:=$DATA_SHARE/$WORLD_FILE"

echo "================================================"
echo " run_generation.sh"
echo "   MODE=$MODE  VOXEL=$VOXEL  DATASET_SIZE=$DATASET_SIZE"
echo "   ROBOT_CONFIG_FILE=${ROBOT_CONFIG_FILE:-<default>}  BASE_LINK=$BASE_LINK"
echo "   WORLD_FILE=${WORLD_FILE:-<default>}  OBJ_MAX=$OBJ_MAX"
echo "   ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-n/a}"
echo "================================================"

case "$MODE" in
  synthetic)
    ros2 launch data_generation generate_data.launch.py \
      voxel_size:=$VOXEL dataset_size:=$DATASET_SIZE batch_size:=$BATCH_SIZE \
      init_batch_size:=$INIT_BATCH_SIZE obj_max:=$OBJ_MAX reach_max:=$REACH_MAX \
      base_link:=$BASE_LINK $ROBOT_ARG $WORLD_ARG
    ;;

  real)
    # cameras.yaml was rendered at staging time (frame_id injected, extrinsics empty
    # so curobo resolves the camera pose from /tf: base_link -> frame).
    CAM_CFG="$DATA_SHARE/cameras.yaml"
    [ -f "$CAM_CFG" ] || { echo "ERROR: missing rendered cameras config $CAM_CFG" >&2; exit 1; }
    echo "Using cameras config: $CAM_CFG"

    # Diversify the start point in the looped bag so each array task captures a
    # different moment of the recording.
    sleep $((RANDOM % 160))

    BAG_DIR=/home/ros2_ws/rosbag   # bind-mounted bag directory (metadata.yaml + .db3)

    ros2 launch data_generation generate_real_data.launch.py \
      voxel_size:=$VOXEL dataset_size:=$DATASET_SIZE batch_size:=$BATCH_SIZE \
      init_batch_size:=$INIT_BATCH_SIZE obj_max:=0 reach_max:=$REACH_MAX \
      use_rosbag_mode:=true scene_stabilization_max_delay_sec:=100.0 \
      base_link:=$BASE_LINK $ROBOT_ARG $WORLD_ARG \
      cameras_config_file:=$CAM_CFG bag_file:=$BAG_DIR
    ;;

  create_rm)
    # (Re)generate the robot's master_ik_data<voxel>.npz into DATA_SHARE.
    ros2 launch data_generation create_reachability_map.launch.py \
      voxel_size:=$VOXEL batch_size:=$BATCH_SIZE init_batch_size:=$INIT_BATCH_SIZE \
      base_link:=$BASE_LINK $ROBOT_ARG
    ;;

  scene)
    # Obstacles come from a scene the user already has as a dense occupancy grid,
    # not from obstacle_adder's random cubes or a rosbag -- see
    # pipeline/load_scene.py. Only curobo_ik + generate_data are launched
    # (generate_from_scene.launch.py): obstacle_adder/scene_manager would fight
    # load_scene.py for control of the world, so neither is started.
    #
    # The scene file itself was already split into one .npz per scene by
    # pipeline/split_scenes.py at staging time (gen_dataset.sh), into
    # DATA_SHARE/scenes/ -- each array task reads only its own small file, so
    # there's no shared HDF5 file for concurrent array tasks to contend over.
    #
    # DATA_SHARE is this task's PRIVATE directory (job.sbatch binds
    # DATA_DIR/.tasks/task_<N> here, not the shared DATA_DIR), so generate_data's
    # wall-clock-second output name cannot collide with another task's, and
    # job.sbatch renames the single .h5 left behind once the container exits.
    SCENE_INDEX=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
    SCENE_NPZ="$DATA_SHARE/scenes/scene_${SCENE_INDEX}.npz"
    [ -f "$SCENE_NPZ" ] || { echo "ERROR: $SCENE_NPZ not found (bad --array size vs. number of staged scenes?)" >&2; exit 1; }
    export VOXEL_SCENE_PATH="$DATA_SHARE/voxel_scene_task${SLURM_ARRAY_TASK_ID:-0}.npz"

    ros2 launch data_generation generate_from_scene.launch.py \
      voxel_size:=$VOXEL init_batch_size:=$INIT_BATCH_SIZE \
      base_link:=$BASE_LINK $ROBOT_ARG $WORLD_ARG &
    LAUNCH_PID=$!

    # curobo_ik (IKSolver/MotionGen warmup on GPU) and generate_data (loading
    # master_ik_data*.npz) both take a while to come up -- poll instead of a fixed
    # sleep. Bail out early if the launch itself dies (generate_from_scene.launch.py
    # gives generate_data on_exit=Shutdown(), so a crashed node takes the launch with
    # it) rather than polling a corpse for five minutes.
    ready=0
    for i in $(seq 1 60); do
      if ros2 service list 2>/dev/null | grep -q '^/generate_rm$'; then ready=1; break; fi
      if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "ERROR: ros2 launch exited before /generate_rm came up" >&2
        wait "$LAUNCH_PID" 2>/dev/null || true
        exit 1
      fi
      sleep 5
    done
    [ "$ready" = "1" ] || { echo "ERROR: /generate_rm never came up" >&2; kill "$LAUNCH_PID" 2>/dev/null || true; exit 1; }

    # load_scene.py waits on /generate_rm with a deliberate 24 h timeout (a real RM
    # takes >1 h and a short timeout would throw away finished work). That means a
    # crash on the ROS side would otherwise hang the task until the SLURM wall clock,
    # holding a whole GPU idle -- so supervise the launch while the call is in flight
    # and abort as soon as it dies.
    python3 "$DATA_SHARE/load_scene.py" \
      --scene-npz "$SCENE_NPZ" \
      --voxel-size "$VOXEL" --batch-size "$BATCH_SIZE" &
    LOAD_PID=$!

    # NOTE: every kill/wait below is `|| true`-guarded. Under `set -e`, waiting on a
    # process we just SIGTERM'd returns 143 and would abort the script *before*
    # `exit $STATUS` -- silently reporting failure on a perfectly good run.
    while kill -0 "$LOAD_PID" 2>/dev/null; do
      if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "ERROR: ros2 launch died while load_scene.py was running (generate_data crashed?)" >&2
        kill "$LOAD_PID" 2>/dev/null || true
        wait "$LOAD_PID" 2>/dev/null || true
        exit 1
      fi
      sleep 10
    done
    set +e
    wait "$LOAD_PID"
    STATUS=$?
    set -e

    kill "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    exit $STATUS
    ;;

  *)
    echo "ERROR: unknown MODE='$MODE' (expected: synthetic | real | create_rm)" >&2
    exit 1
    ;;
esac
