#!/usr/bin/env bash
#
# smoke_test.sh — check a generation .sif on a workstation with a GPU, outside SLURM, the
# way job.sbatch runs it (apptainer exec -C --net --network=none --nv, same mounts).
#
#   images/generation/smoke_test.sh --sif <sif> --master-map-dir <dir> [options]
#
#   --sif PATH             the image to test
#   --master-map-dir DIR   holds master_ik_data<voxel>.npz and .h5 of the Robot (m1013)
#   --mode MODE            synthetic (default) | create_rm
#   --voxel V              voxel size (default 0.02)
#   --half-extent H        Station scene_half_extent (default 1.52, leeloo)
#   --batch-size N         IK poses per batch (default 10000; lower it on a small GPU)
#   --obj-max N            max random cubes per Scene (default 20)
#   --work DIR             work directory (default: a new one under /var/tmp)
#
# synthetic: generates one Scene (DATASET_SIZE=1) and checks the .h5: group/0/voxel_grid and
# group/0/reachability_map are N³ with N = ceil(2H/V), values in [0, 1], a non-empty map.
# create_rm: writes master_ik_data<V>.npz/.h5 into <work>/data (the --master-map-dir is
# not used), for comparison with the current master map.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
SIF=""; MM_DIR=""; MODE=synthetic; VOXEL=0.02; HALF=1.52; BATCH=10000; OBJ_MAX=20; WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sif) SIF="$2"; shift 2;;
    --master-map-dir) MM_DIR="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --voxel) VOXEL="$2"; shift 2;;
    --half-extent) HALF="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --obj-max) OBJ_MAX="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    -h|--help) sed -n '2,21p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
die() { echo "smoke_test.sh: $*" >&2; exit 1; }
[ -f "$SIF" ] || die "--sif: no such file '$SIF'"
SIF="$(realpath "$SIF")"
WORK="${WORK:-$(mktemp -d /var/tmp/heron-smoke.XXXXXX)}"
mkdir -p "$WORK"/{data,tmp,warp}
WORK="$(realpath "$WORK")"
DATA="$WORK/data"
CONTAINER_SHARE=/home/ros2_ws/install/data_generation/share/data_generation

echo "== 1. torch and cuRobo see the GPU"
apptainer exec -C --nv "$SIF" python3 -c \
  "import torch, curobo; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'curobo', curobo.__version__); assert torch.cuda.is_available()"

echo "== 2. generation ($MODE, voxel $VOXEL, half extent $HALF) in $WORK"
cp "$REPO_ROOT/data_generation/pipeline/run_generation.sh" "$DATA/"
BINDS=(
  --bind "$DATA":"$CONTAINER_SHARE/data"
  --bind "$REPO_ROOT/data_generation/launch":"$CONTAINER_SHARE/launch"
  --bind "$WORK/tmp":/tmp
  --bind "$WORK/warp":"$HOME/.cache/warp"
)
if [ "$MODE" = synthetic ]; then
  for ext in npz h5; do
    f="$MM_DIR/master_ik_data${VOXEL}.${ext}"
    [ -f "$f" ] || die "missing $f"
    : > "$DATA/master_ik_data${VOXEL}.${ext}"      # mount point, read-only bind below
    BINDS+=( --bind "$(realpath "$f")":"$CONTAINER_SHARE/data/master_ik_data${VOXEL}.${ext}":ro )
  done
fi
export APPTAINERENV_MODE="$MODE" APPTAINERENV_VOXEL="$VOXEL" APPTAINERENV_DATASET_SIZE=1 \
       APPTAINERENV_BATCH_SIZE="$BATCH" APPTAINERENV_INIT_BATCH_SIZE="$BATCH" \
       APPTAINERENV_OBJ_MAX="$OBJ_MAX" APPTAINERENV_REACH_MAX=1.3 APPTAINERENV_ROBOT_CONFIG_FILE= \
       APPTAINERENV_BASE_LINK=base_0 APPTAINERENV_WORLD_FILE= APPTAINERENV_CAMERA_FRAME= \
       APPTAINERENV_SLURM_ARRAY_TASK_ID=1 APPTAINERENV_HERON_SCENE_HALF_EXTENT="$HALF"
start=$(date +%s)
set +e
apptainer exec -C --net --network=none --nv "${BINDS[@]}" "$SIF" \
  bash "$CONTAINER_SHARE/data/run_generation.sh" 2>&1 | tee "$WORK/generation.log"
status=${PIPESTATUS[0]}
set -e
echo "== generation exited $status after $(( $(date +%s) - start )) s (log: $WORK/generation.log)"

echo "== 3. output"
if [ "$MODE" = create_rm ]; then
  ls -l "$DATA"/master_ik_data* || die "no master map written"
  exit "$status"
fi
shopt -s nullglob
outs=()
for f in "$DATA"/*.h5; do
  case "$(basename "$f")" in master_ik_data*) ;; *) outs+=( "$f" );; esac
done
[ ${#outs[@]} -ge 1 ] || die "no .h5 written in $DATA"
CHECK='
import sys, math, h5py, numpy as np
path, voxel, half = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
n = math.ceil(2 * half / voxel - 1e-3)
with h5py.File(path, "r") as f:
    vg = f["group/0/voxel_grid"][()]; rm = f["group/0/reachability_map"][()]
    attrs = dict(f["group/0/voxel_grid"].attrs)
print(path); print("  voxel_grid", vg.shape, "occupied", int((vg > 0).sum()), "attrs", attrs)
print("  reachability_map", rm.shape, "min", rm.min(), "max", rm.max(), "non-zero", int((rm > 0).sum()))
ok = vg.shape == (n, n, n) and rm.shape == (n, n, n)
eps = 1e-9   # a voxel reachable in all 50 orientations sums 50 x 1/50 = 1.0000000000000004
ok &= -eps <= rm.min() and rm.max() <= 1 + eps and 0.0 <= vg.min() and vg.max() <= 1.0 and (rm > 0).any()
print("OK" if ok else f"FAILED (expected {n}^3, values in [0, 1], a non-empty map)")
sys.exit(0 if ok else 1)
'
for out in "${outs[@]}"; do
  if python3 -c "import h5py" 2>/dev/null; then
    python3 -c "$CHECK" "$out" "$VOXEL" "$HALF"
  else
    apptainer exec -C --bind "$DATA" "$SIF" python3 -c "$CHECK" "$out" "$VOXEL" "$HALF"
  fi
done
exit "$status"
