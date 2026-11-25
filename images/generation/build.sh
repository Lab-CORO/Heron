#!/usr/bin/env bash
#
# build.sh — build the Heron generation image (ROS 2 Jazzy + cuRobo v2) and convert it
# to an Apptainer .sif. Non-interactive; amd64 only. See docs/images.md, "Generation image".
#
#   images/generation/build.sh [--out DIR] [--heron-commit SHA] [--skip-base] [--skip-sif] [--base-only]
#                              [--low-disk] [--from-image heron_data_generation_jazzy:<tag>]
#
# Steps:
#   1. clone Lab-CORO/curobo_ros at CUROBO_ROS_COMMIT;
#   2. build its docker/x86_jazzy.dockerfile (cuRobo v0.8.0), after the pinning patch
#      below, as curobo_ros:jazzy-<commit>;
#   3. build images/generation/Dockerfile on top of it (Heron's data_generation);
#   4. apptainer build --fakeroot <out>/heron_data_generation_jazzy.sif docker-daemon://…
#      That needs about three times the image size free (~90 GB); --low-disk converts with
#      docker2sif.sh instead (about twice the squashfs, ~25 GB). --from-image skips 1-3 and
#      converts an image already built.
#
# Environment (all optional):
#   OUT_DIR / --out        work and output directory          (default /var/tmp/heron-build)
#   HERON_REPO, CUROBO_ROS_REPO  repositories the image clones Heron and curobo_ros from: a URL,
#                          or a local checkout for commits not pushed yet (label heron.sources=local)
#   HERON_REPO             repository the image clones Heron from
#                                                   (default https://github.com/Lab-CORO/Heron.git)
#   HERON_COMMIT           Heron commit to build   (default: HEAD of this checkout; it must be
#                          reachable from HERON_REPO, i.e. pushed)
#   APPTAINER_CACHEDIR, APPTAINER_TMPDIR           (default <out>/apptainer/{cache,tmp})
#   MAKEFLAGS              parallel jobs of the base build (default -j4)
set -euo pipefail

# ---- Pinned versions (each one checked when it was written here) -----------------
# curobo_ros main of 2026-08-09 (06aef20) + fix/ik-batch-size (ik_batch failed for any batch
# of more than one pose) + feat/collision-cache-params (collision cache sized by ROS params).
# Its curobo_doosan submodule (m1013 config) is pinned by it. The base image is built from
# the same commit.
CUROBO_ROS_REPO="${CUROBO_ROS_REPO:-https://github.com/Lab-CORO/curobo_ros.git}"
CUROBO_ROS_COMMIT=927e410467936ac3481a88335c60ff2e3bb604cc
CUROBO_MSGS_REPO=https://github.com/Lab-CORO/curobo_msgs.git
CUROBO_MSGS_COMMIT=39ceb494666f023cc9ab11132cc537d85a428ad9
CUROBO_REF=v0.8.0                     # NVlabs/curobo tag, commit 4ea77366
# nvcr.io/nvidia/pytorch:24.11-py3 (Ubuntu 24.04, Python 3.12, CUDA 12.6, torch 2.6.0a0)
NGC_IMAGE=nvcr.io/nvidia/pytorch:24.11-py3
NGC_DIGEST=sha256:832d94d665baebc0f3f6cfa3168f21d75f7aa8f02ade5ebb65e102e55df79243
TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"    # A100, RTX 30xx, H100
CUDA_ARCH="80;86;90"
# Pins added to curobo_ros's base recipe, which leaves these unpinned:
CUDA_SAMPLES_TAG=v12.5                # helper_*.h headers (CUDA 12.6 in the base)
ROBOMETRICS_COMMIT=81e3d1d605de84100d8ab880b43096aba221a48b
ROS_APT_SOURCE_VERSION=1.3.0
ROS2_NUMPY_COMMIT=6233f90b80dc636ff2a5f1eae4f856a8e4eb5676
TRAJECTORY_PREVIEW_COMMIT=f3f30cb167ba60a66b655988ec8c216f5b4bfdd8
PIP_VERSION=26.3
# cuRobo's runtime dependencies, as of its v0.8.0 release (2026-04-18)
WARP_VERSION=1.12.1                   # CUDA 12.9 build; 1.19 removed wp.torch, which cuRobo uses
CUDA_CORE_VERSION=0.7.0
CUDA_BINDINGS_VERSION=12.9.6
CUDA_PATHFINDER_VERSION=1.5.3
OPEN3D_VERSION=0.20.0
# The base forces numpy==1.26.4; the latest scipy/pandas need numpy>=2, so pin the last
# releases that accept 1.26.4.
SCIPY_VERSION=1.17.0
PANDAS_VERSION=2.3.3
SKLEARN_VERSION=1.9.1
PYARROW_VERSION=26.0.1
PYREALSENSE2_VERSION=2.59.2.11152
TRANSFORMS3D_VERSION=0.4.2

IMAGE_NAME=heron_data_generation_jazzy

# ---- Arguments --------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-/var/tmp/heron-build}"
HERON_REPO="${HERON_REPO:-https://github.com/Lab-CORO/Heron.git}"
HERON_COMMIT="${HERON_COMMIT:-}"
SKIP_BASE=0
SKIP_SIF=0
BASE_ONLY=0
LOW_DISK=0
FROM_IMAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out)          OUT_DIR="$2"; shift 2;;
    --heron-commit) HERON_COMMIT="$2"; shift 2;;
    --skip-base)    SKIP_BASE=1; shift;;     # reuse an existing curobo_ros:jazzy-<commit>
    --skip-sif)     SKIP_SIF=1; shift;;
    --base-only)    BASE_ONLY=1; SKIP_SIF=1; shift;;   # stop after the base image
    --low-disk)     LOW_DISK=1; shift;;
    --from-image)   FROM_IMAGE="$2"; shift 2;;
    -h|--help)      sed -n '2,27p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

die() { echo "build.sh: $*" >&2; exit 1; }
[ "$(uname -m)" = "x86_64" ] || die "amd64 only (this machine is $(uname -m))"
command -v docker >/dev/null || die "docker not found"
[ "$SKIP_SIF" = 1 ] || command -v apptainer >/dev/null || die "apptainer not found"

if [ "$BASE_ONLY" = 1 ] || [ -n "$FROM_IMAGE" ]; then
  HERON_COMMIT=0000000000000000000000000000000000000000   # unused
elif [ -z "$HERON_COMMIT" ]; then
  [ -z "$(git -C "$REPO_ROOT" status --porcelain)" ] || die "uncommitted changes in $REPO_ROOT: commit them (the image clones a pushed commit)"
  HERON_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi
[ "$BASE_ONLY" = 1 ] || [ -n "$FROM_IMAGE" ] || [[ "$HERON_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "HERON_COMMIT must be a full 40-character SHA, got '$HERON_COMMIT'"
# The image clones this commit from HERON_REPO: fail now, not 30 minutes into the build.
probe="$(mktemp -d)"
if [ "$BASE_ONLY" = 0 ] && [ -z "$FROM_IMAGE" ] && ! { git -C "$probe" init -q && git -C "$probe" fetch -q --depth 1 "$HERON_REPO" "$HERON_COMMIT" 2>/dev/null; }; then
  rm -rf "$probe"; die "commit $HERON_COMMIT is not in $HERON_REPO (push it first, or set HERON_REPO to a local checkout)"
fi
rm -rf "$probe"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
LOG_DIR="$OUT_DIR/logs"; mkdir -p "$LOG_DIR"
BASE_TAG="curobo_ros:jazzy-${CUROBO_ROS_COMMIT:0:7}"
BUILT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
IMAGE_TAG="${IMAGE_NAME}:$(date -u +%Y%m%d)-${HERON_COMMIT:0:7}"
[ -z "$FROM_IMAGE" ] || IMAGE_TAG="$FROM_IMAGE"
echo "== Heron generation image: $IMAGE_TAG (base $BASE_TAG), work dir $OUT_DIR"

# ---- 1-2. Base image: curobo_ros's Jazzy recipe, pinned ------------------------------
if [ -z "$FROM_IMAGE" ]; then
if [ "$SKIP_BASE" = 0 ]; then
  SRC="$OUT_DIR/src/curobo_ros"
  rm -rf "$SRC"; mkdir -p "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" fetch -q --depth 1 "$CUROBO_ROS_REPO" "$CUROBO_ROS_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$CUROBO_ROS_COMMIT" ] || die "curobo_ros checkout mismatch"

  # curobo_ros's docker/x86_jazzy_prod.dockerfile still installs the cuRobo v1 fork
  # (Lab-CORO/curobo@lab-coro), which this curobo_ros cannot import; x86_jazzy.dockerfile
  # is the v0.8.0 recipe. It is used as is, except for this patch: each replacement must
  # match exactly once, so a change upstream stops the build instead of being ignored.
  BASE_DOCKERFILE="$OUT_DIR/base.dockerfile"
  export NGC_DIGEST ROBOMETRICS_COMMIT CUDA_SAMPLES_TAG ROS_APT_SOURCE_VERSION ROS2_NUMPY_COMMIT \
         TRAJECTORY_PREVIEW_COMMIT PIP_VERSION OPEN3D_VERSION SCIPY_VERSION PANDAS_VERSION \
         SKLEARN_VERSION PYARROW_VERSION PYREALSENSE2_VERSION TRANSFORMS3D_VERSION \
         WARP_VERSION CUDA_CORE_VERSION CUDA_BINDINGS_VERSION CUDA_PATHFINDER_VERSION
  python3 "$HERE/patch_base.py" "$SRC/docker/x86_jazzy.dockerfile" "$BASE_DOCKERFILE"
  diff -u "$SRC/docker/x86_jazzy.dockerfile" "$BASE_DOCKERFILE" > "$OUT_DIR/base.dockerfile.diff" || true
  echo "== Base recipe patched ($OUT_DIR/base.dockerfile.diff). Building $BASE_TAG (log: $LOG_DIR/base.log)"
  CTX="$OUT_DIR/base-context"; rm -rf "$CTX"; mkdir -p "$CTX"   # the recipe copies nothing in
  docker build --platform linux/amd64 --progress=plain \
    -f "$BASE_DOCKERFILE" -t "$BASE_TAG" \
    --build-arg CUDA_ARCH="$CUDA_ARCH" \
    --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    --build-arg CUROBO_REF="$CUROBO_REF" \
    --build-arg MAKEFLAGS="${MAKEFLAGS:--j4}" \
    --build-arg PRECOMPILE_CUROBO=false \
    "$CTX" 2>&1 | tee "$LOG_DIR/base.log"
fi
docker image inspect "$BASE_TAG" >/dev/null 2>&1 || die "base image $BASE_TAG missing"
[ "$BASE_ONLY" = 0 ] || { echo "== Base image $BASE_TAG ready (--base-only)"; exit 0; }

# ---- 3. Heron layer -------------------------------------------------------------------
# A local checkout cannot be reached from docker build: push the pinned commit into a bare
# repository in the build context (sources/<name>.git), which the Dockerfile fetches from.
CTX="$OUT_DIR/heron-context"; rm -rf "$CTX"; mkdir -p "$CTX/sources"
SOURCES=github
stage() {  # stage <name> <repo> <commit>: prints the repository the Dockerfile fetches from
  if [ -d "$2" ]; then
    git init -q --bare "$CTX/sources/$1.git"
    git -C "$2" push -q "$CTX/sources/$1.git" "$3:refs/heads/pinned"
    echo "/tmp/heron-sources/$1.git"
  else
    echo "$2"
  fi
}
DOCKER_CUROBO_ROS_REPO="$(stage curobo_ros "$CUROBO_ROS_REPO" "$CUROBO_ROS_COMMIT")"
DOCKER_HERON_REPO="$(stage Heron "$HERON_REPO" "$HERON_COMMIT")"
[ -d "$CUROBO_ROS_REPO" ] || [ -d "$HERON_REPO" ] && SOURCES=local
HIGHFIVE_TAG="$(sed -n 's/^ARG HIGHFIVE_TAG=//p' "$HERE/Dockerfile")"
echo "== Building $IMAGE_TAG from Heron $HERON_COMMIT (log: $LOG_DIR/heron.log)"
docker build --platform linux/amd64 --progress=plain \
  -f "$HERE/Dockerfile" -t "$IMAGE_TAG" \
  --build-arg BASE_IMAGE="$BASE_TAG" \
  --build-arg HERON_REPO="$DOCKER_HERON_REPO" \
  --build-arg HERON_COMMIT="$HERON_COMMIT" \
  --build-arg CUROBO_ROS_REPO="$DOCKER_CUROBO_ROS_REPO" \
  --build-arg CUROBO_ROS_COMMIT="$CUROBO_ROS_COMMIT" \
  --build-arg CUROBO_MSGS_REPO="$CUROBO_MSGS_REPO" \
  --build-arg CUROBO_MSGS_COMMIT="$CUROBO_MSGS_COMMIT" \
  --label heron.image=generation \
  --label heron.ros=jazzy \
  --label heron.curobo="$CUROBO_REF" \
  --label heron.curobo_ros="$CUROBO_ROS_COMMIT" \
  --label heron.curobo_msgs="$CUROBO_MSGS_COMMIT" \
  --label heron.heron="$HERON_COMMIT" \
  --label heron.highfive="$HIGHFIVE_TAG" \
  --label heron.cuda_arch="${TORCH_CUDA_ARCH_LIST// /;}" \
  --label heron.base="${NGC_IMAGE}@${NGC_DIGEST}" \
  --label heron.built="$BUILT" \
  --label heron.sources="$SOURCES" \
  "$CTX" 2>&1 | tee "$LOG_DIR/heron.log"
docker tag "$IMAGE_TAG" "${IMAGE_NAME}:latest"
fi   # FROM_IMAGE

# ---- 4. Apptainer -----------------------------------------------------------------
if [ "$SKIP_SIF" = 0 ]; then
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$OUT_DIR/apptainer/cache}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$OUT_DIR/apptainer/tmp}"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  SIF="$OUT_DIR/${IMAGE_NAME}.sif"
  echo "== Converting to $SIF"
  if [ "$LOW_DISK" = 1 ]; then
    "$HERE/docker2sif.sh" "$IMAGE_TAG" "$SIF" "$APPTAINER_TMPDIR/docker2sif" 2>&1 | tee "$LOG_DIR/sif.log"
  else
    apptainer build --force --fakeroot "$SIF" "docker-daemon://${IMAGE_TAG}" 2>&1 | tee "$LOG_DIR/sif.log"
  fi
  ls -lh "$SIF"
  apptainer inspect --labels "$SIF" | grep '^heron\.' || die "no heron.* label in $SIF"
fi
echo "== Done: $IMAGE_TAG"
