#!/usr/bin/env python3
"""Pin curobo_ros's docker/x86_jazzy.dockerfile for the Heron generation image.

    patch_base.py <x86_jazzy.dockerfile> <out.dockerfile>

The pinned versions come from the environment (build.sh exports them). Each replacement
must match exactly once: if curobo_ros changes its recipe, the build stops here instead
of silently building something unpinned.
"""
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
env = os.environ
s = open(src).read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"patch_base.py: expected {count} match(es), found {n}:\n{old}")
    s = s.replace(old, new)


BS = "\\"  # Dockerfile line continuation

# Base image by digest, not by tag. The NGC image's /etc/pip.conf adds an extra index,
# pypi.ngc.nvidia.com, whose name no longer resolves: every pip install then spends minutes
# retrying it. Ignore that file (its other settings are PyPI, the default, and no-cache-dir,
# set again here) and give pip a few retries for a flaky network.
sub("FROM nvcr.io/nvidia/pytorch:24.11-py3 AS torch_cuda_base",
    f"FROM nvcr.io/nvidia/pytorch:24.11-py3@{env['NGC_DIGEST']} AS torch_cuda_base\n"
    "ENV PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5")

robometrics = "git+https://github.com/fishbotics/robometrics.git"
sub(f'RUN pip install "robometrics[evaluator] @ {robometrics}"',
    f'RUN pip install "robometrics[evaluator] @ {robometrics}@{env["ROBOMETRICS_COMMIT"]}"')
sub(f'RUN python -m pip install "robometrics[evaluator] @ {robometrics}"',
    f'RUN python -m pip install "robometrics[evaluator] @ {robometrics}@{env["ROBOMETRICS_COMMIT"]}"')

sub("git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git /pkgs/cuda-samples",
    f"git clone --depth 1 -b {env['CUDA_SAMPLES_TAG']} https://github.com/NVIDIA/cuda-samples.git /pkgs/cuda-samples")

# cuRobo only asks for warp-lang>=0.10.0 and cuda-core>=0.7: unpinned, pip takes today's
# releases, and warp 1.19 (no more wp.torch) breaks cuRobo v0.8.0. Pin them, and the CUDA
# Python packages they pull, to their last releases before v0.8.0 (2026-04-18).
sub("RUN MAX_JOBS=${MAKEFLAGS##*-j} pip3 install -e .[cu12,dev,usd] --no-build-isolation",
    "RUN MAX_JOBS=${MAKEFLAGS##*-j} pip3 install -e .[cu12,dev,usd] --no-build-isolation "
    f"warp-lang=={env['WARP_VERSION']} cuda-core=={env['CUDA_CORE_VERSION']} "
    f"cuda-bindings=={env['CUDA_BINDINGS_VERSION']} cuda-pathfinder=={env['CUDA_PATHFINDER_VERSION']}")

# cuRobo v1 leftover: v0.8.0 has no src/curobo/curobolib/cpp (its kernels are compiled at
# run time by NVRTC from curobo/_src/curobolib/kernels), so this cp fails the build.
sub("RUN cp /pkgs/curobo_src/src/curobo/curobolib/cpp/*.h /usr/local/cuda/include/\n", "")

sub(f"RUN python -m pip install {BS}\n    pyrealsense2 {BS}\n    transforms3d\n",
    f"RUN python -m pip install {BS}\n    pyrealsense2=={env['PYREALSENSE2_VERSION']} {BS}\n"
    f"    transforms3d=={env['TRANSFORMS3D_VERSION']}\n")

sub("export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/"
    "ros-apt-source/releases/latest | grep -F \"tag_name\" | awk -F\\\" '{print $4}')",
    f"export ROS_APT_SOURCE_VERSION={env['ROS_APT_SOURCE_VERSION']}")

sub(f"RUN git clone https://github.com/Box-Robotics/ros2_numpy.git || {BS}\n"
    "    git clone -b main https://github.com/Box-Robotics/ros2_numpy.git",
    "RUN git clone https://github.com/Box-Robotics/ros2_numpy.git && "
    f"git -C ros2_numpy checkout {env['ROS2_NUMPY_COMMIT']}")

sub("RUN git clone https://github.com/swri-robotics/trajectory_preview.git\n",
    "RUN git clone https://github.com/swri-robotics/trajectory_preview.git && "
    f"git -C trajectory_preview checkout {env['TRAJECTORY_PREVIEW_COMMIT']}\n")

sub(f"python3 -m pip install --no-cache-dir --upgrade pip && {BS}\n"
    "    python3 -m pip install --no-cache-dir --upgrade open3d",
    f"python3 -m pip install --no-cache-dir pip=={env['PIP_VERSION']} && {BS}\n"
    f"    python3 -m pip install --no-cache-dir open3d=={env['OPEN3D_VERSION']}")

# The recipe forces numpy==1.26.4; the latest scipy/pandas require numpy>=2.
sub(f'pip3 install --no-cache-dir --force-reinstall "numpy==1.26.4" scipy && {BS}\n'
    f"    python3 -m pip install --no-cache-dir --force-reinstall --no-deps {BS}\n"
    "    pandas scikit-learn pyarrow",
    f'pip3 install --no-cache-dir --force-reinstall "numpy==1.26.4" scipy=={env["SCIPY_VERSION"]} && {BS}\n'
    f"    python3 -m pip install --no-cache-dir --force-reinstall --no-deps {BS}\n"
    f"    pandas=={env['PANDAS_VERSION']} scikit-learn=={env['SKLEARN_VERSION']} "
    f"pyarrow=={env['PYARROW_VERSION']}")

with open(dst, "w") as f:
    f.write(s)
