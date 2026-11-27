# Heron generation image: data_generation (ROS 2 Jazzy) on curobo_ros + cuRobo v0.8.0.
#
# Built by images/generation/build.sh, which first builds BASE_IMAGE from curobo_ros's
# docker/x86_jazzy.dockerfile (pinned), then passes every commit below and the heron.*
# labels. Contract: docs/images.md, "Generation image".
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# ---- System dependencies of data_generation (CMakeLists.txt) -------------------------
# pcl_ros, nav2_msgs, rmw_fastrtps_cpp and rosbag2 already come with the base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libboost-dev \
        zlib1g-dev \
        ros-jazzy-octomap \
        ros-jazzy-pcl-ros \
        ros-jazzy-tf2-geometry-msgs \
        ros-jazzy-rosbag2-interfaces \
        ros-jazzy-rosbag2-storage-default-plugins \
        ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# ---- HighFive (HDF5 C++), the version the Humble image used --------------------------
ARG HIGHFIVE_TAG=v2.8.0
ARG HIGHFIVE_COMMIT=88fcc898970f93a63be8e25760d6b9f33589690f
RUN git clone --depth 1 -b ${HIGHFIVE_TAG} https://github.com/BlueBrain/HighFive.git /pkgs/HighFive \
    && test "$(git -C /pkgs/HighFive rev-parse HEAD)" = "${HIGHFIVE_COMMIT}" \
    && cmake -S /pkgs/HighFive -B /pkgs/HighFive/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DHIGHFIVE_USE_BOOST=OFF -DHIGHFIVE_UNIT_TESTS=OFF -DHIGHFIVE_EXAMPLES=OFF \
        -DHIGHFIVE_BUILD_DOCS=OFF \
    && cmake --build /pkgs/HighFive/build && cmake --install /pkgs/HighFive/build

# ---- cnpy (its header is included by create_reachability_map) ------------------------
ARG CNPY_COMMIT=4e8810b1a8637695171ed346ce68f6984e585ef4
RUN git clone https://github.com/rogersce/cnpy.git /pkgs/cnpy \
    && git -C /pkgs/cnpy checkout ${CNPY_COMMIT} \
    && cmake -S /pkgs/cnpy -B /pkgs/cnpy/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /pkgs/cnpy/build && cmake --install /pkgs/cnpy/build && ldconfig

# ---- Sources, each at a pinned commit -------------------------------------------------
ARG CUROBO_ROS_REPO=https://github.com/Lab-CORO/curobo_ros.git
ARG CUROBO_ROS_COMMIT
ARG CUROBO_MSGS_REPO=https://github.com/Lab-CORO/curobo_msgs.git
ARG CUROBO_MSGS_COMMIT
ARG HERON_REPO=https://github.com/Lab-CORO/Heron.git
ARG HERON_COMMIT
# Commits not pushed yet arrive as bare repositories (build.sh, sources/); empty otherwise.
COPY sources/ /tmp/heron-sources/
WORKDIR /home/ros2_ws/src
# fetch <repo> <commit> <dir>: exactly that commit, nothing else.
RUN fetch() { mkdir -p "$3" && git -C "$3" init -q && git -C "$3" fetch -q --depth 1 "$1" "$2" \
              && git -C "$3" checkout -q FETCH_HEAD && test "$(git -C "$3" rev-parse HEAD)" = "$2"; } \
    && test -n "${CUROBO_ROS_COMMIT}" && test -n "${CUROBO_MSGS_COMMIT}" && test -n "${HERON_COMMIT}" \
    && fetch "${CUROBO_ROS_REPO}" "${CUROBO_ROS_COMMIT}" curobo_ros \
    && git -C curobo_ros submodule update --init --recursive --depth 1 \
    && fetch "${CUROBO_MSGS_REPO}" "${CUROBO_MSGS_COMMIT}" curobo_msgs \
    && fetch "${HERON_REPO}" "${HERON_COMMIT}" Heron \
    && rm -rf /tmp/heron-sources

# ---- Build: curobo_msgs, curobo_ros, data_generation ----------------------------------
WORKDIR /home/ros2_ws
RUN source /opt/ros/jazzy/setup.bash && source /home/ros2_ws/install/setup.bash \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --packages-select curobo_msgs curobo_ros data_generation \
    && test -x install/data_generation/lib/data_generation/generate_data \
    && test -x install/curobo_ros/lib/curobo_ros/curobo_trajectory_planner

# ---- What ended up in the image (the base leaves apt and most pip packages unpinned) --
RUN mkdir -p /opt/heron \
    && pip freeze > /opt/heron/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/heron/dpkg.txt \
    && for d in /pkgs/curobo_src /pkgs/cuda-samples /pkgs/HighFive /pkgs/cnpy \
                /home/ros2_ws/src/*/ /home/ros2_ws/src/curobo_ros/curobo_doosan; do \
         echo "$(git -C "$d" rev-parse HEAD 2>/dev/null || echo '?') $d"; \
       done > /opt/heron/git.txt \
    && { python3 -c "import curobo, torch; print('curobo', curobo.__version__, 'torch', torch.__version__)" \
         || echo "curobo import failed without a GPU (checked at run time instead)"; } >> /opt/heron/git.txt \
    && cat /opt/heron/git.txt

# ---- Runtime contract ---------------------------------------------------------------
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    PYTHONFAULTHANDLER=1
WORKDIR /home/ros2_ws
