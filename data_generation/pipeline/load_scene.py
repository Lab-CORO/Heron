#!/usr/bin/env python3
"""
load_scene.py -- CapaciNet 'scene' pipeline mode.

Reads one scene (a dense occupancy voxel grid) out of a shared HDF5 file, injects
it into the running curobo_ik node's collision world, and triggers RM generation
via the existing `generate_data` node -- so the rest of the pipeline (HDF5 output
with voxel_grid + reachability_map, format_data.py, etc.) is unchanged.

Two ways to inject the scene, tried in order:

  1. Native ESDF path: curobo.geom.types.VoxelGrid via the new
     `/curobo_ik/load_voxel_scene` service (see pipeline/curobo_patch/), which only
     exists if that patch is bind-mounted over curobo_ros's installed files. This
     avoids decomposing the scene into obstacles entirely. Confirmed viable on GPU
     by pipeline/verify_voxel_esdf.py (WorldConfig.voxel + CollisionCheckerType.VOXEL
     IS enforced by WorldVoxelCollision once fed via update_voxel_data()).

  2. Cuboid fallback: if the service above is unavailable or fails (patch not
     applied, or the native path errors for any reason), decompose the same
     occupancy grid into axis-aligned boxes via greedy multi-resolution merging
     (adapted from curobo_ros's own MeshBloxilization.build_levels/extract_cubes,
     minus the open3d/mesh front end that isn't needed when the input is already a
     voxel array) and inject each box through the existing, unmodified
     `/curobo_ik/add_object` service. The merge is exact -- total box volume equals
     total occupied volume, no bounding-box over-approximation.

     CAVEAT, and the reason path 1 is preferred: the merge only helps where the
     occupancy is blocky. On a measured/noisy scene it degrades badly -- a 64^3 test
     scene with speckle noise gave only a ~5x reduction, i.e. ~2000 cuboids for
     ~10k occupied voxels. That is far above ConfigWrapperIK's default
     collision_cache of {"obb": 100}, so cuRobo will silently rebuild a much larger
     OBB cache, and each box costs a round-trip through /add_object (which also
     re-runs update_world_config each time). Expect this path to be slow and to use
     a lot of GPU memory on realistic scenes; it exists to keep the pipeline
     working, not to be the normal route. If you find yourself relying on it,
     fix the patch instead.

This script reads one scene from a *pre-split* .npz (see pipeline/split_scenes.py
-- data_generation.sif has no h5py, so the original shared scenes.h5 is split into
per-scene .npz files on the host, once, before the array job runs; this script
never touches the original HDF5 file at all).

split_scenes.py already resolves the source file's occupancy sign convention
(via --occupied-value, since it varies by source -- e.g. reachability_map.h5
uses -1=occupied but a grid recorded straight from a ROS topic may use
0=occupied) and writes a canonical boolean 'occupied' array into the .npz, so
this script never has to know or guess a sign convention.

Axis order assumption: (x, y, z) with z innermost/fastest, matching
data_generator.cpp's own nested-loop write order.
"""
import argparse
import math
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from curobo_msgs.srv import AddObject, GenerateRM


# ---------------------------------------------------------------------------
# Scene reading
# ---------------------------------------------------------------------------

def read_scene(scene_npz_path: str):
    """Read a single pre-split scene written by pipeline/split_scenes.py.
    Returns (occupied: bool array, voxel_size, origin) -- occupancy is already
    canonical (True=occupied), see module docstring."""
    data = np.load(scene_npz_path)
    return data["occupied"].astype(bool), float(data["voxel_size"]), data["origin"]


# ---------------------------------------------------------------------------
# Native ESDF path
# ---------------------------------------------------------------------------

def occupancy_to_esdf(occ: np.ndarray, voxel_size: float) -> np.ndarray:
    """Signed distance in meters, + inside obstacles, - in free space (curobo convention,
    confirmed in verify_voxel_esdf.py)."""
    inside = distance_transform_edt(occ)
    outside = distance_transform_edt(~occ)
    return ((inside - outside) * voxel_size).astype(np.float32)


def matching_dims(shape, voxel_size):
    """
    Pick [dx,dy,dz] such that curobo.geom.types.VoxelGrid(dims=..., voxel_size=...)
    .get_grid_shape() reproduces `shape` exactly. Their formula is
    N = 1 + floor(dim / voxel_size), which is not a clean inverse of an integer N,
    so derive dims from N instead of the other way around, with a tiny epsilon to
    protect against floating point flooring one voxel short.
    """
    from curobo.geom.types import VoxelGrid
    dims = [(n - 1) * voxel_size + voxel_size * 1e-6 for n in shape]
    got, _, _ = VoxelGrid("check", pose=[0, 0, 0, 1, 0, 0, 0], dims=dims, voxel_size=voxel_size).get_grid_shape()
    if list(got) != list(shape):
        raise RuntimeError(f"matching_dims failed: wanted {shape}, got {got} for dims={dims}")
    return dims


def write_voxel_scene_npz(path: str, feature: np.ndarray, dims, voxel_size: float, pose=None):
    pose = pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    np.savez(path, feature=feature, dims=np.array(dims, dtype=np.float64),
             voxel_size=np.float64(voxel_size), pose=np.array(pose, dtype=np.float64))


def grid_center_pose(occ_shape, voxel_size: float, origin: np.ndarray):
    """
    cuRobo places a VoxelGrid *centered on its pose*: with pose p, voxel i sits at
    p + (-N*voxel_size/2) + (i+0.5)*voxel_size (see VoxelGrid.create_xyzr_tensor).
    Our scene instead states its own corner `origin`, with voxel i at
    origin + (i+0.5)*voxel_size. Equating the two gives p = origin + N*voxel_size/2.

    Do NOT simplify this to the identity pose: it only happens to be identity when
    the scene is symmetric about the robot base (origin == -N*voxel_size/2, which is
    the case for the pipeline's own 128^3 @ -1.28 output). A scene with any other
    origin would silently land in the wrong place.
    """
    n = np.asarray(occ_shape, dtype=np.float64)
    center = np.asarray(origin, dtype=np.float64) + n * voxel_size / 2.0
    return [float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0]


def try_native_path(node: "SceneLoaderNode", occ, voxel_size, origin, timeout=10.0) -> bool:
    if not node.load_voxel_scene_client.wait_for_service(timeout_sec=2.0):
        node.get_logger().warn("/curobo_ik/load_voxel_scene not available (patch not applied?) "
                                "-- falling back to cuboid decomposition.")
        return False

    dims = matching_dims(occ.shape, voxel_size)
    feature = occupancy_to_esdf(occ, voxel_size)
    pose = grid_center_pose(occ.shape, voxel_size, origin)
    scene_path = os.environ.get("VOXEL_SCENE_PATH")
    if not scene_path:
        node.get_logger().error("VOXEL_SCENE_PATH is not set; cannot use the native ESDF path.")
        return False
    write_voxel_scene_npz(scene_path, feature, dims, voxel_size, pose=pose)

    req = Trigger.Request()
    future = node.load_voxel_scene_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        node.get_logger().warn("load_voxel_scene call timed out -- falling back.")
        return False
    resp = future.result()
    if not resp.success:
        node.get_logger().warn(f"load_voxel_scene reported failure: {resp.message} -- falling back.")
        return False
    node.get_logger().info(f"Native ESDF scene loaded: {resp.message}")
    return True


# ---------------------------------------------------------------------------
# Cuboid fallback: greedy multi-resolution box merge over the occupancy array.
# Adapted from curobo_ros's ObstacleManager.MeshBloxilization (build_levels /
# merge_level / extract_cubes), which does the same thing starting from a mesh
# via open3d raycasting -- we already have the occupancy array, so that front
# end is skipped and this has no open3d dependency.
# ---------------------------------------------------------------------------

def _pad_to_power_of_two(occ: np.ndarray) -> np.ndarray:
    padded_shape = [1 << math.ceil(math.log2(max(s, 1))) for s in occ.shape]
    if list(padded_shape) == list(occ.shape):
        return occ
    padded = np.zeros(padded_shape, dtype=bool)
    padded[:occ.shape[0], :occ.shape[1], :occ.shape[2]] = occ
    return padded


def _merge_level(prev: np.ndarray) -> np.ndarray:
    nx, ny, nz = prev.shape
    m = prev.reshape(nx // 2, 2, ny // 2, 2, nz // 2, 2)
    return m.all(axis=(1, 3, 5))


def _build_levels(occ: np.ndarray):
    levels = [occ]
    while True:
        prev = levels[-1]
        if prev.shape[0] % 2 or prev.shape[1] % 2 or prev.shape[2] % 2:
            break
        merged = _merge_level(prev)
        if not merged.any():
            break
        levels.append(merged)
    return levels


def _extract_cubes(levels):
    base_shape = levels[0].shape
    coverage = np.zeros(base_shape, dtype=bool)
    cubes = []  # (corner_origin_in_voxels, size_in_voxels)
    for lvl in range(len(levels) - 1, -1, -1):
        grid = levels[lvl]
        factor = 2 ** lvl
        for (i, j, k) in np.argwhere(grid):
            si, sj, sk = i * factor, j * factor, k * factor
            ei, ej, ek = si + factor, sj + factor, sk + factor
            if coverage[si:ei, sj:ej, sk:ek].all():
                continue
            cubes.append((np.array([si, sj, sk]), factor))
            coverage[si:ei, sj:ej, sk:ek] = True
    return cubes


def occupancy_to_cuboids(occ: np.ndarray, voxel_size: float, origin: np.ndarray):
    """
    Greedy multi-resolution merge: prefer large cubes, only subdividing near
    occupancy boundaries. A single solid blob becomes one (or a few) cuboids
    instead of one-per-voxel or one-per-connected-component, which keeps the
    obstacle count low even for a fragmented/noisy real-world scan.

    Returns list of (pose_xyz_meters, dims_xyz_meters) in the scene's own frame
    (same frame as `origin`).
    """
    padded = _pad_to_power_of_two(occ)
    levels = _build_levels(padded)
    cubes = _extract_cubes(levels)
    result = []
    for corner_vox, size_vox in cubes:
        size_m = size_vox * voxel_size
        center = origin + (corner_vox + size_vox / 2.0) * voxel_size
        result.append((center.tolist(), [size_m, size_m, size_m]))
    return result


def add_cuboids(node: "SceneLoaderNode", cuboids, timeout=10.0) -> int:
    added = 0
    for i, (pos, dims) in enumerate(cuboids):
        req = AddObject.Request()
        req.type = AddObject.Request.CUBOID
        req.name = f"scene_obj_{i}"
        req.pose.position.x, req.pose.position.y, req.pose.position.z = pos
        req.pose.orientation.w = 1.0
        req.dimensions.x, req.dimensions.y, req.dimensions.z = dims
        req.color.r, req.color.a = 1.0, 1.0

        future = node.add_object_client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        if future.done() and future.result() is not None and future.result().success:
            added += 1
        else:
            msg = future.result().message if future.done() and future.result() else "timeout"
            node.get_logger().warn(f"Failed to add {req.name}: {msg}")
    return added


# ---------------------------------------------------------------------------
# Orchestration node
# ---------------------------------------------------------------------------

class SceneLoaderNode(Node):
    def __init__(self):
        super().__init__("load_scene")
        self.remove_all_client = self.create_client(Trigger, "/curobo_ik/remove_all_objects")
        self.add_object_client = self.create_client(AddObject, "/curobo_ik/add_object")
        self.load_voxel_scene_client = self.create_client(Trigger, "/curobo_ik/load_voxel_scene")
        self.generate_rm_client = self.create_client(GenerateRM, "/generate_rm")

    def wait_core_services(self, timeout=60.0):
        for client, name in [(self.remove_all_client, "remove_all_objects"),
                              (self.add_object_client, "add_object"),
                              (self.generate_rm_client, "generate_rm")]:
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"Service {name} did not become available within {timeout}s")

    def remove_all_objects(self, timeout=10.0):
        future = self.remove_all_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None or not future.result().success:
            self.get_logger().warn("remove_all_objects did not report success (continuing anyway).")

    def generate_rm(self, batch_size: int, resolution: float, timeout=86400.0) -> bool:
        # One RM means len(master_ik_data)/batch_size IK service calls (~4000 batches
        # of 10000 at voxel 0.02), each ~1.2 s on an A100 -> well over an hour per
        # scene. Don't put a "reasonable-looking" timeout here: the real bound is the
        # job's SLURM --time, and a short timeout just throws away work already done.
        req = GenerateRM.Request()
        req.batch_size = batch_size
        req.resolution = resolution
        future = self.generate_rm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("generate_rm timed out.")
            return False
        return bool(future.result().success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-npz", required=True, help="one file written by split_scenes.py, e.g. scenes/scene_<i>.npz")
    ap.add_argument("--voxel-size", type=float, required=True, help="pipeline operating voxel_size (pose sampling resolution)")
    ap.add_argument("--batch-size", type=int, default=10000)
    args = ap.parse_args()

    occ, scene_voxel_size, origin = read_scene(args.scene_npz)
    n_occ = int(occ.sum())
    print(f"[load_scene] {args.scene_npz}: shape={occ.shape} "
          f"voxel_size={scene_voxel_size} occupied={n_occ}/{occ.size}", flush=True)

    rclpy.init()
    node = SceneLoaderNode()
    try:
        node.wait_core_services()
        node.remove_all_objects()

        used_native = False
        if n_occ > 0:
            try:
                used_native = try_native_path(node, occ, scene_voxel_size, origin)
            except Exception as e:
                node.get_logger().warn(f"Native ESDF path raised {e!r} -- falling back to cuboids.")
                used_native = False

        if not used_native and n_occ > 0:
            cuboids = occupancy_to_cuboids(occ, scene_voxel_size, origin)
            print(f"[load_scene] cuboid fallback: {len(cuboids)} boxes "
                  f"(from {n_occ} occupied voxels)", flush=True)
            added = add_cuboids(node, cuboids)
            if added < len(cuboids):
                node.get_logger().warn(f"Only {added}/{len(cuboids)} cuboids were accepted.")

        ok = node.generate_rm(batch_size=args.batch_size, resolution=args.voxel_size)
        if not ok:
            print("[load_scene] generate_rm FAILED", flush=True)
            sys.exit(1)
        print("[load_scene] generate_rm OK", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
