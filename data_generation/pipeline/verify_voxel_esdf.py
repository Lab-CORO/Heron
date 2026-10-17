#!/usr/bin/env python3
"""
Standalone GPU check: does cuRobo's native VoxelGrid/ESDF obstacle
(curobo.geom.types.VoxelGrid, WorldConfig.voxel, CollisionCheckerType.VOXEL)
actually get enforced by WorldVoxelCollision.get_sphere_distance?

No ROS2 involved on purpose -- this tests the raw curobo API exactly as the
planned curobo_ros patch (obstacle_manager.py / ros_service_manager.py) would
use it, so a pass/fail here tells us whether that patch is worth writing.

Run inside the data_generation.sif container with a GPU:
    apptainer exec --nv docker_images/data_generation.sif \
        python3 CapaciNet/data_generation/pipeline/verify_voxel_esdf.py
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from curobo.geom.types import VoxelGrid, WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.types.base import TensorDeviceType


def build_scene_voxel_grid(tensor_args, dims=(1.0, 1.0, 1.0), voxel_size=0.05):
    """One solid occupied cube in the middle of an otherwise free 1m^3 grid."""
    vg = VoxelGrid(name="scene", pose=[0, 0, 0, 1, 0, 0, 0], dims=list(dims), voxel_size=voxel_size)
    grid_shape, low, high = vg.get_grid_shape()
    nx, ny, nz = grid_shape
    print(f"grid_shape={grid_shape} low={low} high={high}")

    occ = np.zeros((nx, ny, nz), dtype=bool)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    r = max(1, nx // 6)
    occ[cx - r:cx + r, cy - r:cy + r, cz - r:cz + r] = True
    n_occupied = int(occ.sum())
    print(f"occupied voxels: {n_occupied} / {occ.size}")

    inside_dist = distance_transform_edt(occ)   # >0 inside obstacle, 0 outside
    outside_dist = distance_transform_edt(~occ)  # >0 outside obstacle, 0 inside
    sdf = (inside_dist - outside_dist) * voxel_size  # curobo convention: + inside, - outside

    feature = torch.as_tensor(sdf.reshape(-1, 1), dtype=torch.float32, device=tensor_args.device)
    vg.feature_tensor = feature
    return vg, grid_shape, occ, voxel_size


def make_query(points_xyz, radius, tensor_args):
    pts = torch.as_tensor(points_xyz, dtype=tensor_args.dtype, device=tensor_args.device)
    r = torch.full((pts.shape[0], 1), radius, dtype=tensor_args.dtype, device=tensor_args.device)
    sph = torch.cat([pts, r], dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,n,4]
    return sph


def main():
    tensor_args = TensorDeviceType(device="cuda", dtype=torch.float32)

    vg, grid_shape, occ, voxel_size = build_scene_voxel_grid(tensor_args)
    world_cfg = WorldConfig(voxel=[vg])

    coll_cfg = WorldCollisionConfig(
        tensor_args=tensor_args,
        world_model=world_cfg,
        checker_type=CollisionCheckerType.VOXEL,
        cache={"voxel": {"layers": 1, "dims": vg.dims, "voxel_size": vg.voxel_size,
                          "feature_dtype": torch.float32}},
        max_esdf_distance=1.0,
    )
    checker = WorldVoxelCollision(coll_cfg)
    checker.load_collision_model(world_cfg)
    print("collision_types after load_collision_model:", checker.collision_types)

    # feature values are NOT copied by load_collision_model -- push them explicitly
    checker.update_voxel_data(vg, env_idx=0)

    # Query points: one dead center of the occupied blob, one far outside it.
    query_points = np.array([
        [0.0, 0.0, 0.0],     # center -> should be INSIDE the occupied cube
        [0.45, 0.45, 0.45],  # near a corner, outside the cube -> should be FREE
    ])
    query = make_query(query_points, radius=0.0, tensor_args=tensor_args)

    from curobo.geom.sdf.world import CollisionQueryBuffer
    buf = CollisionQueryBuffer.initialize_from_shape(query.shape, tensor_args, checker.collision_types)
    weight = tensor_args.to_device([1.0])
    act_dist = tensor_args.to_device([0.0])
    env_query_idx = torch.zeros((query.shape[0],), device=tensor_args.device, dtype=torch.int32)

    dist = checker.get_sphere_distance(
        query, buf, weight, act_dist, env_query_idx=env_query_idx, compute_esdf=True
    )
    dist = dist.flatten().tolist()
    print(f"ESDF at center (expect > 0, inside obstacle): {dist[0]:.4f}")
    print(f"ESDF at corner (expect < 0, free space):       {dist[1]:.4f}")

    center_ok = dist[0] > 0
    corner_ok = dist[1] < 0
    print()
    if center_ok and corner_ok:
        print("PASS: WorldConfig.voxel + CollisionCheckerType.VOXEL correctly reports "
              "collision from an externally supplied ESDF array. Native path is viable.")
    else:
        print("FAIL: native VoxelGrid/ESDF path did not behave as expected "
              "(center_ok=%s, corner_ok=%s). Fall back to cuboid decomposition." % (center_ok, corner_ok))


if __name__ == "__main__":
    main()
