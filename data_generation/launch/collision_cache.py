"""Collision cache sizes for curobo_ros's unified_planner, from what a run can hold.

cuRobo v2 checks every pre-allocated cache slot, even empty ones, so the cache is sized to the
run instead of curobo_ros's default 100 cuboids + 100 meshes (4x slower IK). Too small is
refused loudly by add_object (success=False), never ignored.
"""
import yaml


def _next_pow2(n: int) -> int:
    return 0 if n <= 0 else 1 << (n - 1).bit_length()


def world_obstacles(world_file: str) -> tuple[int, int]:
    """(cuboids, meshes) of a world file; (0, 0) without one."""
    if not world_file:
        return 0, 0
    with open(world_file) as f:
        world = yaml.safe_load(f) or {}
    return len(world.get('cuboid') or {}), len(world.get('mesh') or {})


def cache_params(world_file: str, extra_cuboids: int, voxel_layers: int) -> dict:
    """unified_planner parameters: the world's obstacles plus `extra_cuboids` added at run
    time, rounded up to a power of 2; `voxel_layers` ESDF layers (0 = none)."""
    cuboids, meshes = world_obstacles(world_file)
    return {
        'collision_cache_cuboid': _next_pow2(cuboids + max(0, extra_cuboids)),
        'collision_cache_mesh': _next_pow2(meshes),
        'collision_cache_voxel': voxel_layers,
    }
