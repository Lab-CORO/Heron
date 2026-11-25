"""Collision cache sizes the launch files give curobo_ros (data_generation/launch/collision_cache.py)."""
import importlib.util
from pathlib import Path

LAUNCH = Path(__file__).resolve().parents[1] / "data_generation" / "launch"
spec = importlib.util.spec_from_file_location("collision_cache", LAUNCH / "collision_cache.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def test_leeloo_synthetic_holds_its_world_and_obj_max_cubes():
    assert cc.cache_params(str(LAUNCH.parent / "config" / "leeloo_world.yaml"), 20, 1) == {
        "collision_cache_cuboid": 32, "collision_cache_mesh": 0, "collision_cache_voxel": 1}


def test_no_world_and_no_obstacle_means_no_cache():
    assert cc.cache_params("", 0, 0) == {
        "collision_cache_cuboid": 0, "collision_cache_mesh": 0, "collision_cache_voxel": 0}


def test_sizes_round_up_to_a_power_of_two_and_never_below_the_need(tmp_path):
    world = tmp_path / "w.yaml"
    world.write_text("cuboid:\n  a: {dims: [1, 1, 1], pose: [0, 0, 0, 1, 0, 0, 0]}\n"
                     "mesh:\n  m: {file_path: x.stl, pose: [0, 0, 0, 1, 0, 0, 0]}\n")
    for extra in range(0, 70):
        p = cc.cache_params(str(world), extra, 1)
        n = p["collision_cache_cuboid"]
        assert n >= 1 + extra and n & (n - 1) == 0
        assert p["collision_cache_mesh"] == 1
