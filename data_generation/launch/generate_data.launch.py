"""Synthetic generation: random cubes (obstacle_adder) around the robot, one Reachability map per Scene.

curobo_ros (ROS 2 Jazzy, cuRobo v2) serves IK, obstacles and the voxel grid from one node,
`curobo_trajectory_planner`, started here as `unified_planner`. Our nodes reach it through
their `curobo_node` parameter.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collision_cache import cache_params  # noqa: E402  (next to this file)

CUROBO_NODE = 'unified_planner'


def scene_half_extent():
    """Half side of the Scene cube (m): the Station's scene_half_extent, 1.52 by default."""
    return float(os.environ.get('HERON_SCENE_HALF_EXTENT') or 1.52)


def curobo_node(context):
    """The cuRobo node; its collision cache depends on world_file and obj_max."""
    voxel_size = LaunchConfiguration('voxel_size')
    world_file = LaunchConfiguration('world_file')
    cameras_config_file = LaunchConfiguration('cameras_config_file')
    robot_config_file = LaunchConfiguration('robot_config_file')
    base_link = LaunchConfiguration('base_link')
    half = scene_half_extent()
    return [
        Node(
            package='curobo_ros',
            executable='curobo_trajectory_planner',
            name=CUROBO_NODE,
            output='screen',
            additional_env={'PYTHONFAULTHANDLER': '1'},
            parameters=[{
                'robot': 'doosan_m1013',
                'robot_config_file': robot_config_file,
                'base_link': base_link,
                'world_file': world_file,
                'cameras_config_file': cameras_config_file,
                'voxel_size': ParameterValue(voxel_size, value_type=float),
                # The voxel grid (get_voxel_grid) is this cube, centred on the base:
                # ceil(2 * half / voxel_size) voxels per side, origin at -half.
                'mapper_extent_xyz': [2.0 * half] * 3,
                'mapper_grid_center': [0.0, 0.0, 0.0],
                # obstacle_adder rejects a cube closer than this to the robot (5 cm, as
                # under Humble): get_collision_distance is > 0 within this distance.
                'collision_activation_distance': 0.05,
                # Nobody reads the sparse voxel topic here; it would rebuild the full grid
                # several times a second.
                'sparse_voxel_publish_rate': 0.0,
                # Sized to the world file plus the cubes obstacle_adder adds (collision_cache.py).
                **cache_params(world_file.perform(context),
                               int(LaunchConfiguration('obj_max').perform(context)),
                               voxel_layers=1),
            }],
        ),
    ]


def generate_launch_description():
    voxel_size = LaunchConfiguration('voxel_size')
    dataset_size = LaunchConfiguration('dataset_size')
    batch_size = LaunchConfiguration('batch_size')
    reach_max = LaunchConfiguration('reach_max')
    obj_max = LaunchConfiguration('obj_max')
    use_rosbag_mode = LaunchConfiguration('use_rosbag_mode')
    rosbag_player_name = LaunchConfiguration('rosbag_player_name')
    scene_stabilization_max_delay_sec = LaunchConfiguration('scene_stabilization_max_delay_sec')
    world_file = LaunchConfiguration('world_file')
    cameras_config_file = LaunchConfiguration('cameras_config_file')
    robot_config_file = LaunchConfiguration('robot_config_file')
    base_link = LaunchConfiguration('base_link')
    half = scene_half_extent()

    args = [
        DeclareLaunchArgument('voxel_size', default_value="0.02"),
        # Kept for the callers' command lines; cuRobo v2 warms its IK solver up on the first
        # GenerateRM request, for that request's batch_size.
        DeclareLaunchArgument('init_batch_size', default_value="1000"),
        DeclareLaunchArgument('dataset_size', default_value="5"),
        DeclareLaunchArgument('batch_size', default_value="1000"),
        DeclareLaunchArgument('reach_max', default_value="1.3"),
        DeclareLaunchArgument('obj_max', default_value="20"),
        DeclareLaunchArgument('use_rosbag_mode', default_value="false"),
        DeclareLaunchArgument('rosbag_player_name', default_value="rosbag2_player"),
        DeclareLaunchArgument('scene_stabilization_max_delay_sec', default_value="0.5"),
        # Empty: no fixed obstacles (the Station has no world file).
        DeclareLaunchArgument('world_file', default_value=""),
        # Empty: no camera (synthetic Scenes only hold the random cubes).
        DeclareLaunchArgument('cameras_config_file', default_value=""),
        # Empty: curobo_ros's built-in config of the `robot` descriptor (doosan_m1013).
        DeclareLaunchArgument('robot_config_file', default_value=""),
        DeclareLaunchArgument('base_link', default_value="base_0"),
    ]
    return LaunchDescription(args + [
        OpaqueFunction(function=curobo_node),
        Node(
            package='data_generation',
            executable='obstacle_adder',
            name='obstacle_adder',
            output='screen',
            parameters=[{'curobo_node': CUROBO_NODE}],
        ),
        Node(
            package='data_generation',
            executable='generate_data',
            name='generate_data',
            output='screen',
            parameters=[{
                'curobo_node': CUROBO_NODE,
                'voxel_size': ParameterValue(voxel_size, value_type=float),
                'scene_half_extent': half,
                'dataset_size': dataset_size,
                'batch_size': batch_size,
                'reach_max': reach_max,
                'obj_max': obj_max,
            }],
        ),
        Node(
            package='data_generation',
            executable='scene_manager',
            name='scene_manager',
            output='screen',
            # scene_manager exits once dataset_size Scenes are done: end the whole launch
            # then, or the cuRobo node keeps the task (and its GPU) until the time limit.
            on_exit=Shutdown(),
            parameters=[{
                'voxel_size': ParameterValue(voxel_size, value_type=float),
                'dataset_size': dataset_size,
                'batch_size': batch_size,
                'reach_max': reach_max,
                'obj_max': obj_max,
                'use_rosbag_mode': use_rosbag_mode,
                'rosbag_player_name': rosbag_player_name,
                'scene_stabilization_max_delay_sec': scene_stabilization_max_delay_sec,
            }],
        ),
    ])
