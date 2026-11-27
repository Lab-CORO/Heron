"""Real generation: Scenes seen by the Station's camera in a replayed rosbag.

The bag is played in a loop; scene_manager pauses it, asks for a Reachability map of the
current Scene, then resumes it. curobo_ros (ROS 2 Jazzy, cuRobo v2) builds the Scene from
the depth camera described by `cameras_config_file`, and resolves the camera pose from the
bag's /tf (base_link parameter -> the camera frame) when the config has `extrinsics: []`.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collision_cache import cache_params  # noqa: E402  (next to this file)
from stop_on_death import stop_on_death  # noqa: E402

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
            on_exit=stop_on_death('curobo_trajectory_planner'),
            additional_env={'PYTHONFAULTHANDLER': '1'},
            parameters=[{
                'robot': 'doosan_m1013',
                'robot_config_file': robot_config_file,
                'base_link': base_link,
                'world_file': world_file,
                'cameras_config_file': cameras_config_file,
                'voxel_size': ParameterValue(voxel_size, value_type=float),
                # Scene cube centred on the base: ceil(2 * half / voxel_size) voxels per side.
                'mapper_extent_xyz': [2.0 * half] * 3,
                'mapper_grid_center': [0.0, 0.0, 0.0],
                'mapper_voxel_size': ParameterValue(voxel_size, value_type=float),
                'collision_activation_distance': 0.05,
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
    bag_file = LaunchConfiguration('bag_file')
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
        DeclareLaunchArgument('use_rosbag_mode', default_value="true"),
        DeclareLaunchArgument('rosbag_player_name', default_value="rosbag2_player"),
        DeclareLaunchArgument('scene_stabilization_max_delay_sec', default_value="0.5"),
        DeclareLaunchArgument('world_file', default_value=""),
        # Required: the rendered cameras.yaml (run_generation.sh passes it).
        DeclareLaunchArgument('cameras_config_file'),
        DeclareLaunchArgument('bag_file'),
        DeclareLaunchArgument('robot_config_file', default_value=""),
        DeclareLaunchArgument('base_link', default_value="base_0"),
    ]
    return LaunchDescription(args + [
        # /tf and /tf_static are replayed so the camera pose resolves from TF.
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_file, '--loop',
                 '--topics',
                 '/masked_depth_image/camera_info', '/masked_depth_image/image_raw',
                 '/tf', '/tf_static'],
            output='screen',
            on_exit=stop_on_death('ros2 bag play'),
        ),
        OpaqueFunction(function=curobo_node),
        Node(
            package='data_generation',
            executable='obstacle_adder',
            name='obstacle_adder',
            output='screen',
            on_exit=stop_on_death('obstacle_adder'),
            parameters=[{'curobo_node': CUROBO_NODE}],
        ),
        Node(
            package='data_generation',
            executable='generate_data',
            name='generate_data',
            output='screen',
            on_exit=stop_on_death('generate_data'),
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
