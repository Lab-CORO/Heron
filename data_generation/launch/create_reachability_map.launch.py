from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    voxel_size = LaunchConfiguration('voxel_size')
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value="0.08",  # Set the default value
    )
    batch_size = LaunchConfiguration('batch_size')
    batch_size_arg = DeclareLaunchArgument(
        'batch_size',
        default_value="1000",  # Set the default value
    )
    init_batch_size = LaunchConfiguration('init_batch_size')
    init_batch_size_arg = DeclareLaunchArgument(
        'init_batch_size',
        default_value="1000",  # Set the default value
    )
    # Robot selection: empty robot_config_file -> curobo falls back to its default (m1013 / leeloo)
    robot_config_file = LaunchConfiguration('robot_config_file')
    robot_config_file_arg = DeclareLaunchArgument(
        'robot_config_file',
        default_value="",
    )
    base_link = LaunchConfiguration('base_link')
    base_link_arg = DeclareLaunchArgument(
        'base_link',
        default_value="base_0",  # curobo ConfigManager legacy default (m1013/leeloo)
    )


    return LaunchDescription([
        voxel_size_arg,
        batch_size_arg,
        init_batch_size_arg,
        robot_config_file_arg,
        base_link_arg,
        Node(
            package='curobo_ros',
            executable='curobo_ik',
            name='curobo_ik',
            output='screen',
            parameters=[
                    {"voxel_size": voxel_size,
                    "init_batch_size": init_batch_size,
                    "robot_config_file": robot_config_file,
                    "base_link": base_link}
                ]
        ),
        # create_reachability_map
        Node(
        package='data_generation',
        executable='create_reachability_map',
        name='create_reachability_map',
        output='screen',
        parameters=[
                {"voxel_size": voxel_size,
                "batch_size": batch_size,}
        ])
        ])