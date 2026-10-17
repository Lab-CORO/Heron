from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    curobo_ik + generate_data only, for the 'scene' mode of the unified pipeline
    (obstacles come from a user-supplied HDF5 scene file, injected at runtime by
    pipeline/load_scene.py via ROS2 services -- not from obstacle_adder's random
    cubes or scene_manager's rosbag orchestration, so neither node is started here).
    """
    voxel_size = LaunchConfiguration('voxel_size')
    init_batch_size = LaunchConfiguration('init_batch_size')
    world_file = LaunchConfiguration('world_file')
    cameras_config_file = LaunchConfiguration('cameras_config_file')
    robot_config_file = LaunchConfiguration('robot_config_file')
    base_link = LaunchConfiguration('base_link')

    voxel_size_arg = DeclareLaunchArgument('voxel_size', default_value="0.02")
    init_batch_size_arg = DeclareLaunchArgument('init_batch_size', default_value="1000")
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value="/home/ros2_ws/src/CapaciNet/data_generation/config/leeloo_world.yaml",
    )
    cameras_config_file_arg = DeclareLaunchArgument(
        'cameras_config_file',
        default_value="/home/ros2_ws/src/CapaciNet/data_generation/config/cameras.yaml",
    )
    # Robot selection: empty robot_config_file -> curobo falls back to its default (m1013 / leeloo)
    robot_config_file_arg = DeclareLaunchArgument('robot_config_file', default_value="")
    base_link_arg = DeclareLaunchArgument('base_link', default_value="base_0")

    return LaunchDescription([
        voxel_size_arg,
        init_batch_size_arg,
        world_file_arg,
        cameras_config_file_arg,
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
                 "world_file": world_file,
                 "cameras_config_file": cameras_config_file,
                 "robot_config_file": robot_config_file,
                 "base_link": base_link}
            ]
        ),
        Node(
            package='data_generation',
            executable='generate_data',
            name='generate_data',
            output='screen',
            parameters=[{"voxel_size": voxel_size}],
            # Without this, a dead generate_data leaves curobo_ik running and holding
            # the GPU: the /generate_rm client then blocks until the SLURM wall clock,
            # burning a whole A100 idle. Shutting the launch down instead lets
            # run_generation.sh notice and abort within seconds.
            on_exit=Shutdown(),
        ),
    ])
