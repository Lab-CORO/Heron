"""Master map: IK of every pose of the Robot's reach cube, no obstacle (master_ik_data<voxel>.npz/.h5).

curobo_ros (ROS 2 Jazzy, cuRobo v2) is started as `unified_planner`; create_reachability_map
calls its ik_batch service through the `curobo_node` parameter.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CUROBO_NODE = 'unified_planner'


def generate_launch_description():
    voxel_size = LaunchConfiguration('voxel_size')
    batch_size = LaunchConfiguration('batch_size')
    robot_config_file = LaunchConfiguration('robot_config_file')
    base_link = LaunchConfiguration('base_link')

    args = [
        DeclareLaunchArgument('voxel_size', default_value="0.08"),
        DeclareLaunchArgument('batch_size', default_value="1000"),
        # Kept for the callers' command lines; the IK solver is warmed up for batch_size.
        DeclareLaunchArgument('init_batch_size', default_value="1000"),
        # Empty: curobo_ros's built-in config of the `robot` descriptor (doosan_m1013).
        DeclareLaunchArgument('robot_config_file', default_value=""),
        DeclareLaunchArgument('base_link', default_value="base_0"),
    ]
    return LaunchDescription(args + [
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
                'world_file': '',            # a master map has no obstacle
                'cameras_config_file': '',
                'voxel_size': ParameterValue(voxel_size, value_type=float),
                'sparse_voxel_publish_rate': 0.0,
            }],
        ),
        Node(
            package='data_generation',
            executable='create_reachability_map',
            name='create_reachability_map',
            output='screen',
            parameters=[{
                'curobo_node': CUROBO_NODE,
                'voxel_size': ParameterValue(voxel_size, value_type=float),
                'batch_size': ParameterValue(batch_size, value_type=int),
            }],
        ),
    ])
