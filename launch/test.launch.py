from ament_index_python.packages import get_package_share_path
from launch_ros.actions import Node, PushROSNamespace

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    launch_description = LaunchDescription()
    arg = DeclareLaunchArgument('vehicle_name')
    launch_description.add_action(arg)

    package_path = get_package_share_path('final_project')
    mapping_params_file_path = str(package_path / 'config/mapping_params.yaml')
    urdf_file_path = str(package_path / 'test.urdf')

    scenario_arg = DeclareLaunchArgument(
        name='scenario',
        default_value=str(1),
        description='The number of the scenario')
    launch_description.add_action(scenario_arg)
    viewpoints_arg = DeclareLaunchArgument(
        name='viewpoints',
        default_value=str(0),
        description='The number of the viewpoints file')
    launch_description.add_action(viewpoints_arg)

    group = GroupAction([
        PushROSNamespace(LaunchConfiguration('vehicle_name')),
        Node(
            executable='mapper.py',
            package='final_project',
            parameters=[
                LaunchConfiguration('mapping_params',
                                    default=mapping_params_file_path)
            ],
        ),
        Node(
            executable='scenario_node.py',
            package='final_project',
            parameters=[
                {
                    'scenario': LaunchConfiguration('scenario'),
                    'viewpoints': LaunchConfiguration('viewpoints'),
                },
            ],
        ),
        Node(executable='path_planner.py',
             package='final_project',
             arguments=['--ros-args', '--log-level', 'warn']),
        Node(
            executable='path_follower.py',
            package='final_project',
        ),
        Node(
            executable='position_controller.py',
            package='final_project',
        ),
        Node(
            executable='yaw_controller.py',
            package='final_project',
        ),
        Node(
            executable='robot_marker_publisher.py',
            package='final_project',
        ),
    ])
    launch_description.add_action(group)
    return launch_description
