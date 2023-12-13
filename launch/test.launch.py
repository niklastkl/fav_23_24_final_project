from launch_ros.actions import Node, PushRosNamespace

from ament_index_python.packages import get_package_share_path
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

    group = GroupAction([
        PushRosNamespace(LaunchConfiguration('vehicle_name')),
        Node(executable='mapper.py',
             package='final_project',
             parameters=[
                 LaunchConfiguration('mapping_params',
                                     default=mapping_params_file_path)
             ]),
        Node(executable='scenario_node.py', package='final_project'),
    ])
    launch_description.add_action(group)
    return launch_description
