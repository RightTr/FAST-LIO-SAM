import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def load_ros1_yaml_as_params(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten_dict(config)


rviz_cfg = os.path.join(
    get_package_share_directory("fast_lio_sam"),
    "rviz_cfg",
    "sam_ros2.rviz",
)

config_file = os.path.join(
    get_package_share_directory("fast_lio_sam"),
    "config",
    "mapping",
    "m3dgr.yaml",
)

yaml_params = load_ros1_yaml_as_params(config_file)

fast_lio_params = [
    {'sam_enable': True},
    {'feature_extract_enable': False},
    {'point_filter_num': 3},
    {'max_iteration': 3},
    {'filter_size_surf': 0.5},
    {'filter_size_map': 0.5},
    {'cube_side_length': 1000.0},
    yaml_params,
]


def generate_launch_description():
    fast_lio_sam = Node(
        package='fast_lio_sam',
        executable='fastlio_mapping',
        output='screen',
        parameters=fast_lio_params,
    )

    fast_lio_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
    )

    return LaunchDescription([
        fast_lio_sam,
        fast_lio_rviz,
    ])
