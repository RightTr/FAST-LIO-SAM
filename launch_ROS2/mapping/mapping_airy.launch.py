import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def flatten_params(values, prefix=''):
    params = {}
    for key, value in values.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(value, dict):
            params.update(flatten_params(value, name))
        else:
            params[name] = value
    return params


def launch_nodes(context):
    input_mode = LaunchConfiguration('input').perform(context)
    bag_path = LaunchConfiguration('bag').perform(context)
    if input_mode not in ('online', 'offline'):
        raise ValueError("input must be 'online' or 'offline'")
    if input_mode == 'offline' and not bag_path:
        raise ValueError('bag is required when input:=offline')

    share = get_package_share_directory('fast_lio_sam')
    with open(os.path.join(share, 'config', 'mapping', 'airy.yaml'), encoding='utf-8') as stream:
        params = flatten_params(yaml.safe_load(stream))
    params.update({
        'sam_enable': True,
        'feature_extract_enable': False,
        'point_filter_num': 4,
        'max_iteration': 3,
        'filter_size_surf': 0.5,
        'filter_size_map': 0.5,
        'cube_side_length': 1000.0,
        'offline/input_mode': input_mode,
        'offline/bag_path': bag_path if input_mode == 'offline' else '',
    })
    return [
        Node(package='fast_lio_sam', executable='fastlio_mapping',
             output='screen', parameters=[params]),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', os.path.join(share, 'rviz_cfg', 'sam_ros2.rviz')]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input', default_value='online',
                              description='Input mode: online or offline'),
        DeclareLaunchArgument('bag', default_value='',
                              description='ROS 2 rosbag directory for offline input'),
        OpaqueFunction(function=launch_nodes),
    ])
