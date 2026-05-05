import launch
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
import yaml

def load_ros1_yaml_as_params(yaml_file_path):
    """Load ROS1 format YAML and convert to ROS2 parameter dict"""
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

rviz_cfg = os.path.join(get_package_share_directory("fast_lio_sam"), "rviz_cfg", "sam_ros2.rviz" )

print(rviz_cfg)

config_file = os.path.join(
    get_package_share_directory("fast_lio_sam"), "config", "mapping", "mid360.yaml"
)

# Load ROS1 format YAML and convert params
yaml_params = load_ros1_yaml_as_params(config_file)

fast_lio_params = [
    {'sam_enable': True},
    {'feature_extract_enable': False},
    {'point_filter_num': 3},
    {'max_iteration': 3},
    {'filter_size_surf': 0.5},
    {'filter_size_map': 0.5},
    {'cube_side_length': 1000.0},
    {'runtime_pos_log_enable': False},
    yaml_params,
]

def generate_launch_description():
    fast_lio_sam = Node(
        package='fast_lio_sam',
        executable='fastlio_mapping',
        output='screen',
        parameters=fast_lio_params,
        prefix='xterm -e gdb -ex run --args'
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
        fast_lio_rviz
    ])