import launch
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

rviz_cfg = os.path.join( get_package_share_directory("fast_lio_sam"), "rviz_cfg", "loam_livox_ros2.rviz" )

print(rviz_cfg)

config_path = PathJoinSubstitution(
    [get_package_share_directory("fast_lio_sam"), "config_ROS2", "reloc", "mid360.yaml"]
)

fast_lio_params = [
    {'feature_extract_enable': False},
    {'point_filter_num': 3},
    {'max_iteration': 3},
    {'filter_size_surf': 0.5},
    {'filter_size_map': 0.5},
    {'cube_side_length': 1000.0},
    {'runtime_pos_log_enable': False},
    config_path
]

def generate_launch_description():
    fast_lio_sam = Node(
        package='fast_lio_sam',
        executable='fastlio_mapping',
        name='laserMapping',
        output='screen',
        parameters=fast_lio_params
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