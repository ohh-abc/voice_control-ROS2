from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_control',
            executable='voice_control',
            name='voice_control_node',
            output='screen'
        )
    ])
