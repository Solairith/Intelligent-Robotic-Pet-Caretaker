#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import math
import time

class SpinNode(Node):
    def __init__(self):
        super().__init__('spin_360_node')
        self.pub = self.create_publisher(
            TwistStamped, '/cmd_vel', 10)

    def spin(self, angular_speed=0.5):
        # Calculate time needed for full 360 at given speed
        duration = (2 * math.pi) / angular_speed
        start = time.time()
        msg = TwistStamped()
        msg.header.frame_id = ''
        while time.time() - start < duration:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.angular.z = angular_speed
            self.pub.publish(msg)
            time.sleep(0.1)
        # Stop
        msg.twist.angular.z = 0.0
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = SpinNode()
    node.spin(angular_speed=0.5)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
