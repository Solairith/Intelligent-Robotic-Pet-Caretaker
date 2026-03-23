#!/usr/bin/env python3
import math
import time
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped

HOME_X   =  0.110
HOME_Y   = -0.027
HOME_YAW =  0.014546297382826012

def main():
    rclpy.init()
    node = rclpy.create_node('initial_pose_publisher')
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    for _ in range(10):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.pose.position.x = HOME_X
        msg.pose.pose.position.y = HOME_Y
        msg.pose.pose.orientation.z = math.sin(HOME_YAW / 2)
        msg.pose.pose.orientation.w = math.cos(HOME_YAW / 2)
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.06853891945200942
        pub.publish(msg)
        time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
