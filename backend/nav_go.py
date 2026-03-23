#!/usr/bin/env python3
import os
import math
import signal
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped

stop_requested = False

def handle_stop(signum, frame):
    global stop_requested
    print('Stop signal received, cancelling task...')
    stop_requested = True

signal.signal(signal.SIGUSR1, handle_stop)

def make_pose(navigator, x, y, yaw=0.0):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose

def main():
    rclpy.init()
    navigator = BasicNavigator()

    goal_x   = float(os.environ.get('NAV_GOAL_X',   '0.0'))
    goal_y   = float(os.environ.get('NAV_GOAL_Y',   '0.0'))
    goal_yaw = float(os.environ.get('NAV_GOAL_YAW', '0.0'))

    print(f'Navigating to x={goal_x}, y={goal_y}, yaw={goal_yaw}')

    # send goal to Nav2
    goal = make_pose(navigator, goal_x, goal_y, yaw=goal_yaw)
    navigator.goToPose(goal)

    while not navigator.isTaskComplete():
        if stop_requested:
            navigator.cancelTask()
            break
        navigator.getFeedback()


    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Arrived at goal successfully!')
    elif result == TaskResult.CANCELED:
        print('Navigation was cancelled')
    elif result == TaskResult.FAILED:
        print('Navigation failed')

    rclpy.shutdown()

if __name__ == '__main__':
    main()
