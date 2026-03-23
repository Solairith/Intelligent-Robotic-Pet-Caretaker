import os
import rclpy
import signal
import math
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
    

    # Hardcoded start position — update via app map feature in future
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.747
    initial_pose.pose.position.y = -0.240
    yaw = -3.1413
    initial_pose.pose.orientation.z = math.sin(yaw / 2)
    initial_pose.pose.orientation.w = math.cos(yaw / 2)

    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()

    goal_x   = float(os.environ.get('NAV_GOAL_X',   '-3.649'))
    goal_y   = float(os.environ.get('NAV_GOAL_Y',   '-0.658'))
    goal_yaw = float(os.environ.get('NAV_GOAL_YAW', '-3.3'))
    
    goal = make_pose(navigator, goal_x, goal_y, yaw=goal_yaw)
    navigator.goToPose(goal)

    while not navigator.isTaskComplete():
        if stop_requested:
            navigator.cancelTask()
            break
        feedback = navigator.getFeedback()
        if feedback:
            print(f'Distance remaining: {round(feedback.distance_remaining, 2)} metres')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Robot arrived at goal successfully!')
    elif result == TaskResult.CANCELED:
        print('Navigation was cancelled')
    elif result == TaskResult.FAILED:
        print('Navigation failed')

    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
