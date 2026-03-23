#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/solairith/turtlebot3_ws/install/setup.bash

echo "Waiting for RPi bringup..."
until ros2 topic list 2>/dev/null | grep -q "/odom"; do
    echo "RPi not ready yet, retrying in 2s..."
    sleep 2
done
echo "RPi is ready, starting Nav2..."

# Launch Nav2 in background
ros2 launch turtlebot3_navigation2 navigation2_headless.launch.py map:=/home/solairith/map.yaml &
NAV2_PID=$!

# Wait until AMCL is actually running before publishing initial pose
echo "Waiting for AMCL to be ready..."
until ros2 node list 2>/dev/null | grep -q "/amcl"; do
    echo "AMCL not ready yet, retrying in 2s..."
    sleep 2
done
echo "AMCL is ready, publishing initial pose..."
sleep 3
python3 /home/solairith/pet_caretaker_robot/publish_initial_pose.py
echo "Initial pose published."

# Keep script alive with Nav2 process
wait $NAV2_PID
