# Intelligent Robotic Pet Caretaker

A final year engineering project that transforms a TurtleBot3 Burger into an autonomous pet caretaker robot. The system autonomously patrols a home environment, monitors a pet's food bowl, detects the presence of a pet, and provides real-time remote monitoring and control via a Flutter Android application.

---

## System Architecture

| Component | Hardware | Role |
|-----------|----------|------|
| Robot | TurtleBot3 Burger (Raspberry Pi 3B, OpenCR 1.0) | Onboard robot operations |
| Remote PC | Ubuntu 24.04, Nvidia GPU | Navigation, inference, backend server |
| Mobile App | Flutter (Android) | Remote monitoring and control |
| Network | Tailscale VPN | Secure remote connectivity |

---

## Features

- **Autonomous Patrol** — Robot navigates through user-defined waypoints at configurable intervals
- **360° Waypoint Inspection** — Performs a full rotation at each waypoint for complete area coverage
- **Food Bowl Monitoring** — YOLO26 model detects bowl status (full, low, empty) using majority vote across 15 frames
- **Pet Detection** — YOLO26 model detects cats and dogs during patrol using COCO dataset classes
- **Push Notifications** — Firebase Cloud Messaging delivers alerts when bowl is low/empty or pet is not detected
- **Live Camera Stream** — MJPEG stream served via FastAPI with optional AI annotation overlay
- **Remote Control** — Teleoperation via rosbridge WebSocket connection
- **Battery Monitoring** — Patrol automatically stops when battery drops below 25%
- **Emergency Stop** — Immediate halt of all navigation from the mobile app

---

## Repository Structure

```
Intelligent-Robotic-Pet-Caretaker/
├── backend/
│   ├── server.py               # FastAPI backend server
│   ├── yolo_inference.py       # YOLO26 ROS 2 inference node
│   ├── nav_go.py               # Nav2 navigation script
│   ├── spin_360.py             # 360° waypoint spin behaviour
│   ├── publish_initial_pose.py # AMCL initialisation at boot
│   ├── start_nav2.sh           # Nav2 startup script
│   ├── coords.json             # Home and bowl coordinates
│   ├── waypoints.json          # Patrol waypoints
│   └── requirements.txt        # Python dependencies
└── flutter_app/
    └── lib/
        ├── main.dart                      # App entry point, Firebase init
        ├── teleop_manager.dart            # Rosbridge teleoperation
        ├── screens/
        │   ├── map_screen.dart            # Map UI with drag-and-drop markers
        │   └── patrol_screen.dart         # Patrol control screen
        └── services/
            ├── api_service.dart           # FastAPI HTTP client
            └── robot_state_provider.dart  # App state management
```

---

## Hardware Components

- TurtleBot3 Burger
  - Raspberry Pi 3B
  - OpenCR 1.0
  - LDS-01 LiDAR
  - Pi Camera V2
- Remote PC with Ubuntu 24.04 and Nvidia GPU
- Android smartphone

---

## Software Requirements

### Remote PC
- ROS 2 Jazzy
- Nav2
- Cartographer
- Python 3.12

### Python Dependencies
See `backend/requirements.txt`. Install with:
```bash
pip install -r backend/requirements.txt --break-system-packages
```

### Flutter App
- Flutter SDK
- Android SDK

---

## Setup

### Prerequisites
Before proceeding, complete the official ROBOTIS TurtleBot3 setup guide for your platform:
[https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)

This includes setting up ROS 2 on both the Remote PC and Raspberry Pi, installing TurtleBot3 packages, and configuring the OpenCR board.

### 1. SLAM Map
Build a map of your environment using Cartographer before running the system:
```bash
ros2 launch turtlebot3_cartographer cartographer.launch.py
```

### 2. Configure Coordinates
Update `backend/coords.json` with your home and bowl positions, and `backend/waypoints.json` with your patrol waypoints.

### 3. YOLO Models
Place your trained models in `backend/models/`:
- `best.pt` — Fine-tuned food bowl detection model
- `yolo26m.pt` — COCO pretrained model for pet detection

### 4. Firebase Setup
Create a Firebase project and place your `firebase_credentials.json` in the `backend/` directory and `google-services.json` in `flutter_app/android/app/`.

### 5. Start Services
On the Remote PC, the following systemd services should be enabled:
- `turtlebot3-nav2.service`
- `turtlebot3-server.service`
- `rosbridge.service`

On the RPi:
- `turtlebot3-bringup.service`
- `v4l2-camera.service`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Robot status and battery |
| GET | `/stream` | Live MJPEG camera stream |
| POST | `/check_bowl` | Navigate to bowl and check food level |
| POST | `/return_home` | Navigate robot to home position |
| POST | `/stop` | Emergency stop |
| POST | `/patrol/start` | Start autonomous patrol |
| POST | `/patrol/stop` | Stop autonomous patrol |
| GET | `/patrol/status` | Current patrol state |
| POST | `/fcm/register` | Register FCM token for notifications |

---

## Acknowledgements

- [ROBOTIS TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/)
- [Nav2](https://nav2.org/)
- [Ultralytics YOLO26](https://docs.ultralytics.com/)
- [Cartographer ROS](https://google-cartographer-ros.readthedocs.io/)
