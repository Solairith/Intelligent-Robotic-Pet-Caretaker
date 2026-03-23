import json
import os
import subprocess
import threading
import signal
import time
import math
import rclpy
import httpx
import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Initialise rclpy once at startup
if not rclpy.ok():
    rclpy.init()

app = FastAPI()

# File paths
COORDS_FILE     = os.path.expanduser('~/pet_caretaker_robot/coords.json')
WAYPOINTS_FILE  = os.path.expanduser('~/pet_caretaker_robot/waypoints.json')
NAV_GO_SCRIPT   = os.path.expanduser('~/pet_caretaker_robot/nav_go.py')
FCM_TOKEN_FILE  = os.path.expanduser('~/pet_caretaker_robot/fcm_token.json')
FIREBASE_CREDS  = os.path.expanduser('~/pet_caretaker_robot/firebase_credentials.json')

# Firebase initialisation
import firebase_admin
from firebase_admin import credentials, messaging

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDS)
    firebase_admin.initialize_app(cred)

# FCM token storage
def load_fcm_token() -> Optional[str]:
    if os.path.exists(FCM_TOKEN_FILE):
        with open(FCM_TOKEN_FILE, 'r') as f:
            data = json.load(f)
            return data.get('token')
    return None

def save_fcm_token(token: str):
    with open(FCM_TOKEN_FILE, 'w') as f:
        json.dump({'token': token}, f)

def send_notification(title: str, body: str):
    token = load_fcm_token()
    if token is None:
        print('No FCM token stored, skipping notification')
        return
    try:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token,
        )
        response = messaging.send(message)
        print(f'Notification sent: {response}')
    except Exception as e:
        print(f'Notification error: {e}')

# Default coordinates
DEFAULT_COORDS = {
    'bowl': {'x': -3.649, 'y': -0.658, 'yaw': -3.3},
    'home': {'x':  0.747, 'y': -0.240, 'yaw': -3.1413},
}

BATTERY_LOW_THRESHOLD = 25.0
PET_CLASSES = {15, 16}  # COCO: 15=cat, 16=dog
HEARTBEAT_TIMEOUT_SEC = 60
SPIN_DURATION_SEC = 15
PET_NOT_SEEN_THRESHOLD_SEC = 7200  # 2 hours

# Robot state
robot_state = {
    'status': 'idle',
    'bowl_status': None,
    'battery': None,
}
nav_process     = None
_stop_requested = False

# Patrol state
patrol_state = {
    'active': False,
    'interval_minutes': 20,
    'current_waypoint': None,
    'cycle': 0,
    'cycles_without_pet': 0,
    'pet_detected_this_cycle': False
}
patrol_stop_event = threading.Event()
patrol_thread = None

# Heartbeat
last_heartbeat_time = None

# YOLO inference node
_yolo_node = None
_yolo_lock = threading.Lock()

def get_yolo_node():
    global _yolo_node
    with _yolo_lock:
        if _yolo_node is None:
            from yolo_inference import start_node, get_node
            start_node()
            _yolo_node = get_node()
        return _yolo_node

# ROS helpers
_ros_node = None
_ros_lock = threading.Lock()

def get_ros_node():
    global _ros_node
    with _ros_lock:
        if _ros_node is None:
            _ros_node = rclpy.create_node('fastapi_server_node')
        return _ros_node

def get_battery_percentage() -> Optional[float]:
    try:
        result = subprocess.run(
            ['ros2', 'topic', 'echo', '--once', '/battery_state',
             '--field', 'percentage'],
            capture_output=True, text=True, timeout=5,
            env=os.environ.copy()
        )
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            try:
                value = float(line)
                return value * 100 if value <= 1.0 else value
            except ValueError:
                continue
        return None
    except Exception as e:
        print(f'Battery read error: {e}')
        return None

def get_robot_position() -> dict:
    try:
        result = subprocess.run(
            ['ros2', 'topic', 'echo', '--once', '/amcl_pose',
             '--field', 'pose.pose.position'],
            capture_output=True, text=True, timeout=5,
            env=os.environ.copy()
        )
        lines = result.stdout.strip().split('\n')
        x = float([l for l in lines if 'x:' in l][0].split(':')[1])
        y = float([l for l in lines if 'y:' in l][0].split(':')[1])
        return {'x': x, 'y': y}
    except Exception as e:
        print(f'Position read error: {e}')
        return {'x': 0.0, 'y': 0.0}

def spin_360():
    global nav_process
    try:
        spin_script = os.path.expanduser('~/pet_caretaker_robot/spin_360.py')
        nav_process = subprocess.Popen(['python3', spin_script])
        nav_process.wait(timeout=SPIN_DURATION_SEC + 5)
    except Exception as e:
        print(f'Spin error: {e}')
    finally:
        nav_process = None  

# Coordinate helpers
def load_coords() -> dict:
    if os.path.exists(COORDS_FILE):
        with open(COORDS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_COORDS.copy()

def save_coords(coords: dict):
    with open(COORDS_FILE, 'w') as f:
        json.dump(coords, f, indent=2)

def load_waypoints() -> list:
    if os.path.exists(WAYPOINTS_FILE):
        with open(WAYPOINTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_waypoints(waypoints: list):
    with open(WAYPOINTS_FILE, 'w') as f:
        json.dump(waypoints, f, indent=2)

def distance(p1: dict, p2: dict) -> float:
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def sort_waypoints_by_nearest(waypoints: list, current_pos: dict) -> list:
    remaining = waypoints.copy()
    sorted_wps = []
    pos = current_pos
    while remaining:
        nearest = min(remaining, key=lambda wp: distance(pos, wp))
        sorted_wps.append(nearest)
        remaining.remove(nearest)
        pos = nearest
    return sorted_wps

# Navigation helpers
def _nav_to(goal_x, goal_y, goal_yaw, status_label):
    global nav_process
    robot_state['status'] = status_label
    env = os.environ.copy()
    env['NAV_GOAL_X']   = str(goal_x)
    env['NAV_GOAL_Y']   = str(goal_y)
    env['NAV_GOAL_YAW'] = str(goal_yaw)
    nav_process = subprocess.Popen(['python3', NAV_GO_SCRIPT], env=env)
    nav_process.wait()
    nav_process = None

def _run_nav(goal_x, goal_y, goal_yaw, status_label, run_bowl_check=False):
    global _stop_requested
    _nav_to(goal_x, goal_y, goal_yaw, status_label)

    if run_bowl_check and not _stop_requested:
        robot_state['status'] = 'analysing_bowl'
        node = get_yolo_node()
        time.sleep(1.0)  # wait for inference to be ready
        node.start_bowl_check()
        timeout, start = 30, time.time()
        while time.time() - start < timeout:
            if _stop_requested:
                break
            decision = node.get_bowl_decision()
            if decision is not None:
                robot_state['bowl_status'] = decision
                if decision == 'foodbowl_empty':
                    threading.Thread(
                        target=send_notification,
                        args=('🍽️ Bowl Empty',
                              'The food bowl is empty. Please refill it.'),
                        daemon=True
                    ).start()
                elif decision == 'foodbowl_low':
                    threading.Thread(
                        target=send_notification,
                        args=('🍽️ Bowl Low',
                              'The food bowl is running low. Please refill it soon.'),
                        daemon=True
                    ).start()
                break
            time.sleep(0.5)

        if not _stop_requested:
            coords = load_coords()
            home = coords['home']
            _nav_to(home['x'], home['y'], home['yaw'], 'returning_home')

    robot_state['status'] = 'idle'

# Patrol loop
def _patrol_loop(interval_minutes: int):
    node = get_yolo_node()
    node.start_inference()
    node.start_pet_inference()

    coords = load_coords()
    bowl   = coords['bowl']
    patrol_state['cycle']                = 0
    patrol_state['cycles_without_pet']   = 0
    patrol_state['pet_detected_this_cycle'] = False

    while not patrol_stop_event.is_set():
        patrol_state['cycle'] += 1
        patrol_state['pet_detected_this_cycle'] = False

        waypoints = load_waypoints()
        if not waypoints:
            print('No waypoints set, stopping patrol')
            break

        all_points = waypoints + [{'x': bowl['x'], 'y': bowl['y'],
                                    'yaw': bowl['yaw'], 'is_bowl': True}]
        current_pos = get_robot_position()
        sorted_points = sort_waypoints_by_nearest(all_points, current_pos)

        for wp in sorted_points:
            if patrol_stop_event.is_set():
                break

            battery = get_battery_percentage()
            if battery is not None:
                robot_state['battery'] = round(battery, 1)
                if battery < BATTERY_LOW_THRESHOLD:
                    print(f'Battery low ({battery}%), stopping patrol')
                    patrol_stop_event.set()
                    patrol_state['active'] = False
                    _nav_to(coords['home']['x'], coords['home']['y'],
                            coords['home']['yaw'], 'returning_home')
                    robot_state['status'] = 'idle'
                    node.stop_pet_inference()
                    return

            is_bowl = wp.get('is_bowl', False)
            patrol_state['current_waypoint'] = wp

            _nav_to(wp['x'], wp['y'], wp['yaw'], 'patrolling')

            if patrol_stop_event.is_set():
                break

            if is_bowl:
                robot_state['status'] = 'analysing_bowl'
                time.sleep(1.0)
                node.start_bowl_check()
                timeout, start = 30, time.time()
                while time.time() - start < timeout:
                    decision = node.get_bowl_decision()
                    if decision is not None:
                        robot_state['bowl_status'] = decision
                        print(f'Bowl status: {decision}')
                        if decision == 'foodbowl_empty':
                            threading.Thread(
                                target=send_notification,
                                args=('🍽️ Bowl Empty',
                                      'The food bowl is empty. Please refill it.'),
                                daemon=True).start()
                        elif decision == 'foodbowl_low':
                            threading.Thread(
                                target=send_notification,
                                args=('🍽️ Bowl Low',
                                      'The food bowl is running low. Please refill it soon.'),
                                daemon=True).start()
                        break
                    time.sleep(0.5)
                else:
                    robot_state['bowl_status'] = 'unknown'
                    print('Bowl check timed out')
            else:
                robot_state['status'] = 'spinning'
                spin_360()

            # Check pet detection after each waypoint
            _check_pet_detection(node)

        if patrol_stop_event.is_set():
            break

        # End of cycle, check if pet was seen this cycle
        if not patrol_state['pet_detected_this_cycle']:
            patrol_state['cycles_without_pet'] += 1
            print(f'No pet detected this cycle. '
                  f'Consecutive cycles without pet: '
                  f'{patrol_state["cycles_without_pet"]}')
            if patrol_state['cycles_without_pet'] >= 1: 
                print('Pet not detected for 2 cycles — sending notification')
                patrol_state['cycles_without_pet'] = 0
                threading.Thread(
                    target=send_notification,
                    args=('🐾 Where is your pet?',
                          'Your pet has not been seen for 1 patrol cycles. '
                          'Want to check up on them?'),
                    daemon=True).start()
        else:
            patrol_state['cycles_without_pet'] = 0

        print(f'Patrol cycle {patrol_state["cycle"]} complete, returning home')
        coords = load_coords()
        _nav_to(coords['home']['x'], coords['home']['y'],
                coords['home']['yaw'], 'returning_home')

        print(f'Waiting {interval_minutes} minutes before next cycle')
        robot_state['status'] = 'idle'
        for _ in range(interval_minutes * 60):
            if patrol_stop_event.is_set():
                break
            time.sleep(1)

    patrol_state['active'] = False
    robot_state['status']  = 'idle'
    node.stop_pet_inference()

def _check_pet_detection(node):
    detected = node.check_and_clear_pet_detected()
    if detected:
        patrol_state['pet_detected_this_cycle'] = True
        print('Pet detected this cycle')

# Camera streamer
from sensor_msgs.msg import Image as RosImage

class CameraStreamer:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self._started = False
        self._fps = 0
        self._frame_count = 0
        self._fps_timer = time.time()

    def start(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        from cv_bridge import CvBridge
        bridge = CvBridge()
        node = rclpy.create_node('camera_streamer_node')

        def cb(msg):
            try:
                frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                with self.lock:
                    self.latest_frame = frame
                    self._frame_count += 1
                    now = time.time()
                    elapsed = now - self._fps_timer
                    if elapsed >= 1.0:
                        self._fps = round(self._frame_count / elapsed)
                        self._frame_count = 0
                        self._fps_timer = now
            except Exception as e:
                print(f'Camera frame error: {e}')

        node.create_subscription(RosImage, '/camera/image_raw', cb, 10)
        rclpy.spin(node)

    def get_frame(self, annotate=False):
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()

        if annotate and _yolo_node is not None:
            try:
                # Run both models and combine annotations
                bowl_results = _yolo_node.bowl_model(frame, verbose=False)
                pet_results  = _yolo_node.pet_model(frame, verbose=False)

                # Update latest bowl detection for inference status endpoint
                detection = _yolo_node._get_top_bowl_detection(bowl_results)
                with _yolo_node.lock:
                    _yolo_node.latest_result = detection

                
                # Combine annotations from both models
                frame = bowl_results[0].plot()

                # Filter pet results to only cat and dog before plotting
                import torch
                pet_boxes = pet_results[0].boxes
                if pet_boxes is not None and len(pet_boxes) > 0:
                    mask = torch.tensor([
                        int(c) in PET_CLASSES
                        for c in pet_boxes.cls.cpu().numpy()
                    ])
                    if mask.any():
                        pet_results[0].boxes = pet_boxes[mask]
                        frame = pet_results[0].plot(img=frame)
            except Exception as e:
                print(f'Inference error: {e}')
        elif _yolo_node is not None:
            try:
                # Non annotated stream still update latest bowl detection
                bowl_results = _yolo_node.bowl_model(frame, verbose=False)
                detection    = _yolo_node._get_top_bowl_detection(bowl_results)
                with _yolo_node.lock:
                    _yolo_node.latest_result = detection
            except Exception as e:
                print(f'Inference error: {e}')

        return frame

    def get_fps(self):
        with self.lock:
            return self._fps

camera_streamer = CameraStreamer()
camera_streamer.start()

# Heartbeat monitor
def _heartbeat_monitor():
    global last_heartbeat_time
    while True:
        time.sleep(10)
        if last_heartbeat_time is None:
            continue
        if time.time() - last_heartbeat_time > HEARTBEAT_TIMEOUT_SEC:
            last_heartbeat_time = None
            if robot_state['status'] == 'idle':
                pos = get_robot_position()
                coords = load_coords()
                home = coords['home']
                dist = distance(pos, home)
                if dist > 0.3:
                    print('Heartbeat lost, returning home')
                    t = threading.Thread(
                        target=_run_nav,
                        args=(home['x'], home['y'], home['yaw'], 'returning_home'),
                        daemon=True)
                    t.start()
                else:
                    print('Heartbeat lost but robot already at home, staying')
            else:
                print('Heartbeat lost but robot is busy, ignoring')

threading.Thread(target=_heartbeat_monitor, daemon=True).start()

# Models
class CoordUpdate(BaseModel):
    x: float
    y: float
    yaw: float = 0.0

class NavGoal(BaseModel):
    x: float
    y: float
    yaw: float = 0.0

class Waypoint(BaseModel):
    x: float
    y: float
    yaw: float = 0.0
    name: Optional[str] = None

class PatrolStart(BaseModel):
    interval_minutes: int = 20

class FcmTokenUpdate(BaseModel):
    token: str

# Status
@app.get('/status')
def get_status():
    battery = get_battery_percentage()
    if battery is not None:
        robot_state['battery'] = round(battery, 1)
    robot_state['stream_fps'] = camera_streamer.get_fps()
    return robot_state

# Coordinates
@app.get('/coords')
def get_coords():
    return load_coords()

@app.post('/coords/bowl')
def set_bowl(coord: CoordUpdate):
    coords = load_coords()
    coords['bowl'] = coord.model_dump()
    save_coords(coords)
    return {'message': 'Bowl coordinates updated', 'bowl': coords['bowl']}

@app.post('/coords/home')
def set_home(coord: CoordUpdate):
    coords = load_coords()
    coords['home'] = coord.model_dump()
    save_coords(coords)
    return {'message': 'Home coordinates updated', 'home': coords['home']}

# Waypoints
@app.get('/waypoints')
def get_waypoints():
    return load_waypoints()

@app.post('/waypoints')
def set_waypoints(waypoints: List[Waypoint]):
    wps = [wp.model_dump() for wp in waypoints]
    save_waypoints(wps)
    return {'message': f'{len(wps)} waypoints saved', 'waypoints': wps}

# Navigation
@app.post('/navigate')
def navigate(goal: NavGoal):
    if robot_state['status'] != 'idle':
        raise HTTPException(status_code=409, detail=f"Robot is busy: {robot_state['status']}")
    t = threading.Thread(
        target=_run_nav, args=(goal.x, goal.y, goal.yaw, 'navigating'), daemon=True)
    t.start()
    return {'message': 'Navigating to goal', 'goal': goal.model_dump()}

@app.post('/check_bowl')
def check_bowl():
    global _stop_requested
    if robot_state['status'] != 'idle':
        raise HTTPException(status_code=409, detail=f"Robot is busy: {robot_state['status']}")
    _stop_requested = False
    coords = load_coords()
    bowl = coords['bowl']
    t = threading.Thread(
        target=_run_nav,
        args=(bowl['x'], bowl['y'], bowl['yaw'], 'checking_bowl', True),
        daemon=True)
    t.start()
    return {'message': 'Bowl check started', 'bowl': bowl}

@app.post('/return_home')
def return_home():
    global _stop_requested
    if robot_state['status'] != 'idle':
        raise HTTPException(status_code=409, detail=f"Robot is busy: {robot_state['status']}")
    _stop_requested = False
    coords = load_coords()
    home = coords['home']
    t = threading.Thread(
        target=_run_nav, args=(home['x'], home['y'], home['yaw'], 'returning_home'),
        daemon=True)
    t.start()
    return {'message': 'Returning home', 'home': home}

@app.post('/stop')
def emergency_stop():
    global nav_process, _stop_requested
    _stop_requested = True

    if patrol_state['active']:
        patrol_stop_event.set()
        patrol_state['active'] = False

    if nav_process and nav_process.poll() is None:
        nav_process.send_signal(signal.SIGUSR1)
        try:
            nav_process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            nav_process.kill()
    nav_process = None
    robot_state['status'] = 'idle'
    return {'message': 'Robot stopped'}

# Patrol
@app.post('/patrol/start')
def start_patrol(params: PatrolStart):
    global patrol_thread

    if patrol_state['active']:
        raise HTTPException(status_code=409, detail='Patrol already active')
    if robot_state['status'] != 'idle':
        raise HTTPException(status_code=409, detail=f"Robot is busy: {robot_state['status']}")

    waypoints = load_waypoints()
    if not waypoints:
        raise HTTPException(status_code=400, detail='No waypoints set. Add waypoints first.')

    interval = max(20, params.interval_minutes)
    patrol_stop_event.clear()
    patrol_state['active'] = True
    patrol_state['interval_minutes'] = interval

    patrol_thread = threading.Thread(
        target=_patrol_loop, args=(interval,), daemon=True)
    patrol_thread.start()

    return {'message': 'Patrol started', 'interval_minutes': interval,
            'waypoints': len(waypoints)}

@app.post('/patrol/stop')
def stop_patrol():
    if not patrol_state['active']:
        raise HTTPException(status_code=409, detail='Patrol is not active')
    patrol_stop_event.set()
    patrol_state['active'] = False
    return {'message': 'Patrol stopping, returning home'}

@app.get('/patrol/status')
def get_patrol_status():
    return {
        'active': patrol_state['active'],
        'cycle': patrol_state['cycle'],
        'interval_minutes': patrol_state['interval_minutes'],
        'current_waypoint': patrol_state['current_waypoint'],
        'cycles_without_pet':   patrol_state['cycles_without_pet'],
        'pet_detected_this_cycle': patrol_state['pet_detected_this_cycle'],
    }

# YOLO inference toggle
@app.post('/inference/start')
def start_inference():
    node = get_yolo_node()
    node.start_inference()
    return {'message': 'Inference started'}

@app.post('/inference/stop')
def stop_inference():
    node = get_yolo_node()
    node.stop_inference()
    return {'message': 'Inference stopped'}

@app.get('/inference/status')
def inference_status():
    node = get_yolo_node()
    return {
        'inference_active': node.inference_active,
        'latest_result': node.get_latest_pet_result(),
        'bowl_status': robot_state.get('bowl_status'),
    }

# FCM
@app.post('/fcm/register')
def register_fcm_token(body: FcmTokenUpdate):
    save_fcm_token(body.token)
    print(f'FCM token saved: {body.token[:20]}...')
    return {'message': 'FCM token registered'}

# Map
@app.get('/map')
def get_map():
    map_path = os.path.expanduser('~/map.png')
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail='Map not found')
    return FileResponse(map_path, media_type='image/png')

# Heartbeat
@app.post('/heartbeat')
def heartbeat():
    global last_heartbeat_time
    last_heartbeat_time = time.time()
    return {'message': 'ok'}

# Stream
@app.get('/stream')
async def stream(annotate: bool = False):
    def generate():
        while True:
            frame = camera_streamer.get_frame(annotate=annotate)
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
