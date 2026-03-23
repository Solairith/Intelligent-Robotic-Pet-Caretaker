#!/usr/bin/env python3
import threading
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

BOWL_MODEL_PATH = '/home/solairith/pet_caretaker_robot/models/best.pt'
PET_MODEL_PATH  = '/home/solairith/pet_caretaker_robot/models/yolo26m.pt'
DECISION_FRAMES = 15
PET_CLASSES     = {15, 16}  # class in COCO dataset for cat and dog

class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        self.bridge = CvBridge()

        # Load both models
        self.bowl_model = YOLO(BOWL_MODEL_PATH)
        self.pet_model  = YOLO(PET_MODEL_PATH)
        self.get_logger().info('YOLO models loaded')

        # Bowl check state
        self.inference_active  = False
        self.collecting        = False
        self.collected_frames  = []
        self.bowl_decision     = None

        # Pet detection state
        self.pet_inference_active = False
        self.latest_pet_result    = None 
        self.pet_detected_since_last_check = False

        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_annotated = self.create_publisher(
            Image, '/camera/image_annotated', 10)

        self.get_logger().info('YOLO inference node ready')

    def image_callback(self, msg):
        
        with self.lock:
            bowl_active = self.inference_active
            collecting = self.collecting
            pet_active = self.pet_inference_active

        if not bowl_active and not collecting and not pet_active:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Bowl model inference
        bowl_results = None
        if bowl_active or collecting:
            bowl_results = self.bowl_model(frame, verbose=False)
            detection = self._get_top_bowl_detection(bowl_results)

            with self.lock:
                if self.collecting:
                    if detection:
                        self.collected_frames.append(detection)
                    else:
                        self.collected_frames.append(
                            {'class': 'none', 'confidence': 0.0})
                    if len(self.collected_frames) >= DECISION_FRAMES:
                        self.bowl_decision = self._majority_vote(
                            self.collected_frames)
                        self.collected_frames = []
                        self.collecting = False
                        self.get_logger().info(
                            f'Bowl decision: {self.bowl_decision}')

        # Pet model inference
        pet_results = None
        if pet_active:
            pet_results = self.pet_model(frame, verbose=False)
            pet_detection = self._get_top_pet_detection(pet_results)
            with self.lock:
                self.latest_pet_result = pet_detection
                if pet_detection is not None:
                    self.pet_detected_since_last_check = True

        # Publish annotated frame
        if bowl_active or pet_active:
            annotated = frame.copy()
            if bowl_results is not None:
                annotated = bowl_results[0].plot(img=annotated)
            if pet_results is not None:
                annotated = pet_results[0].plot(img=annotated)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)

    def _get_top_bowl_detection(self, results):
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        confidences = boxes.conf.cpu().numpy()
        best_idx    = int(np.argmax(confidences))
        class_id    = int(boxes.cls[best_idx].cpu().numpy())
        confidence  = float(confidences[best_idx])
        class_name  = self.bowl_model.names[class_id]
        return {'class': class_name, 'confidence': round(confidence, 3)}

    def _get_top_pet_detection(self, results):
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        # Filter to only cat and dog classes
        pet_detections = []
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].cpu().numpy())
            if class_id in PET_CLASSES:
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = self.pet_model.names[class_id]
                pet_detections.append(
                    {'class': class_name, 'confidence': round(confidence, 3)})
        if not pet_detections:
            return None
        return max(pet_detections, key=lambda d: d['confidence'])

    def _majority_vote(self, detections):
        counter = Counter(d['class'] for d in detections)
        result  = counter.most_common(1)[0][0]
        return 'unknown' if result == 'none' else result

    # Bowl check controls
    def start_inference(self):
        with self.lock:
            self.inference_active = True
        self.get_logger().info('Bowl inference started')

    def stop_inference(self):
        with self.lock:
            self.inference_active = False
        self.get_logger().info('Bowl inference stopped')

    def start_bowl_check(self):
        with self.lock:
            self.collected_frames = []
            self.bowl_decision    = None
            self.collecting       = True
            self.inference_active = True
        self.get_logger().info('Bowl check collection started')

    def get_bowl_decision(self):
        with self.lock:
            return self.bowl_decision

    # Pet detection controls
    def start_pet_inference(self):
        with self.lock:
            self.pet_inference_active = True
        self.get_logger().info('Pet inference started')

    def stop_pet_inference(self):
        with self.lock:
            self.pet_inference_active = False
        self.get_logger().info('Pet inference stopped')

    def get_latest_pet_result(self):
        with self.lock:
            return self.latest_pet_result
    
    def check_and_clear_pet_detected(self):
        with self.lock:
            detected = self.pet_detected_since_last_check
            self.pet_detected_since_last_check = False
            return detected


# Global node instance
_node   = None
_thread = None

def get_node():
    return _node

def start_node():
    global _node, _thread
    if _node is not None:
        return
    _node = YoloInferenceNode()

    def spin_node():
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(_node)
        try:
            executor.spin()
        except Exception as e:
            print(f'YOLO executor error: {e}')

    _thread = threading.Thread(target=spin_node, daemon=True)
    _thread.start()

if __name__ == '__main__':
    start_node()
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
