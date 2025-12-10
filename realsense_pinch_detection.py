#!/usr/bin/env python3
"""
Requirements:
    pip install pyrealsense2 opencv-python numpy mediapipe
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONFIDENCE_THRESHOLD = 0.5
THUMB_TIP = 4
INDEX_FINGER_TIP = 8

def initialize_hand_detector():
    import os
    import urllib.request
    
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=HAND_CONFIDENCE_THRESHOLD
    )
    return vision.HandLandmarker.create_from_options(options)

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline, rs.colorizer()

def get_realsense_frames(pipeline, colorizer):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_data = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())
    depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    return depth_data, depth_colorized, color

def detect_hands(detector, color_frame):
    rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect(mp_image)


def extract_hand_joints(detection_result, color_shape):
    hands_data = []
    if not detection_result.hand_landmarks:
        return hands_data
    
    h, w = color_shape[:2]
    
    for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
        hand_info = {'hand_id': hand_idx, 'joints': []}
        
        for landmark_idx in [THUMB_TIP, INDEX_FINGER_TIP]:
            landmark = landmarks[landmark_idx]
            x = np.clip(int(landmark.x * w), 0, w - 1)
            y = np.clip(int(landmark.y * h), 0, h - 1)
            
            hand_info['joints'].append({
                'landmark_id': landmark_idx,
                'pixel_x': x,
                'pixel_y': y,
                'depth': None
            })
        
        hands_data.append(hand_info)
    
    return hands_data


def associate_depth_with_joints(hands_data, depth_data, depth_scale=0.001):
    for hand in hands_data:
        for joint in hand['joints']:
            x, y = joint['pixel_x'], joint['pixel_y']
            joint['depth'] = depth_data[y, x] * depth_scale
    return hands_data


def detect_pinch(hands_data, threshold=0.04):
    """Detect if thumb and index finger are pinching (3D distance < threshold in meters)."""
    pinch_status = []
    
    for hand in hands_data:
        if len(hand['joints']) < 2:
            continue
            
        thumb = hand['joints'][0]  # THUMB_TIP
        index = hand['joints'][1]  # INDEX_FINGER_TIP
        
        # Calculate 2D pixel distance as fallback
        pixel_dist = np.sqrt(
            (thumb['pixel_x'] - index['pixel_x'])**2 +
            (thumb['pixel_y'] - index['pixel_y'])**2
        )
        
        # Calculate 3D distance if depth is available
        if thumb['depth'] and index['depth'] and thumb['depth'] > 0 and index['depth'] > 0:
            # Simple 3D distance using depth values
            depth_dist = abs(thumb['depth'] - index['depth'])
            # Combine pixel and depth distance for robust detection
            is_pinching = pixel_dist < 30 and depth_dist < threshold
        else:
            # Fallback to pixel distance only
            is_pinching = pixel_dist < 30
        
        pinch_status.append({
            'hand_id': hand['hand_id'],
            'is_pinching': is_pinching,
            'pixel_distance': pixel_dist
        })
    
    return pinch_status


def draw_hand_joints(frame, hands_data, pinch_status=None):
    for hand in hands_data:
        # Determine color based on pinch status
        is_pinching = False
        if pinch_status:
            for status in pinch_status:
                if status['hand_id'] == hand['hand_id']:
                    is_pinching = status['is_pinching']
                    break
        
        color = (0, 0, 255) if is_pinching else (0, 255, 0)  # Red if pinching, green otherwise
        
        for joint in hand['joints']:
            x, y = joint['pixel_x'], joint['pixel_y']
            depth = joint['depth']
            cv2.circle(frame, (x, y), 5, color, 2)
            if depth and depth > 0:
                cv2.putText(frame, f"{depth:.2f}m", (x + 10, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw line between thumb and index
        if len(hand['joints']) >= 2:
            thumb = hand['joints'][0]
            index = hand['joints'][1]
            cv2.line(frame, (thumb['pixel_x'], thumb['pixel_y']),
                    (index['pixel_x'], index['pixel_y']), color, 2)
        
        # Draw pinch status text
        if is_pinching and len(hand['joints']) > 0:
            x = hand['joints'][0]['pixel_x']
            y = hand['joints'][0]['pixel_y']
            cv2.putText(frame, "PINCH", (x - 30, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame



def main():
    pipeline, colorizer = initialize_realsense()
    detector = initialize_hand_detector()
    
    try:
        while True:
            depth_data, depth_colorized, color = get_realsense_frames(pipeline, colorizer)
            
            detection_result = detect_hands(detector, color)
            hands_data = extract_hand_joints(detection_result, color.shape)
            hands_data = associate_depth_with_joints(hands_data, depth_data)
            
            # Detect pinch gesture
            pinch_status = detect_pinch(hands_data)
            
            # Draw with pinch visualization
            color = draw_hand_joints(color, hands_data, pinch_status)
            
            display = np.hstack([color, depth_colorized])
            cv2.imshow("RealSense Hand Tracking", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
