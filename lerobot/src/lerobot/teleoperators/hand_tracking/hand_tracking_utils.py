#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand landmark indices
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
WRIST = 0


class HandTrackingSystem:
    """Handles hand tracking using MediaPipe and RealSense."""

    def __init__(
        self,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        hand_confidence_threshold: float = 0.5,
        num_hands: int = 2,
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.hand_confidence_threshold = hand_confidence_threshold
        self.num_hands = num_hands

        self.pipeline = None
        self.colorizer = None
        self.detector = None
        self.align = None
        self.intrinsics = None
        self.depth_scale = None

    def initialize(self):
        """Initialize RealSense pipeline and hand detector."""
        self._initialize_realsense()
        self._initialize_hand_detector()

    def _initialize_realsense(self):
        """Initialize RealSense camera pipeline with alignment and intrinsics."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable depth and color streams
        config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.camera_fps)
        config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.camera_fps)
        
        # Start pipeline and get the profile
        profile = self.pipeline.start(config)
        
        # 1. Get Depth Scale (converts raw 16-bit values to meters)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 2. Setup Alignment (Align depth TO color)
        # This ensures depth[y, x] corresponds to color[y, x]
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # 3. Get Intrinsics from the color stream
        # This is required for rs2_deproject_pixel_to_point
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        self.colorizer = rs.colorizer()

    def _initialize_hand_detector(self):
        """Initialize MediaPipe hand detector."""
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.hand_confidence_threshold,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ALIGNED depth, colorized depth, and color frames."""
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline not initialized.")

        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None

        depth_data = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())
        depth_colorized = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return depth_data, depth_colorized, color

    def detect_hands(self, color_frame: np.ndarray) -> object:
        """Detect hands in the color frame using MediaPipe."""
        if self.detector is None:
            raise RuntimeError("Hand detector not initialized. Call initialize() first.")

        rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)

    def extract_hand_landmarks(self, detection_result, color_shape: tuple, landmark_indices: list[int]) -> list[dict]:
        """Extract specific hand landmarks from detection results."""
        hands_data = []

        if not detection_result.hand_landmarks:
            return hands_data

        h, w = color_shape[:2]

        for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
            hand_info = {"hand_id": hand_idx, "landmarks": []}

            for landmark_idx in landmark_indices:
                landmark = landmarks[landmark_idx]
                x = np.clip(int(landmark.x * w), 0, w - 1)
                y = np.clip(int(landmark.y * h), 0, h - 1)

                # Get confidence, defaulting to 1.0 if presence is None or doesn't exist
                confidence = 1.0
                if hasattr(landmark, "presence") and landmark.presence is not None:
                    confidence = landmark.presence

                hand_info["landmarks"].append(
                    {
                        "landmark_id": landmark_idx,
                        "pixel_x": x,
                        "pixel_y": y,
                        "depth": None,
                        "confidence": confidence,
                    }
                )

            hands_data.append(hand_info)

        return hands_data

    def associate_depth_with_landmarks(self, hands_data: list[dict], depth_data: np.ndarray, depth_scale: float = 0.001) -> list[dict]:
        """Associate depth values with hand landmarks."""
        for hand in hands_data:
            for landmark in hand["landmarks"]:
                x, y = landmark["pixel_x"], landmark["pixel_y"]
                # Ensure indices are within bounds
                if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
                    landmark["depth"] = depth_data[y, x] * depth_scale

        return hands_data

    def detect_pinch(self, hands_data: list[dict], pixel_threshold: int = 15) -> list[dict]:
        """
        Detect if thumb and index finger are pinching.
        
        Args:
            hands_data: List of hand data with landmarks
            pixel_threshold: Pixel distance threshold for pinch detection
            depth_threshold: 3D distance threshold in meters
            
        Returns:
            List of pinch status for each hand
        """
        pinch_status = []

        for hand in hands_data:
            if len(hand["landmarks"]) < 2:
                pinch_status.append({"hand_id": hand["hand_id"], "is_pinching": False, "distance": float("inf")})
                continue

            # Assume first landmark is thumb, second is index
            thumb = hand["landmarks"][0]
            index = hand["landmarks"][1]

            # Calculate 2D pixel distance
            pixel_dist = np.sqrt((thumb["pixel_x"] - index["pixel_x"]) ** 2 + (thumb["pixel_y"] - index["pixel_y"]) ** 2)
            is_pinching = pixel_dist < pixel_threshold

            pinch_status.append({"hand_id": hand["hand_id"], "is_pinching": is_pinching, "distance": pixel_dist})

        return pinch_status

    def get_hand_position_3d(self, hand_data: dict, reference_landmark_idx: int = WRIST) -> Optional[tuple[float, float, float]]:
        """De-projects 2D pixels to true 3D coordinates in meters."""
        if not hand_data.get("landmarks") or self.intrinsics is None:
            return None

        for landmark in hand_data["landmarks"]:
            if landmark["landmark_id"] == reference_landmark_idx:
                depth_in_meters = landmark["depth"]
                
                if depth_in_meters is not None and depth_in_meters > 0:
                    px_x = landmark["pixel_x"]
                    px_y = landmark["pixel_y"]

                    # Use SDK function to convert (pixel_x, pixel_y, depth) -> (X, Y, Z) in meters
                    # X: Left/Right, Y: Up/Down, Z: Forward/Back
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        self.intrinsics, [px_x, px_y], depth_in_meters
                    )
                    return tuple(point_3d)
                break
        return None

    def is_hand_valid(self, hand_data: dict, min_confidence: float = 0.5) -> bool:
        """Check if hand detection is valid based on landmark confidence."""
        if not hand_data.get("landmarks"):
            return False

        # Check if we have valid depth for at least one landmark
        has_valid_depth = any(lm.get("depth") and lm["depth"] > 0 for lm in hand_data["landmarks"])

        # Check confidence - filter out None values and use default 1.0
        confidences = [lm.get("confidence", 1.0) for lm in hand_data["landmarks"]]
        confidences = [c for c in confidences if c is not None]
        
        if not confidences:
            # If no valid confidence values, default to 1.0 (valid)
            avg_confidence = 1.0
        else:
            avg_confidence = np.mean(confidences)

        return has_valid_depth and avg_confidence >= min_confidence

    def shutdown(self):
        """Shutdown RealSense pipeline."""
        if self.pipeline is not None:
            self.pipeline.stop()

    def draw_hand_landmarks(self, image: np.ndarray, hands_data: list[dict], pinch_status: list[dict] = None) -> np.ndarray:
        """
        Draw hand landmarks and pinch status on the image.
        
        Args:
            image: Input image (BGR format)
            hands_data: List of hand data with landmarks
            pinch_status: Optional list of pinch status for each hand
            
        Returns:
            Image with landmarks drawn on it
        """
        image_copy = image.copy()
        
        # Define landmark colors and names
        landmark_colors = {
            THUMB_TIP: (0, 255, 0),      # Green for thumb
            INDEX_FINGER_TIP: (255, 0, 0),  # Blue for index
            MIDDLE_FINGER_TIP: (0, 165, 255),  # Orange for middle
            WRIST: (255, 255, 0),        # Cyan for wrist
        }
        
        landmark_names = {
            THUMB_TIP: "Thumb",
            INDEX_FINGER_TIP: "Index",
            MIDDLE_FINGER_TIP: "Middle",
            WRIST: "Wrist",
        }
        
        for hand_idx, hand_data in enumerate(hands_data):
            # Draw landmarks
            for landmark in hand_data["landmarks"]:
                lm_id = landmark["landmark_id"]
                x, y = landmark["pixel_x"], landmark["pixel_y"]
                depth = landmark["depth"]
                
                # Get color for this landmark type
                color = landmark_colors.get(lm_id, (200, 200, 200))
                name = landmark_names.get(lm_id, f"L{lm_id}")
                
                # Draw circle at landmark
                cv2.circle(image_copy, (x, y), 8, color, -1)
                cv2.circle(image_copy, (x, y), 8, (0, 0, 0), 2)
                
                # Draw text label with depth
                if depth is not None:
                    text = f"{name} ({depth:.2f}m)"
                else:
                    text = name
                cv2.putText(image_copy, text, (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw connection between thumb and index (pinch line)
            if len(hand_data["landmarks"]) >= 2:
                thumb = hand_data["landmarks"][0]
                index = hand_data["landmarks"][1]
                thumb_pos = (thumb["pixel_x"], thumb["pixel_y"])
                index_pos = (index["pixel_x"], index["pixel_y"])
                
                # Draw line between thumb and index
                cv2.line(image_copy, thumb_pos, index_pos, (200, 0, 200), 2)
                
                # If pinch status is available, show if pinching
                if pinch_status:
                    pinch = pinch_status[hand_idx] if hand_idx < len(pinch_status) else None
                    if pinch and pinch["is_pinching"]:
                        # Draw text indicating pinch
                        mid_x = (thumb_pos[0] + index_pos[0]) // 2
                        mid_y = (thumb_pos[1] + index_pos[1]) // 2
                        cv2.putText(image_copy, "PINCHING", (mid_x - 30, mid_y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
        
        return image_copy

    def display_frame(self, image: np.ndarray, window_name: str = "Hand Tracking") -> bool:
        """
        Display the image in a window.
        
        Args:
            image: Image to display
            window_name: Name of the window
            
        Returns:
            True if window is still open, False if closed with 'q' or ESC
        """
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), 27)  # 27 is ESC key
