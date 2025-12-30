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

from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from .configuration_hand_tracking import HandTrackingTeleopConfig
from .hand_tracking_utils import THUMB_TIP, INDEX_FINGER_TIP, WRIST, HandTrackingSystem


class HandTrackingTeleop(Teleoperator):
    """
    Teleoperator for controlling a robot arm using hand tracking via RealSense camera.
    
    Features:
    - Hand position tracking for end-effector goal poses
    - Pinch gesture detection (index + thumb) for gripper control
    - Robust hand detection with timeout handling
    - Automatic motion halting when hand is not detected
    """

    config_class = HandTrackingTeleopConfig
    name = "hand_tracking"

    def __init__(self, config: HandTrackingTeleopConfig):
        super().__init__(config)
        self.config = config

        # Hand tracking system
        self.hand_tracker = HandTrackingSystem(
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            camera_fps=config.camera_fps,
            hand_confidence_threshold=config.hand_confidence_threshold,
            num_hands=config.num_hands,
        )

        # State tracking
        self._is_connected = False
        self._frames_without_detection = 0
        self._last_valid_position = np.array([0.5, 0.5, 0.0])  # Normalized coordinates
        self._is_hand_detected = False
        self._last_gripper_pos = 0.0  # Track last valid gripper position
        self._gripper_smoothing_factor = 0.3  # Exponential moving average factor (0-1)

    @property
    def action_features(self) -> dict[str, type]:
        """Return action feature specification matching robot motor positions."""
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        """Return feedback feature specification."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if hand tracking teleoperator is connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Hand tracking doesn't require calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the hand tracking system (RealSense camera).
        
        Args:
            calibrate: Ignored, as hand tracking doesn't require calibration.
        """
        try:
            self.hand_tracker.initialize()
            self._is_connected = True
            print("Hand tracking teleoperator connected successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect hand tracking system: {e}")

    def calibrate(self) -> None:
        """No calibration needed for hand tracking."""
        pass

    def configure(self) -> None:
        """Apply any configuration to the teleoperator."""
        # Additional configuration can be added here
        pass

    def get_action(self) -> dict[str, Any]:
        """
        Get the current action from hand tracking.
        
        Returns:
            Dictionary with motor position keys for all joints.
            Returns zero motion if hand is not detected for timeout period.
        """
        if not self._is_connected:
            raise RuntimeError("Hand tracking teleoperator not connected. Call connect() first.")

        # Get frames from RealSense
        depth_data, _, color = self.hand_tracker.get_frames()

        # Detect hands
        try:
            detection_result = self.hand_tracker.detect_hands(color)
        except Exception as e:
            print(f"Error detecting hands: {e}")
            self._frames_without_detection += 1
            return self._handle_hand_detection_loss()

        # Extract hand landmarks
        hands_data = self.hand_tracker.extract_hand_landmarks(
            detection_result, color.shape, [THUMB_TIP, INDEX_FINGER_TIP, WRIST]
        )

        # Associate depth with landmarks
        hands_data = self.hand_tracker.associate_depth_with_landmarks(hands_data, depth_data)

        # Check if we have valid hand detection
        if not hands_data or not self.hand_tracker.is_hand_valid(hands_data[0]):
            self._frames_without_detection += 1
            # Display frame even when no hand detected
            viz_image = self.hand_tracker.draw_hand_landmarks(color, hands_data)
            self.hand_tracker.display_frame(viz_image, "Hand Tracking Teleop")
            return self._handle_hand_detection_loss()

        # Reset detection timeout counter
        self._frames_without_detection = 0
        self._is_hand_detected = True

        # Get hand position (use first hand detected)
        hand_data = hands_data[0]
        hand_position_3d = self.hand_tracker.get_hand_position_3d(hand_data, reference_landmark_idx=WRIST)

        if hand_position_3d is None:
            self._frames_without_detection += 1
            viz_image = self.hand_tracker.draw_hand_landmarks(color, hands_data)
            self.hand_tracker.display_frame(viz_image, "Hand Tracking Teleop")
            return self._handle_hand_detection_loss()

        # Update last valid position
        self._last_valid_position = np.array(hand_position_3d)

        # Map 3D hand position to robot joint positions
        # hand_position_3d is (x, y, z) - only 3 dimensions
        # Map to the 6 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        action_dict = {
            "shoulder_pan.pos": float(hand_position_3d[0]) * self.config.movement_scale,
            "shoulder_lift.pos": float(hand_position_3d[1]) * self.config.movement_scale,
            "elbow_flex.pos": float(hand_position_3d[2]) * self.config.movement_scale,
            "wrist_flex.pos": 0.0,  # Keep at default
            "wrist_roll.pos": 0.0,  # Keep at default
        }

        # Check for pinch gesture for gripper control
        gripper_pos = 0.0  # Default to fully closed
        if self.config.use_gripper:
            pinch_status = self.hand_tracker.detect_pinch(
                hands_data,
                pixel_threshold=self.config.pinch_pixel_threshold
            )

            if pinch_status and pinch_status[0]:
                # Map pixel distance between thumb and index to gripper position (0-100)
                # Small pixel distance (pinch) -> gripper closed (0)
                # Large pixel distance (open hand) -> gripper open (100)
                pinch_distance = pinch_status[0].get("distance", 0.0)
                
                # Normalize pixel distance to 0-100 range
                # Use pinch_pixel_threshold as the baseline for fully closed
                max_distance = self.config.pinch_pixel_threshold * 2
                raw_gripper_pos = (pinch_distance / max_distance) * 100.0
                raw_gripper_pos = min(100.0, max(0.0, raw_gripper_pos))
                
                # Apply exponential moving average smoothing
                gripper_pos = (self._gripper_smoothing_factor * raw_gripper_pos + 
                              (1 - self._gripper_smoothing_factor) * self._last_gripper_pos)
        
        # Update last valid gripper position
        self._last_gripper_pos = gripper_pos
        action_dict["gripper.pos"] = gripper_pos

        # Visualize hand landmarks and pinch status
        viz_image = self.hand_tracker.draw_hand_landmarks(color, hands_data, pinch_status)
        self.hand_tracker.display_frame(viz_image, "Hand Tracking Teleop")

        return action_dict

    def _handle_hand_detection_loss(self) -> dict[str, Any]:
        """
        Handle the case when hand detection is lost.
        Halt robot motion after timeout period.
        """
        if self._frames_without_detection >= self.config.hand_detection_timeout:
            # Timeout exceeded - halt motion
            self._is_hand_detected = False
            print(f"Hand detection lost for {self._frames_without_detection} frames. Halting robot motion.")
            return self._get_zero_action()
        else:
            # Still within timeout - return last known position
            return self._get_last_position_action()

    def _get_zero_action(self) -> dict[str, Any]:
        """Return a zero action (no motion, gripper open)."""
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 1.0,
        }

    def _get_last_position_action(self) -> dict[str, Any]:
        """Return action using last known valid hand position and gripper position."""
        # _last_valid_position is (x, y, z) - only 3 dimensions
        action_dict = {
            "shoulder_pan.pos": float(self._last_valid_position[0]) * self.config.movement_scale,
            "shoulder_lift.pos": float(self._last_valid_position[1]) * self.config.movement_scale,
            "elbow_flex.pos": float(self._last_valid_position[2]) * self.config.movement_scale,
            "wrist_flex.pos": 0.0,  # Keep at default
            "wrist_roll.pos": 0.0,  # Keep at default
            "gripper.pos": self._last_gripper_pos,  # Use last valid gripper position
        }
        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the hand tracking system."""
        self.hand_tracker.shutdown()
        self._is_connected = False
        print("Hand tracking teleoperator disconnected.")