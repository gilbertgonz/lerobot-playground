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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("hand_tracking")
@dataclass
class HandTrackingTeleopConfig(TeleoperatorConfig):
    """Configuration for hand tracking teleoperator."""

    # Camera settings
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Hand detection settings
    hand_confidence_threshold: float = 0.5
    min_hand_detection_confidence: float = 0.5
    num_hands: int = 2
    
    # Pinch detection settings
    pinch_pixel_threshold: int = 30  # Pixel distance for pinch detection
    
    # Movement settings
    movement_scale: float = 1000  # Scale factor for hand position to robot movement
    max_velocity: float = 0.1  # Max velocity in m/s
    
    # Hand detection timeout (frames without detection before halting)
    hand_detection_timeout: int = 30  # frames
    
    # Gripper control
    use_gripper: bool = True
    gripper_close_threshold: float = 0.5  # Confidence threshold for closed gripper
