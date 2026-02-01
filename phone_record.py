# !/usr/bin/env python

import time
import traceback
import cv2
import torch
import numpy as np
from pathlib import Path

# LeRobot Core
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)

# Teleop
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep

# Cameras & Dataset
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- GLOBAL CONFIGURATION ---
FPS = 30
TASK_DESCRIPTION = "Pick up the black object and place it inside the green cup."

USE_WEBCAM = True
USE_REALSENSE = False

def main():
    repo_id = "gilberto/so101_training_data_v2"
    local_root = Path("outputs/datasets").resolve()
    urdf_path = "/home/gilberto/projects/lerobot-playground/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

    # 1. HARDWARE CONFIGURATION
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0", 
        id="gil_follower_arm", 
        use_degrees=True
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)

    # 2. CAMERA SETUP
    webcam = None
    realsense = None

    if USE_WEBCAM:
        cam_web_cfg = OpenCVCameraConfig(
            index_or_path=4, fps=FPS, width=640, height=480, color_mode="bgr"
        )
        webcam = OpenCVCamera(cam_web_cfg)

    if USE_REALSENSE:
        cam_rs_cfg = RealSenseCameraConfig(
            serial_number_or_name="146222253839", fps=FPS, width=640, height=480, color_mode="bgr"
        )
        realsense = RealSenseCamera(cam_rs_cfg)

    # 3. ROBOT & KINEMATICS SETUP
    print("Connecting to robot...")
    robot = SO101Follower(robot_config)
    robot.connect()
    
    motor_names = list(robot.bus.motors.keys())
    
    # Set max velocities
    max_velocity = 1000
    for motor_name in motor_names:
        if 'gripper' in motor_name:
            robot.bus.write("Goal_Velocity", motor_name, 1500)
        else:
            robot.bus.write("Goal_Velocity", motor_name, max_velocity)

    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # 4. PROCESSOR PIPELINE
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=motor_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 5. DATASET INITIALIZATION
    # Build dynamic features dict based on available cameras
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": motor_names},
        "action": {"dtype": "float32", "shape": (6,), "names": motor_names},
    }
    
    if USE_WEBCAM:
        features["observation.images.webcam"] = {
            "dtype": "video", "shape": [3, 480, 640], "names": ["channels", "height", "width"]
        }
    if USE_REALSENSE:
        features["observation.images.realsense"] = {
            "dtype": "video", "shape": [3, 480, 640], "names": ["channels", "height", "width"]
        }

    if local_root.exists() and (local_root / repo_id).exists():
        print(f"Loading existing dataset from {repo_id}...")
        dataset = LeRobotDataset(repo_id, root=local_root)
    else:
        print(f"Creating new dataset at {repo_id}...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=local_root,
            fps=FPS,
            robot_type="so101",
            features=features
        )

    # 6. CONNECT HARDWARE
    print("Connecting teleop...")
    teleop_device = Phone(teleop_config)
    teleop_device.connect()

    if USE_WEBCAM:
        print("Connecting Webcam...")
        webcam.connect()
    if USE_REALSENSE:
        print("Connecting RealSense...")
        realsense.connect()

    is_recording = False
    print("\n--- READY ---")
    print("Controls: [SPACE] Record/Stop | [Q] Quit")

    try:
        while True:
            t0 = time.perf_counter()

            # A. Observations & Teleop
            robot_obs = robot.get_observation()
            phone_obs = teleop_device.get_action()
            img_web = None
            img_rs = None
            
            if USE_WEBCAM:
                img_web = webcam.read()
            if USE_REALSENSE:
                img_rs = realsense.read()

            # B. Compute & Send Action
            joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))
            robot.send_action(joint_action)

            # C. Display
            display_images = []
            
            if USE_REALSENSE and img_rs is not None:
                display_images.append(img_rs)
            
            if USE_WEBCAM and img_web is not None:
                # Resize webcam to match RealSense height if both are present
                if USE_REALSENSE and img_rs is not None:
                    h = img_rs.shape[0]
                    img_web_res = cv2.resize(img_web, (int(img_web.shape[1] * (h / img_web.shape[0])), h))
                    display_images.append(img_web_res)
                else:
                    display_images.append(img_web)

            # Stack whatever images we have horizontally
            display = np.hstack(display_images)

            if is_recording:
                cv2.putText(display, f"RECORDING EPISODE {dataset.num_episodes}", (60, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, "STANDBY", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("SO-101 Teleop Recorder", display)

            # D. Handle Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not is_recording: # start recording
                    is_recording = True
                    print(f"\nREC START: Episode {dataset.num_episodes}")
                else: # stop recording
                    is_recording = False
                    dataset.save_episode()
                    print(f"REC STOP: Episode {dataset.num_episodes - 1} saved.")
            elif key == ord('q'): # terminate and dont use episode if recording
                break

            # E. Recording Data
            if is_recording:
                frame_data = {
                    "observation.state": torch.tensor([robot_obs[f"{n}.pos"] for n in motor_names], dtype=torch.float32),
                    "action": torch.tensor([joint_action[f"{n}.pos"] for n in motor_names], dtype=torch.float32),
                    "task": TASK_DESCRIPTION,
                }
                if USE_WEBCAM:
                    frame_data["observation.images.webcam"] = torch.from_numpy(img_web).permute(2, 0, 1)
                if USE_REALSENSE:
                    frame_data["observation.images.realsense"] = torch.from_numpy(img_rs).permute(2, 0, 1)

                dataset.add_frame(frame_data)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0)) # Control Loop Timing

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        cv2.destroyAllWindows()
        robot.disconnect()
        if USE_WEBCAM and webcam:
            webcam.disconnect()
        if USE_REALSENSE and realsense:
            realsense.disconnect()
        print(f"Dataset finalized at: {local_root}")

if __name__ == "__main__":
    main()