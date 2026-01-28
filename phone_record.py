import time
import torch
import cv2
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
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep

# Camera & Dataset
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import traceback

FPS = 30
TASK_DESCRIPTION = "Pick up the black object and place it inside the green cup."

HOME_JOINTS = {
    "shoulder_pan.pos": 1.1,
    "shoulder_lift.pos": -100.0,
    "elbow_flex.pos": 81.5,
    "wrist_flex.pos": 79.6,
    "wrist_roll.pos": 10.7,
    "gripper.pos": 0.0
}

def main():
    repo_id = "gilberto/so101_training_data"
    local_root = Path("outputs/datasets").resolve()
    
    # 1. HARDWARE CONFIGURATION
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0", 
        id="gil_follower_arm", 
        use_degrees=True
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)
    
    # Cameras
    cam_web_cfg = OpenCVCameraConfig(index_or_path=4, fps=FPS, width=640, height=480, color_mode="bgr")
    cam_rs_cfg = RealSenseCameraConfig(serial_number_or_name="146222253839", fps=FPS, width=640, height=480, color_mode="bgr")
    webcam = OpenCVCamera(cam_web_cfg)
    realsense = RealSenseCamera(cam_rs_cfg)

    # 2. ROBOT & KINEMATICS SETUP
    print("Connecting to robot...")
    robot = SO101Follower(robot_config)
    robot.connect()
    motor_names = list(robot.bus.motors.keys())
    kinematics_solver = RobotKinematics(
        urdf_path="/home/gilberto/projects/lerobot-playground/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )
    
    max_velocity = 1000
    for motor_name in motor_names:
        if 'gripper' in motor_name:
            max_velocity = 1500
        robot.bus.write("Goal_Velocity", motor_name, max_velocity)

    # 3. PROCESSOR PIPELINE
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

    # 4. DATASET INITIALIZATION
    if local_root.exists():
        print(f"Loading existing dataset from {repo_id}...")
        dataset = LeRobotDataset(repo_id, root=local_root)
    else:
        print(f"Creating new dataset at {repo_id}...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=local_root,
            fps=FPS,
            robot_type="so101",
            features={
                "observation.images.webcam": {"dtype": "video", "shape": [3, 480, 640], "names": ["channels", "height", "width"]},
                "observation.images.realsense": {"dtype": "video", "shape": [3, 480, 640], "names": ["channels", "height", "width"]},
                "observation.state": {"dtype": "float32", "shape": (6,), "names": motor_names},
                "action": {"dtype": "float32", "shape": (6,), "names": motor_names},
            }
        )

    # 5. CONNECT REMAINING HARDWARE
    print("Connecting teleop and cameras...")
    teleop_device = Phone(teleop_config)
    teleop_device.connect()
    webcam.connect()
    realsense.connect()

    is_recording = False
    try:
        while True:
            t0 = time.perf_counter()

            # A. Observations & Teleop
            robot_obs = robot.get_observation()
            img_web = webcam.read()
            img_rs = realsense.read()
            phone_obs = teleop_device.get_action()

            # B. Compute & Send Action
            joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))
            robot.send_action(joint_action)

            # C. Display
            h, w = img_rs.shape[:2]
            img_web_res = cv2.resize(img_web, (int(img_web.shape[1] * (h / img_web.shape[0])), h))
            display = np.hstack([img_rs, img_web_res])
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
                    "observation.images.webcam": torch.from_numpy(img_web).permute(2, 0, 1),
                    "observation.images.realsense": torch.from_numpy(img_rs).permute(2, 0, 1),
                    "observation.state": torch.tensor([robot_obs[f"{n}.pos"] for n in motor_names], dtype=torch.float32),
                    "action": torch.tensor([joint_action[f"{n}.pos"] for n in motor_names], dtype=torch.float32),
                    "task": TASK_DESCRIPTION,
                }
                dataset.add_frame(frame_data)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0)) # Control Loop Timing

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nShutting down safely...")
        cv2.destroyAllWindows()
        robot.disconnect()
        webcam.disconnect()
        realsense.disconnect()
        print(f"Dataset finalized at: {local_root}")

if __name__ == "__main__":
    main()