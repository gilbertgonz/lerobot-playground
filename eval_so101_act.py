import time
import torch
import cv2
import numpy as np
from pathlib import Path
import traceback

# LeRobot Core
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower

# Camera & Utilities
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.utils.robot_utils import precise_sleep

# --- CONFIGURATION ---
FPS = 30
MODEL_PATH = "gilbertgonz/so101-act-models"
DATASET_REPO_ID = "gilbertgonz/so101_training_data"  # Needed for normalization stats
DATASET_LOCAL_ROOT = Path("outputs/datasets").resolve()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. HARDWARE SETUP
    print("Connecting to SO-101 and Cameras...")
    robot_config = SO100FollowerConfig(port="/dev/ttyACM0", id="gil_follower_arm", use_degrees=True)
    robot = SO100Follower(robot_config)
    robot.connect()
    motor_names = list(robot.bus.motors.keys())

    cam_web_cfg = OpenCVCameraConfig(index_or_path=11, fps=FPS, width=480, height=640, rotation=-90)
    cam_rs_cfg = RealSenseCameraConfig(serial_number_or_name="146222253839", fps=FPS, width=640, height=480)
    webcam = OpenCVCamera(cam_web_cfg)
    realsense = RealSenseCamera(cam_rs_cfg)
    webcam.connect()
    realsense.connect()

    # 2. LOAD DATASET STATS (Crucial for normalizing live observations)
    print(f"Loading stats from {DATASET_REPO_ID}...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_LOCAL_ROOT)
    stats = dataset.meta.stats

    # 3. LOAD POLICY
    print(f"Loading ACT Policy from {MODEL_PATH}...")
    policy = ACTPolicy.from_pretrained(MODEL_PATH).to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=MODEL_PATH,
        dataset_stats=stats,
    )

    print("\n--- INFERENCE STARTING ---")
    try:
        while True:
            t0 = time.perf_counter()

            # A. Get Live Observations
            robot_obs = robot.get_observation()
            img_web = webcam.read()
            img_rs = realsense.read()

            # B. Prepare Observation Frame. Tensors must be (Batch, Channel, Height, Width)
            obs_frame = {
                "observation.images.webcam": torch.from_numpy(img_web).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0,    # Convert uint8 images to float32 (0.0-1.0 range)
                "observation.images.realsense": torch.from_numpy(img_rs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0,  # Convert uint8 images to float32 (0.0-1.0 range)
                "observation.state": torch.tensor([[robot_obs[f"{n}.pos"] for n in motor_names]], dtype=torch.float32).to(device),
            }

            # C. Model Inference
            with torch.no_grad():
                # Apply normalization (pre) -> Run Model -> Apply un-normalization (post)
                normalized_obs = preprocessor(obs_frame)
                output_action = policy.select_action(normalized_obs)
                unnormalized_action = postprocessor(output_action)

            # D. Extract Action and Send to Robot
            joint_values = unnormalized_action[0].cpu().numpy()
            target_joints = {f"{name}.pos": val for name, val in zip(motor_names, joint_values)}
            
            robot.send_action(target_joints)

            # E. Visualization - Concatenate both camera feeds with letter-boxing
            h, w = img_rs.shape[:2]
            img_web_res = cv2.resize(img_web, (int(img_web.shape[1] * (h / img_web.shape[0])), h))
            display = np.hstack([img_rs, img_web_res])
            cv2.putText(display, f"RUNNING INFERENCE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("SO-101 Model Eval", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Maintain consistent loop frequency
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except Exception:
        traceback.print_exc()
    finally:
        print("Shutting down...")
        robot.disconnect()
        webcam.disconnect()
        realsense.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
