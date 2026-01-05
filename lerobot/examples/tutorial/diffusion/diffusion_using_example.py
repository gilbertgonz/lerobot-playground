import time
import torch
import cv2
import numpy as np
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 100
FPS = 30


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained diffusion model
    model_path = Path("/home/BOSDYN/ggonzalez/test/lerobot-playground/outputs/train/diffusion_so101")
    print(f"Loading model from {model_path}...")
    model = DiffusionPolicy.from_pretrained(model_path).to(device)
    model.eval()

    # Load dataset metadata for normalization stats
    dataset_id = "gilberto/so101_training_data"
    print(f"Loading dataset metadata from {dataset_id}...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    
    preprocess, postprocess = make_pre_post_processors(
        model.config, str(model_path), dataset_stats=dataset_metadata.stats
    )

    # Robot configuration (from phone_record.py)
    robot_config = SO100FollowerConfig(
        port="/dev/ttyACM0", 
        id="gil_follower_arm", 
        use_degrees=True
    )
    
    # Camera configuration matching training data
    cam_web_cfg = OpenCVCameraConfig(index_or_path=11, fps=FPS, width=480, height=640, rotation=-90)
    cam_rs_cfg = RealSenseCameraConfig(serial_number_or_name="146222253839", fps=FPS, width=640, height=480)
    
    # Initialize hardware
    print("Connecting to robot and cameras...")
    robot = SO100Follower(robot_config)
    robot.connect()
    
    webcam = OpenCVCamera(cam_web_cfg)
    realsense = RealSenseCamera(cam_rs_cfg)
    webcam.connect()
    realsense.connect()
    
    motor_names = list(robot.bus.motors.keys())

    try:
        print("\nStarting inference rollouts...")
        print("Press 'q' in the display window to quit\n")
        
        for episode in range(MAX_EPISODES):
            print(f"Episode {episode + 1}/{MAX_EPISODES}")
            
            for step in range(MAX_STEPS_PER_EPISODE):
                t0 = time.perf_counter()
                
                # Get observations
                robot_obs = robot.get_observation()
                img_web = webcam.read()
                img_rs = realsense.read()
                
                # Build observation frame for model
                obs_frame = build_inference_frame(
                    observation=robot_obs, 
                    ds_features=dataset_metadata.features, 
                    device=device
                )
                
                # Add images to observation frame
                obs_frame["observation.images.webcam"] = torch.from_numpy(img_web).permute(2, 0, 1).unsqueeze(0).to(device)
                obs_frame["observation.images.realsense"] = torch.from_numpy(img_rs).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Preprocess and run inference
                obs_preprocessed = preprocess(obs_frame)
                with torch.no_grad():
                    action = model.select_action(obs_preprocessed)
                
                # Postprocess and send action
                action = postprocess(action)
                action = make_robot_action(action, dataset_metadata.features)
                robot.send_action(action)
                
                # Display
                h, w = img_rs.shape[:2]
                img_web_res = cv2.resize(img_web, (int(img_web.shape[1] * (h / img_web.shape[0])), h))
                display = np.hstack([img_rs, img_web_res])
                cv2.putText(display, f"Episode {episode + 1}/{MAX_EPISODES} | Step {step + 1}/{MAX_STEPS_PER_EPISODE}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Diffusion Policy Inference", display)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nQuitting...")
                    raise KeyboardInterrupt
                
                # Control loop timing
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

            print(f"Episode {episode + 1} finished! Starting new episode...")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        cv2.destroyAllWindows()
        robot.disconnect()
        webcam.disconnect()
        realsense.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
