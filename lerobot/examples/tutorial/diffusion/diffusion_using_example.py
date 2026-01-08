import torch
import numpy as np
import cv2
import time
import traceback

from torchvision.transforms import Resize, InterpolationMode

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep


# CONFIG
FPS = 30
MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 1000

MODEL_REPO_ID = "gilbertgonz/so101-diffusion-models"
DATASET_ID = "gilbertgonz/so101_training_data"

TARGET_HEIGHT = 224
TARGET_WIDTH = 224

resizer = Resize(
    (TARGET_HEIGHT, TARGET_WIDTH),
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model + metadata
    model = DiffusionPolicy.from_pretrained(MODEL_REPO_ID).to(device)
    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)

    # model.config.n_obs_steps = 2
    # model.config.horizon = 40
    # model.config.n_action_steps = 30

    # Pre/Post processors
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        dataset_stats=dataset_metadata.stats,
    )

    # Robot config
    robot_cfg = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="gil_follower_arm",
        use_degrees=True,
        # max_relative_target=10.0,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    motor_names = list(robot.bus.motors.keys())

    # Camera configs
    cam_web_cfg = OpenCVCameraConfig(
        index_or_path=11,
        fps=FPS,
        width=480,
        height=640,
        rotation=-90,
    )

    cam_rs_cfg = RealSenseCameraConfig(
        serial_number_or_name="146222253839",
        fps=FPS,
        width=640,
        height=480,
    )

    webcam = OpenCVCamera(cam_web_cfg)
    realsense = RealSenseCamera(cam_rs_cfg)

    webcam.connect()
    realsense.connect()

    try:
        for episode in range(MAX_EPISODES):
            print(f"\nEpisode {episode + 1}")
            model.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                t0 = time.perf_counter()
                robot_obs = robot.get_observation()
                img_web = webcam.read()
                img_rs = realsense.read()

                # Build raw observations
                raw_obs = {
                    "webcam": img_web,
                    "realsense": img_rs,
                }

                for name in motor_names:
                    raw_obs[f"{name}.pos"] = np.array(
                        robot_obs[f"{name}.pos"],
                        dtype=np.float32,
                    )

                # Build inference frame
                obs_frame = build_inference_frame(
                    observation=raw_obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )

                # Image resizing
                obs_frame["observation.images.webcam"] = resizer(
                    obs_frame["observation.images.webcam"]
                )
                obs_frame["observation.images.realsense"] = resizer(
                    obs_frame["observation.images.realsense"]
                )
                
                # Inference
                with torch.no_grad():
                    obs = preprocess(obs_frame)
                    action = model.select_action(obs)
                    action = postprocess(action)
                action_dict = make_robot_action(action, dataset_metadata.features)
                robot.send_action(action_dict)

                # Viz
                h = img_rs.shape[0]
                img_web_resized = cv2.resize(img_web, (int(img_web.shape[1] * (h / img_web.shape[0])), h))
                display = np.hstack([img_rs, img_web_resized])
                cv2.putText(display, f"Episode {episode + 1} | Step {step + 1}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Diffusion Policy Inference", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

                precise_sleep(
                    max(1.0 / FPS - (time.perf_counter() - t0), 0.0)
                )

    except (KeyboardInterrupt, SystemExit):
        print("\nStopping...")

    except Exception:
        traceback.print_exc()

    finally:
        cv2.destroyAllWindows()
        robot.disconnect()
        webcam.disconnect()
        realsense.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
