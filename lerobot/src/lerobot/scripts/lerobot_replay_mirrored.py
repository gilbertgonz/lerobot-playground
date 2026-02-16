import logging
import time
import torch
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

@dataclass
class DatasetReplayConfig:
    repo_id: str
    episode: int
    root: str | Path | None = None
    fps: int = 30

@dataclass
class ReplayConfig:
    robot: RobotConfig  # This allows --robot.type, --robot.port, etc.
    dataset: DatasetReplayConfig
    urdf_path: str = "/home/gilberto/projects/lerobot-playground/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    play_sounds: bool = True

@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 1. Setup Robot and Dataset
    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    
    # Filter for the specific episode
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)
    motor_names = dataset.features[ACTION]["names"]

    # 2. Initialize Kinematics Solver
    kinematics = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    robot.connect()
    log_say("Starting IK mirrored replay", cfg.play_sounds, blocking=True)

    try:
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            # 1. Get joints as standard Python floats
            recorded_joints_raw = actions[idx][ACTION]
            recorded_joints_list = [float(x) for x in recorded_joints_raw]
            
            # We convert to numpy because most C++ wrappers (like Placo) 
            # handle numpy scalars as standard doubles automatically.
            recorded_joints_np = np.array(recorded_joints_list, dtype=np.float64)

            # 2. Forward Kinematics - PASS THE NUMPY ARRAY
            # This should resolve the Boost.Python.ArgumentError
            ee_pose = kinematics.forward_kinematics(recorded_joints_np)

            # 3. Mirroring (Y -> -Y)
            # ee_pose comes back as a Tensor from LeRobot's wrapper
            mirrored_pose = ee_pose.copy()

            # Flip Y translation
            mirrored_pose[1, 3] *= -1.0

            # Flip Rotation for XZ-plane reflection
            # Negate the 2nd row (Y-related mappings)
            mirrored_pose[1, 0:3] *= -1.0
            # Negate the 2nd column (Y-axis basis vector)
            mirrored_pose[0:3, 1] *= -1.0

            # 4. Inverse Kinematics
            # Use the numpy array as the guess, converted to the shape it expects
            mirrored_joints = kinematics.inverse_kinematics(
                recorded_joints_np, 
                mirrored_pose
            )

            # 5. Execute
            action_dict = {}
            for i, name in enumerate(motor_names):
                if "gripper" in name:
                    action_dict[name] = float(recorded_joints_list[i])
                else:
                    action_dict[name] = float(mirrored_joints[i])

            robot.bus.sync_write("Goal_Position", action_dict)

            dt = time.perf_counter() - t0
            precise_sleep(max(1 / dataset.fps - dt, 0))
            
    except KeyboardInterrupt:
        print("\nReplay interrupted.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")

if __name__ == "__main__":
    replay()