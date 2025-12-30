import time

from src.lerobot.teleoperators.hand_tracking.teleop_hand_tracking import HandTrackingTeleop
from src.lerobot.teleoperators.hand_tracking.configuration_hand_tracking import HandTrackingTeleopConfig
from src.lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
import pprint

# Configure the robot
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="gil_follower_arm",
)

# Configure the hand tracking teleoperator
teleop_config = HandTrackingTeleopConfig(
    camera_width=640,
    camera_height=480,
    camera_fps=30,
    hand_confidence_threshold=0.5,
    num_hands=2,
    movement_scale=0.01,
    use_gripper=True,
    hand_detection_timeout=30,
)

robot = SO101Follower(robot_config)
teleop_device = HandTrackingTeleop(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    pp = pprint.PrettyPrinter(indent=4)
    print("Action:")
    pp.pprint(action)
    robot.send_action(action)
    time.sleep(0.01)