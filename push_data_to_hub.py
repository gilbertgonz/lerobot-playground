from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("gilbertgonz/so101_training_data_wrist_only_v2", root="outputs/datasets/gilbertgonz/so101_training_data_wrist_only_v2")
dataset.push_to_hub()
