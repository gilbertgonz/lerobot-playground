import logging
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# Assuming your provided file is saved as 'dataset_utils.py' in the same folder
from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
def cleanup_last_episode():
    # 1. Setup Paths
    repo_id = "gilberto/so101_training_data"
    local_root = Path("outputs/datasets").resolve()
    
    print(f"Loading dataset from {local_root / repo_id}...")
    
    # 2. Load the current dataset
    dataset = LeRobotDataset(repo_id, root=local_root)
    
    if dataset.meta.total_episodes == 0:
        print("Dataset is already empty.")
        return

    # 3. Identify the last episode index
    last_episode_idx = 6 #dataset.meta.total_episodes - 1
    print(f"Attempting to delete Episode Index: {last_episode_idx}")

    # 4. Use the official utility to delete and re-index
    # We set repo_id to the same name and output_dir to the same path 
    # to "overwrite" the current dataset with the cleaned version.
    try:
        new_dataset = delete_episodes(
            dataset=dataset,
            episode_indices=[last_episode_idx],
            repo_id=repo_id,
            output_dir=local_root / repo_id
        )
        print(f"Successfully deleted episode. New total: {new_dataset.meta.total_episodes}")
    except Exception as e:
        print(f"Error during deletion: {e}")

if __name__ == "__main__":
    # Set logging to INFO to see the progress bars from your utils file
    logging.basicConfig(level=logging.INFO)
    cleanup_last_episode()
