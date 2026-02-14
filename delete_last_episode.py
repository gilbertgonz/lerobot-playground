import logging
import shutil # Added for moving the folder back
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

def cleanup_last_episode():
    repo_id = "gilbertgonz/so101_training_data_wrist_only_v2"
    local_root = Path("outputs/datasets").resolve()
    current_path = local_root / repo_id
    temp_path = local_root / f"{repo_id}_tmp" # Temporary path
    
    print(f"Loading dataset from {current_path}...")
    dataset = LeRobotDataset(str(current_path))
    
    if dataset.meta.total_episodes == 0:
        print("Dataset is already empty.")
        return

    last_episode_idx = dataset.meta.total_episodes - 1
    print(f"Attempting to delete Episode Index: {last_episode_idx}")

    try:
        # Create the new dataset in a temporary location
        new_dataset = delete_episodes(
            dataset=dataset,
            episode_indices=[last_episode_idx],
            repo_id=repo_id,
            output_dir=temp_path # Write to tmp
        )
        
        print(f"Successfully created cleaned version. Swapping folders...")
        
        # 5. Swap the folders: Delete the old one and move the new one in
        shutil.rmtree(current_path)
        temp_path.rename(current_path)
        
        print(f"Cleanup complete. New total: {new_dataset.meta.total_episodes}")
        
    except Exception as e:
        print(f"Error during deletion: {e}")
        if temp_path.exists():
            print(f"Cleaning up failed temp directory: {temp_path}")
            shutil.rmtree(temp_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_last_episode()