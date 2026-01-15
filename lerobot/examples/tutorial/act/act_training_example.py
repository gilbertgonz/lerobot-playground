"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    output_directory = Path("outputs/train/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"

    dataset_id = "gilbertgonz/so101_training_data"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root="outputs/datasets")
    features = dataset_to_policy_features(dataset_metadata.features)

    target_description = "Pick up the black object and place it inside the green cup."
    matching_episode_indices = []
    
    # Filter episodes based on task description
    for i in range(dataset_metadata.total_episodes):
        episode_data = dataset_metadata.episodes[i]
        episode_tasks = episode_data.get('tasks', [])
        if any(target_description.lower() == t.lower() for t in episode_tasks):
            matching_episode_indices.append(i)

    print(f"Found {len(matching_episode_indices)} episodes for: '{target_description}'") 

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # Instantiate the dataset
    dataset = LeRobotDataset(
        dataset_id, 
        delta_timestamps=delta_timestamps, 
        root="outputs/datasets", 
        episodes=matching_episode_indices if len(matching_episode_indices) > 0 else None
    )

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 6
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Training Parameters
    training_steps = 20000 
    log_freq = 100    
    save_freq = 4000

    scaler = torch.amp.GradScaler()

    # Run training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = preprocessor(batch)

            # Use autocast for the forward pass
            with torch.amp.autocast("cuda"):
                loss, _ = policy.forward(batch)

            # Scale the loss and step the optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            # Checkpoint Saving
            if step > 0 and step % save_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint_{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Saving checkpoint to {checkpoint_dir}...")
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)

            step += 1
            if step >= training_steps:
                done = True
                break

    # Final Save
    final_dir = output_directory / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Finished! Saving final model to {final_dir}")
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    # Save all assets to the Hub
    # policy.push_to_hub("<user>/robot_learning_tutorial_act")
    # preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    # postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")


if __name__ == "__main__":
    main()