"""This script demonstrates how to train Diffusion Policy on a real-world dataset."""

from pathlib import Path
import torch
from torchvision.transforms import Resize, Compose, InterpolationMode

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def main():
    output_directory = Path("outputs/robot_learning_tutorial/diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    # (Since you are on an XPS 15 with NVIDIA, try "cuda" if possible, else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}")

    dataset_id = "gilberto/so101_training_data"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, root="outputs/datasets")
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 1. Define Target Size
    target_h, target_w = 224, 224

    # 2. Update Config Shapes
    for key, ft in input_features.items():
        if ft.type is FeatureType.VISUAL:
            ft.shape = (3, target_h, target_w)

    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    
    # [IMPORTANT] Create the preprocessors using the updated config
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # This forces every image from the disk to become 224x224 before the model sees it.
    resize_transform = Resize((target_h, target_w), interpolation=InterpolationMode.BILINEAR, antialias=True)

    # Instantiate the dataset WITH the transform
    dataset = LeRobotDataset(
        dataset_id, 
        delta_timestamps=delta_timestamps, 
        root="outputs/datasets",
        image_transforms=resize_transform
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    
    # Reduced batch size slightly just in case you hit memory limits on the XPS GPU
    batch_size = 32 
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    training_steps = 10000
    log_freq = 100
    save_freq = 2000 # Added save frequency

    step = 0
    done = False
    print("Starting Training Loop...")
    
    while not done:
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            
            # Save checkpoint occasionally
            if step > 0 and step % save_freq == 0:
                policy.save_pretrained(output_directory / f"checkpoint_{step}")
                
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print("Training Complete.")

if __name__ == "__main__":
    main()