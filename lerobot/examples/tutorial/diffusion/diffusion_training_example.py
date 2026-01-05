from pathlib import Path
import torch
from torchvision.transforms import Resize, InterpolationMode
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from datetime import datetime

# Paths and Dataset Configuration
OUTPUT_DIRECTORY = "outputs/train/diffusion_so101"
DATASET_ID = "gilbertgonz/so101_training_data"

# Image Resolution
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

# Policy Configuration
USE_SEPARATE_RGB_ENCODER_PER_CAMERA = True

# DataLoader Configuration
BATCH_SIZE = 32
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = True
NUM_WORKERS = 8

# Training Configuration
TRAINING_STEPS = 20000
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 5000

# ============================================================================

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None: return [0]
    return [i / fps for i in delta_indices]

def main():
    # 1. Setup paths and device
    output_directory = Path(OUTPUT_DIRECTORY)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata and force re-calculation of features for target resolution
    dataset_metadata = LeRobotDatasetMetadata(DATASET_ID)
    features = dataset_to_policy_features(dataset_metadata.features)

    # 2. Force input features to target resolution to match our intended training resolution
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    for key, ft in input_features.items():
        if ft.type is FeatureType.VISUAL:
            ft.shape = (3, TARGET_HEIGHT, TARGET_WIDTH)

    # 3. Initialize Policy with separate encoders for Webcam vs Realsense
    cfg = DiffusionConfig(
        input_features=input_features, 
        output_features=output_features,
        use_separate_rgb_encoder_per_camera=USE_SEPARATE_RGB_ENCODER_PER_CAMERA
    )
    policy = DiffusionPolicy(cfg)
    
    # Create processors using updated metadata
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    # 4. Setup Timestamps and Data Loading
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    delta_timestamps |= {k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps) for k in cfg.image_features}

    # This transform fixes the "stack expects each tensor to be equal size" error
    resize_transform = Resize((TARGET_HEIGHT, TARGET_WIDTH), interpolation=InterpolationMode.BILINEAR, antialias=True)

    dataset = LeRobotDataset(
        DATASET_ID, 
        delta_timestamps=delta_timestamps, 
        image_transforms=resize_transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, num_workers=NUM_WORKERS
    )

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    
    # 5. Training Loop
    step = 0
    print("Starting Training...")
    
    while step < TRAINING_STEPS:
        for batch in dataloader:
            # Move to GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Normalize and Forward
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % LOG_INTERVAL == 0:
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Step: {step} | Loss: {loss.item():.4f}")
            
            if step > 0 and step % CHECKPOINT_INTERVAL == 0:
                policy.save_pretrained(output_directory / f"checkpoint_{step}")
                
            step += 1
            if step >= TRAINING_STEPS: break

    # 6. Final Save
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"Finished! Model saved to {output_directory}")

if __name__ == "__main__":
    main()