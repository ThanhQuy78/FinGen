import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.controlnet_baseline import ControlNetTransformerBaseline
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.data.dataset import CrossSensorFingerprintDataset


def main():
    print("=" * 60)
    print("TRAINING CONTROLNET TRANSFORMER BASELINE")
    print("=" * 60)

    config_path = "./configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model & Trajectory Manager
    model = ControlNetTransformerBaseline().to(device)
    traj_manager = RectifiedFlowTrajectoryManager()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    dataset = CrossSensorFingerprintDataset(config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Running Baseline Training Demo (3 Steps)...")
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= 3:
            break

        # Dummy VAE encoding (4, H_lat, W_lat)
        target_lat = torch.randn(batch["img_B"].shape[0], 4, 32, 32, device=device)
        struct_map = batch["S_aligned"].to(device)
        sensor_b = batch["sensor_B"].to(device)

        # Flow Trajectory
        x_t, t, v_target, x_0 = traj_manager.sample_trajectory(target_lat)

        optimizer.zero_grad()
        v_pred = model(x_t, t, sensor_b, struct_map)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        loss.backward()
        optimizer.step()

        print(f"  Step {step+1}/3 - Baseline Velocity MSE Loss: {loss.item():.6f}")

    print("Baseline execution completed successfully.")


if __name__ == "__main__":
    main()
