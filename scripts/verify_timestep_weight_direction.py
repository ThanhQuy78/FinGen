"""
Empirically test whether x1-estimate reliability actually increases with t
(this codebase's convention: t=0 noise, t=1 clean target), as the analytical
derivation predicted -- to check whether loss_builder.py's timestep weight
(1-t) for L_Identity/L_Orient is applying full weight in the LEAST reliable
regime and near-zero weight in the MOST reliable regime.

For each of N real (source, target) pairs:
  1. Fix one random noise draw x_0 (per pair).
  2. For t in {0.1, 0.3, 0.5, 0.7, 0.9}: build x_t = (1-t)*x_0 + t*x_1_true
     (straight line toward the TRUE target latent -- simulates "the model is
     given a state that's genuinely on-trajectory at this t").
  3. Run the real trained model once (single forward, no sampling loop) to
     get v_pred at (x_t, t, sensor_b, S_aligned).
  4. Compute x1_est = x_t + (1-t)*v_pred (RectifiedFlowTrajectoryManager's
     own compute_x0_estimate).
  5. Compare x1_est to the TRUE target latent x_1_true (latent MSE), and
     decode both to pixel space for a DMD identity-cosine-similarity check
     against the real target image.
Report mean error per t, alongside the weight loss_builder.py actually
assigns there (post-fix: t; pre-fix it was 1-t, backwards -- see
loss_builder.py's _get_timestep_weight_scale).

Usage:
    python scripts/verify_timestep_weight_direction.py
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.data.dataset import CrossSensorFingerprintDataset
from src.losses.identity_loss import IdentityCosineLoss


def parse_args():
    p = argparse.ArgumentParser(description="Check x1-estimate reliability vs timestep t")
    p.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--checkpoint", type=str, default="./outputs/training_unet_controlnet/latest.pt")
    p.add_argument("--n_pairs", type=int, default=24)
    return p.parse_args()


args = parse_args()
CONFIG_PATH = args.config
CKPT_PATH = args.checkpoint

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"

vae = FingerprintVAE(latent_channels=config["vae"]["latent_channels"], base_channels=config["vae"]["base_channels"]).to(device)
vae.load_pretrained(config["vae"]["weights"])
vae.eval()

ckpt = torch.load(CKPT_PATH, map_location=device)
model = UNetControlNetDenoiser(**ckpt["model_kwargs"]).to(device)
ema_sd = ckpt["ema_state_dict"]
model_sd = model.state_dict()
for name in model_sd:
    if name in ema_sd:
        model_sd[name] = ema_sd[name]
model.load_state_dict(model_sd)
model.eval()
print(f"Loaded {CKPT_PATH} (epoch {ckpt.get('epoch')})")

id_loss_fn = IdentityCosineLoss(embedder_type="dmd", checkpoint_path=config["losses"]["identity_checkpoint"]).to(device)
id_loss_fn.eval()

dataset = CrossSensorFingerprintDataset(config, split="all")
traj = RectifiedFlowTrajectoryManager()
latent_size = config["unet_model"]["latent_size"]

torch.manual_seed(0)
idxs = torch.randperm(len(dataset))[:args.n_pairs].tolist()
t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

results = {t: {"latent_mse": [], "dmd_cos_sim": []} for t in t_values}

with torch.no_grad():
    for idx in idxs:
        sample = dataset[idx]
        img_a = sample["img_A"].unsqueeze(0).to(device)
        img_b = sample["img_B"].unsqueeze(0).to(device)
        S_aligned = sample["S_aligned"].unsqueeze(0).to(device)
        sensor_b = sample["sensor_B"].unsqueeze(0).to(device)

        if img_a.shape[-1] != 256:
            img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
        if img_b.shape[-1] != 256:
            img_b = F.interpolate(img_b, size=(256, 256), mode="bilinear", align_corners=False)
        S_for_model = F.interpolate(S_aligned, size=(latent_size, latent_size), mode="bilinear", align_corners=False)

        x1_true = vae.encode(img_b)  # true target latent
        x0_noise = torch.randn_like(x1_true)  # fixed noise draw for this pair

        for t_val in t_values:
            t = torch.full((1,), t_val, device=device)
            t_expand = t.view(1, 1, 1, 1)
            x_t = (1.0 - t_expand) * x0_noise + t_expand * x1_true

            v_pred = model(x_t, t, sensor_b, S_for_model)
            x1_est = traj.compute_x0_estimate(x_t, v_pred, t)

            mse = F.mse_loss(x1_est, x1_true).item()
            results[t_val]["latent_mse"].append(mse)

            gen_img = vae.decode(x1_est).clamp(0, 1)
            cos_sim = 1.0 - id_loss_fn(gen_img, img_b).item()  # vs REAL target image
            results[t_val]["dmd_cos_sim"].append(cos_sim)

print(f"\n{'t':>6} | {'old weight=1-t':>14} | {'latent MSE (est vs true target)':>32} | {'DMD cos-sim (est vs real target)':>33}")
print("(old weight column shows what the pre-fix `1-t` formula assigned, for reference -- current code uses `t`)")
print("-" * 95)
for t_val in t_values:
    mse_mean = np.mean(results[t_val]["latent_mse"])
    mse_std = np.std(results[t_val]["latent_mse"])
    sim_mean = np.mean(results[t_val]["dmd_cos_sim"])
    sim_std = np.std(results[t_val]["dmd_cos_sim"])
    w = 1.0 - t_val  # pre-fix formula, shown for comparison against the MSE/cos-sim reliability trend
    print(f"{t_val:>6.1f} | {w:>14.2f} | {mse_mean:>18.4f} +/- {mse_std:<10.4f} | {sim_mean:>18.4f} +/- {sim_std:<10.4f}")
