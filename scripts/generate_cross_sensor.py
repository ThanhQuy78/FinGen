"""
Cross-sensor fingerprint transfer using the trained UNet+ControlNet+Flow-Matching
pipeline (unet_controlnet.py) — for each dataset pair (source image I_A's
structural map -> target sensor), render the same identity as it would look
captured on the target sensor. Runs over the *entire* SD302a dataset
(--split all, bypassing the train/val subject split CrossSensorFingerprintDataset
otherwise applies), sensor H excluded (already dropped at the dataset level
via configs/unet_controlnet_config.yaml's `exclude_sensors: ["H"]`).

The real on-disk naming convention (dataset.py's docstring) is
`{SUBJECT}_{DEVICE}_{IMPRESSION}_{FRGP}.png` = id_sensor_roll_fingerposition
(IMPRESSION is always "roll" here — configs/*.yaml's `sd302a_impressions:
["roll"]` is the only one indexed). FRGP (finger position: 01-10, one code
per finger of the hand — confirmed 10 distinct codes / 2000 (subject, FRGP)
combos across the 200 subjects here) is the actual identity unit
biometrically, not the subject alone: a person's ten fingers are ten
different fingerprints. So the output groups by id_fingerposition, each
holding every sensor rendering generated so far for that specific finger,
named to match the real convention:

    outputs/cross_sensor_gen/<subject_id>_<frgp>/<subject_id>_<SENSOR>_roll_<frgp>.png

Idempotent / resumable by design: if that file already exists, generation
for that (finger, sensor) pair is skipped entirely (not regenerated, not
overwritten) — rerunning only fills in sensors a finger doesn't have yet.

Usage:
    python scripts/generate_cross_sensor.py --checkpoint outputs/training_unet_controlnet/latest.pt
"""

import sys
import os
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.data.dataset import CrossSensorFingerprintDataset
from src.data.sd302a_inspector import SENSOR_TO_INDEX

INDEX_TO_SENSOR = {v: k for k, v in SENSOR_TO_INDEX.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Cross-sensor fingerprint transfer, organized per identity")
    p.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--checkpoint", type=str, default="./outputs/training_unet_controlnet/latest.pt")
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "all"],
                    help="'all' = entire SD302a dataset, bypassing the train/val subject split")
    p.add_argument("--num_samples", type=int, default=-1, help="-1 = all samples in the split")
    p.add_argument("--nfe_steps", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="./outputs/cross_sensor_gen")
    return p.parse_args()


def save_image_tensor(tensor: torch.Tensor, path: str):
    import cv2
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    vae = FingerprintVAE(
        latent_channels=config["vae"]["latent_channels"], base_channels=config["vae"]["base_channels"]
    ).to(device)
    vae.load_pretrained(config["vae"]["weights"])
    vae.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = UNetControlNetDenoiser(**ckpt["model_kwargs"]).to(device)
    ema_sd = ckpt["ema_state_dict"]
    model_sd = model.state_dict()
    for name in model_sd:
        if name in ema_sd:
            model_sd[name] = ema_sd[name]
    model.load_state_dict(model_sd)
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    dataset = CrossSensorFingerprintDataset(config, split=args.split)
    num_samples = args.num_samples if args.num_samples > 0 else len(dataset)
    num_samples = min(num_samples, len(dataset))
    print(f"Scanning {num_samples} {args.split} pairs...")

    traj_manager = RectifiedFlowTrajectoryManager()
    latent_size = config["unet_model"]["latent_size"]
    in_channels = config["unet_model"]["in_channels"]

    t_start = time.time()
    generated, skipped = 0, 0

    for i in range(num_samples):
        subject_id = dataset.samples[i]["subject_id"]
        # sample_id is f"sd302a_{subject}_{frgp}_{s_a}2{s_b}" (dataset.py's
        # _index_sd302a) — frgp (finger position, 01-10) is the true identity
        # unit, not the subject alone: ten different fingers, ten different
        # fingerprints. subject_id itself is purely numeric (no underscores),
        # so splitting on "_" and taking index 2 reliably recovers it.
        frgp = dataset.samples[i]["sample_id"].split("_")[2]
        sensor_b_idx = int(dataset.samples[i]["sensor_B"])
        sensor_b_letter = INDEX_TO_SENSOR.get(sensor_b_idx, f"idx{sensor_b_idx}")

        finger_dir = os.path.join(args.output_dir, f"{subject_id}_{frgp}")
        out_path = os.path.join(finger_dir, f"{subject_id}_{sensor_b_letter}_roll_{frgp}.png")

        if os.path.exists(out_path):
            skipped += 1
            continue

        sample = dataset[i]
        img_a = sample["img_A"].unsqueeze(0).to(device)
        S_aligned = sample["S_aligned"].unsqueeze(0).to(device)
        sensor_b = sample["sensor_B"].unsqueeze(0).to(device)

        if img_a.shape[-1] != 256:
            img_a = F.interpolate(img_a, size=(256, 256), mode="bilinear", align_corners=False)
        S_for_model = F.interpolate(S_aligned, size=(latent_size, latent_size),
                                    mode="bilinear", align_corners=False)

        with torch.no_grad():
            gen_lat = traj_manager.sample_euler(
                model, shape=(1, in_channels, latent_size, latent_size),
                c=sensor_b, struct_map=S_for_model, steps=args.nfe_steps,
            )
            gen_img = vae.decode(gen_lat)

        os.makedirs(finger_dir, exist_ok=True)
        save_image_tensor(gen_img, out_path)
        generated += 1

        if generated % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{num_samples}] generated={generated} skipped={skipped} "
                  f"({elapsed:.0f}s, {elapsed/max(generated,1)*1000:.0f}ms/generated)")

    elapsed = time.time() - t_start
    finger_dirs = len([d for d in os.listdir(args.output_dir)
                       if os.path.isdir(os.path.join(args.output_dir, d))])
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  Generated: {generated} new (identity, sensor) images")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  id_fingerposition folders with at least one image: {finger_dirs}")
    print(f"  Output: {args.output_dir}/<subject_id>_<frgp>/<subject_id>_<SENSOR>_roll_<frgp>.png")


if __name__ == "__main__":
    main()
