"""
Render ONE SD302a finger on every sensor A-G (H excluded, same as training),
using the trained UNet+ControlNet+Flow-Matching pipeline (unet_controlnet.py).

This is the same underlying mechanism as generate_missing_sensors.py — the
model conditions on the Stage-1 structural map S_aligned (derived from ONE
real source image of the finger, any sensor it already has) and a plain
sensor-embedding index (SENSOR_TO_INDEX), so any target sensor A-G can be
requested regardless of whether that finger has real ground truth for it —
just scoped down to a single (subject, frgp) instead of a whole split.

For sensors the finger already has a REAL capture for, that real file is
copied through as-is (no need to regenerate a worse synthetic version of
something you already have for free). Only genuinely missing sensors are
generated.

NOTE on quality: earlier evaluation (see the extended-training-set
discussion) found the generator extrapolates poorly to (identity, sensor)
combinations it never saw paired during training — synthetic missing-sensor
renders came out noticeably blurrier / lower-contrast than real captures
(Laplacian variance ~1,900 vs ~10,800) and visibly lack real ridge-flow
structure. Treat this script's *generated* outputs as a quick look /
prototyping aid, not production-grade cross-sensor data, unless the
generator has since been improved.

Output:
    <output_dir>/<subject_id>_<frgp>/<subject_id>_<SENSOR>_roll_<frgp>.png
    (one file per sensor A-G; a `source.json` alongside records which files
    are real copies vs generated, and which sensor was used as the source)

Usage:
    python scripts/generate_one_fingerprint_all_sensors.py --subject 00002303 --frgp 01
"""

import sys
import os
import json
import shutil
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet_controlnet import UNetControlNetDenoiser
from src.models.vae import FingerprintVAE
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.data.dataset import CrossSensorFingerprintDataset, image_cache_key
from src.data.sd302a_inspector import SD302aInspector, SENSOR_TO_INDEX, NIST_SD302A_SENSOR_TECH

ALL_SENSORS = sorted(NIST_SD302A_SENSOR_TECH)


def parse_args():
    p = argparse.ArgumentParser(description="Render one finger on every sensor A-G")
    p.add_argument("--subject", type=str, required=True, help="Subject id, e.g. 00002303")
    p.add_argument("--frgp", type=str, required=True, help="Finger position code, e.g. 01")
    p.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--checkpoint", type=str, default="./outputs/training_unet_controlnet/latest.pt")
    p.add_argument("--source_sensor", type=str, default=None,
                    help="Force this real sensor as the source for generation "
                         "(default: the finger's sharpest real capture, by Laplacian variance)")
    p.add_argument("--nfe_steps", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="./outputs/single_fingerprint_all_sensors")
    p.add_argument("--force", action="store_true", help="Regenerate/recopy even if the output file already exists")
    return p.parse_args()


def save_image_tensor(tensor: torch.Tensor, path: str):
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def pick_sharpest_source(by_sensor: dict, candidates: list) -> str:
    """
    Real captures vary wildly in native sharpness across sensor technologies
    (e.g. a capacitive sensor's raw image can be 10-50x higher Laplacian
    variance than an optical one for the same finger) -- and since the
    ControlNet's identity conditioning (S_aligned) is derived from whichever
    source image is picked, a soft/low-contrast source visibly degrades
    generation quality more than picking a sharper alternative does.
    Ties broken alphabetically for determinism.
    """
    def sharpness(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return -1.0
        return cv2.Laplacian(img, cv2.CV_64F).var()

    return max(sorted(candidates), key=lambda s: sharpness(by_sensor[s]))


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    ds_cfg = config.get("dataset", {})

    frgp = f"{int(args.frgp):02d}"
    finger_key = f"{args.subject}_{frgp}"

    root = ds_cfg.get("sd302a_root", "")
    impressions = ds_cfg.get("sd302a_impressions") or None
    contact_only = bool(ds_cfg.get("contact_only", True))
    exclude_sensors = {s.upper() for s in ds_cfg.get("exclude_sensors", [])}
    target_sensors = [s for s in ALL_SENSORS if s not in exclude_sensors]

    inspector = SD302aInspector(root, impressions=impressions)
    finger_index = inspector.build_finger_index()
    contact = set(inspector.get_contact_sensors())

    by_sensor = finger_index.get(finger_key)
    if not by_sensor:
        print(f"ERROR: finger {finger_key} not found under sd302a_root={root!r}")
        sys.exit(1)
    if contact_only:
        by_sensor = {s: p for s, p in by_sensor.items() if s in contact}
    if exclude_sensors:
        by_sensor = {s: p for s, p in by_sensor.items() if s not in exclude_sensors}
    real_sensors = sorted(by_sensor)

    if not real_sensors:
        print(f"ERROR: finger {finger_key} has no usable real sensor (after contact_only/exclude_sensors filtering)")
        sys.exit(1)

    source_sensor = args.source_sensor.upper() if args.source_sensor else pick_sharpest_source(by_sensor, real_sensors)
    if source_sensor not in real_sensors:
        print(f"ERROR: --source_sensor {source_sensor} has no real capture for {finger_key}. "
              f"Real sensors available: {real_sensors}")
        sys.exit(1)

    missing_sensors = [s for s in target_sensors if s not in real_sensors]

    print(f"Finger:          {finger_key}")
    print(f"Real sensors:    {real_sensors}")
    print(f"Missing sensors: {missing_sensors or '(none — finger already has all of ' + str(target_sensors) + ')'}")
    print(f"Source sensor:   {source_sensor}")

    out_dir = os.path.join(args.output_dir, finger_key)
    os.makedirs(out_dir, exist_ok=True)
    provenance = {}

    # ── copy through real sensors as-is ──
    for sensor in real_sensors:
        if sensor not in target_sensors:
            continue  # e.g. H, excluded from the target set
        out_path = os.path.join(out_dir, f"{args.subject}_{sensor}_roll_{frgp}.png")
        provenance[sensor] = "real"
        if os.path.exists(out_path) and not args.force:
            continue
        shutil.copyfile(by_sensor[sensor], out_path)

    if not missing_sensors:
        with open(os.path.join(out_dir, "source.json"), "w") as f:
            json.dump({"finger_key": finger_key, "source_sensor": source_sensor, "provenance": provenance}, f, indent=2)
        print(f"\nNothing to generate — {finger_key} already has every target sensor. Output: {out_dir}")
        return

    # ── generate the missing ones ──
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    dataset = CrossSensorFingerprintDataset(config, split="all")
    traj_manager = RectifiedFlowTrajectoryManager()
    latent_size = config["unet_model"]["latent_size"]
    in_channels = config["unet_model"]["in_channels"]

    cache_key_A = image_cache_key(source_sensor, args.subject, frgp)
    stub_sample = {"cache_key_A": cache_key_A, "sample_id": f"single_{finger_key}_src{source_sensor}"}
    S_aligned, _is_aligned = dataset._load_stage1(stub_sample)
    S_for_model = F.interpolate(
        S_aligned.unsqueeze(0).to(device), size=(latent_size, latent_size),
        mode="bilinear", align_corners=False,
    )

    for target_sensor in missing_sensors:
        out_path = os.path.join(out_dir, f"{args.subject}_{target_sensor}_roll_{frgp}.png")
        provenance[target_sensor] = "generated"
        if os.path.exists(out_path) and not args.force:
            print(f"  [skip, exists] {target_sensor}")
            continue

        sensor_b = torch.tensor([SENSOR_TO_INDEX[target_sensor]], dtype=torch.long, device=device)
        with torch.no_grad():
            gen_lat = traj_manager.sample_euler(
                model, shape=(1, in_channels, latent_size, latent_size),
                c=sensor_b, struct_map=S_for_model, steps=args.nfe_steps,
            )
            gen_img = vae.decode(gen_lat)
        save_image_tensor(gen_img, out_path)
        print(f"  [generated] {target_sensor} -> {out_path}")

    with open(os.path.join(out_dir, "source.json"), "w") as f:
        json.dump({"finger_key": finger_key, "source_sensor": source_sensor, "provenance": provenance}, f, indent=2)

    print(f"\nDone. {len(real_sensors)} real + {len(missing_sensors)} generated = "
          f"{len(real_sensors) + len(missing_sensors)} sensors. Output: {out_dir}")


if __name__ == "__main__":
    main()
