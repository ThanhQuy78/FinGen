"""
Generate sensors a finger was NEVER captured with in real SD302a data, using
the trained UNet+ControlNet+Flow-Matching pipeline (unet_controlnet.py).

Unlike scripts/generate_cross_sensor.py — which only reconstructs a target
sensor's image from a source image when BOTH sensors already have real
ground-truth captures for that finger (used to evaluate the generator
against real data, see that script's docstring and src/data/dataset.py's
`_index_sd302a`) — this script targets *every* sensor A-G (H excluded, same
as training: configs/*.yaml's `exclude_sensors: ["H"]`) regardless of
whether that finger has a real capture for it.

Why this is architecturally valid (not just approximation):
  - Identity/structure conditioning is the Stage-1 structural map S_aligned,
    derived from ONE real source image of the finger (any sensor it already
    has) — see dataset.py:95's "S_aligned derived from I_A". It never reads
    the target sensor's ground truth.
  - The target sensor is a plain learned class-embedding index
    (SENSOR_TO_INDEX -> TimestepSensorEmbedder), decoupled from whether that
    (id, sensor) pair exists on disk.
So nothing in the model requires target-sensor ground truth; the restriction
in generate_cross_sensor.py is purely how it builds its (source, target)
pair list, not a model capability limit.

Scope: only fingers (subject_id, frgp) present in --id_list_dir (default:
data/sd302a_811/train's 1600-id ImageFolder split: subject_<ID>_frgp_<NN>/).

For each such finger:
  1. Determine its real captured sensors, using the SAME contact_only /
     exclude_sensors filtering as training (configs/*.yaml's dataset
     section) — so "real" here means "real AND was eligible during
     training", keeping the source image in-distribution.
  2. missing = {A..G} - real_sensors; skip the finger if empty (nothing to
     fill in).
  3. source sensor = the finger's sharpest real capture (by Laplacian
     variance — real captures vary 10-50x in native sharpness across sensor
     technologies, and a soft source visibly degrades output quality). Must
     have a cached Stage-1 map — run scripts/run_offline_preprocessing.py
     first if missing, or pass --skip_uncached to skip instead of failing).
  4. Generate every missing target sensor from that one source.

Idempotent / resumable by design: skips (finger, sensor) pairs whose output
file already exists — rerunning only fills in gaps left by an interrupted run.

Output:
    outputs/cross_sensor_gen_missing/<subject_id>_<frgp>/<subject_id>_<SENSOR>_roll_<frgp>.png

(Deliberately a separate tree from outputs/cross_sensor_gen/, so images with
no real ground truth to compare against are never mixed in with that
script's real-pair reconstructions.)

Usage:
    # See how much work this would be without loading the model:
    python scripts/generate_missing_sensors.py --dry_run

    # Actually generate:
    python scripts/generate_missing_sensors.py --checkpoint outputs/training_unet_controlnet/latest.pt
"""

import sys
import os
import argparse
import time
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

INDEX_TO_SENSOR = {v: k for k, v in SENSOR_TO_INDEX.items()}
ALL_SENSORS = sorted(NIST_SD302A_SENSOR_TECH)


def parse_args():
    p = argparse.ArgumentParser(description="Generate sensors a finger was never captured with")
    p.add_argument("--config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--checkpoint", type=str, default="./outputs/training_unet_controlnet/latest.pt")
    p.add_argument("--id_list_dir", type=str, default="./data/sd302a_811/train",
                    help="ImageFolder dir (subject_<ID>_frgp_<NN>/...) whose id list scopes generation")
    p.add_argument("--nfe_steps", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="./outputs/cross_sensor_gen_missing")
    p.add_argument("--skip_uncached", action="store_true",
                    help="Skip fingers whose chosen source image has no Stage-1 cache, "
                         "instead of failing (require_cache in the dataset config still applies otherwise)")
    p.add_argument("--dry_run", action="store_true",
                    help="Only report how many (finger, sensor) images would be generated; loads no model")
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


def load_target_finger_keys(id_list_dir: str):
    """subject_<ID>_frgp_<NN> directory names -> {'<ID>_<NN>', ...}"""
    keys = set()
    unrecognized = 0
    for name in sorted(os.listdir(id_list_dir)):
        if not os.path.isdir(os.path.join(id_list_dir, name)):
            continue
        parts = name.split("_")
        if len(parts) != 4 or parts[0] != "subject" or parts[2] != "frgp":
            unrecognized += 1
            continue
        keys.add(f"{parts[1]}_{parts[3]}")
    if unrecognized:
        print(f"  [warn] {unrecognized} entries under {id_list_dir} didn't match "
              f"'subject_<ID>_frgp_<NN>' and were skipped")
    return keys


def plan_generation(config, id_list_dir, output_dir):
    """
    Returns list of (finger_key, subject, frgp, source_sensor, [missing_sensors]),
    plus stats, without touching torch/model.
    """
    ds_cfg = config.get("dataset", {})
    root = ds_cfg.get("sd302a_root", "")
    impressions = ds_cfg.get("sd302a_impressions") or None
    contact_only = bool(ds_cfg.get("contact_only", True))
    exclude_sensors = {s.upper() for s in ds_cfg.get("exclude_sensors", [])}
    target_sensors = [s for s in ALL_SENSORS if s not in exclude_sensors]

    inspector = SD302aInspector(root, impressions=impressions)
    finger_index = inspector.build_finger_index()
    contact = set(inspector.get_contact_sensors())

    target_keys = load_target_finger_keys(id_list_dir)

    plan = []
    n_no_finger, n_no_source, n_already_complete = 0, 0, 0
    n_missing_total = 0

    for finger_key in sorted(target_keys):
        by_sensor = finger_index.get(finger_key)
        if not by_sensor:
            n_no_finger += 1
            continue
        if contact_only:
            by_sensor = {s: p for s, p in by_sensor.items() if s in contact}
        if exclude_sensors:
            by_sensor = {s: p for s, p in by_sensor.items() if s not in exclude_sensors}
        real_sensors = sorted(by_sensor)

        missing = [s for s in target_sensors if s not in real_sensors]
        if not missing:
            n_already_complete += 1
            continue
        if not real_sensors:
            n_no_source += 1
            continue

        subject, frgp = finger_key.split("_", 1)
        source_sensor = pick_sharpest_source(by_sensor, real_sensors)

        # drop targets already generated by a previous run
        finger_dir = os.path.join(output_dir, finger_key)
        remaining = [
            s for s in missing
            if not os.path.exists(os.path.join(finger_dir, f"{subject}_{s}_roll_{frgp}.png"))
        ]
        if not remaining:
            n_already_complete += 1
            continue

        n_missing_total += len(remaining)
        plan.append((finger_key, subject, frgp, source_sensor, remaining))

    stats = {
        "target_fingers": len(target_keys),
        "not_found_in_sd302a": n_no_finger,
        "no_usable_source": n_no_source,
        "already_complete": n_already_complete,
        "fingers_to_process": len(plan),
        "images_to_generate": n_missing_total,
        "target_sensor_set": target_sensors,
    }
    return plan, stats


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Scoping generation to ids in: {args.id_list_dir}")
    plan, stats = plan_generation(config, args.id_list_dir, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Generation plan")
    print(f"{'='*60}")
    print(f"  Target sensor set (H excluded): {stats['target_sensor_set']}")
    print(f"  Fingers in id list:              {stats['target_fingers']}")
    print(f"  Not found in SD302a root:        {stats['not_found_in_sd302a']}")
    print(f"  No usable real source sensor:    {stats['no_usable_source']}")
    print(f"  Already complete (nothing to do):{stats['already_complete']:>6}")
    print(f"  Fingers to process:              {stats['fingers_to_process']}")
    print(f"  Images to generate:              {stats['images_to_generate']}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run — stopping before loading any model.")
        return

    if not plan:
        print("Nothing to generate.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── models ──
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

    # Reuse CrossSensorFingerprintDataset purely for its Stage-1 cache loader
    # (self._load_stage1) and config-driven knobs (cached_dir, image_size,
    # require_cache) — NOT for its (source, target) pair index, which is
    # exactly the real-pairs-only restriction this script exists to bypass.
    dataset = CrossSensorFingerprintDataset(config, split="all")

    traj_manager = RectifiedFlowTrajectoryManager()
    latent_size = config["unet_model"]["latent_size"]
    in_channels = config["unet_model"]["in_channels"]

    t_start = time.time()
    generated, skipped_existing, skipped_uncached = 0, 0, 0

    for finger_key, subject, frgp, source_sensor, missing in plan:
        cache_key_A = image_cache_key(source_sensor, subject, frgp)
        stub_sample = {
            "cache_key_A": cache_key_A,
            "sample_id": f"missing_{subject}_{frgp}_src{source_sensor}",
        }
        try:
            S_aligned, _is_aligned = dataset._load_stage1(stub_sample)
        except FileNotFoundError as e:
            if args.skip_uncached:
                print(f"  [skip] {finger_key}: {e}")
                skipped_uncached += len(missing)
                continue
            raise

        S_for_model = F.interpolate(
            S_aligned.unsqueeze(0).to(device), size=(latent_size, latent_size),
            mode="bilinear", align_corners=False,
        )

        finger_dir = os.path.join(args.output_dir, finger_key)
        os.makedirs(finger_dir, exist_ok=True)

        for target_sensor in missing:
            out_path = os.path.join(finger_dir, f"{subject}_{target_sensor}_roll_{frgp}.png")
            if os.path.exists(out_path):
                skipped_existing += 1
                continue

            sensor_b = torch.tensor([SENSOR_TO_INDEX[target_sensor]], dtype=torch.long, device=device)

            with torch.no_grad():
                gen_lat = traj_manager.sample_euler(
                    model, shape=(1, in_channels, latent_size, latent_size),
                    c=sensor_b, struct_map=S_for_model, steps=args.nfe_steps,
                )
                gen_img = vae.decode(gen_lat)

            save_image_tensor(gen_img, out_path)
            generated += 1

            if generated % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  generated={generated}/{stats['images_to_generate']} "
                      f"skipped_existing={skipped_existing} skipped_uncached={skipped_uncached} "
                      f"({elapsed:.0f}s, {elapsed/max(generated,1)*1000:.0f}ms/img)")

    elapsed = time.time() - t_start
    finger_dirs = len([d for d in os.listdir(args.output_dir)
                       if os.path.isdir(os.path.join(args.output_dir, d))])
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  Generated: {generated} new (finger, sensor) images")
    print(f"  Skipped (already existed): {skipped_existing}")
    print(f"  Skipped (no Stage-1 cache): {skipped_uncached}")
    print(f"  Finger folders with at least one image: {finger_dirs}")
    print(f"  Output: {args.output_dir}/<subject_id>_<frgp>/<subject_id>_<SENSOR>_roll_<frgp>.png")


if __name__ == "__main__":
    main()
