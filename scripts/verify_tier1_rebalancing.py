"""
Check the effect of the Tier-1 dataset-rebalancing knobs (min_pairs_per_finger,
sensor_b_oversample -- see src/data/dataset.py's CrossSensorFingerprintDataset)
by comparing the pair index built with and without them: sensor_B frequency
distribution and pairs-per-finger distribution, before and after.

Usage:
    python scripts/verify_tier1_rebalancing.py --config configs/unet_controlnet_config_tier1.yaml
"""
import sys
import os
import argparse
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np

from src.data.dataset import CrossSensorFingerprintDataset
from src.data.sd302a_inspector import SENSOR_TO_INDEX

IDX_TO_SENSOR = {v: k for k, v in SENSOR_TO_INDEX.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Verify Tier-1 dataset rebalancing")
    p.add_argument("--base_config", type=str, default="./configs/unet_controlnet_config.yaml")
    p.add_argument("--config", type=str, default="./configs/unet_controlnet_config_tier1.yaml")
    return p.parse_args()


def report(ds, label):
    print(f"\n{label}")
    print(f"  total pairs: {len(ds.samples)}")
    b_counts = Counter(IDX_TO_SENSOR[s["sensor_B"]] for s in ds.samples)
    print(f"  sensor_B distribution: {dict(sorted(b_counts.items()))}")

    finger_pair_counts = Counter()
    for s in ds.samples:
        parts = s["sample_id"].split("_")
        finger_pair_counts[f"{parts[1]}_{parts[2]}"] += 1
    counts = np.array(list(finger_pair_counts.values()))
    print(f"  pairs/finger: min={counts.min()} max={counts.max()} mean={counts.mean():.2f}")


def main():
    args = parse_args()

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)
    report(CrossSensorFingerprintDataset(base_config, split="all"), "BASELINE (no rebalancing)")

    with open(args.config) as f:
        tier1_config = yaml.safe_load(f)
    report(CrossSensorFingerprintDataset(tier1_config, split="all"), "TIER-1 (rebalanced)")


if __name__ == "__main__":
    main()
