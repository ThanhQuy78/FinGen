"""
Stage 1: CoarseNet feature extraction + TPS alignment caching.

Caches one `{cache_key}_stage1.pt` per SOURCE IMAGE (not per pair) — the structural
map depends only on I_A, so ~13.6k SD302a images cover all ~80k training pairs.

Usage:
    python scripts/run_offline_preprocessing.py                     # full dataset
    python scripts/run_offline_preprocessing.py --limit 50          # smoke test
    python scripts/run_offline_preprocessing.py --split val --device cuda
    python scripts/run_offline_preprocessing.py --overwrite
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import CrossSensorFingerprintDataset
from src.preprocessing.fingernet_extractor import build_extractor
from src.preprocessing.offline_preprocess import Stage1OfflinePreprocessor


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 offline preprocessing")
    p.add_argument("--config", type=str, default="./configs/default_config.yaml")
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "all"])
    p.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all)")
    p.add_argument("--device", type=str, default="", help="cuda | cpu (default: auto)")
    p.add_argument("--overwrite", action="store_true", help="Recompute already-cached items")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--max_canvas", type=int, default=0,
                   help="Override preprocessing.max_canvas (CoarseNet input side cap)")
    p.add_argument("--no_minutiae", action="store_true",
                   help="Skip the minutiae branch (~3x faster, zeroes S channels 3:6)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)

    cached_dir = config["dataset"]["cached_prep_dir"]

    print("=" * 68)
    print("STAGE 1 - COARSENET FEATURE EXTRACTION & TPS ALIGNMENT CACHING")
    print("=" * 68)
    print(f"Device      : {device}")
    print(f"Cache dir   : {cached_dir}")
    print(f"Split       : {args.split}")
    print(f"Minutiae    : {'off' if args.no_minutiae else 'on'}")
    print(f"Max canvas  : {args.max_canvas or config['preprocessing'].get('max_canvas', 768)} px")

    dataset = CrossSensorFingerprintDataset(config, split=args.split)
    items = dataset.unique_source_images()
    if args.limit:
        items = items[: args.limit]

    print(f"Pairs       : {len(dataset):,}")
    print(f"Images      : {len(items):,} unique source images to cache")
    print("-" * 68)

    extractor = build_extractor(config, device=device, with_minutiae=not args.no_minutiae)
    preprocessor = Stage1OfflinePreprocessor(
        cache_dir=cached_dir,
        device=device,
        extractor=extractor,
        output_size=config["dataset"].get("image_size", 256),
        max_canvas=args.max_canvas or config["preprocessing"].get("max_canvas", 768),
    )

    t0 = time.time()
    stats = preprocessor.process_many(items, overwrite=args.overwrite)
    elapsed = time.time() - t0

    print("-" * 68)
    print(f"Done in {elapsed/60:.1f} min - processed={stats['processed']} "
          f"skipped={stats['skipped']} failed={stats['failed']}")
    if stats["processed"]:
        print(f"Throughput: {stats['processed']/elapsed:.2f} img/s")
    print(f"Cache: {cached_dir}")

    if stats["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
