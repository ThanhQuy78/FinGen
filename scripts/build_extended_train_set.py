"""
Build an "extended" ImageFolder training set: every real image from
data/sd302a_811/train, plus every generated missing-sensor image from
outputs/cross_sensor_gen_missing (sensors that finger had no real capture
for — see scripts/generate_missing_sensors.py), symlinked together under one
directory with the same id list as the real split.

Idempotent: safe to rerun after regenerating outputs/cross_sensor_gen_missing
with a different source-selection heuristic — it recreates every symlink
each time, so stale links from a previous version never linger.

Usage:
    python scripts/build_extended_train_set.py
    python scripts/build_extended_train_set.py --output_dir data/sd302a_811_extended_v2/train
"""

import os
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build real + missing-sensor-synthetic training set")
    p.add_argument("--train_dir", type=str, default="data/sd302a_811/train")
    p.add_argument("--missing_dir", type=str, default="outputs/cross_sensor_gen_missing")
    p.add_argument("--output_dir", type=str, default="data/sd302a_811_extended/train")
    return p.parse_args()


def link(src: Path, dst_dir: Path):
    link_path = dst_dir / src.name
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(src, dst_dir))


def main():
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    missing_dir = Path(args.missing_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ids = n_real = n_added = n_collisions = n_ids_with_addition = 0

    for id_dir in sorted(train_dir.iterdir()):
        if not id_dir.is_dir():
            continue
        name = id_dir.name  # subject_<SUBJECT>_frgp_<FRGP>
        parts = name.split("_")
        sub, frgp = parts[1], parts[3]

        dst_id_dir = out_dir / name
        dst_id_dir.mkdir(parents=True, exist_ok=True)
        n_ids += 1

        existing_names = set()
        for f in sorted(id_dir.iterdir()):
            if not f.is_file():
                continue
            link(f, dst_id_dir)
            existing_names.add(f.name)
            n_real += 1

        missing_id_dir = missing_dir / f"{sub}_{frgp}"
        if not missing_id_dir.is_dir():
            continue

        added_here = 0
        for f in sorted(missing_id_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name in existing_names:
                n_collisions += 1
                continue
            link(f, dst_id_dir)
            existing_names.add(f.name)
            n_added += 1
            added_here += 1
        if added_here:
            n_ids_with_addition += 1

    print(f"id folders:               {n_ids}")
    print(f"real image links:         {n_real}")
    print(f"added synthetic links:    {n_added}")
    print(f"ids that got 1+ addition: {n_ids_with_addition}")
    print(f"filename collisions (skipped): {n_collisions}")
    print(f"total images:             {n_real + n_added}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
