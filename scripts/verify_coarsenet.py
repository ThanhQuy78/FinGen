"""
Sanity-check the converted CoarseNet extractor on real SD302a images.

Renders, per sensor, a strip of [input | segmentation | ridge phase | orientation |
minutiae overlay] so the structural maps can be eyeballed before 13k images are
committed to the Stage-1 cache.

Usage:
    python scripts/verify_coarsenet.py
    python scripts/verify_coarsenet.py --sensors A C D --device cuda --out outputs/coarsenet_check
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import load_canonical_image
from src.data.sd302a_inspector import NIST_SD302A_SENSOR_TECH, SD302aInspector
from src.preprocessing.fingernet_extractor import build_extractor


def parse_args():
    p = argparse.ArgumentParser(description="Verify converted CoarseNet weights")
    p.add_argument("--config", type=str, default="./configs/default_config.yaml")
    p.add_argument("--sensors", nargs="*", default=sorted(NIST_SD302A_SENSOR_TECH))
    p.add_argument("--device", type=str, default="")
    p.add_argument("--out", type=str, default="./outputs/coarsenet_check")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--max_canvas", type=int, default=0,
                   help="Override preprocessing.max_canvas (0 = use config)")
    return p.parse_args()


def orientation_quiver(img: np.ndarray, orient: np.ndarray, seg: np.ndarray,
                       stride: int = 16) -> np.ndarray:
    """Draws the (cos2θ, sin2θ) ridge-flow field over the input image."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    theta = 0.5 * np.arctan2(orient[1], orient[0])
    r = stride * 0.45
    for y in range(stride // 2, img.shape[0], stride):
        for x in range(stride // 2, img.shape[1], stride):
            if seg[y, x] < 0.5:
                continue
            dx, dy = r * np.cos(theta[y, x]), r * np.sin(theta[y, x])
            cv2.line(vis, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)),
                     (0, 200, 255), 1, cv2.LINE_AA)
    return vis


def minutiae_overlay(img: np.ndarray, minutiae: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, angle, score in minutiae:
        x, y = int(x), int(y)
        cv2.circle(vis, (x, y), 6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (x, y),
                 (int(x + 14 * np.cos(angle)), int(y - 14 * np.sin(angle))),
                 (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def to_bgr(gray01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(gray01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    max_canvas = args.max_canvas or config["preprocessing"].get("max_canvas", 768)
    print(f"Device: {device} | max_canvas: {max_canvas} px")
    extractor = build_extractor(config, device=device)

    inspector = SD302aInspector(config["dataset"]["sd302a_root"],
                                impressions=config["dataset"].get("sd302a_impressions"))
    records = inspector.scan_files()
    if not records:
        sys.exit(f"No images under {config['dataset']['sd302a_root']!r}")

    by_sensor = {}
    for rec in records:
        by_sensor.setdefault(rec["sensor"], rec)

    print(f"{'sensor':<7}{'tech':<18}{'canvas':>10}{'fg%':>8}{'minutiae':>10}{'sec':>8}")
    print("-" * 61)

    for sensor in args.sensors:
        rec = by_sensor.get(sensor)
        if rec is None:
            print(f"{sensor:<7}(no images)")
            continue

        # Same canvas rule as Stage 1, so what is checked here is what gets cached.
        canvas = load_canonical_image(rec["path"], image_size=None)
        side = canvas.shape[0]
        if max_canvas and side > max_canvas:
            side = max_canvas
            canvas = cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA)
        side -= side % 8
        canvas = canvas[:side, :side]
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, None].to(device)

        t0 = time.time()
        with torch.no_grad():
            out = extractor(tensor)
        minutiae = extractor.extract_minutiae(feats=out)[0]
        elapsed = time.time() - t0

        seg = out["segmentation_map"][0, 0].float().cpu().numpy()
        mask = extractor.clean_foreground_mask(out["segmentation_map"][0])
        orient = out["orientation_map"][0].float().cpu().numpy()

        strip = np.hstack([
            label(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), f"{sensor} input {side}px"),
            label(to_bgr(seg), "foreground prob"),
            label(to_bgr(mask), "cleaned mask"),
            label(orientation_quiver(canvas, orient, mask), "orientation"),
            label(minutiae_overlay(canvas, minutiae), f"minutiae ({len(minutiae)})"),
        ])
        cv2.imwrite(os.path.join(args.out, f"{sensor}_{os.path.basename(rec['path'])}"), strip)

        tech = NIST_SD302A_SENSOR_TECH[sensor]["type"]
        print(f"{sensor:<7}{tech:<18}{side:>10}{mask.mean()*100:>7.1f}%"
              f"{len(minutiae):>10}{elapsed:>8.1f}")

    print("-" * 61)
    print(f"Wrote visualisations to {args.out}")
    print("Expect: segmentation tracking the print, phase showing clean ridge stripes,\n"
          "        orientation flowing along ridges, ~40-150 minutiae on a rolled print.")


if __name__ == "__main__":
    main()
