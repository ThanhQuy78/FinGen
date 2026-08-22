"""
Convert MinutiaeNet CoarseNet pretrained Keras (.h5) weights to PyTorch (.pt).

Usage:
  python scripts/convert_minutiaenet_weights.py                       # uses config paths
  python scripts/convert_minutiaenet_weights.py --h5 CoarseNet.h5 --output weights/coarsenet_pytorch.pt

Requirements: h5py (no TensorFlow needed).

The mapping lives in `FingerNetExtractor.build_keras_state_dict` — each PyTorch
module carries the Keras layer name it mirrors, so conversion is exact rather than
name-guessed. The script fails loudly if any tensor is left unmapped.

Reference: https://github.com/luannd/MinutiaeNet
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.fingernet_extractor import FingerNetExtractor


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert MinutiaeNet CoarseNet Keras .h5 weights to PyTorch .pt")
    p.add_argument("--config", type=str, default="./configs/default_config.yaml")
    p.add_argument("--h5", type=str, default="",
                   help="Keras CoarseNet .h5 (default: preprocessing.coarsenet_keras_h5)")
    p.add_argument("--output", type=str, default="",
                   help="Output .pt (default: preprocessing.coarsenet_weights)")
    p.add_argument("--verify_image", type=str, default="",
                   help="Optional fingerprint image to sanity-check the converted model")
    return p.parse_args()


def main():
    args = parse_args()

    h5_path, out_path = args.h5, args.output
    if not h5_path or not out_path:
        import yaml
        with open(args.config, "r") as f:
            prep = yaml.safe_load(f).get("preprocessing", {})
        h5_path = h5_path or prep.get("coarsenet_keras_h5", "")
        out_path = out_path or prep.get("coarsenet_weights", "weights/coarsenet_pytorch.pt")

    if not h5_path:
        sys.exit("ERROR: no .h5 path given and preprocessing.coarsenet_keras_h5 is empty")
    if not os.path.exists(h5_path):
        sys.exit(f"ERROR: file not found: {h5_path}")

    print("=" * 68)
    print("COARSENET KERAS -> PYTORCH WEIGHT CONVERSION")
    print("=" * 68)
    print(f"Source : {h5_path}  ({os.path.getsize(h5_path)/1024/1024:.1f} MB)")
    print(f"Target : {out_path}")
    print("-" * 68)

    model = FingerNetExtractor()
    stats = model.load_pretrained_keras(h5_path, save_pt_path=out_path, strict=True)

    print("-" * 68)
    print(f"Tensors mapped     : {stats['loaded']}")
    print(f"Unmapped (PyTorch) : {stats['unmapped_pytorch']}")
    print(f"Missing (Keras)    : {stats['missing_keras']}")
    print(f"Parameters         : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Saved              : {out_path} "
          f"({os.path.getsize(out_path)/1024/1024:.1f} MB)")

    # Round-trip check: the .pt must reproduce the .h5-loaded model exactly.
    reloaded = FingerNetExtractor()
    reloaded.load_pretrained_pytorch(out_path)
    a = dict(model.state_dict())
    b = dict(reloaded.state_dict())
    assert a.keys() == b.keys(), "round-trip key mismatch"
    max_delta = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
    print(f"Round-trip max diff: {max_delta:.3e}")
    if max_delta > 0:
        sys.exit("ERROR: round-trip mismatch - saved weights differ from loaded ones")

    if args.verify_image:
        import numpy as np
        from src.data.dataset import load_canonical_image

        model.eval()
        canvas = load_canonical_image(args.verify_image, image_size=None)
        side = canvas.shape[0] - (canvas.shape[0] % 8)
        canvas = canvas[:side, :side]
        t = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, None]
        with torch.no_grad():
            out = model(t)
        seg = out["segmentation_map"][0, 0]
        mnt = model.extract_minutiae(feats=out)[0]
        print("-" * 68)
        print(f"Sanity check on {args.verify_image} ({side}x{side}):")
        print(f"  foreground fraction : {(seg > 0.5).float().mean():.3f}")
        print(f"  minutiae detected   : {len(mnt)}")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
