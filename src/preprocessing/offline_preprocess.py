"""
Stage 1 offline preprocessing.

For every source fingerprint image:
  1. place it on the canonical square canvas at NATIVE scale (`to_square_canvas`),
  2. run CoarseNet to get segmentation / orientation / minutiae maps,
  3. optionally TPS-register the maps onto a target geometry,
  4. resize to the training resolution and cache as `{cache_key}_stage1.pt`.

Running CoarseNet at native scale matters: its Gabor bank is tuned for ~500 dpi
ridge spacing, so downsampling the image first would wreck the enhancement and
hence the minutiae head. Only the resulting maps are downsampled.
"""

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import load_canonical_image
from src.preprocessing.fingernet_extractor import FingerNetExtractor, build_extractor
from src.preprocessing.tps_aligner import ThinPlateSplineAligner


class Stage1OfflinePreprocessor:
    """Extracts CoarseNet structural features, TPS-aligns them, and caches to disk."""

    def __init__(
        self,
        cache_dir: str = "./data/cached_stage1",
        device: str = "cpu",
        config: Optional[Dict] = None,
        extractor: Optional[FingerNetExtractor] = None,
        output_size: int = 256,
        store_full_maps: bool = False,
        max_canvas: int = 768,
    ):
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self.output_size = output_size
        self.store_full_maps = store_full_maps
        # Upper bound on the CoarseNet input side. SD302a canvases run from 512 px
        # (sensor A) to 1248 px (sensor H, a ~1000 ppi device); capping at 768 keeps
        # activations inside a 4 GB GPU and simultaneously pulls the 1000 ppi sensors
        # back towards the ~500 ppi ridge spacing CoarseNet's Gabor bank expects.
        self.max_canvas = max_canvas
        os.makedirs(self.cache_dir, exist_ok=True)

        if extractor is not None:
            self.extractor = extractor.to(self.device).eval()
        elif config is not None:
            self.extractor = build_extractor(config, device=device)
        else:
            raise ValueError("Provide either `config` or a prebuilt `extractor`.")

        self.tps_aligner = ThinPlateSplineAligner()

    # -- helpers -------------------------------------------------------------

    def cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}_stage1.pt")

    def is_cached(self, cache_key: str) -> bool:
        return os.path.exists(self.cache_path(cache_key))

    def _to_output_size(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] == self.output_size and tensor.shape[-2] == self.output_size:
            return tensor
        return F.interpolate(tensor, size=(self.output_size, self.output_size),
                             mode="bilinear", align_corners=False)

    # -- core ----------------------------------------------------------------

    @torch.no_grad()
    def process_image_tensor(
        self,
        img: torch.Tensor,
        cache_key: str,
        src_minutiae_pts: Optional[np.ndarray] = None,
        dst_minutiae_pts: Optional[np.ndarray] = None,
        save: bool = True,
    ) -> Dict:
        """
        img: (1, 1, H, W) canvas-sized tensor in [0, 1]. H, W must be multiples of 8.
        """
        img = img.to(self.device)
        feats = self.extractor(img)

        S_raw = feats["combined_structure"]
        S_aligned, is_aligned = self.tps_aligner.align_structure_tensor(
            S_raw, src_pts=src_minutiae_pts, dst_pts=dst_minutiae_pts
        )

        cached: Dict = {
            "cache_key": cache_key,
            "S_aligned": self._to_output_size(S_aligned).squeeze(0).cpu(),
            "is_aligned": bool(is_aligned),
            "source_size": tuple(img.shape[-2:]),
        }
        if self.store_full_maps:
            cached.update({
                "orientation_map": self._to_output_size(feats["orientation_map"]).cpu(),
                "minutiae_map": self._to_output_size(feats["minutiae_map"]).cpu(),
                "segmentation_map": self._to_output_size(feats["segmentation_map"]).cpu(),
            })

        if save:
            torch.save(cached, self.cache_path(cache_key))
        return cached

    def process_image_file(self, path: str, cache_key: str, overwrite: bool = False) -> Dict:
        """Loads `path` onto the canonical canvas and caches its Stage-1 features."""
        if not overwrite and self.is_cached(cache_key):
            return torch.load(self.cache_path(cache_key), map_location="cpu")

        canvas = load_canonical_image(path, image_size=None)   # native scale, square
        side = canvas.shape[0]

        if self.max_canvas and side > self.max_canvas:
            side = self.max_canvas
            canvas = cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA)

        if side % 8:                                            # crop to a multiple of 8
            side -= side % 8
            canvas = canvas[:side, :side]

        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None, None]
        return self.process_image_tensor(tensor, cache_key)

    def process_many(
        self,
        items: List[Dict],
        overwrite: bool = False,
        progress_every: int = 25,
    ) -> Dict[str, int]:
        """
        items: [{'cache_key': str, 'path': str}, ...] — typically
        `CrossSensorFingerprintDataset.unique_source_images()`.
        """
        import time

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        t0 = time.time()

        for i, item in enumerate(items):
            key, path = item["cache_key"], item["path"]
            if not overwrite and self.is_cached(key):
                stats["skipped"] += 1
                continue
            try:
                self.process_image_file(path, key, overwrite=overwrite)
                stats["processed"] += 1
            except Exception as exc:                            # noqa: BLE001 - keep going
                stats["failed"] += 1
                print(f"  ! {key}: {type(exc).__name__}: {exc}")

            done = i + 1
            if progress_every and done % progress_every == 0:
                elapsed = time.time() - t0
                rate = stats["processed"] / max(elapsed, 1e-6)
                remaining = len(items) - done
                eta = remaining / rate if rate > 0 else float("nan")
                print(f"  [{done}/{len(items)}] processed={stats['processed']} "
                      f"skipped={stats['skipped']} failed={stats['failed']} | "
                      f"{rate:.2f} img/s | ETA {eta/60:.1f} min")
        return stats


if __name__ == "__main__":
    import yaml

    with open("./configs/default_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre = Stage1OfflinePreprocessor(cache_dir="./data/cached_stage1_demo",
                                    device=device, config=cfg)
    out = pre.process_image_tensor(torch.rand(1, 1, 256, 256), "sample_001")
    print(f"S_aligned {tuple(out['S_aligned'].shape)}, is_aligned={out['is_aligned']}")
