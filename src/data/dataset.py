"""
Unified PyTorch Dataset for Contact-Based Cross-Sensor Fingerprint Generation & Transfer.

Primary source is NIST SD302a in its real on-disk layout
(`{root}/{DEVICE}/{IMPRESSION}/png/{SUBJECT}_{DEVICE}_{IMPRESSION}_{FRGP}.png`);
unpaired domain datasets (FVC, SD14, SD4, PrintsGAN) can be mixed in.

Geometry contract (important — Stage 1 caching depends on it):
    Every image is first placed on a SQUARE canvas of side `max(H, W)` (centered,
    padded with the border intensity), then resized to `image_size`.
    `Stage1OfflinePreprocessor` applies the *same* canvas transform before running
    CoarseNet, so the cached structural map S is pixel-aligned with `img_A`.
"""

import os
import glob
import hashlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.sd302a_inspector import SD302aInspector, SENSOR_TO_INDEX


# ---------------------------------------------------------------------------
# Canonical geometry — shared by the dataset and the offline preprocessor
# ---------------------------------------------------------------------------

def to_square_canvas(img: np.ndarray) -> np.ndarray:
    """
    Centers `img` on a square canvas of side max(H, W), padding with the median
    border intensity (fingerprint backgrounds are light, so this keeps the
    padding visually inert). Returns the canvas at NATIVE scale — no resampling,
    which matters because CoarseNet's Gabor bank is tuned for ~500 dpi ridges.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    if h == side and w == side:
        return img

    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    fill = int(np.median(border))

    canvas = np.full((side, side), fill, dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def load_canonical_image(path: str, image_size: Optional[int] = None) -> np.ndarray:
    """
    Reads a grayscale fingerprint, applies `to_square_canvas`, and optionally
    resizes to `image_size`. Raises FileNotFoundError / ValueError on bad input.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    img = to_square_canvas(img)
    if image_size is not None and img.shape[0] != image_size:
        interp = cv2.INTER_AREA if img.shape[0] > image_size else cv2.INTER_LINEAR
        img = cv2.resize(img, (image_size, image_size), interpolation=interp)
    return img


def image_cache_key(sensor: str, subject: str, frgp: str) -> str:
    """Stable per-IMAGE cache key (Stage 1 features depend on the image, not the pair)."""
    return f"sd302a_{sensor}_{subject}_{frgp}"


def subject_split(subject_id: str, val_fraction: float, seed: int = 42) -> str:
    """
    Deterministic subject-level train/val assignment.

    Splitting by subject (not by sample) prevents the same identity appearing in
    both halves, which would make the cross-identity generalisation numbers
    meaningless.
    """
    digest = hashlib.md5(f"{seed}:{subject_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrossSensorFingerprintDataset(Dataset):
    """
    Yields cross-sensor pairs (I_A, I_B) of the *same* physical finger together
    with the Stage-1 structural map S_aligned derived from I_A.

    Args:
        config:          parsed default_config.yaml
        split:           'train' | 'val' | 'all'
        cached_prep_dir: overrides `dataset.cached_prep_dir`
        allow_synthetic: if True, fall back to dummy samples when no real data is
                         found (used by tests / demo_pipeline). Real training
                         scripts should leave this False so a mis-configured path
                         fails loudly instead of silently training on noise.
    """

    def __init__(
        self,
        config: Dict,
        split: str = "train",
        cached_prep_dir: Optional[str] = None,
        allow_synthetic: bool = False,
    ):
        self.config = config
        self.split = split
        ds_cfg = config.get("dataset", {})

        self.image_size = ds_cfg.get("image_size", 256)
        self.cached_dir = cached_prep_dir or ds_cfg.get("cached_prep_dir", "./data/cached_stage1")
        self.val_fraction = float(ds_cfg.get("val_fraction", 0.1))
        self.split_seed = int(ds_cfg.get("split_seed", 42))
        self.contact_only = bool(ds_cfg.get("contact_only", True))
        self.directed_pairs = bool(ds_cfg.get("directed_pairs", True))
        self.max_pairs_per_finger = int(ds_cfg.get("max_pairs_per_finger", 0))  # 0 = unlimited
        self.allow_synthetic = allow_synthetic
        self.require_cache = bool(ds_cfg.get("require_cache", False))

        self.samples = self._load_sample_index()

    # -- index construction --------------------------------------------------

    def _load_sample_index(self) -> List[Dict]:
        samples: List[Dict] = []
        ds_cfg = self.config.get("dataset", {})

        samples += self._index_sd302a(ds_cfg)
        samples += self._index_unpaired(ds_cfg)

        if not samples:
            if not self.allow_synthetic:
                raise RuntimeError(
                    "No dataset samples found.\n"
                    f"  dataset.sd302a_root = {ds_cfg.get('sd302a_root')!r}\n"
                    "Point it at the directory containing the A..H device folders "
                    "(e.g. './archive/images/challengers'), or construct the dataset "
                    "with allow_synthetic=True for smoke tests."
                )
            samples = self._synthetic_samples()
        return samples

    def _index_sd302a(self, ds_cfg: Dict) -> List[Dict]:
        root = ds_cfg.get("sd302a_root", "")
        if not root or not os.path.isdir(root):
            return []

        impressions = ds_cfg.get("sd302a_impressions") or None
        inspector = SD302aInspector(root, impressions=impressions)
        finger_index = inspector.build_finger_index()

        contact = set(inspector.get_contact_sensors())
        samples: List[Dict] = []

        for finger_key in sorted(finger_index):
            by_sensor = finger_index[finger_key]
            if self.contact_only:
                by_sensor = {s: p for s, p in by_sensor.items() if s in contact}
            sensors = sorted(by_sensor)
            if len(sensors) < 2:
                continue

            subject, frgp = finger_key.split("_", 1)
            if self.split != "all" and \
               subject_split(subject, self.val_fraction, self.split_seed) != self.split:
                continue

            pairs: List[Tuple[str, str]] = []
            for i in range(len(sensors)):
                for j in range(i + 1, len(sensors)):
                    pairs.append((sensors[i], sensors[j]))
                    if self.directed_pairs:
                        pairs.append((sensors[j], sensors[i]))

            if self.max_pairs_per_finger > 0:
                # Deterministic subsample so epochs stay reproducible.
                rng = np.random.RandomState(
                    int(hashlib.md5(finger_key.encode()).hexdigest()[:8], 16)
                )
                keep = rng.permutation(len(pairs))[: self.max_pairs_per_finger]
                pairs = [pairs[k] for k in sorted(keep)]

            for s_a, s_b in pairs:
                samples.append({
                    "sample_id": f"sd302a_{subject}_{frgp}_{s_a}2{s_b}",
                    "path_A": by_sensor[s_a],
                    "path_B": by_sensor[s_b],
                    "cache_key_A": image_cache_key(s_a, subject, frgp),
                    "sensor_A": SENSOR_TO_INDEX[s_a],
                    "sensor_B": SENSOR_TO_INDEX[s_b],
                    "subject_id": subject,
                    "is_paired": True,
                })
        return samples

    def _index_unpaired(self, ds_cfg: Dict) -> List[Dict]:
        roots = list(ds_cfg.get("fvc_roots") or [])
        for key in ("multisensor_root", "sd14_root", "sd4_root", "printsgan_root"):
            extra = ds_cfg.get(key, "")
            if extra:
                roots.append(extra)

        samples: List[Dict] = []
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for img_path in sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True)):
                if not img_path.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.bmp')):
                    continue
                basename = os.path.basename(img_path)
                stem = os.path.splitext(basename)[0]
                subject = stem.split("_")[0]
                if self.split != "all" and \
                   subject_split(subject, self.val_fraction, self.split_seed) != self.split:
                    continue
                tag = f"unpaired_{os.path.basename(os.path.normpath(root))}_{stem}"
                samples.append({
                    "sample_id": tag,
                    "path_A": img_path,
                    "path_B": img_path,   # unpaired: style is learned via the sensor label
                    "cache_key_A": tag,
                    "sensor_A": 0, "sensor_B": 1,
                    "subject_id": subject,
                    "is_paired": False,
                })
        return samples

    def _synthetic_samples(self) -> List[Dict]:
        return [{
            "sample_id": f"pair_{i:04d}",
            "path_A": "", "path_B": "",
            "cache_key_A": f"pair_{i:04d}",
            "sensor_A": 0, "sensor_B": 1,
            "subject_id": f"sub_{i % 20:03d}",
            "is_paired": True,
            "synthetic": True,
        } for i in range(100)]

    # -- item access ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_tensor(self, path: str, synthetic: bool) -> torch.Tensor:
        """Grayscale fingerprint -> (1, H, W) float tensor in [0, 1]."""
        if synthetic:
            img = np.random.randint(50, 200, (self.image_size, self.image_size), dtype=np.uint8)
        else:
            img = load_canonical_image(path, self.image_size)
        return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

    def _load_stage1(self, sample: Dict) -> Tuple[torch.Tensor, bool]:
        """Loads the cached Stage-1 structural map for this sample's source image."""
        cache_path = os.path.join(self.cached_dir, f"{sample['cache_key_A']}_stage1.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            S = cached["S_aligned"]
            if S.dim() == 4:
                S = S.squeeze(0)
            if S.shape[-1] != self.image_size:
                S = torch.nn.functional.interpolate(
                    S.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
            return S, bool(cached.get("is_aligned", False))

        if self.require_cache:
            raise FileNotFoundError(
                f"Stage-1 cache missing for {sample['sample_id']}: {cache_path}\n"
                "Run: python scripts/run_offline_preprocessing.py"
            )
        # Uncached fallback — random structure. Fine for shape/smoke tests, useless
        # for real training, hence the `require_cache` guard above.
        return torch.randn(6, self.image_size, self.image_size), False

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        synthetic = sample.get("synthetic", False)

        img_A = self._load_image_tensor(sample["path_A"], synthetic)
        img_B = self._load_image_tensor(sample["path_B"], synthetic)
        S_aligned, is_aligned = self._load_stage1(sample)

        return {
            "img_A": img_A,                          # (1, H, W) source image I_A
            "img_B": img_B,                          # (1, H, W) target image I_B
            "S_aligned": S_aligned,                  # (6, H, W) structural map from I_A
            "sensor_A": torch.tensor(sample["sensor_A"], dtype=torch.long),
            "sensor_B": torch.tensor(sample["sensor_B"], dtype=torch.long),
            "is_aligned": torch.tensor(int(is_aligned), dtype=torch.long),
            "sample_id": sample["sample_id"],
        }

    # -- helpers -------------------------------------------------------------

    def unique_source_images(self) -> List[Dict]:
        """
        De-duplicated list of {cache_key, path} over every image referenced as a
        source. Stage-1 preprocessing iterates this, not the pair list — 13.6k
        images instead of ~80k pairs.
        """
        seen: Dict[str, str] = {}
        for s in self.samples:
            if s.get("synthetic"):
                continue
            seen.setdefault(s["cache_key_A"], s["path_A"])
        return [{"cache_key": k, "path": v} for k, v in sorted(seen.items())]


if __name__ == "__main__":
    import yaml

    with open("./configs/default_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    for split in ("train", "val"):
        ds = CrossSensorFingerprintDataset(cfg, split=split)
        subjects = {s["subject_id"] for s in ds.samples}
        print(f"{split:5s}: {len(ds):6d} pairs | {len(subjects):3d} subjects | "
              f"{len(ds.unique_source_images()):5d} unique source images")

    ds = CrossSensorFingerprintDataset(cfg, split="train")
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
    print("\nBatch check:")
    for k in ("img_A", "img_B", "S_aligned"):
        print(f"  {k:10s} {tuple(batch[k].shape)}  "
              f"[{batch[k].min():.3f}, {batch[k].max():.3f}]")
    print("  sensor_A:", batch["sensor_A"].tolist(), "sensor_B:", batch["sensor_B"].tolist())
