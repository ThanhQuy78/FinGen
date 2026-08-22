"""
PyTorch port of MinutiaeNet CoarseNet.

Reference: "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge"
           Dinh-Luan Nguyen, Kai Cao, Anil K. Jain — ICB 2018
           https://arxiv.org/pdf/1712.09401.pdf
Original Keras implementation: https://github.com/luannd/MinutiaeNet

This is a *layer-for-layer* transcription of `CoarseNetmodel()` so that the released
Keras `CoarseNet.h5` weights load exactly:

    stem  1_0(5x5,64) 1_1 1_2                              -> pool  (1/2)
    blk1  res(64->128) res(128) res(128)  '2_*'             -> pool  (1/4)  conv_block1
    blk2  res(128->256) res(256)          '3_*'             -> pool  (1/8)  conv_block2
    blk3  res(256->512) + 3_4c(512->256)  '3_*c'                    (1/8)  conv_block3
    ASPP  4_1(d=1 on blk3)  4_2(d=4 on blk2)  4_3(d=8 on blk2)
          -> ori_{1,2,3}_* (90 bins) and seg_{1,2,3}_* (1)  summed then sigmoid
    enh   Gabor bank (25x25, 90 orientations) x one-hot orientation -> ridge phase image
    mnt   9x9 -> 5x5 -> 3x3 residual stack on [phase, seg] -> mnt_{o,w,h,s} heads (1/8)

Every Keras layer name is carried on the module as `keras_name`, and
`build_keras_state_dict()` turns the .h5 tree into a PyTorch state_dict with zero
name guessing. Load with `load_pretrained_keras("CoarseNet.h5")`.

Notes on faithfulness:
  * BatchNorm uses eps=1e-3 (Keras default), NOT PyTorch's 1e-5.
  * PReLU has one alpha per channel (Keras `shared_axes=[1, 2]`).
  * The input is min/max-agnostic: `img_normalization` makes the trunk scale
    invariant and `atan2` makes the enhancement branch scale invariant, so
    images may be passed in [0, 1] or [0, 255].
  * Input H, W must be multiples of 8 (the branch runs at 1/8 and upsamples x8).
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_BN_EPS = 1e-3
K_EPSILON = 1e-7


# =============================================================================
# Building blocks (mirroring Keras conv_bn / conv_bn_prelu)
# =============================================================================

class ConvBnPReLU(nn.Module):
    """Conv2D('same') -> BatchNorm(eps=1e-3) -> PReLU(one alpha per channel)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, keras_name: str = ""):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding,
                              dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_ch, eps=KERAS_BN_EPS)
        self.prelu = nn.PReLU(num_parameters=out_ch, init=0.0)
        self.keras_name = keras_name
        # Keras names dilated convs 'atrousconv<name>' instead of 'conv<name>'.
        self.keras_conv_prefix = "conv" if dilation == 1 else "atrousconv"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class KerasConv(nn.Module):
    """Bare Conv2D('same') matching a named Keras output layer (no BN/activation)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, keras_name: str = ""):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              padding=(kernel_size - 1) // 2, bias=True)
        self.keras_name = keras_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResUnit(nn.Module):
    """
    MinutiaeNet residual unit:
        s = conv_1(x);  y = conv_3(conv_2(s));  out = y + s
    `conv_1` doubles as the channel projection, which is why there is no
    separate projection layer (an earlier port added one and broke the mapping).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, names: Tuple[str, str, str]):
        super().__init__()
        self.conv1 = ConvBnPReLU(in_ch, out_ch, kernel_size, keras_name=names[0])
        self.conv2 = ConvBnPReLU(out_ch, out_ch, kernel_size, keras_name=names[1])
        self.conv3 = ConvBnPReLU(out_ch, out_ch, kernel_size, keras_name=names[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.conv1(x)
        out = self.conv3(self.conv2(shortcut))
        return out + shortcut


# =============================================================================
# Keras lambda equivalents
# =============================================================================

def img_normalization(x: torch.Tensor, m0: float = 0.0, var0: float = 1.0) -> torch.Tensor:
    """Per-image zero-mean/unit-variance normalisation (sign preserved about the mean)."""
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    after = torch.sqrt(var0 * (x - m) ** 2 / (var + 1e-12))
    return torch.where(x > m, m0 + after, m0 - after)


def select_max(x: torch.Tensor) -> torch.Tensor:
    """One-hot (over channels) selection of the dominant orientation bin."""
    x = x / (x.amax(dim=1, keepdim=True) + K_EPSILON)
    x = torch.where(x > 0.999, x, torch.zeros_like(x))
    return x / (x.sum(dim=1, keepdim=True) + K_EPSILON)


def gausslabel(length: int = 180, stride: int = 2) -> np.ndarray:
    """
    Gaussian smoothing kernel over orientation bins — Keras shape (1, 1, N, N),
    returned here already in PyTorch layout (N_out, N_in, 1, 1).
    """
    from scipy import signal
    gaussian_pdf = signal.windows.gaussian(length + 1, 3)
    n = length // stride
    centers = np.arange(stride // 2, length, stride)
    label = centers.reshape(-1, 1)
    y = centers.reshape(1, -1)
    delta = np.abs(label - y).astype(int)
    delta = np.minimum(delta, length - delta) + length // 2
    kernel = gaussian_pdf[delta]                       # (N_in, N_out)
    assert kernel.shape == (n, n)
    return kernel.T.reshape(n, n, 1, 1).astype(np.float32)   # (N_out, N_in, 1, 1)


# =============================================================================
# CoarseNet
# =============================================================================

class CoarseNetBackbone(nn.Module):
    """Trunk producing conv_block2 (1/8) and conv_block3 (1/8)."""

    def __init__(self):
        super().__init__()
        self.stem1 = ConvBnPReLU(1, 64, 5, keras_name="1_0")
        self.stem2 = ConvBnPReLU(64, 64, 3, keras_name="1_1")
        self.stem3 = ConvBnPReLU(64, 64, 3, keras_name="1_2")
        self.pool = nn.MaxPool2d(2, 2)

        self.blk1_a = ResUnit(64, 128, 3, ("2_1", "2_2", "2_3"))
        self.blk1_b = ResUnit(128, 128, 3, ("2_1b", "2_2b", "2_3b"))
        self.blk1_c = ResUnit(128, 128, 3, ("2_1c", "2_2c", "2_3c"))

        self.blk2_a = ResUnit(128, 256, 3, ("3_1", "3_2", "3_3"))
        self.blk2_b = ResUnit(256, 256, 3, ("3_1b", "3_2b", "3_3b"))

        self.blk3_a = ResUnit(256, 512, 3, ("3_1c", "3_2c", "3_3c"))
        self.blk3_reduce = ConvBnPReLU(512, 256, 3, keras_name="3_4c")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem3(self.stem2(self.stem1(x)))
        x = self.pool(x)                              # 1/2

        x = self.blk1_c(self.blk1_b(self.blk1_a(x)))
        x = self.pool(x)                              # 1/4  conv_block1

        x = self.blk2_b(self.blk2_a(x))
        conv_block2 = self.pool(x)                    # 1/8  conv_block2

        conv_block3 = self.blk3_reduce(self.blk3_a(conv_block2))   # 1/8
        return conv_block2, conv_block3


class ASPPHeads(nn.Module):
    """
    Multi-scale ASPP: dilation 1 on conv_block3, dilations 4 and 8 on conv_block2
    (both already at 1/8). Orientation (90 bins) and segmentation (1) branches
    are sum-fused then passed through a sigmoid.
    """

    def __init__(self):
        super().__init__()
        self.level2 = ConvBnPReLU(256, 256, 3, dilation=1, keras_name="4_1")
        self.level3 = ConvBnPReLU(256, 256, 3, dilation=4, keras_name="4_2")
        self.level4 = ConvBnPReLU(256, 256, 3, dilation=8, keras_name="4_3")

        for i in (1, 2, 3):
            setattr(self, f"ori{i}_1", ConvBnPReLU(256, 128, 1, keras_name=f"ori_{i}_1"))
            setattr(self, f"ori{i}_2", KerasConv(128, 90, 1, keras_name=f"ori_{i}_2"))
            setattr(self, f"seg{i}_1", ConvBnPReLU(256, 128, 1, keras_name=f"seg_{i}_1"))
            setattr(self, f"seg{i}_2", KerasConv(128, 1, 1, keras_name=f"seg_{i}_2"))

    def forward(self, conv_block2: torch.Tensor,
                conv_block3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        levels = [self.level2(conv_block3), self.level3(conv_block2), self.level4(conv_block2)]

        ori = sum(getattr(self, f"ori{i+1}_2")(getattr(self, f"ori{i+1}_1")(lv))
                  for i, lv in enumerate(levels))
        seg = sum(getattr(self, f"seg{i+1}_2")(getattr(self, f"seg{i+1}_1")(lv))
                  for i, lv in enumerate(levels))
        return torch.sigmoid(ori), torch.sigmoid(seg)


class MinutiaeBranch(nn.Module):
    """Residual stack over [ridge-phase, segmentation] -> mnt_{o,w,h,s} heads at 1/8."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.b1_a = ResUnit(2, 64, 9, ("mnt_1_1", "mnt_1_2", "mnt_1_3"))
        self.b1_b = ResUnit(64, 64, 9, ("mnt_1_1b", "mnt_1_2b", "mnt_1_3b"))

        self.b2_a = ResUnit(64, 128, 5, ("mnt_2_1", "mnt_2_2", "mnt_2_3"))
        self.b2_b = ResUnit(128, 128, 5, ("mnt_2_1b", "mnt_2_2b", "mnt_2_3b"))

        # Block 3 has a non-standard double-skip, so it is spelled out.
        self.b3_1 = ConvBnPReLU(128, 256, 3, keras_name="mnt_3_1")
        self.b3_2 = ConvBnPReLU(256, 256, 3, keras_name="mnt_3_2")
        self.b3_3 = ConvBnPReLU(256, 256, 3, keras_name="mnt_3_3")
        self.b3_4 = ConvBnPReLU(256, 256, 3, keras_name="mnt_3_4")

        self.o_1 = ConvBnPReLU(256 + 90, 256, 1, keras_name="mnt_o_1_1")
        self.o_2 = KerasConv(256, 180, 1, keras_name="mnt_o_1_2")
        self.w_1 = ConvBnPReLU(256, 256, 1, keras_name="mnt_w_1_1")
        self.w_2 = KerasConv(256, 8, 1, keras_name="mnt_w_1_2")
        self.h_1 = ConvBnPReLU(256, 256, 1, keras_name="mnt_h_1_1")
        self.h_2 = KerasConv(256, 8, 1, keras_name="mnt_h_1_2")
        self.s_1 = ConvBnPReLU(256, 256, 1, keras_name="mnt_s_1_1")
        self.s_2 = KerasConv(256, 1, 1, keras_name="mnt_s_1_2")

    def forward(self, enh_seg: torch.Tensor, ori_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pool(self.b1_b(self.b1_a(enh_seg)))          # 1/2
        x = self.pool(self.b2_b(self.b2_a(x)))                # 1/4

        c1 = self.b3_1(x)
        c2 = self.b3_2(c1)
        c3 = self.b3_3(c2) + c1
        c4 = self.b3_4(c3) + c2
        x = self.pool(c4)                                     # 1/8

        o = torch.sigmoid(self.o_2(self.o_1(torch.cat([x, ori_out], dim=1))))
        return {
            "mnt_o": o,                                        # (B, 180, h, w)
            "mnt_w": torch.sigmoid(self.w_2(self.w_1(x))),     # (B, 8, h, w)
            "mnt_h": torch.sigmoid(self.h_2(self.h_1(x))),     # (B, 8, h, w)
            "mnt_s": torch.sigmoid(self.s_2(self.s_1(x))),     # (B, 1, h, w)
        }


class FingerNetExtractor(nn.Module):
    """
    CoarseNet-based structural feature extractor.

    Outputs (all upsampled to the input resolution):
      segmentation_map    (B, 1, H, W)  FOREGROUND probability (already inverted from
                                        the network's background-high `seg_out`)
      orientation_map     (B, 2, H, W)  unit (cos2θ, sin2θ) ridge-flow field
      minutiae_map        (B, 3, H, W)  [score, cos φ, sin φ]
      combined_structure  (B, 6, H, W)  concat([seg, orient, minutiae]) — the S map
      enhanced_phase      (B, 1, H, W)  Gabor ridge-phase image (diagnostic)

    Args:
      input_range: 'unit' if images arrive in [0, 1] (default), 'byte' for [0, 255].
    """

    ORI_BINS = 90
    MNT_BINS = 180

    def __init__(self, in_channels: int = 1, pretrained_path: Optional[str] = None,
                 with_minutiae: bool = True, input_range: str = "unit"):
        super().__init__()
        if input_range not in ("unit", "byte"):
            raise ValueError("input_range must be 'unit' ([0,1]) or 'byte' ([0,255])")
        self.with_minutiae = with_minutiae
        self.input_range = input_range

        self.backbone = CoarseNetBackbone()
        self.aspp = ASPPHeads()

        # Gabor bank — 90 orientations, weights come from the .h5 (never re-derived).
        self.enh_real = KerasConv(1, 90, 25, keras_name="enh_img_real_1")
        self.enh_imag = KerasConv(1, 90, 25, keras_name="enh_img_imag_1")

        if with_minutiae:
            self.minutiae = MinutiaeBranch()

        # Fixed (non-learned) buffers.
        self.register_buffer("ori_peak_kernel", torch.from_numpy(gausslabel(180, 2)))
        # Ridge orientation θ_i for bin i, in the doubled-angle representation.
        theta = (torch.arange(self.ORI_BINS, dtype=torch.float32) * 2.0 + 1.0) * math.pi / 180.0
        self.register_buffer("ori_cos2", torch.cos(2 * theta).view(1, -1, 1, 1))
        self.register_buffer("ori_sin2", torch.sin(2 * theta).view(1, -1, 1, 1))
        # Minutiae direction φ_j for bin j (see MinutiaeNet `label2mnt`).
        phi = (torch.arange(self.MNT_BINS, dtype=torch.float32) * 2.0 - 89.0) * math.pi / 180.0
        phi = (-phi) % (2 * math.pi)
        self.register_buffer("mnt_cos", torch.cos(phi).view(1, -1, 1, 1))
        self.register_buffer("mnt_sin", torch.sin(phi).view(1, -1, 1, 1))

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    # -- forward -------------------------------------------------------------

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """img: (B, 1, H, W) in the range given by `input_range`; H, W multiples of 8."""
        B, _, H, W = img.shape
        if H % 8 or W % 8:
            raise ValueError(f"Input spatial size must be a multiple of 8, got {(H, W)}")

        # MinutiaeNet feeds raw 0-255 images (its deploy script explicitly comments out
        # the /255). The trunk is scale invariant via img_normalization, but the Gabor
        # layer was fine-tuned and carries a large DC term + bias, so scale matters there.
        if self.input_range == "unit":
            img = img * 255.0

        conv_block2, conv_block3 = self.backbone(img_normalization(img))
        ori_out, seg_bg = self.aspp(conv_block2, conv_block3)       # (B,90,h,w), (B,1,h,w)

        # `seg_bg` is high on BACKGROUND — MinutiaeNet's deploy path uses 1 - round(seg).
        seg_fg = 1.0 - seg_bg

        # Gaussian-smoothed histogram -> one-hot dominant orientation bin. This is the
        # network's own orientation read-out; a weighted average over the raw per-bin
        # sigmoids has almost no resultant (~0.36) and decodes to noise.
        ori_peak = select_max(F.conv2d(ori_out, self.ori_peak_kernel))

        # ---- Gabor enhancement -> ridge phase image (full resolution) ----
        up_ori = F.interpolate(ori_peak, size=(H, W), mode="nearest")
        up_seg = F.interpolate(F.softsign(seg_bg), size=(H, W), mode="nearest")

        real = (self.enh_real(img) * up_ori).sum(dim=1, keepdim=True)
        imag = (self.enh_imag(img) * up_ori).sum(dim=1, keepdim=True)
        phase = torch.atan2(imag, real + K_EPSILON)

        # ---- structural maps at input resolution ----
        orient_lr = torch.cat([(ori_peak * self.ori_cos2).sum(1, keepdim=True),
                               (ori_peak * self.ori_sin2).sum(1, keepdim=True)], dim=1)
        orient_map = F.interpolate(orient_lr, size=(H, W), mode="bilinear", align_corners=False)
        orient_map = F.normalize(orient_map, p=2, dim=1, eps=1e-8)
        seg_map = F.interpolate(seg_fg, size=(H, W), mode="bilinear", align_corners=False)

        if self.with_minutiae:
            mnt = self.minutiae(torch.cat([phase, up_seg], dim=1), ori_out)
            p = mnt["mnt_o"] / (mnt["mnt_o"].sum(dim=1, keepdim=True) + K_EPSILON)
            mnt_lr = torch.cat([
                mnt["mnt_s"],
                (p * self.mnt_cos).sum(1, keepdim=True),
                (p * self.mnt_sin).sum(1, keepdim=True),
            ], dim=1)
            minutiae_map = F.interpolate(mnt_lr, size=(H, W), mode="bilinear", align_corners=False)
        else:
            minutiae_map = torch.zeros(B, 3, H, W, device=img.device, dtype=orient_map.dtype)
            mnt = {}

        out = {
            "segmentation_map": seg_map,
            "orientation_map": orient_map,
            "minutiae_map": minutiae_map,
            "combined_structure": torch.cat([seg_map, orient_map, minutiae_map], dim=1),
            "enhanced_phase": phase,
        }
        out.update(mnt)
        return out

    # -- post-processing -----------------------------------------------------

    @staticmethod
    def clean_foreground_mask(seg_map: torch.Tensor, blur_sigma: float = 12.0,
                              threshold: float = 0.35) -> np.ndarray:
        """
        Binary foreground mask. The raw probability oscillates around ~0.5 at ridge
        scale inside the print, so MinutiaeNet's round()-then-morphology recipe leaves
        blobs; blurring to block scale first gives a clean finger-shaped mask. The
        morphology stage is kept (close 10x10 -> open 7x7 -> dilate 5x5).

        seg_map: (1, H, W) or (H, W) FOREGROUND probability. Returns float32 {0, 1}.
        """
        import cv2

        arr = seg_map.detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        if blur_sigma > 0:
            arr = cv2.GaussianBlur(arr, (0, 0), blur_sigma)
        mask = (arr > threshold).astype(np.float32)
        rect = cv2.getStructuringElement
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect(cv2.MORPH_RECT, (10, 10)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (7, 7)))
        return cv2.dilate(mask, rect(cv2.MORPH_RECT, (5, 5)))

    # -- minutiae decoding ---------------------------------------------------

    @torch.no_grad()
    def extract_minutiae(self, img: Optional[torch.Tensor] = None, threshold: float = 0.5,
                         feats: Optional[Dict[str, torch.Tensor]] = None) -> List[np.ndarray]:
        """
        Per-image minutiae list, (N, 4) = [x, y, angle_rad, score].
        Mirrors MinutiaeNet's `label2mnt` (argmax decoding, 8 px cell + sub-cell offset).

        Pass `feats` from an earlier `forward()` to avoid a second full pass.
        """
        if not self.with_minutiae:
            raise RuntimeError("Extractor was built with with_minutiae=False")
        if feats is None:
            if img is None:
                raise ValueError("Provide either `img` or precomputed `feats`")
            feats = self(img)
        s = feats["mnt_s"][:, 0].cpu().numpy()
        w = feats["mnt_w"].argmax(dim=1).cpu().numpy()
        h = feats["mnt_h"].argmax(dim=1).cpu().numpy()
        o = feats["mnt_o"].argmax(dim=1).cpu().numpy()

        results = []
        for b in range(s.shape[0]):
            rows, cols = np.nonzero(s[b] > threshold)
            if rows.size == 0:
                results.append(np.zeros((0, 4), dtype=np.float32))
                continue
            angle = (o[b][rows, cols] * 2.0 - 89.0) / 180.0 * np.pi
            angle = np.where(angle < 0, angle + 2 * np.pi, angle)
            angle = (-angle) % (2 * np.pi)
            results.append(np.stack([
                cols * 8 + w[b][rows, cols],
                rows * 8 + h[b][rows, cols],
                angle,
                s[b][rows, cols],
            ], axis=1).astype(np.float32))
        return results

    # -- weight loading ------------------------------------------------------

    def _keras_layer_specs(self) -> List[Tuple[str, nn.Module]]:
        """(module_path, module) for every module carrying a Keras layer name."""
        return [(path, mod) for path, mod in self.named_modules()
                if getattr(mod, "keras_name", "")]

    def build_keras_state_dict(self, h5_path: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Reads the Keras .h5 weight tree and returns (state_dict, missing_layer_names).
        Keras Conv2D (H, W, Cin, Cout) -> PyTorch (Cout, Cin, H, W);
        BatchNorm gamma/beta/moving_* -> weight/bias/running_*;
        PReLU alpha (1, 1, C) -> (C,).
        """
        import h5py

        raw: Dict[str, np.ndarray] = {}
        with h5py.File(h5_path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    raw[name] = np.array(obj)
            f.visititems(visit)

        def get(layer: str, tensor: str) -> Optional[np.ndarray]:
            return raw.get(f"{layer}/{layer}/{tensor}:0")

        state: Dict[str, torch.Tensor] = {}
        missing: List[str] = []

        for path, mod in self._keras_layer_specs():
            name = mod.keras_name
            prefix = f"{path}." if path else ""

            if isinstance(mod, KerasConv):
                conv_layer = name
            else:  # ConvBnPReLU
                conv_layer = f"{mod.keras_conv_prefix}{name}"

            kernel = get(conv_layer, "kernel")
            if kernel is None:
                missing.append(conv_layer)
                continue
            state[f"{prefix}conv.weight"] = torch.from_numpy(
                kernel.transpose(3, 2, 0, 1).copy())
            bias = get(conv_layer, "bias")
            state[f"{prefix}conv.bias"] = torch.from_numpy(
                bias.copy() if bias is not None else np.zeros(kernel.shape[3], np.float32))

            if isinstance(mod, KerasConv):
                continue

            bn = f"bn-{name}"
            for pt_key, keras_key in (("bn.weight", "gamma"), ("bn.bias", "beta"),
                                      ("bn.running_mean", "moving_mean"),
                                      ("bn.running_var", "moving_variance")):
                arr = get(bn, keras_key)
                if arr is None:
                    missing.append(f"{bn}/{keras_key}")
                else:
                    state[f"{prefix}{pt_key}"] = torch.from_numpy(arr.copy())
            state[f"{prefix}bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

            alpha = get(f"prelu-{name}", "alpha")
            if alpha is None:
                missing.append(f"prelu-{name}/alpha")
            else:
                state[f"{prefix}prelu.weight"] = torch.from_numpy(
                    alpha.reshape(-1).copy())

        return state, missing

    def load_pretrained_keras(self, h5_path: str,
                              save_pt_path: Optional[str] = None,
                              strict: bool = True) -> Dict[str, int]:
        """Loads original Keras CoarseNet .h5 weights. Requires h5py."""
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Keras .h5 not found: {h5_path}")

        state, missing_keras = self.build_keras_state_dict(h5_path)
        result = self.load_state_dict(state, strict=False)

        # Buffers and (optionally) the minutiae branch are legitimately absent.
        unmapped = [k for k in result.missing_keys
                    if not k.startswith(("ori_peak_kernel", "ori_cos", "ori_sin",
                                         "mnt_cos", "mnt_sin"))]
        stats = {"loaded": len(state), "unmapped_pytorch": len(unmapped),
                 "missing_keras": len(missing_keras),
                 "unexpected": len(result.unexpected_keys)}

        if strict and (unmapped or missing_keras or result.unexpected_keys):
            raise RuntimeError(
                "CoarseNet weight conversion is incomplete — architecture/.h5 mismatch.\n"
                f"  unmapped PyTorch params: {unmapped[:10]}\n"
                f"  missing Keras layers:    {missing_keras[:10]}\n"
                f"  unexpected keys:         {list(result.unexpected_keys)[:10]}"
            )

        print(f"[FingerNetExtractor] Loaded {stats['loaded']} tensors from {h5_path} "
              f"(unmapped={stats['unmapped_pytorch']}, missing={stats['missing_keras']})")

        if save_pt_path:
            os.makedirs(os.path.dirname(save_pt_path) or ".", exist_ok=True)
            torch.save({"state_dict": self.state_dict(), "source_h5": os.path.abspath(h5_path)},
                       save_pt_path)
            print(f"[FingerNetExtractor] Saved PyTorch weights to {save_pt_path}")
        return stats

    def load_pretrained_pytorch(self, pt_path: str) -> None:
        """Loads weights previously converted by `load_pretrained_keras`."""
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"CoarseNet weights not found: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.load_state_dict(state, strict=False)
        print(f"[FingerNetExtractor] Loaded pretrained weights from {pt_path}")

    def load_pretrained(self, path: str) -> None:
        """Dispatches on file extension (.h5 -> Keras, otherwise PyTorch)."""
        if path.lower().endswith((".h5", ".hdf5")):
            self.load_pretrained_keras(path)
        else:
            self.load_pretrained_pytorch(path)


def build_extractor(config: Dict, device: str = "cpu",
                    with_minutiae: bool = True) -> FingerNetExtractor:
    """
    Constructs a FingerNetExtractor from `preprocessing.*` config, preferring the
    converted .pt and falling back to the original Keras .h5.
    """
    prep = config.get("preprocessing", {})
    extractor = FingerNetExtractor(with_minutiae=with_minutiae)

    pt_path = prep.get("coarsenet_weights", "")
    h5_path = prep.get("coarsenet_keras_h5", "")
    if pt_path and os.path.exists(pt_path):
        extractor.load_pretrained_pytorch(pt_path)
    elif h5_path and os.path.exists(h5_path):
        extractor.load_pretrained_keras(h5_path)
    else:
        raise FileNotFoundError(
            "No CoarseNet weights available. Set preprocessing.coarsenet_keras_h5 to "
            "CoarseNet.h5 and run: python scripts/convert_minutiaenet_weights.py"
        )

    extractor.to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    return extractor


if __name__ == "__main__":
    model = FingerNetExtractor()
    out = model(torch.rand(2, 1, 256, 256))
    print("FingerNetExtractor (CoarseNet) output shapes:")
    for k, v in out.items():
        print(f"  {k:20s} {tuple(v.shape)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
