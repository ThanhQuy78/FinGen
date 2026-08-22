"""
Quick-Start Demo: Chạy toàn bộ pipeline end-to-end với dữ liệu synthetic.
Không cần dataset thật — chỉ cần PyTorch + dependencies cơ bản.

Usage:
    cd d:/CV/FinGen/DMCS
    python scripts/demo_pipeline.py

Pipeline demo gồm:
  1. Stage 1: Feature extraction (CoarseNet) + TPS alignment
  2. Stage 2-3: MM-DiT training loop (2 steps)
  3. Inference: Euler sampling sinh ảnh vân tay
  4. Evaluation: Orientation RMSE + Minutiae metrics
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.fingernet_extractor import FingerNetExtractor
from src.preprocessing.tps_aligner import ThinPlateSplineAligner
from src.preprocessing.orientation import compute_gradient_orientation
from src.models.mm_dit import DualStreamMMDiT
from src.models.flow_matching import RectifiedFlowTrajectoryManager
from src.losses.loss_builder import Stage4CompositeLossBuilder
from src.evaluation.eval_metrics import FingerprintEvaluator


def make_synthetic_fingerprint(batch_size: int = 2, size: int = 128) -> torch.Tensor:
    """Tạo ảnh vân tay synthetic (concentric ridges) để demo."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    # Concentric sine waves simulating ridge pattern
    r = torch.sqrt(x**2 + y**2)
    img = 0.5 + 0.4 * torch.sin(15 * r + 2 * x)
    img = img.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    # Add slight noise variation per sample
    img = img + 0.05 * torch.randn_like(img)
    return img.clamp(0, 1)


def main():
    device = "cpu"
    torch.manual_seed(42)

    print("=" * 70)
    print("  DMCS Pipeline Demo - Cross-Sensor Fingerprint Generation & Transfer")
    print("=" * 70)

    # =====================================================================
    # STAGE 1: Feature Extraction + Alignment (Offline, cached)
    # =====================================================================
    print("\n" + "-" * 70)
    print("STAGE 1: Feature Extraction (CoarseNet) + TPS Alignment")
    print("-" * 70)

    t0 = time.time()

    # Tạo cặp vân tay synthetic (sensor A vs sensor B)
    img_a = make_synthetic_fingerprint(batch_size=2, size=128)
    img_b = make_synthetic_fingerprint(batch_size=2, size=128)
    img_b = img_b * 0.8 + 0.1  # Simulate sensor B contrast difference

    print(f"  Input I_A shape: {img_a.shape}  (Sensor A: optical)")
    print(f"  Input I_B shape: {img_b.shape}  (Sensor B: capacitive)")

    # Feature extraction via CoarseNet (random init — no pretrained weights for demo)
    extractor = FingerNetExtractor()
    extractor.eval()
    with torch.no_grad():
        feats = extractor(img_a)

    seg_map = feats["segmentation_map"]
    orient_map = feats["orientation_map"]
    minutiae_map = feats["minutiae_map"]
    S_raw = feats["combined_structure"]  # (B, 6, H, W)

    print(f"  CoarseNet outputs:")
    print(f"    Segmentation:  {seg_map.shape}")
    print(f"    Orientation:   {orient_map.shape}  (cos2t, sin2t)")
    print(f"    Minutiae:      {minutiae_map.shape}  (presence, cos, sin)")
    print(f"    Combined S_A:  {S_raw.shape}")

    # TPS Alignment (demo with dummy points -> returns unaligned)
    aligner = ThinPlateSplineAligner()
    S_aligned, is_aligned = aligner.align_structure_tensor(S_raw)
    print(f"  TPS alignment: is_aligned={is_aligned} (no paired minutiae for demo)")
    print(f"  S_aligned shape: {S_aligned.shape}")

    t1 = time.time()
    print(f"  Time: Stage 1 completed in {t1 - t0:.1f}s")

    # =====================================================================
    # STAGE 2-3: MM-DiT Training Loop (mini demo)
    # =====================================================================
    print("\n" + "-" * 70)
    print("STAGE 2-3: MM-DiT Training (2 steps x 2 epochs)")
    print("-" * 70)

    # Lightweight MM-DiT config for CPU demo
    model = DualStreamMMDiT(
        in_channels=4, struct_channels=6,
        hidden_size=128, depth=2, num_heads=4,
        patch_size=2, num_sensors=8
    ).to(device)

    traj_manager = RectifiedFlowTrajectoryManager()
    loss_builder = Stage4CompositeLossBuilder(
        warmup_epochs=1, l_identity_weight=0.1, l_orient_weight=0.05
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  MM-DiT params: {total_params:,}")

    # Simulate latent space (pretend VAE encoder produced 4-channel 16×16 latents)
    target_lat = torch.randn(2, 4, 16, 16, device=device)
    struct_map_resized = F.interpolate(S_aligned, size=(32, 32), mode="bilinear", align_corners=False).to(device)
    sensor_b = torch.tensor([1, 2], dtype=torch.long, device=device)

    for epoch in [0, 2]:  # Epoch 0 = warmup (L_Diff only), Epoch 2 = post-warmup
        model.train()
        for step in range(2):
            # Sample trajectory
            x_t, t, v_target, x_0 = traj_manager.sample_trajectory(target_lat)

            optimizer.zero_grad()
            v_pred, y_cache = model(
                x_t, t, sensor_b, struct_map_resized, is_aligned=(not is_aligned)
            )

            # Estimate x̂₁ for identity/orient losses
            x0_est_lat = traj_manager.compute_x0_estimate(x_t, v_pred, t)
            x0_est_img = F.interpolate(
                x0_est_lat.mean(dim=1, keepdim=True),
                size=(128, 128), mode="bilinear", align_corners=False
            )

            loss_dict = loss_builder(
                v_pred, v_target, x0_est_img,
                img_a.to(device), S_aligned.to(device),
                t, epoch=epoch
            )
            loss_dict["loss_total"].backward()
            optimizer.step()

            status = "WARMUP" if epoch < 1 else "FULL LOSS"
            print(f"  [{status}] Epoch {epoch} Step {step+1} | "
                  f"L_total={loss_dict['loss_total'].item():.4f} | "
                  f"L_diff={loss_dict['loss_diff'].item():.4f} | "
                  f"L_id(w={loss_dict['weight_identity'].item():.3f})={loss_dict['loss_identity'].item():.4f} | "
                  f"L_ori(w={loss_dict['weight_orient'].item():.3f})={loss_dict['loss_orient'].item():.4f}")

    t2 = time.time()
    print(f"  Time: Stage 2-3 training completed in {t2 - t1:.1f}s")

    # =====================================================================
    # INFERENCE: Euler ODE Sampling
    # =====================================================================
    print("\n" + "-" * 70)
    print("INFERENCE: Euler Sampling (noise -> fingerprint)")
    print("-" * 70)

    model.eval()
    nfe_steps = 10  # Fewer steps for demo speed

    print(f"  Sampling with NFE={nfe_steps} Euler steps...")
    t3 = time.time()

    generated_lat = traj_manager.sample_euler(
        model,
        shape=(1, 4, 16, 16),
        c=torch.tensor([1], device=device),
        struct_map=struct_map_resized[:1],
        is_aligned=True,
        steps=nfe_steps
    )
    print(f"  Generated latent shape: {generated_lat.shape}")

    # Decode latent to image (mock: channel mean → grayscale)
    generated_img = generated_lat.mean(dim=1, keepdim=True)
    generated_img = F.interpolate(generated_img, size=(128, 128), mode="bilinear", align_corners=False)
    generated_img = torch.sigmoid(generated_img)  # Normalize to [0,1]

    print(f"  Generated image shape: {generated_img.shape}")
    print(f"  Pixel range: [{generated_img.min().item():.3f}, {generated_img.max().item():.3f}]")

    t4 = time.time()
    print(f"  Time: Inference completed in {t4 - t3:.1f}s")

    # =====================================================================
    # Y-STREAM CACHING DEMO
    # =====================================================================
    print("\n" + "-" * 70)
    print("Y-STREAM CACHING: Speedup from caching Stream Y across steps")
    print("-" * 70)

    x_test = torch.randn(1, 4, 16, 16, device=device)
    s_test = struct_map_resized[:1]
    c_test = torch.tensor([1], device=device)

    # Pass 1: no cache
    t_start = time.time()
    with torch.no_grad():
        _, y_cache = model(x_test, torch.tensor([0.5]), c_test, s_test, is_aligned=True)
    t_no_cache = time.time() - t_start

    # Pass 2: with cached Y
    t_start = time.time()
    with torch.no_grad():
        _, _ = model(x_test, torch.tensor([0.6]), c_test, s_test, is_aligned=True, cached_y_kv_list=y_cache)
    t_with_cache = time.time() - t_start

    speedup = t_no_cache / max(t_with_cache, 1e-6)
    print(f"  Without cache: {t_no_cache*1000:.1f}ms")
    print(f"  With cache:    {t_with_cache*1000:.1f}ms")
    print(f"  Speedup:       {speedup:.2f}x")

    # =====================================================================
    # EVALUATION
    # =====================================================================
    print("\n" + "-" * 70)
    print("EVALUATION: Metrics on generated vs target")
    print("-" * 70)

    evaluator = FingerprintEvaluator()
    target_img = img_b[:1]

    rmse = evaluator.compute_orientation_rmse(generated_img, target_img)
    print(f"  Orientation RMSE: {rmse:.2f} deg")

    # Synthetic minutiae points for demo
    gen_pts = np.array([[30, 40], [60, 70], [100, 110], [50, 90]])
    tgt_pts = np.array([[31, 41], [62, 68], [120, 130], [50, 91]])
    prec, rec, f1 = evaluator.compute_minutiae_precision_recall(gen_pts, tgt_pts, distance_threshold=15)
    print(f"  Minutiae - Precision: {prec:.2f}  Recall: {rec:.2f}  F1: {f1:.2f}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    t_total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  [OK] Full pipeline demo completed in {t_total:.1f}s")
    print("=" * 70)
    print("""
  To train with real dataset:

  1. Download dataset:
     - NIST SD302a -> ./data/NIST_SD302a/
     - FVC2000-2006 -> ./data/FVC2000/, etc.

  2. (Optional) Download MinutiaeNet pretrained weights:
     - https://github.com/luannd/MinutiaeNet
     - Convert: python scripts/convert_minutiaenet_weights.py --h5 CoarseNet.h5 --output weights/coarsenet_pytorch.pt

  3. Run offline preprocessing (Stage 1):
     python scripts/run_offline_preprocessing.py

  4. Train MM-DiT:
     python scripts/train_mmdit.py

  5. Evaluate:
     python scripts/evaluate.py

  6. Run ablation studies:
     python scripts/run_ablations.py --quick
""")


if __name__ == "__main__":
    main()
