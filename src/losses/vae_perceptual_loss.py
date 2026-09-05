"""
DMD-based perceptual loss for VAE reconstruction (Phase 1).

Plain MSE under-penalizes blur: a mean-texture reconstruction that gets every
pixel "close enough" scores nearly as well as one that keeps sharp ridge lines,
which is why the current `FingerprintVAE` (MSE + small KL) converges to smeared,
ridge-less output (see analysis_results.md). This module reuses **DMD** (Dense
Minutia Descriptor, IJCB 2024) — already vendored at `src/losses/dmd/` and
pretrained on real fingerprints (NIST SD14, `weights/dmd.pt`) for `L_Identity` —
as a fixed feature extractor and penalizes the distance between its *dense*
per-patch features (`feat_t` texture branch, `feat_f` minutiae branch) on the
reconstruction vs. the target, instead of DMD's single pooled identity vector.
Because DMD was trained to describe fingerprint ridge/minutiae structure, this
pushes the decoder to keep that structure rather than just minimizing average
pixel error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.identity_loss import IdentityCosineLoss


class VAEPerceptualLoss(nn.Module):
    """
    L_Perceptual = mean |feat_f(recon) - feat_f(target)| + mean |feat_t(recon) - feat_t(target)|,
    masked by DMD's own foreground mask (computed on the target) so background
    noise outside the fingerprint area doesn't dilute the signal.

    Frozen throughout — gradients flow into `recon` only.
    """

    def __init__(self, checkpoint_path: str = "./weights/dmd.pt", antialias_sigma: float = 0.8):
        super().__init__()
        # Reuses IdentityCosineLoss's loading path (handles the DMD checkpoint's
        # nested state dict) rather than duplicating it; we only need the dense
        # backbone underneath, not its pooled-vector `forward`.
        id_loss = IdentityCosineLoss(embedder_type="dmd", checkpoint_path=checkpoint_path)
        self.dmd = id_loss.embedder.model
        for p in self.dmd.parameters():
            p.requires_grad = False
        self.dmd.eval()

        # DMD downsamples 16x through four stride-2 conv stages (see model_zoo.py)
        # with no anti-aliasing filter — feeding it a full-res image and backprop-ing
        # feature error to pixels is a textbook way to induce grid/checkerboard-style
        # artifacts (aliasing from the strided convs shows up as a periodic gradient
        # signal), independent of anything in the VAE's own decoder. Blurring *only
        # the copy fed to DMD* (not the reconstruction itself, not the recon/KL
        # losses) knocks down the high-frequency content that aliases, without
        # discouraging the decoder from keeping real sharp ridge detail — unlike a
        # generic TV penalty, which can't tell that apart from an artifact (measured:
        # a real fingerprint crop has *higher* TV than one of our checkerboard-y
        # reconstructions, so TV would push toward blur, undoing this loss's point).
        k = max(3, int(2 * round(3 * antialias_sigma) + 1))
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-(ax ** 2) / (2 * antialias_sigma ** 2))
        g = (g / g.sum()).view(1, 1, -1)
        kernel_2d = (g.transpose(1, 2) @ g).view(1, 1, k, k)
        self.register_buffer("_blur_kernel", kernel_2d)
        self._blur_pad = k // 2

    def _antialias(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._blur_kernel, padding=self._blur_pad)

    def train(self, mode: bool = True):
        # Stay in eval mode always — frozen BN running stats, not affected by
        # the caller (VAE) switching between train()/eval().
        return super().train(False)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """recon, target: (B, 1, H, W) in [0, 1]."""
        if recon.shape[1] > 1:
            recon = recon.mean(dim=1, keepdim=True)
        if target.shape[1] > 1:
            target = target.mean(dim=1, keepdim=True)

        recon_in = self._antialias(recon) * 2.0 - 1.0
        target_in = self._antialias(target) * 2.0 - 1.0

        with torch.no_grad():
            out_t = self.dmd(target_in)
        out_r = self.dmd(recon_in)

        mask = out_t["mask_f"].detach()  # (B, 1, h, w), DMD's own foreground estimate
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)

        def masked_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff = (a - b).abs().mean(dim=1, keepdim=True)  # (B, 1, h, w)
            return ((diff * mask).sum(dim=(2, 3)) / denom).mean()

        loss_f = masked_l1(out_r["feat_f"], out_t["feat_f"])
        loss_t = masked_l1(out_r["feat_t"], out_t["feat_t"])
        return loss_f + loss_t


if __name__ == "__main__":
    loss_fn = VAEPerceptualLoss()
    recon = torch.rand(2, 1, 256, 256, requires_grad=True)
    target = torch.rand(2, 1, 256, 256)
    loss = loss_fn(recon, target)
    loss.backward()
    print(f"Perceptual loss: {loss.item():.4f}, recon grad norm: {recon.grad.norm().item():.4f}")
