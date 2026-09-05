import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from src.losses.identity_loss import IdentityCosineLoss
from src.losses.orientation_loss import OrientationCoherenceLoss


class Stage4CompositeLossBuilder(nn.Module):
    """
    Composite Stage 4 Loss Manager:
    L_total = L_Diff + w_id(t, epoch) * L_Identity + w_ori(t, epoch) * L_Orient
    
    Includes:
    - Warm-up schedule across initial epochs (e.g. epochs 0..warmup_epochs run L_Diff only).
    - Timestep weight decay (t large -> x0-estimate is blurry -> decrease identity/orient loss weights).
    """

    def __init__(
        self,
        l_diff_weight: float = 1.0,
        l_identity_weight: float = 0.1,
        l_orient_weight: float = 0.05,
        warmup_epochs: int = 5,
        timestep_decay: bool = True,
        identity_embedder: str = "dmd",
        identity_checkpoint: str = "./weights/dmd.pt",
    ):
        super().__init__()
        self.w_diff = l_diff_weight
        self.w_id_max = l_identity_weight
        self.w_ori_max = l_orient_weight
        self.warmup_epochs = warmup_epochs
        self.timestep_decay = timestep_decay

        self.identity_loss_fn = IdentityCosineLoss(
            embedder_type=identity_embedder,
            checkpoint_path=identity_checkpoint or None,
        )
        self.orient_loss_fn = OrientationCoherenceLoss()

    def _get_epoch_warmup_factor(self, epoch: int) -> float:
        """Returns warmup factor in [0, 1] based on current epoch."""
        if epoch < self.warmup_epochs:
            return 0.0
        # Ramp up linearly over 5 epochs after warmup
        ramp_epochs = 5
        progress = (epoch - self.warmup_epochs) / float(ramp_epochs)
        return min(1.0, max(0.0, progress))

    def _get_timestep_weight_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns timestep weight multiplier t in [0, 1].

        This codebase's rectified-flow trajectory (flow_matching.py's
        sample_trajectory) runs t=0 (noise, x_0) -> t=1 (clean target, x_1) --
        the OPPOSITE of the standard DDPM convention this function used to
        assume. x1_est = x_t + (1-t)*v_pred (compute_x0_estimate), so a fixed
        model error in v_pred is scaled by (1-t): the x1-estimate is least
        reliable near t=0 and most reliable near t=1. Empirically (see
        verify_timestep_weight_direction.py), DMD identity cosine similarity
        of x1_est against the real target rises from ~0.46 at t=0.1 to ~0.77
        at t=0.9. So identity/orientation loss should be weighted UP as t -> 1
        (reliable estimate, worth pulling toward), not down -- the previous
        `1 - t` here did the opposite: near-max weight on the noisiest,
        least-identity-preserving estimates, near-zero weight on the best ones.
        """
        if not self.timestep_decay:
            return torch.ones_like(t)
        return torch.clamp(t, min=0.0, max=1.0)

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        x0_est: torch.Tensor,
        img_A: torch.Tensor,
        S_aligned: torch.Tensor,
        t: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates Stage 4 composite losses.
        
        Input:
          - v_pred: Model predicted velocity (B, C, H, W)
          - v_target: Ground truth velocity (B, C, H, W)
          - x0_est: Estimated clean target image (B, 1, H, W)
          - img_A: Source image I_A (B, 1, H, W)
          - S_aligned: Pre-registered structural map (B, 6, H, W)
          - t: Denoising timestep tensor (B,)
          - epoch: Current training epoch
        """
        # 1. Diffusion Velocity MSE Loss
        l_diff = F.mse_loss(v_pred, v_target)

        # Warmup and Timestep Weight Scaling
        warmup_factor = self._get_epoch_warmup_factor(epoch)
        t_weight = self._get_timestep_weight_scale(t).mean()

        # 2. Differentiable Identity Loss
        if warmup_factor > 0 and self.w_id_max > 0:
            l_identity = self.identity_loss_fn(x0_est, img_A)
            w_id = self.w_id_max * warmup_factor * t_weight
        else:
            l_identity = torch.tensor(0.0, device=v_pred.device)
            w_id = 0.0

        # 3. Differentiable Orientation Coherence Loss
        if warmup_factor > 0 and self.w_ori_max > 0:
            l_orient = self.orient_loss_fn(x0_est, S_aligned)
            w_ori = self.w_ori_max * warmup_factor * t_weight
        else:
            l_orient = torch.tensor(0.0, device=v_pred.device)
            w_ori = 0.0

        l_total = (self.w_diff * l_diff) + (w_id * l_identity) + (w_ori * l_orient)

        return {
            "loss_total": l_total,
            "loss_diff": l_diff,
            "loss_identity": l_identity,
            "loss_orient": l_orient,
            "weight_identity": torch.tensor(float(w_id), device=v_pred.device),
            "weight_orient": torch.tensor(float(w_ori), device=v_pred.device),
            "warmup_factor": torch.tensor(float(warmup_factor), device=v_pred.device)
        }


if __name__ == "__main__":
    builder = Stage4CompositeLossBuilder(warmup_epochs=2)
    v_p = torch.randn(2, 4, 32, 32, requires_grad=True)
    v_t = torch.randn(2, 4, 32, 32)
    x0_e = torch.randn(2, 1, 256, 256, requires_grad=True)
    img_a = torch.randn(2, 1, 256, 256)
    S_alg = torch.randn(2, 6, 256, 256)
    t_step = torch.tensor([0.2, 0.8])

    # Test epoch 0 (warmup)
    res_ep0 = builder(v_p, v_t, x0_e, img_a, S_alg, t_step, epoch=0)
    print("Epoch 0 total loss:", res_ep0["loss_total"].item(), "Warmup:", res_ep0["warmup_factor"].item())

    # Test epoch 5 (post warmup)
    res_ep5 = builder(v_p, v_t, x0_e, img_a, S_alg, t_step, epoch=5)
    print("Epoch 5 total loss:", res_ep5["loss_total"].item(), "Identity w:", res_ep5["weight_identity"].item())
