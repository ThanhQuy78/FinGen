import torch
import torch.nn as nn
import torch.nn.functional as F
from src.preprocessing.orientation import compute_gradient_orientation, compute_orientation_coherence


class OrientationCoherenceLoss(nn.Module):
    """
    Differentiable Orientation Coherence Loss L_Orient.
    Extracts continuous (cos2θ, sin2θ) orientation map from estimated x_0 and compares
    it with target structural map S_aligned.
    """

    def __init__(self, block_size: int = 16):
        super().__init__()
        self.block_size = block_size

    def forward(self, x0_est: torch.Tensor, S_aligned: torch.Tensor) -> torch.Tensor:
        """
        Input:
          - x0_est: Estimated image tensor (B, 1, H, W)
          - S_aligned: Pre-registered structural map (B, 6, H, W) where channels [1, 2] contain (cos2θ, sin2θ)
        Output:
          - Coherence loss (1 - cosine_similarity) averaged over spatial grid
        """
        if x0_est.shape[1] > 1:
            x0_est = x0_est.mean(dim=1, keepdim=True)

        # Extract gradient orientation from x0_est
        orient_x0 = compute_gradient_orientation(x0_est, block_size=self.block_size) # (B, 2, h_blk, w_blk)

        # Extract reference orientation from S_aligned
        orient_ref_raw = S_aligned[:, 1:3, :, :] # (B, 2, H, W)
        orient_ref = F.avg_pool2d(orient_ref_raw, kernel_size=self.block_size, stride=self.block_size) # (B, 2, h_blk, w_blk)

        # Compute cosine distance
        loss = compute_orientation_coherence(orient_x0, orient_ref)
        return loss


if __name__ == "__main__":
    loss_fn = OrientationCoherenceLoss()
    x0_est = torch.randn(2, 1, 256, 256, requires_grad=True)
    S_aligned = torch.randn(2, 6, 256, 256)
    
    l_ori = loss_fn(x0_est, S_aligned)
    l_ori.backward()
    print(f"Orientation loss: {l_ori.item():.4f}, x0 grad norm: {x0_est.grad.norm().item():.4f}")
