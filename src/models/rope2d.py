import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding (2D-RoPE) for DiT / MM-DiT backbones.
    Applies spatial rotary position encodings to Q and K tensors at each attention block.
    
    Supports:
    - Shared 2D spatial grid (for aligned / pre-registered Stream X & Stream Y).
    - Offset position IDs (Δ shift, OminiControl style) for unaligned Stream Y data
      to prevent RoPE spatial alignment bias.
    """

    def __init__(self, dim: int = 64, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Half dimension for H, half dimension for W
        self.dim_h = dim // 2
        self.dim_w = dim // 2
        
        inv_freq_h = 1.0 / (theta ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        inv_freq_w = 1.0 / (theta ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
        
        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)

    def _get_spatial_grid(
        self,
        H: int,
        W: int,
        device: torch.device,
        offset_delta: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates 2D grid coordinates for H and W with optional offset Δ.
        """
        pos_h = torch.arange(H, device=device).float()
        pos_w = torch.arange(W, device=device).float() + offset_delta

        # Compute raw angle products, then apply sin/cos
        angles_h = torch.einsum("i,j->ij", pos_h, self.inv_freq_h)
        angles_w = torch.einsum("i,j->ij", pos_w, self.inv_freq_w)

        sin_h = torch.sin(angles_h)
        cos_h = torch.cos(angles_h)
        sin_w = torch.sin(angles_w)
        cos_w = torch.cos(angles_w)

        # Expand grid across 2D spatial dimensions H x W
        sin_h_grid = sin_h[:, None, :].expand(H, W, -1)
        cos_h_grid = cos_h[:, None, :].expand(H, W, -1)
        sin_w_grid = sin_w[None, :, :].expand(H, W, -1)
        cos_w_grid = cos_w[None, :, :].expand(H, W, -1)

        sin_2d = torch.cat([sin_h_grid, sin_w_grid], dim=-1).reshape(H * W, -1)
        cos_2d = torch.cat([cos_h_grid, cos_w_grid], dim=-1).reshape(H * W, -1)

        # Duplicate for sin/cos rotation pair
        sin_2d = torch.cat([sin_2d, sin_2d], dim=-1)
        cos_2d = torch.cat([cos_2d, cos_2d], dim=-1)
        return sin_2d, cos_2d

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        offset_delta: int = 0
    ) -> torch.Tensor:
        """
        Applies 2D-RoPE to tensor x of shape (B, num_heads, N, head_dim) where N = H * W.
        """
        sin, cos = self._get_spatial_grid(H, W, x.device, offset_delta=offset_delta)
        # Reshape for broadcasting with heads: (1, 1, N, dim)
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        return (x * cos) + (self._rotate_half(x) * sin)


if __name__ == "__main__":
    rope = RotaryEmbedding2D(dim=64)
    q = torch.randn(2, 8, 256, 64) # Batch=2, Heads=8, N=256 (16x16 grid), HeadDim=64
    q_rope_aligned = rope.apply_rope(q, H=16, W=16, offset_delta=0)
    q_rope_offset = rope.apply_rope(q, H=16, W=16, offset_delta=100)
    print("RoPE output shape:", q_rope_aligned.shape)
    print("RoPE offset difference norm:", torch.norm(q_rope_aligned - q_rope_offset).item())
