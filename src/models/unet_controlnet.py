"""
Conditional UNet denoiser + ControlNet side-network, for the UNet backbone
alternative to `mm_dit.py`'s DualStreamMMDiT (see README discussion in the
project chat: MM-DiT's dual-stream joint-attention has no direct UNet
equivalent, since it depends on both streams being transformer token
sequences — a conv UNet needs a different mechanism to inject the aligned
structural map `S_aligned` and keep fingerprint identity).

Design mirrors `controlnet_baseline.py`'s pattern one level up (conv instead
of linear projections, spatial instead of token-sequence blocks):
  - ControlNetUNet is a small trainable copy of the first `control_levels`
    down-resolution stages of the main UNet.
  - Each of its blocks' output goes through a zero-initialized 1x1 conv
    (`control_zero_convs`) and is added into the *same-shape* main-UNet
    down-block output at that index — additive residual injection, exactly
    like `ControlNetTransformerBaseline.forward`'s
    `out_x = out_x + control_outputs[i]` loop, just expressed spatially.
  - Zero-init means the control branch contributes nothing at step 0, so
    training starts identical to an unconditioned UNet and the structural
    signal fades in as those convs move off zero — the standard ControlNet
    stability trick (Zhang et al. 2023).

Conditioning:
  - Timestep t (continuous, in [0, 1], rectified-flow convention — see
    `flow_matching.py`) + sensor domain id -> sinusoidal + embedding MLP,
    same shape of embedding as `mm_dit.py`'s TimestepSensorEmbedder but
    injected into ResBlocks via FiLM (scale/shift), not AdaLN on tokens —
    the natural conditioning mechanism for a conv UNet.
  - `struct_map` (6ch, pixel-aligned to the latent grid) is the only
    identity/structure signal — carried entirely through the ControlNet
    side-branch, no cross-attention path (see project discussion: for
    pixel-precise minutiae/ridge alignment, cross-attention on a spatial map
    is too soft; ControlNet's per-block spatial injection is the right tool).

Call signature intentionally matches what `flow_matching.py`'s
`RectifiedFlowTrajectoryManager.sample_euler` already expects for any
non-MM-DiT model (the `else: v_pred = model(x, t, c, struct_map)` branch,
see flow_matching.py:107) — this model drops in with zero changes to the
flow-matching / sampling code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


def zero_init(module: nn.Module) -> nn.Module:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class TimestepSensorEmbedder(nn.Module):
    """Sinusoidal timestep + sensor-domain embedding -> MLP, for FiLM conditioning."""

    def __init__(self, base_channels: int, time_embed_dim: int, num_sensors: int = 8):
        super().__init__()
        self.base_channels = base_channels
        self.sensor_embedder = nn.Embedding(num_sensors, base_channels)
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        half_dim = self.base_channels // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        c_emb = self.sensor_embedder(c)
        return self.mlp(t_emb + c_emb)


class ResBlock(nn.Module):
    """GroupNorm/SiLU/Conv residual block with FiLM (scale, shift) timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(time_embed_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = zero_init(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        # NOT inplace: `emb` is the same tensor object passed into every ResBlock
        # in the network (down/mid/up/control all share one embedding per step),
        # so self.act(emb) here can't mutate it in place — that broke autograd
        # ("modified by an inplace operation") once more than one ResBlock's
        # backward needed the original emb values.
        self.act = nn.SiLU(inplace=False)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.emb_proj(self.act(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act(h))
        return self.skip(x) + h


class AttnBlock2D(nn.Module):
    """Spatial multi-head self-attention over flattened (H*W) tokens."""

    def __init__(self, channels: int, num_heads: int, groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = zero_init(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(attn)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNetControlNetDenoiser(nn.Module):
    """
    Conditional UNet velocity/noise predictor + additive ControlNet side-branch
    for the structural/identity conditioning map. See module docstring.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        struct_channels: int = 6,
        base_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        num_heads: int = 8,
        num_sensors: int = 8,
        latent_size: int = 64,
        control_levels: int = 2,
    ):
        super().__init__()
        assert 1 <= control_levels <= len(channel_mult), \
            f"control_levels must be in [1, {len(channel_mult)}], got {control_levels}"

        self.num_res_blocks = num_res_blocks
        self.control_levels = control_levels
        self.channels = [base_channels * m for m in channel_mult]
        attn_res = set(attn_resolutions)

        time_embed_dim = base_channels * 4
        self.time_sensor_embed = TimestepSensorEmbedder(base_channels, time_embed_dim, num_sensors)

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Main UNet down path ──
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = base_channels
        resolution = latent_size
        skip_channels = []
        for level, out_ch in enumerate(self.channels):
            for _ in range(num_res_blocks):
                self.down_res.append(ResBlock(in_ch, out_ch, time_embed_dim))
                self.down_attn.append(AttnBlock2D(out_ch, num_heads) if resolution in attn_res else nn.Identity())
                in_ch = out_ch
                skip_channels.append(out_ch)
            is_last = level == len(self.channels) - 1
            self.downsamplers.append(nn.Identity() if is_last else Downsample(out_ch))
            if not is_last:
                resolution //= 2

        # ── Mid ──
        mid_ch = self.channels[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, time_embed_dim)
        self.mid_attn = AttnBlock2D(mid_ch, num_heads)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, time_embed_dim)

        # ── Main UNet up path (mirrors down path, consumes skip connections) ──
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level in reversed(range(len(self.channels))):
            out_ch = self.channels[level]
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                self.up_res.append(ResBlock(in_ch + skip_ch, out_ch, time_embed_dim))
                self.up_attn.append(AttnBlock2D(out_ch, num_heads) if resolution in attn_res else nn.Identity())
                in_ch = out_ch
            self.upsamplers.append(nn.Identity() if level == 0 else Upsample(out_ch))
            if level != 0:
                resolution *= 2

        self.out_norm = nn.GroupNorm(min(32, in_ch), in_ch)
        self.out_conv = zero_init(nn.Conv2d(in_ch, out_channels, 3, padding=1))
        self.act = nn.SiLU(inplace=False)

        # ── ControlNet side-branch: trainable copy of the first `control_levels`
        #    down-resolution stages, fed by struct_map instead of the latent ──
        self.control_stem = nn.Conv2d(struct_channels, base_channels, 3, padding=1)
        self.control_res = nn.ModuleList()
        self.control_attn = nn.ModuleList()
        self.control_downsamplers = nn.ModuleList()
        self.control_zero_convs = nn.ModuleList()
        in_ch_c = base_channels
        resolution_c = latent_size
        for level in range(control_levels):
            out_ch = self.channels[level]
            for _ in range(num_res_blocks):
                self.control_res.append(ResBlock(in_ch_c, out_ch, time_embed_dim))
                self.control_attn.append(AttnBlock2D(out_ch, num_heads) if resolution_c in attn_res else nn.Identity())
                self.control_zero_convs.append(zero_init(nn.Conv2d(out_ch, out_ch, 1)))
                in_ch_c = out_ch
            is_last = level == control_levels - 1
            self.control_downsamplers.append(nn.Identity() if is_last else Downsample(out_ch))
            if not is_last:
                resolution_c //= 2

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        struct_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
          - x: Noisy/interpolated latent (B, in_channels, H, W)
          - t: Flow-matching timestep in [0, 1] (B,)
          - c: Target sensor class id (B,)
          - struct_map: Pre-registered structural/identity map (B, struct_channels, H, W)
        Output:
          - velocity / noise estimate (B, out_channels, H, W)
        """
        emb = self.time_sensor_embed(t, c)
        n_control = self.control_levels * self.num_res_blocks

        # ControlNet branch — zero-convs are the only thing keeping this from
        # being pure noise at init, so it contributes ~0 until they train off zero.
        hc = self.control_stem(struct_map)
        control_outputs = []
        idx_c = 0
        for level in range(self.control_levels):
            for _ in range(self.num_res_blocks):
                hc = self.control_res[idx_c](hc, emb)
                hc = self.control_attn[idx_c](hc)
                control_outputs.append(self.control_zero_convs[idx_c](hc))
                idx_c += 1
            hc = self.control_downsamplers[level](hc)

        # Main UNet down path, with additive control injection on the first
        # n_control blocks (mirrors ControlNetTransformerBaseline.forward).
        h = self.stem(x)
        skips = []
        block_idx = 0
        for level in range(len(self.channels)):
            for _ in range(self.num_res_blocks):
                h = self.down_res[block_idx](h, emb)
                h = self.down_attn[block_idx](h)
                if block_idx < n_control:
                    h = h + control_outputs[block_idx]
                skips.append(h)
                block_idx += 1
            h = self.downsamplers[level](h)

        h = self.mid_res1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, emb)

        block_idx = 0
        for up_i, level in enumerate(reversed(range(len(self.channels)))):
            for _ in range(self.num_res_blocks):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.up_res[block_idx](h, emb)
                h = self.up_attn[block_idx](h)
                block_idx += 1
            h = self.upsamplers[up_i](h)

        h = self.out_conv(self.act(self.out_norm(h)))
        return h


if __name__ == "__main__":
    model = UNetControlNetDenoiser(
        base_channels=32, channel_mult=(1, 2), num_res_blocks=1,
        attn_resolutions=(), control_levels=1, latent_size=16,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNetControlNetDenoiser (tiny test config) params: {total_params:,}")

    x_in = torch.randn(2, 4, 16, 16)
    t_in = torch.rand(2)
    c_in = torch.tensor([0, 3])
    struct_in = torch.randn(2, 6, 16, 16)
    pred = model(x_in, t_in, c_in, struct_in)
    print("UNet+ControlNet output shape:", pred.shape)
    assert pred.shape == x_in.shape
