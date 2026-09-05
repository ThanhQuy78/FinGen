"""
Small unconditional flow-matching prior over the structural map itself
(the 6-channel S_aligned representation: segmentation, orientation unit
vector, minutiae score + direction unit vector — see
fingernet_extractor.py:364's `combined_structure`).

This is the missing piece flagged in the project discussion on "how can I
sample a new identity": UNetControlNetDenoiser (unet_controlnet.py) always
needs a *real* struct_map to condition on — it was never trained to generate
one. This model does the opposite: no conditioning at all, just learns
p(S_aligned) over the real dataset so it can sample a brand-new, not-in-
-dataset structural map from pure noise. Feed that sample into the existing
ControlNet pipeline (in place of a real cached S_aligned) to render it as a
full fingerprint image — the two models compose into a real "new identity"
generator; neither one is that on its own.

Architecture reuses the exact building blocks from unet_controlnet.py
(ResBlock, AttnBlock2D, Downsample, Upsample, zero_init) — same FiLM-
conditioned conv UNet design, just no ControlNet side-branch (nothing to
condition on) and no sensor embedding (structural identity isn't tied to a
target sensor — only the final rendering step is). Trained with the same
RectifiedFlowTrajectoryManager (flow_matching.py) used everywhere else in
this repo, since that class already only depends on the model call
signature, not on what the tensor represents.

forward(x, t, c=None, struct_map=None) — the two trailing args exist only so
this model's call signature matches what flow_matching.py's sample_euler
already dispatches to for a non-MM-DiT model (`model(x, t, c, struct_map)`
at flow_matching.py:107); this model is genuinely unconditional and simply
ignores them, letting the existing sampler be reused as-is instead of
writing a second one.
"""

import torch
import torch.nn as nn
from typing import Sequence

from src.models.unet_controlnet import ResBlock, AttnBlock2D, Downsample, Upsample, zero_init


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP. No sensor/class conditioning — unconditional prior."""

    def __init__(self, base_channels: int, time_embed_dim: int):
        super().__init__()
        self.base_channels = base_channels
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.base_channels // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(t_emb)


class StructurePriorUNet(nn.Module):
    """Unconditional flow-matching UNet over the 6-channel structural map."""

    def __init__(
        self,
        struct_channels: int = 6,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        num_heads: int = 8,
        map_size: int = 64,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.channels = [base_channels * m for m in channel_mult]
        attn_res = set(attn_resolutions)

        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedder(base_channels, time_embed_dim)

        self.stem = nn.Conv2d(struct_channels, base_channels, 3, padding=1)

        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = base_channels
        resolution = map_size
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

        mid_ch = self.channels[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, time_embed_dim)
        self.mid_attn = AttnBlock2D(mid_ch, num_heads)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, time_embed_dim)

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
        self.out_conv = zero_init(nn.Conv2d(in_ch, struct_channels, 3, padding=1))
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c=None, struct_map=None) -> torch.Tensor:
        """
        Input:
          - x: Noisy/interpolated structural map (B, struct_channels, H, W)
          - t: Flow-matching timestep in [0, 1] (B,)
          - c, struct_map: unused (see module docstring) — kept only so this
            model's call signature matches flow_matching.py's sampler.
        Output:
          - velocity estimate (B, struct_channels, H, W)
        """
        emb = self.time_embed(t)

        h = self.stem(x)
        skips = []
        block_idx = 0
        for level in range(len(self.channels)):
            for _ in range(self.num_res_blocks):
                h = self.down_res[block_idx](h, emb)
                h = self.down_attn[block_idx](h)
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

        return self.out_conv(self.act(self.out_norm(h)))


if __name__ == "__main__":
    model = StructurePriorUNet(base_channels=32, channel_mult=(1, 2), num_res_blocks=1,
                                attn_resolutions=(), map_size=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"StructurePriorUNet (tiny test config) params: {total_params:,}")
    x_in = torch.randn(2, 6, 16, 16)
    t_in = torch.rand(2)
    pred = model(x_in, t_in)
    print("Output shape:", pred.shape)
    assert pred.shape == x_in.shape
