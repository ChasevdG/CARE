import torch
from math import pi, sqrt
from ..PE_registry import PE_REGISTRY, register_PE

import math
import torch
import torch.nn as nn


class Mixed_RoPE(nn.Module):
    """Mixed Rotary Position Embedding.

    Unlike axial RoPE (which assigns each channel-pair to a single axis),
    mixed RoPE rotates every channel-pair by an angle that mixes all M axes:
        theta[p, k] = sum_m  pos[p, m] * freq[m, k]
    Per-head learned frequencies; `D` must be divisible by 2.
    """

    def __init__(self, embedding_dim, positions=None, pos_dim=None, heads=12):
        super().__init__()

        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, (
                "Mixed_RoPE: need `positions` or `pos_dim`."
            )
            pos_dim = positions.size(-1)
        self.pos_dim = pos_dim
        M = pos_dim
        self.H = heads

        assert D % 2 == 0, f"Mixed_RoPE: D={D} not divisible by 2."

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        d_pair = D // 2
        # Per-head learned frequencies, one per (axis, channel-pair): [H, M, d_pair]
        self.freq = nn.Parameter(torch.rand(heads, M, d_pair))

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, "Mixed_RoPE.forward: no `pos` given and none cached."
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.freq.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def get_frequencies(self):
        return self.freq

    def apply_rope(self, z, pos):
        """
        z   : [B, H, P, D]   (D % 2 == 0; N must equal heads)
        pos : [P, M] or [B, P, M]
        returns: [B, H, P, D]
        """
        assert z.dim() == 4, (
            f"Mixed_RoPE: expected z=[B,H,P,D], got {tuple(z.shape)}."
        )
        B, N, P, D = z.shape
        H, M = self.H, self.pos_dim
        
        assert N == H, f"Mixed_RoPE: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, f"Mixed_RoPE: z D={D} != embedding_dim={self.embedding_dim}."
        assert D % 2 == 0, f"Mixed_RoPE: D={D} not divisible by 2."

        d_pair = D // 2

        if pos.dim() == 2:
            P_, M_ = pos.shape
            assert P_ == P, f"Mixed_RoPE: pos P={P_} != z P={P}."
            assert M_ == M, f"Mixed_RoPE: pos last dim {M_} != pos_dim={M}."
            pos = pos.view(1, 1, P, M, 1)
        elif pos.dim() == 3:
            B_, P_, M_ = pos.shape
            assert P_ == P, f"Mixed_RoPE: pos P={P_} != z P={P}."
            assert M_ == M, f"Mixed_RoPE: pos last dim {M_} != pos_dim={M}."
            assert B_ == B or B_ == 1, f"Mixed_RoPE: pos B={B_} != z B={B} (and not 1)."

            pos = pos.view(B_, 1, P, M, 1)
        else:
            raise ValueError(
                f"Mixed_RoPE: pos must be [P,M] or [B,P,M], got {tuple(pos.shape)}."
            )

        # pos:  [B_or_1, 1, P, M, 1]
        # freq: [H, M, d_pair] -> [1, H, 1, M, d_pair]
        # product:    [B_or_1, H, P, M, d_pair]
        # sum over M: [B_or_1, H, P, d_pair]
        angles = (pos * self.freq.view(1, H, 1, M, d_pair)).sum(dim=-2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # [B, H, P, D] -> [B, H, P, d_pair, 2]
        z = z.view(B, H, P, d_pair, 2)
        z_a = z[..., 0]
        z_b = z[..., 1]

        z_a_rot = z_a * cos - z_b * sin
        z_b_rot = z_a * sin + z_b * cos

        z_rot = torch.stack([z_a_rot, z_b_rot], dim=-1)
        return z_rot.reshape(B, H, P, D)
  
  
register_PE("Mixed RoPE", Mixed_RoPE)