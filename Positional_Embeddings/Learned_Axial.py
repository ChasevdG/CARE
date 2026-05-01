import torch
import torch.nn as nn
from PE_registry import register_PE, PE_REGISTRY
from math import pi
import math

class Learned_Axial_RoPE(nn.Module):
    """Learned Axial Rotary Position Embedding for arbitrary-dim positions.

    Per-head learned frequencies, one per (axis, channel-pair).
    """

    def __init__(self, embedding_dim, positions=None, pos_dim=None,
                 heads=6, uniform_freq=False):
        super().__init__()

        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, (
                "Learned_Axial_RoPE: need `positions` or `pos_dim`."
            )
            pos_dim = positions.size(-1)
        self.pos_dim = pos_dim
        M = pos_dim
        self.H = heads
        self.uniform_freq = uniform_freq

        assert D % (2 * M) == 0, (
            f"Learned_Axial_RoPE: embedding_dim={D} not divisible by 2*pos_dim={2 * M}."
        )

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        d_pair = D // (2 * M)

        # Per-head learned frequencies: [H, M, d_pair]
        if uniform_freq:
            init = torch.full((heads, M, d_pair), 1.0 / math.pi)
        else:
            init = torch.rand(heads, M, d_pair)
        self.freq = nn.Parameter(init, requires_grad=True)

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, "Learned_Axial_RoPE.forward: no `pos` given and none cached."

        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.freq.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def apply_rope(self, z, pos):
        """
        z   : [B, H, P, D]   (the N dim must equal `heads`)
        pos : [P, M] or [B, P, M]
        returns: [B, H, P, D]
        """

        assert z.dim() == 4, f"Learned_Axial_RoPE: expected z=[B,H,P,D], got {tuple(z.shape)}."

        B, N, P, D = z.shape
        H, M = self.H, self.pos_dim

        assert N == H, f"Learned_Axial_RoPE: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"Learned_Axial_RoPE: z D={D} != embedding_dim={self.embedding_dim}."
        )
        d_pair = D // (2 * M)

        if pos.dim() == 2:
            P_, M_ = pos.shape
            assert P_ == P, f"Learned_Axial_RoPE: pos P={P_} != z P={P}."
            assert M_ == M, f"Learned_Axial_RoPE: pos last dim {M_} != pos_dim={M}."
            pos = pos.view(1, 1, P, M, 1)
        elif pos.dim() == 3:
            B_, P_, M_ = pos.shape
            assert P_ == P, f"Learned_Axial_RoPE: pos P={P_} != z P={P}."
            assert M_ == M, f"Learned_Axial_RoPE: pos last dim {M_} != pos_dim={M}."
            assert B_ == B or B_ == 1, f"Learned_Axial_RoPE: pos B={B_} != z B={B} (and not 1)."
            pos = pos.view(B_, 1, P, M, 1)
        else:
            raise ValueError(
                f"Learned_Axial_RoPE: pos must be [P,M] or [B,P,M], "
                f"got {tuple(pos.shape)}."
            )

        # pos:  [B_or_1, 1, P, M, 1]
        # freq: [H, M, d_pair] -> view [1, H, 1, M, d_pair]
        # angles: [B_or_1, H, P, M, d_pair]
        angles = pos * self.freq.view(1, H, 1, M, d_pair)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # [B, H, P, D] -> [B, H, P, M, d_pair, 2]
        z = z.view(B, H, P, M, d_pair, 2)
        z_a = z[..., 0]
        z_b = z[..., 1]

        z_a_rot = z_a * cos - z_b * sin
        z_b_rot = z_a * sin + z_b * cos

        z_rot = torch.stack([z_a_rot, z_b_rot], dim=-1)
        return z_rot.reshape(B, H, P, D)
        
register_PE("Learned Axial RoPE", Learned_Axial_RoPE)