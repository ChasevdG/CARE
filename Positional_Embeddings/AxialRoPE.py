import torch
from math import pi, sqrt
import math
from PE_registry import register_PE, PE_REGISTRY

class Axial_RoPE(torch.nn.Module):
    """Axial Rotary Position Embedding for arbitrary-dimensional positions."""

    def __init__(self, embedding_dim, positions=None, pos_dim=None, uniform_freq=False, heads=None):
        super().__init__()

        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, "Axial_RoPE: need `positions` or `pos_dim`."
            pos_dim = positions.size(-1)
        self.pos_dim = pos_dim
        M = pos_dim

        assert D % (2 * M) == 0, (
            f"Axial_RoPE: embedding_dim={D} not divisible by 2*pos_dim={2 * M}."
        )

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        d_pair = D // (2 * M)
        d = torch.arange(d_pair, dtype=torch.float32)

        if uniform_freq:
            theta_d = torch.full((d_pair,), 1.0 / math.pi)
        else:
            theta_d = 100.0 ** (-2.0 * d / D)

        self.register_buffer('theta_d', theta_d)
        self.uniform_freq = uniform_freq

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, "Axial_RoPE.forward: no `pos` given and none cached."
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.theta_d.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def apply_rope(self, z, pos):
        """
        z   : [B, N, P, D]
        pos : [P, M] or [B, P, M]
        returns: [B, N, P, D]
        """
        assert z.dim() == 4, f"Axial_RoPE: expected z=[B,N,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        M = self.pos_dim
        assert D == self.embedding_dim, (
            f"Axial_RoPE: z dim D={D} != embedding_dim={self.embedding_dim}."
        )
        d_pair = D // (2 * M)

        if pos.dim() == 2:
            P_, M_ = pos.shape
            assert P_ == P, f"Axial_RoPE: pos P={P_} != z P={P}."
            assert M_ == M, f"Axial_RoPE: pos last dim {M_} != pos_dim={M}."
            pos = pos.view(1, 1, P, M, 1)
        elif pos.dim() == 3:
            B_, P_, M_ = pos.shape
            assert P_ == P, f"Axial_RoPE: pos P={P_} != z P={P}."
            assert M_ == M, f"Axial_RoPE: pos last dim {M_} != pos_dim={M}."
            assert B_ == B or B_ == 1, f"Axial_RoPE: pos B={B_} != z B={B} (and not 1)."
            pos = pos.view(B_, 1, P, M, 1)
        else:
            raise ValueError(
                f"Axial_RoPE: pos must be [P,M] or [B,P,M], got {tuple(pos.shape)}."
            )

        # angles: [B_or_1, 1, P, M, d_pair]
        angles = pos * self.theta_d.view(1, 1, 1, 1, -1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # [B, N, P, D] -> [B, N, P, M, d_pair, 2]
        z = z.view(B, N, P, M, d_pair, 2)
        z_a = z[..., 0]
        z_b = z[..., 1]

        z_a_rot = z_a * cos - z_b * sin
        z_b_rot = z_a * sin + z_b * cos

        z_rot = torch.stack([z_a_rot, z_b_rot], dim=-1)
        return z_rot.reshape(B, N, P, D)

class Uniform_Axial_RoPE(Axial_RoPE):
  def __init__(self, embedding_dim, P_x, P_y):
    super().__init__(embedding_dim, P_x, P_y, uniform_freq=True)

register_PE(
        "Axial RoPE", Axial_RoPE)
register_PE(
        "Uniform Axial RoPE", Uniform_Axial_RoPE
    )