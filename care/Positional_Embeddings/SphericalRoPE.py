from ..PE_registry import register_PE, PE_REGISTRY
import torch
from torch import pi

class Spherical_RoPE(torch.nn.Module):
    """Spherical Rotary Position Embedding (fixed geometric frequencies).

    Treats the embedding as D/3 stacked 3-vectors and rotates each one in
    SO(3) via composed yaw + roll rotations whose angles are
    `pos[axis] * theta_d`, with `theta_d = 100^(-2d/D)`.

    Requires `pos_dim == 2` (yaw, roll) and `D % 3 == 0`.
    """

    def __init__(self, embedding_dim, positions=None, pos_dim=2, heads=None):
        super().__init__()

        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, "Spherical_RoPE: need `positions` or `pos_dim`."
            pos_dim = positions.size(-1)
        assert pos_dim == 2, f"Spherical_RoPE: pos_dim must be 2 (yaw, roll), got {pos_dim}."
        self.pos_dim = pos_dim

        assert D % 3 == 0, f"Spherical_RoPE: D={D} not divisible by 3."

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        d_triple = D // 3
        d = torch.arange(d_triple, dtype=torch.float32)
        theta_d = 100.0 ** (-2.0 * d / D)  # [d_triple]
        self.register_buffer('theta_d', theta_d)

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, (
            "Spherical_RoPE.forward: no `pos` given and none cached."
        )
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
        z   : [B, N, P, D]   (D % 3 == 0)
        pos : [P, 2] or [B, P, 2]   (axis 0 -> yaw, axis 1 -> roll)
        returns: [B, N, P, D]
        """
        assert z.dim() == 4, (
            f"Spherical_RoPE: expected z=[B,N,P,D], got {tuple(z.shape)}."
        )
        B, N, P, D = z.shape
        assert D == self.embedding_dim, (
            f"Spherical_RoPE: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 3 == 0, f"Spherical_RoPE: D={D} not divisible by 3."
        d_triple = D // 3

        if pos.dim() == 2:
            P_, M_ = pos.shape
            assert P_ == P, f"Spherical_RoPE: pos P={P_} != z P={P}."
            assert M_ == 2, f"Spherical_RoPE: pos last dim {M_} != 2."
            pos = pos.view(1, 1, P, 2)
        elif pos.dim() == 3:
            B_, P_, M_ = pos.shape
            assert P_ == P, f"Spherical_RoPE: pos P={P_} != z P={P}."
            assert M_ == 2, f"Spherical_RoPE: pos last dim {M_} != 2."
            assert B_ == B or B_ == 1, (
                f"Spherical_RoPE: pos B={B_} != z B={B} (and not 1)."
            )
            pos = pos.view(B_, 1, P, 2)
        else:
            raise ValueError(
                f"Spherical_RoPE: pos must be [P,2] or [B,P,2], "
                f"got {tuple(pos.shape)}."
            )

        # Per-axis positions: [B_or_1, 1, P, 1]
        pos_x = pos[..., 0:1]
        pos_y = pos[..., 1:2]

        # angles: [B_or_1, 1, P, d_triple]
        theta_x = pos_x * self.theta_d.view(1, 1, 1, d_triple)
        theta_y = pos_y * self.theta_d.view(1, 1, 1, d_triple)

        cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
        cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)

        # [B, N, P, D] -> [B, N, P, d_triple, 3]; treat last dim as 3-vectors.
        z = z.view(B, N, P, d_triple, 3)
        vx, vy, vw = z.unbind(dim=-1)  # each [B, N, P, d_triple]

        # Yaw around y-axis (affects x and w)
        vx_yaw = vx * cos_y - vw * sin_y
        vw_yaw = vx * sin_y + vw * cos_y

        # Roll around x-axis (affects y and w_yaw)
        vy_roll = vy * cos_x - vw_yaw * sin_x
        vw_roll = vy * sin_x + vw_yaw * cos_x

        z_rot = torch.stack([vx_yaw, vy_roll, vw_roll], dim=-1)
        return z_rot.reshape(B, N, P, D)
  
register_PE("Spherical RoPE", Spherical_RoPE)
