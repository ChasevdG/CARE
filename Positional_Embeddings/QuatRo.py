import torch
from math import pi, sqrt
from PE_registry import PE_REGISTRY, register_PE
from utils.Cl3_triton import e12, e31
from utils.Cl3_triton import Cl3_triton as Cl3

def quaternion_multiply(q, r):
    """q, r : (..., 4) [w, x, y, z]  ->  (..., 4)"""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def exp_pure_quaternion(q):
    """Exp of a pure quaternion. Accepts (..., 3) or (..., 4) (w is ignored)."""
    v = q[..., 1:] if q.shape[-1] == 4 else q
    if v.shape[-1] != 3:
        raise ValueError("q must have shape (..., 3) or (..., 4)")
    theta = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8
    w = torch.cos(theta)
    xyz = v * (torch.sin(theta) / theta)
    return torch.cat([w, xyz], dim=-1)


def rotor_apply(r, x, normalize=False):
    """Rotate (...,3) vectors `x` by unit quaternion `r` (...,4)."""
    if r.shape[-1] != 4 or x.shape[-1] != 3:
        raise ValueError("r must be (...,4) and x must be (...,3)")
    w = r[..., :1]
    v = r[..., 1:]
    t = 2.0 * torch.cross(v, x, dim=-1)
    return x + w * t + torch.cross(v, t, dim=-1)


# Cl(3) bivector axes mapped to 3-vectors:
#   e12 generates rotation in the xy-plane (about z) -> [0, 0, 1]
#   e31 generates rotation in the zx-plane (about y) -> [0, 1, 0]
E12 = torch.tensor([0.0, 0.0, 1.0])
E31 = torch.tensor([0.0, 1.0, 0.0])

def _normalize_pos(pos, P, expected_M, name):
    """Reshape `pos` ([P,M] or [B,P,M]) to [B_or_1, 1, P, M] with shape checks."""
    if pos.dim() == 2:
        P_, M_ = pos.shape
        assert P_ == P, f"{name}: pos P={P_} != z P={P}."
        assert M_ == expected_M, f"{name}: pos last dim {M_} != {expected_M}."
        return pos.view(1, 1, P, expected_M), 1
    elif pos.dim() == 3:
        B_, P_, M_ = pos.shape
        assert P_ == P, f"{name}: pos P={P_} != z P={P}."
        assert M_ == expected_M, f"{name}: pos last dim {M_} != {expected_M}."
        return pos.view(B_, 1, P, expected_M), B_
    raise ValueError(f"{name}: pos must be [P,M] or [B,P,M], got {tuple(pos.shape)}.")


# ============================================================
# QuatRo — per-head learned 3-vector "axes" (magnitude * direction)
# ============================================================

class QuatRo(torch.nn.Module):
    """Quaternion RoPE.

    Per-head learned 3-vector "rotation axes" (magnitude and direction both
    learned, initialized as random_magnitude * fixed_bivector_axis).
    Requires `pos_dim == 2` and `D % 3 == 0`.
    """

    def __init__(self, embedding_dim, positions=None, pos_dim=2, heads=12):
        super().__init__()
        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, "QuatRo: need `positions` or `pos_dim`."
            pos_dim = positions.size(-1)
        assert pos_dim == 2, f"QuatRo: pos_dim must be 2, got {pos_dim}."
        self.pos_dim = pos_dim
        self.H = heads
        assert D % 3 == 0, f"QuatRo: D={D} not divisible by 3."

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        d_size = D // 3
        mag_x = torch.rand(heads, 1, d_size, 1)
        mag_y = torch.rand(heads, 1, d_size, 1)
        # Initialize directions to principal bivector axes.
        self.theta_x = torch.nn.Parameter(mag_x * E12.view(1, 1, 1, 3))  # [H, 1, d, 3]
        self.theta_y = torch.nn.Parameter(mag_y * E31.view(1, 1, 1, 3))

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, "QuatRo.forward: no `pos` given and none cached."
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.theta_x.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def get_frequencies(self):
        return self.theta_x, self.theta_y

    def apply_rope(self, z, pos):
        """
        z   : [B, H, P, D]   (D % 3 == 0; N must equal heads)
        pos : [P, 2] or [B, P, 2]
        returns: [B, H, P, D]
        """
        assert z.dim() == 4, f"QuatRo: expected z=[B,H,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        H = self.H
        assert N == H, f"QuatRo: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"QuatRo: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 3 == 0, f"QuatRo: D={D} not divisible by 3."
        d_size = D // 3

        pos, _ = _normalize_pos(pos, P, 2, "QuatRo")  # [B_or_1, 1, P, 2]
        pos_x = pos[..., 0:1].unsqueeze(-1)  # [B_or_1, 1, P, 1, 1]
        pos_y = pos[..., 1:2].unsqueeze(-1)

        # rotation 3-vectors: [B_or_1, H, P, d, 3]
        rot_x = pos_x * self.theta_x.view(1, H, 1, d_size, 3)
        rot_y = pos_y * self.theta_y.view(1, H, 1, d_size, 3)

        R_x = exp_pure_quaternion(rot_x / 2.0)  # [..., 4]
        R_y = exp_pure_quaternion(rot_y / 2.0)
        R = quaternion_multiply(R_x, R_y)       # [B_or_1, H, P, d, 4]

        z = z.view(B, H, P, d_size, 3)
        z_rot = rotor_apply(R, z)
        return z_rot.reshape(B, H, P, D)


# ============================================================
# Generic "fixed-axis" QuatRo — magnitude learned, axis is a buffer
# ============================================================

class _FixedAxisQuatRo(torch.nn.Module):
    """Shared implementation for Mixed_QuatRo and Spherical_QuatRo.

    Per-head learned scalar magnitudes; bivector directions are fixed buffers.
    Subclasses pick the (axis_x, axis_y) 3-vectors.
    """

    axis_x: torch.Tensor  # set by subclass
    axis_y: torch.Tensor

    def __init__(self, embedding_dim, positions=None, pos_dim=2, heads=6,
                 axis_x=E12, axis_y=E12):
        super().__init__()
        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, (
                f"{type(self).__name__}: need `positions` or `pos_dim`."
            )
            pos_dim = positions.size(-1)
        assert pos_dim == 2, (
            f"{type(self).__name__}: pos_dim must be 2, got {pos_dim}."
        )
        self.pos_dim = pos_dim
        self.H = heads
        assert D % 3 == 0, f"{type(self).__name__}: D={D} not divisible by 3."

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        d_size = D // 3
        # Per-head learned magnitudes (scalar per pair).
        self.theta_x = torch.nn.Parameter(torch.rand(heads, 1, d_size, 1))  # [H,1,d,1]
        self.theta_y = torch.nn.Parameter(torch.rand(heads, 1, d_size, 1))

        self.register_buffer('x', axis_x.view(1, 1, 1, 3).clone())
        self.register_buffer('y', axis_y.view(1, 1, 1, 3).clone())

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, (
            f"{type(self).__name__}.forward: no `pos` given and none cached."
        )
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.theta_x.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def get_frequencies(self):
        return self.theta_x, self.theta_y

    def apply_rope(self, z, pos):
        name = type(self).__name__
        assert z.dim() == 4, f"{name}: expected z=[B,H,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        H = self.H
        assert N == H, f"{name}: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"{name}: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 3 == 0, f"{name}: D={D} not divisible by 3."
        d_size = D // 3

        pos, _ = _normalize_pos(pos, P, 2, name)
        pos_x = pos[..., 0:1].unsqueeze(-1)  # [B_or_1, 1, P, 1, 1]
        pos_y = pos[..., 1:2].unsqueeze(-1)

        # rotation 3-vectors: pos * magnitude * axis  ->  [B_or_1, H, P, d, 3]
        mag_x = self.theta_x.view(1, H, 1, d_size, 1)
        mag_y = self.theta_y.view(1, H, 1, d_size, 1)
        ax_x = self.x.view(1, 1, 1, 1, 3)
        ax_y = self.y.view(1, 1, 1, 1, 3)
        rot_x = pos_x * mag_x * ax_x
        rot_y = pos_y * mag_y * ax_y

        R_x = exp_pure_quaternion(rot_x / 2.0)
        R_y = exp_pure_quaternion(rot_y / 2.0)
        R = quaternion_multiply(R_x, R_y)

        z = z.view(B, H, P, d_size, 3)
        z_rot = rotor_apply(R, z)
        return z_rot.reshape(B, H, P, D)


class Mixed_QuatRo(_FixedAxisQuatRo):
    """Mixed Quaternion RoPE — both axes initialized to e12 (xy-plane bivector)."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2, heads=6):
        super().__init__(embedding_dim, positions, pos_dim, heads,
                         axis_x=E12, axis_y=E12)


class Spherical_QuatRo(_FixedAxisQuatRo):
    """Spherical Quaternion RoPE — orthogonal bivector axes (e12 for x, e31 for y)."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2, heads=6):
        super().__init__(embedding_dim, positions, pos_dim, heads,
                         axis_x=E12, axis_y=E31)

register_PE("QuatRo", QuatRo)

register_PE("Spherical QuatRo", Spherical_QuatRo,)

register_PE("Mixed QuatRo", Mixed_QuatRo)