import torch
from math import pi, sqrt
from PE_registry import PE_REGISTRY, register_PE

from utils.Cl3_triton import Cl3_triton as Cl3
from utils.Cl3_triton import e12, e31


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _normalize_pos(pos, B, P, name, expected_M=2):
    """Reshape pos ([P,M] or [B,P,M]) to [B_or_1, 1, P, M] with checks."""
    if pos.dim() == 2:
        P_, M_ = pos.shape
        assert P_ == P, f"{name}: pos P={P_} != z P={P}."
        assert M_ == expected_M, f"{name}: pos last dim {M_} != {expected_M}."
        return pos.view(1, 1, P, expected_M)
    if pos.dim() == 3:
        B_, P_, M_ = pos.shape
        assert P_ == P, f"{name}: pos P={P_} != z P={P}."
        assert M_ == expected_M, f"{name}: pos last dim {M_} != {expected_M}."
        assert B_ == B or B_ == 1, (
            f"{name}: pos B={B_} != z B={B} (and not 1 for broadcast)."
        )
        return pos.view(B_, 1, P, expected_M)
    raise ValueError(f"{name}: pos must be [P,M] or [B,P,M], got {tuple(pos.shape)}.")


def _random_unit_axes(heads, d_size):
    v = torch.randn(heads, 1, d_size, 3)
    return v / v.norm(dim=-1, keepdim=True)


# =====================================================================
# CARE — full Cl(3) multivector (8-D) input, learned 3-vector axes
# =====================================================================

class CARE(torch.nn.Module):
    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=12, scale=1.0, random_init=True, uniform_freq=False):
        super().__init__()
        D = embedding_dim
        self.embedding_dim = D

        if pos_dim is None:
            assert positions is not None, "CARE: need `positions` or `pos_dim`."
            pos_dim = positions.size(-1)
        assert pos_dim == 2, f"CARE: pos_dim must be 2, got {pos_dim}."
        self.pos_dim = pos_dim
        self.H = heads
        self.scale = scale

        assert D % 8 == 0, "CARE: D must be divisible by 8"
        self._d_size = D // 8
        d = self._d_size

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        mag_x = torch.rand(heads, 1, d, 1)
        mag_y = torch.rand(heads, 1, d, 1)
        if random_init:
            ax_x = _random_unit_axes(heads, d)
            ax_y = _random_unit_axes(heads, d)
        else:
            ax_x = e12.get_bivector().view(1, 1, 1, 3)  # [0, 0, 1]
            ax_y = e31.get_bivector().view(1, 1, 1, 3)  # [1, 0, 0]

        self.theta_x = torch.nn.Parameter(mag_x * ax_x)
        self.theta_y = torch.nn.Parameter(mag_y * ax_y)

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, "CARE.forward: no `pos` given and none cached."
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.theta_x.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def apply_rope(self, z, pos):
        """
        z   : [B, H, P, D]   (D % 8 == 0; N must equal heads)
        pos : [P, 2] or [B, P, 2]
        """
        assert z.dim() == 4, f"CARE: expected z=[B,H,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        H, d = self.H, self._d_size
        assert N == H, f"CARE: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"CARE: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 8 == 0, "CARE: D must be divisible by 8"

        pos = _normalize_pos(pos, B, P, "CARE")  # [B_or_1, 1, P, 2]
        pos_x = pos[..., 0:1].unsqueeze(-1)      # [B_or_1, 1, P, 1, 1]
        pos_y = pos[..., 1:2].unsqueeze(-1)

        # Per-token bivector fields: [B_or_1, H, P, d, 3]
        Bx = pos_x * self.theta_x.view(1, H, 1, d, 3) / 2
        By = pos_y * self.theta_y.view(1, H, 1, d, 3) / 2

        R_x = Cl3.embed_pure_quaternion(Bx).exp()
        R_y = Cl3.embed_pure_quaternion(By).exp()

        z = Cl3(z.view(B, H, P, d, 8))
        z = (R_y * R_x) @ z
        return z.to_tensor().view(B, H, P, D)


# =====================================================================
# 3-vector Cl(3) variants — share a base
# =====================================================================

class _VectorCARE(torch.nn.Module):
    """Base for the (D % 3 == 0) 3-vector Cl(3) variants.

    Subclasses build the per-token bivector fields via `_bivector_fields`.
    """

    def __init__(self, embedding_dim, positions=None, pos_dim=2, heads=6):
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

        assert D % 3 == 0, f"{type(self).__name__}: D must be divisible by 3"
        self._d_size = D // 3

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, (
            f"{type(self).__name__}.forward: no `pos` given and none cached."
        )
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = next(self.parameters()).device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def _bivector_fields(self, pos_x, pos_y):
        """Return (Bx_field, By_field), shape [B_or_1, H, P, d, 3], divided by 2."""
        raise NotImplementedError

    def apply_rope(self, z, pos):
        name = type(self).__name__
        assert z.dim() == 4, f"{name}: expected z=[B,H,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        H, d = self.H, self._d_size
        assert N == H, f"{name}: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"{name}: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 3 == 0, f"{name}: D must be divisible by 3"

        pos = _normalize_pos(pos, B, P, name)
        pos_x = pos[..., 0:1].unsqueeze(-1)  # [B_or_1, 1, P, 1, 1]
        pos_y = pos[..., 1:2].unsqueeze(-1)

        Bx, By = self._bivector_fields(pos_x, pos_y)
        R_x = Cl3.embed_pure_quaternion(Bx).exp()
        R_y = Cl3.embed_pure_quaternion(By).exp()

        z = Cl3.embed_vector(z.view(B, H, P, d, 3))
        z = (R_x * R_y) @ z
        return z.get_vector().view(B, H, P, D)


class Spherical_CARE(_VectorCARE):
    """Fixed bivector axes e12 (xy-plane) and e31 (zx-plane) — orthogonal."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=6, uniform_freq=False):
        super().__init__(embedding_dim, positions, pos_dim, heads)
        d = self._d_size
        self.theta_x = torch.nn.Parameter(torch.rand(heads, 1, d, 1))
        self.theta_y = torch.nn.Parameter(torch.rand(heads, 1, d, 1))
        self.register_buffer('x', e12.get_bivector().view(1, 1, 1, 3))
        self.register_buffer('y', e31.get_bivector().view(1, 1, 1, 3))

    def _bivector_fields(self, pos_x, pos_y):
        H, d = self.H, self._d_size
        ax_x = self.x.view(1, 1, 1, 1, 3)
        ax_y = self.y.view(1, 1, 1, 1, 3)
        Bx = pos_x * self.theta_x.view(1, H, 1, d, 1) * ax_x / 2
        By = pos_y * self.theta_y.view(1, H, 1, d, 1) * ax_y / 2
        return Bx, By


class Mixed_CARE(_VectorCARE):
    """Both bivector axes fixed to e12 (same plane)."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=6, uniform_freq=False):
        super().__init__(embedding_dim, positions, pos_dim, heads)
        d = self._d_size
        self.theta_x = torch.nn.Parameter(torch.rand(heads, 1, d, 1))
        self.theta_y = torch.nn.Parameter(torch.rand(heads, 1, d, 1))
        self.register_buffer('x', e12.get_bivector().view(1, 1, 1, 3))
        self.register_buffer('y', e12.get_bivector().view(1, 1, 1, 3))

    def _bivector_fields(self, pos_x, pos_y):
        H, d = self.H, self._d_size
        ax_x = self.x.view(1, 1, 1, 1, 3)
        ax_y = self.y.view(1, 1, 1, 1, 3)
        Bx = pos_x * self.theta_x.view(1, H, 1, d, 1) * ax_x / 2
        By = pos_y * self.theta_y.view(1, H, 1, d, 1) * ax_y / 2
        return Bx, By


class Quaternion_CARE(_VectorCARE):
    """Joint magnitude*axis 3-vector parameters (initial axes e12 and e31)."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=12, uniform_freq=False):
        super().__init__(embedding_dim, positions, pos_dim, heads)
        d = self._d_size
        mag_x = torch.rand(heads, 1, d, 1)
        mag_y = torch.rand(heads, 1, d, 1)
        ax_x = e12.get_bivector().view(1, 1, 1, 3)
        ax_y = e31.get_bivector().view(1, 1, 1, 3)
        self.theta_x = torch.nn.Parameter(mag_x.to(ax_x.device) * ax_x)
        self.theta_y = torch.nn.Parameter(mag_y.to(ax_y.device) * ax_y)

    def _bivector_fields(self, pos_x, pos_y):
        H, d = self.H, self._d_size
        Bx = pos_x * self.theta_x.view(1, H, 1, d, 3) / 2
        By = pos_y * self.theta_y.view(1, H, 1, d, 3) / 2
        return Bx, By


# =====================================================================
# Registration
# =====================================================================

for _name, _cls in [
    ("CARE", CARE),
    ("Spherical CARE", Spherical_CARE),
    ("Mixed CARE", Mixed_CARE),
    ("Quaternion CARE", Quaternion_CARE),
]:
    register_PE(_name, _cls)