import torch
from math import pi, sqrt
from ..PE_registry import PE_REGISTRY, register_PE

from ..utils.Cl3_triton import Cl3_triton as Cl3
from ..utils.Cl3_triton import e12, e31


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


def _random_unit_bivectors(M, heads, d_size):
    """M random unit 3-vectors, shape [M, H, 1, d, 3]."""
    v = torch.randn(M, heads, 1, d_size, 3)
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
        self.pos_dim = pos_dim
        M = pos_dim
        self.H = heads
        self.scale = scale

        assert D % 8 == 0, "CARE: D must be divisible by 8"
        self._d_size = D // 8
        d = self._d_size

        if positions is not None:
            self.register_buffer('p', positions)
        else:
            self.p = None

        if M == 2:
            # Two fully learned bivectors per head (original behaviour).
            mag = torch.rand(2, heads, 1, d, 1)
            if random_init:
                ax = _random_unit_bivectors(2, heads, d)           # [2, H, 1, d, 3]
            else:
                ax = torch.stack([
                    e12.get_bivector().view(1, 1, 1, 3),
                    e31.get_bivector().view(1, 1, 1, 3),
                ], dim=0)                                           # [2, 1, 1, 1, 3]
            self.thetas = torch.nn.Parameter(mag * ax)             # [2, H, 1, d, 3]
        else:
            # M fixed random unit axes — not learned.
            self.register_buffer('thetas', _random_unit_bivectors(M, heads, d))

    def forward(self, x, pos=None):
        if pos is None:
            pos = self.p
        assert pos is not None, "CARE.forward: no `pos` given and none cached."
        return self.apply_rope(x, pos)

    def set_positions(self, pos):
        device = self.thetas.device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def apply_rope(self, z, pos):
        """
        z   : [B, H, P, D]   (D % 8 == 0; N must equal heads)
        pos : [P, M] or [B, P, M]
        """
        assert z.dim() == 4, f"CARE: expected z=[B,H,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        H, d, M = self.H, self._d_size, self.pos_dim
        assert N == H, f"CARE: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"CARE: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 8 == 0, "CARE: D must be divisible by 8"

        pos = _normalize_pos(pos, B, P, "CARE", expected_M=M)  # [B_or_1, 1, P, M]

        # Compose rotors right-to-left: R_{M-1} * ... * R_0.
        # For M==2 this reproduces the original (R_y * R_x) @ z.
        R = None
        for m in range(M - 1, -1, -1):
            pos_m = pos[..., m:m+1].unsqueeze(-1)          # [B_or_1, 1, P, 1, 1]
            theta_m = self.thetas[m].view(1, H, 1, d, 3)
            Bm = pos_m * theta_m / 2                        # [B_or_1, H, P, d, 3]
            R_m = Cl3.embed_pure_quaternion(Bm).exp()
            R = R_m if R is None else R * R_m

        z = Cl3(z.view(B, H, P, d, 8))
        z = R @ z
        return z.to_tensor().view(B, H, P, D)


# =====================================================================
# 3-vector Cl(3) variants — share a base
# =====================================================================

class _VectorCARE(torch.nn.Module):
    """Base for the (D % 3 == 0) 3-vector Cl(3) variants.

    Subclasses implement `_get_bivector_m(m, pos_m, H, d)` which returns
    the bivector field (already divided by 2) for positional axis m.
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
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = next(self.buffers()).device
        pos = pos.to(device)
        if isinstance(getattr(self, 'p', None), torch.Tensor):
            self.p = pos
        else:
            self.register_buffer('p', pos)

    def _get_bivector_m(self, m, pos_m, H, d):
        """Return bivector field for axis m: shape [B_or_1, H, P, d, 3] / 2."""
        raise NotImplementedError

    def apply_rope(self, z, pos):
        name = type(self).__name__
        assert z.dim() == 4, f"{name}: expected z=[B,H,P,D], got {tuple(z.shape)}."
        B, N, P, D = z.shape
        H, d, M = self.H, self._d_size, self.pos_dim
        assert N == H, f"{name}: z heads N={N} != heads={H}."
        assert D == self.embedding_dim, (
            f"{name}: z D={D} != embedding_dim={self.embedding_dim}."
        )
        assert D % 3 == 0, f"{name}: D must be divisible by 3"

        pos = _normalize_pos(pos, B, P, name, expected_M=M)  # [B_or_1, 1, P, M]

        # Compose rotors left-to-right: R_0 * R_1 * ... * R_{M-1}.
        # For M==2 this reproduces the original (R_x * R_y) @ z.
        R = None
        for m in range(M):
            pos_m = pos[..., m:m+1].unsqueeze(-1)   # [B_or_1, 1, P, 1, 1]
            Bm = self._get_bivector_m(m, pos_m, H, d)
            R_m = Cl3.embed_pure_quaternion(Bm).exp()
            R = R_m if R is None else R * R_m

        z = Cl3.embed_vector(z.view(B, H, P, d, 3))
        z = R @ z
        return z.get_vector().view(B, H, P, D)


class Spherical_CARE(_VectorCARE):
    """Scalar magnitudes learned; axes fixed (e12/e31 for M==2, random unit for M>2)."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=6, uniform_freq=False):
        super().__init__(embedding_dim, positions, pos_dim, heads)
        d = self._d_size
        M = self.pos_dim
        H = self.H

        self.magnitudes = torch.nn.Parameter(torch.rand(M, H, 1, d, 1))

        if M == 2:
            ax = torch.stack([
                e12.get_bivector().view(1, 1, 1, 3),
                e31.get_bivector().view(1, 1, 1, 3),
            ], dim=0)                                 # [2, 1, 1, 1, 3]
        else:
            v = torch.randn(M, 1, 1, 1, 3)
            ax = v / v.norm(dim=-1, keepdim=True)       # [M, 1, 1, 1, 3]
        self.register_buffer('axes', ax.to(self.magnitudes))

    def _get_bivector_m(self, m, pos_m, H, d):
        mag_m = self.magnitudes[m].view(1, H, 1, d, 1)
        ax_m = self.axes[m].view(1, 1, 1, 1, 3)
        return pos_m * mag_m * ax_m / 2


class Mixed_CARE(_VectorCARE):
    """Scalar magnitudes learned; axes all e12 for M==2, random unit for M>2."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=6, uniform_freq=False):
        super().__init__(embedding_dim, positions, pos_dim, heads)
        d = self._d_size
        M = self.pos_dim
        H = self.H

        self.magnitudes = torch.nn.Parameter(torch.rand(M, H, 1, d, 1))

        if M == 2:
            e12_vec = e12.get_bivector().view(1, 1, 1, 3)
            ax = torch.stack([e12_vec, e12_vec], dim=0)  # [2, 1, 1, 1, 3]
        else:
            v = torch.randn(M, 1, 1, 1, 3)
            ax = v / v.norm(dim=-1, keepdim=True)
        self.register_buffer('axes', ax.to(self.magnitudes))

    def _get_bivector_m(self, m, pos_m, H, d):
        mag_m = self.magnitudes[m].view(1, H, 1, d, 1)
        ax_m = self.axes[m].view(1, 1, 1, 1, 3)
        return pos_m * mag_m * ax_m / 2


class Quaternion_CARE(_VectorCARE):
    """Full 3-vector params for M==2 (init to e12/e31); random unit fixed axes for M>2."""

    def __init__(self, embedding_dim, positions=None, pos_dim=2,
                 heads=12, uniform_freq=False):
        super().__init__(embedding_dim, positions, pos_dim, heads)
        d = self._d_size
        M = self.pos_dim
        H = self.H

        if M == 2:
            mag_x = torch.rand(H, 1, d, 1)
            mag_y = torch.rand(H, 1, d, 1)
            ax_x = e12.get_bivector().view(1, 1, 1, 3)
            ax_y = e31.get_bivector().view(1, 1, 1, 3)
            thetas = torch.stack([
                mag_x.to(ax_x.device) * ax_x,   # [H, 1, d, 3]
                mag_y.to(ax_y.device) * ax_y,
            ], dim=0)                             # [2, H, 1, d, 3]
            self.thetas = torch.nn.Parameter(thetas)
        else:
            v = torch.randn(M, H, 1, d, 3)
            self.register_buffer('thetas', v / v.norm(dim=-1, keepdim=True))

    def _get_bivector_m(self, m, pos_m, H, d):
        theta_m = self.thetas[m].view(1, H, 1, d, 3)
        return pos_m * theta_m / 2


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
