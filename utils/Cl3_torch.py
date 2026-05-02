import torch

class Cl3:
    """
    Fully dense, batch-safe, vectorized Cl(3,0) geometric algebra.
    Multivectors are represented as tensors of shape (..., 8)
    """

    blade_names = ["1", "e1", "e2", "e12", "e3", "e13", "e23", "e123"]
    gp_signs = None
    gp_indices = None
    

    @staticmethod
    def _init_gp_tensor():
        """Precompute geometric product tables (8×8) without Python loops.

        This version computes both the target blade indices and the signs
        using pure tensor ops and bitwise arithmetic:
          - index(i,j) = i XOR j
          - sign(i,j)  = (-1)^{\sum_k [bit_k(i)] * popcount(bits < k of j)}
        """
        # i, j in [0..7] encode the basis blade bitmasks
        i = torch.arange(8, dtype=torch.long)            # (8,)
        j = torch.arange(8, dtype=torch.long)            # (8,)
        I = i[:, None].expand(8, 8)                      # (8,8) rows i
        J = j[None, :].expand(8, 8)                      # (8,8) cols j

        # Geometric-product target blade is just XOR of bitmasks
        indices = (I ^ J).clone()                        # (8,8) in [0..7]

        # Vectorized computation of the sign exponent (parity of swaps)
        # masks for the bit position k and for the strictly-lower bits of k
        masks       = torch.tensor([1, 2, 4], dtype=torch.long)   # bit k
        lower_masks = torch.tensor([0, 1, 3], dtype=torch.long)   # bits < k
        tbl_pop3    = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3], dtype=torch.long)

        # Broadcast to shape (3, 8, 8)
        I3 = I.unsqueeze(0)
        J3 = J.unsqueeze(0)
        M  = masks[:, None, None]
        LM = lower_masks[:, None, None]

        # For each k, contribute popcount(j & lower_mask[k]) iff bit k is set in i
        contrib = ((I3 & M) != 0).long() * tbl_pop3[(J3 & LM)]    # (3,8,8)
        swaps = contrib.sum(dim=0)                                # (8,8)

        # sign = (-1)^{swaps}  ⇒  1 - 2*(swaps % 2)
        signs = (1 - 2 * (swaps & 1)).to(torch.int8)              # (8,8)

        Cl3.gp_signs = signs
        Cl3.gp_indices = indices

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 8, "Tensor must have last dimension 8 (Cl(3,0) multivector)"
        self.data = data.data if isinstance(data, Cl3) else data
        self.shape = self.data.shape
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.device = self.data.device

    def __add__(self, other):
        return Cl3(self.data + other.data)

    def __sub__(self, other):
        return Cl3(self.data - other.data)

    def __mul__(self, other):
        # print("Using Torch GP")
        return self.__torch_gp__(other)
    
    def __triton_mul__(self, other):
        """Geometric product using triton kernel."""
        if isinstance(other, Cl3):
            try:
                return Cl3_GP.apply(self, other)
            except Exception as e:
                print("Error in Cl3 triton GP:", e)
                print("Falling back to pure PyTorch implementation.")
                # Cl3.__mul__ = Cl3.__torch_gp__
                return self.__torch_gp__(other)
        if isinstance(other, (float, int)):
            return Cl3(self.data * other)
        if isinstance(other, torch.Tensor):
            return Cl3(self.data * other)
        return NotImplemented
    
    def __torch_gp__(self, other):
        if isinstance(other, Cl3):
            A = self.data
            B = other.data
            A_b, B_b = torch.broadcast_tensors(A, B)

            # Move the (8×8) tables to the right device/dtypes just-in-time
            signs   = Cl3.gp_signs.to(A_b.device, A_b.dtype)
            indices = Cl3.gp_indices.to(A_b.device)  # long

            # Outer product and signed accumulation
            AB = torch.einsum("...i,...j->...ij", A_b, B_b)
            signed = AB * signs

            flat = signed.reshape(*signed.shape[:-2], 64)
            flat_idx = indices.reshape(64).expand_as(flat)

            out = torch.zeros(*flat.shape[:-1], 8, dtype=A_b.dtype, device=A_b.device)
            out = out.scatter_add(-1, flat_idx, flat)
            return Cl3(out)

        if isinstance(other, (float, int)):
            return Cl3(self.data * other)
        if isinstance(other, torch.Tensor):
            return Cl3(self.data * other)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        """Sandwich product: A @ B = A * B * A⁻¹"""
        if isinstance(other, torch.Tensor):
            other = Cl3.from_tensor(other)
        return self * other * self.inv()

    def inv(self):
        # Inverse: flip signs of grades 2 and 3
        signs = torch.tensor([1, 1, 1, -1, 1, -1, -1, -1],
                             dtype=self.data.dtype, device=self.data.device)
        norm = (self.data**2).sum(dim=-1, keepdim=True)
        norm = torch.where(norm == 0.0, 1.0, norm)
        return Cl3(self.data * signs/norm)

    def to_tensor(self):
        return self.data

    @staticmethod
    def from_tensor(t: torch.Tensor):
        return Cl3(t)

    def clone(self):
        return Cl3(self.data.clone())

    def view(self, *args):
        """
        View the data tensor with new shape.
        Args:
            *args: New shape for the data tensor.
        Returns:
            Cl3 instance with reshaped data.
        """
        return Cl3(self.data.view(*args))

    def __repr__(self):
        if self.data.numel() == 8:
            parts = []
            for i, name in enumerate(self.blade_names):
                c = self.data[..., i].item()
                if abs(c) > 1e-6:
                    parts.append(f"{c:+.3f}{name}")
            return " + ".join(parts) if parts else "0"
        return f"Cl3(..., 8) shape = {self.data.shape}"

    @staticmethod
    def basis_vector(index, batch_shape=()):
        """
        Create batched basis vector: e1, e2, or e3.
        index: 1 → e1, 2 → e2, 3 → e3
        """
        idx_map = {1: 1, 2: 2, 3: 4}
        blade_idx = idx_map[index]
        data = torch.zeros(*batch_shape, 8, dtype=torch.float32)
        data[..., blade_idx] = 1.0
        return Cl3(data)

    @staticmethod
    def grade(blade):
        return bin(blade).count("1")

    def exp(self):
      """
      Fast exponential of a bivector multivector:
      exp(B) = cos(|B|) + sin(|B|)/|B| * B
      Assumes B is a pure bivector (blades 3, 5, 6).
      """
      B = self.data  # (..., 8)

      # Only consider bivector components: e12 (3), e31 (5), e23 (6)
      b3 = B[..., 3]
      b5 = B[..., 5]
      b6 = B[..., 6]

      # B^2 = -(|b3|^2 + |b5|^2 + |b6|^2)
      b_squared = b3**2 + b5**2 + b6**2  # (...,)
      b_norm = torch.sqrt(torch.clamp(b_squared, min=1e-12))  # (...,)

      # sin(x)/x with Taylor expansion fallback
      sin_by_norm = torch.where(
          b_norm > 1e-4,
          torch.sin(b_norm) / b_norm,
          1 - b_squared / 6 + b_squared**2 / 120
      )

      cos_part = torch.cos(b_norm)

      # Build result: scalar + bivector
      result = torch.zeros_like(B)
      result[..., 0] = cos_part             # scalar part
      result[..., 3] = b3 * sin_by_norm     # e12
      result[..., 5] = b5 * sin_by_norm     # e31
      result[..., 6] = b6 * sin_by_norm     # e23

      return Cl3(result)

    def repeat(self, *sizes):
      return Cl3(self.data.repeat(*sizes))

    def get_vector(self):
        return self.data[..., (1, 2, 4)]
    
    def get_bivector(self):
        return self.data[..., (3, 5, 6)]
    
    
    def rotate_vector(self, v):
      """
      Efficient sandwich product v' = a v a^{-1} assuming:
        - self = a is a rotor (unit scalar + bivector only)
        - v is a pure vector (only e1,e2,e3 components)
      Works with arbitrary batch shapes and dtypes/devices.
      Returns a Cl3 with only vector components populated.
      """
      A = self.data if isinstance(self, Cl3) else self
      V = v.data if isinstance(v, Cl3) else v
      assert A.shape[-1] == 8 and V.shape[-1] == 8, "Expect (..., 8) multivectors"

      # Rotor parts
      s = A[..., 0]                                # scalar
      r = torch.stack([A[..., 6], A[..., 5], A[..., 3]], dim=-1)  # (b23, b31, b12)

      # Vector parts
      vec = torch.stack([V[..., 1], V[..., 2], V[..., 4]], dim=-1)  # (v1, v2, v3)

      # Broadcast
      r, vec = torch.broadcast_tensors(r, vec)
      s = s.unsqueeze(-1).expand(r.shape[:-1] + (1,))

      # Quaternion/rotor rotation: v' = v + s*(2 r×v) + r×(2 r×v)
      t = 2.0 * torch.cross(r, vec, dim=-1)
      vec_rot = vec + s * t + torch.cross(r, t, dim=-1)

      # Pack back into Cl3 vector multivector
      out = torch.zeros(*vec_rot.shape[:-1], 8, dtype=A.dtype, device=A.device)
      out[..., 1] = vec_rot[..., 0]
      out[..., 2] = vec_rot[..., 1]
      out[..., 4] = vec_rot[..., 2]
      return Cl3(out)

    @staticmethod
    def embed_vector(vec: torch.Tensor):
        """
        Embed a [..., 3] tensor as a Cl3 vector multivector.
        vec[..., 0] → e1
        vec[..., 1] → e2
        vec[..., 2] → e3
        """
        assert vec.shape[-1] == 3, "Input must have last dimension = 3"
        out = torch.zeros(*vec.shape[:-1], 8, dtype=vec.dtype, device=vec.device)
        out[..., 1] = vec[..., 0]  # e1
        out[..., 2] = vec[..., 1]  # e2
        out[..., 4] = vec[..., 2]  # e3
        return Cl3(out)

    @staticmethod
    def embed_quaternion(quat: torch.Tensor):
        """
        Embed a [..., 4] tensor as a Cl3 rotor multivector.
        quat[..., 0] → scalar
        quat[..., 1] → e12
        quat[..., 2] → e23
        quat[..., 3] → e31
        """
        assert quat.shape[-1] == 4, "Input must have last dimension = 4"
        out = torch.zeros(*quat.shape[:-1], 8, dtype=quat.dtype, device=quat.device)
        out[..., 0] = quat[..., 0]
        out[..., 3] = quat[..., 1]  # e12
        out[..., 5] = quat[..., 2]  # e23
        out[..., 6] = quat[..., 3]  # e31
        return Cl3(out)

    @staticmethod
    def embed_pure_quaternion(quat: torch.Tensor):
        """
        Embed a [..., 3] tensor as a Cl3 rotor multivector.
        0 → scalar
        quat[..., 0] → e12
        quat[..., 1] → e23
        quat[..., 2] → e31
        """
        assert quat.shape[-1] == 3, "Input must have last dimension = 4"
        out = torch.zeros(*quat.shape[:-1], 8, dtype=quat.dtype, device=quat.device)
        out[..., 3] = quat[..., 0]  # e12
        out[..., 5] = quat[..., 1]  # e23
        out[..., 6] = quat[..., 2]  # e31
        return Cl3(out)

    def to(self, device):
        return Cl3(self.data.to(device))

    def view(self, *args):
        """
        View the data tensor with new shape.
        Args:
            *args: New shape for the data tensor.
        Returns:
            Cl3 instance with reshaped data.
        """
        d = self.data.view(*args)
        assert d.shape[-1] == 8, "Last dimension must be 8 for Cl3 multivector"
        return Cl3(d)


Cl3._init_gp_tensor()

# Basis vectors
e1 = Cl3.basis_vector(1)
e2 = Cl3.basis_vector(2)
e3 = Cl3.basis_vector(3)

# Bivectors
e12 = e1 * e2
e31 = e3 * e1
e23 = e2 * e3