import torch

import triton
import triton.language as tl


# Triton kernel for batched Cl(3) geometric product (hand-expanded)
# Order of basis elements: [1, e1, e2, e12, e3, e13, e23, e123]

@triton.jit
def cl3_gp_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_a0: tl.constexpr,
    stride_b0: tl.constexpr,
    stride_c0: tl.constexpr,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    inb = rows < N                                  # (BLOCK_M,)

    a_base = A_ptr + rows * stride_a0               # (BLOCK_M,)
    b_base = B_ptr + rows * stride_b0               # (BLOCK_M,)
    c_base = C_ptr + rows * stride_c0               # (BLOCK_M,)

    # Load A lanes (each is (BLOCK_M,))
    a0   = tl.load(a_base + 0, mask=inb, other=0.0).to(tl.float32)
    a1   = tl.load(a_base + 1, mask=inb, other=0.0).to(tl.float32)
    a2   = tl.load(a_base + 2, mask=inb, other=0.0).to(tl.float32)
    a12  = tl.load(a_base + 3, mask=inb, other=0.0).to(tl.float32)
    a3   = tl.load(a_base + 4, mask=inb, other=0.0).to(tl.float32)
    a13  = tl.load(a_base + 5, mask=inb, other=0.0).to(tl.float32)
    a23  = tl.load(a_base + 6, mask=inb, other=0.0).to(tl.float32)
    a123 = tl.load(a_base + 7, mask=inb, other=0.0).to(tl.float32)

    # Load B lanes (each is (BLOCK_M,))
    b0   = tl.load(b_base + 0, mask=inb, other=0.0).to(tl.float32)
    b1   = tl.load(b_base + 1, mask=inb, other=0.0).to(tl.float32)
    b2   = tl.load(b_base + 2, mask=inb, other=0.0).to(tl.float32)
    b12  = tl.load(b_base + 3, mask=inb, other=0.0).to(tl.float32)
    b3   = tl.load(b_base + 4, mask=inb, other=0.0).to(tl.float32)
    b13  = tl.load(b_base + 5, mask=inb, other=0.0).to(tl.float32)
    b23  = tl.load(b_base + 6, mask=inb, other=0.0).to(tl.float32)
    b123 = tl.load(b_base + 7, mask=inb, other=0.0).to(tl.float32)

    # Compute (each output is (BLOCK_M,))
    e0   = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a23*b23 - a13*b13 - a123*b123
    e1   = a0*b1 + a1*b0 - a2*b12 - a3*b13 + a12*b2 - a23*b123 + a13*b3 - a123*b23
    e2   = a0*b2 + a1*b12 + a2*b0 - a3*b23 - a12*b1 + a23*b3 + a13*b123 + a123*b13
    e3   = a0*b3 + a1*b13 + a2*b23 + a3*b0 - a12*b123 - a23*b2 - a13*b1 - a123*b12
    e12  = a0*b12 + a1*b2 - a2*b1 + a3*b123 + a12*b0 + a23*b13 - a13*b23 + a123*b3
    e23  = a0*b23 + a1*b123 + a2*b3 - a3*b2 - a12*b13 + a23*b0 + a13*b12 + a123*b1
    e13  = a0*b13 + a1*b3 - a2*b123 - a3*b1 + a12*b23 - a23*b12 + a13*b0 - a123*b2
    e123 = a0*b123 + a1*b23 - a2*b13 + a3*b12 + a12*b3 + a23*b1 - a13*b2 + a123*b0

    # Store lanes
    tl.store(c_base + 0, e0.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 1, e1.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 2, e2.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 3, e12.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 4, e3.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 5, e13.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 6, e23.to(OUT_DTYPE), mask=inb)
    tl.store(c_base + 7, e123.to(OUT_DTYPE), mask=inb)

def Cl3_Geometric_Product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape == B.shape and A.shape[-1] == 8
    assert A.is_cuda and B.is_cuda

    A2 = A.contiguous()
    B2 = B.contiguous()

    batch_shape = A2.shape[:-1]
    N = int(A2.numel() // 8)

    A_flat = A2.view(N, 8)
    B_flat = B2.view(N, 8)
    C_flat = torch.empty((N, 8), device=A.device, dtype=A.dtype)

    if A.dtype == torch.float16:
        out_dtype = tl.float16
    elif A.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif A.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype: {A.dtype}")

    BLOCK_M = 128  # try 64/128/256
    grid = (triton.cdiv(N, BLOCK_M),)

    cl3_gp_kernel[grid](
        A_flat, B_flat, C_flat,
        stride_a0=A_flat.stride(0),
        stride_b0=B_flat.stride(0),
        stride_c0=C_flat.stride(0),
        N=N,
        OUT_DTYPE=out_dtype,
        BLOCK_M=BLOCK_M,
        num_warps=4,    # try 2/4/8
        num_stages=2,   # try 2..4
    )

    return C_flat.view(*batch_shape, 8)

@triton.jit
def cl3_gp_kernel_bwd(
    A_ptr, B_ptr, dC_ptr,
    dA_ptr, dB_ptr,
    stride_ab: tl.constexpr, stride_ac: tl.constexpr,   # A strides (B,C)
    stride_bb: tl.constexpr, stride_bc: tl.constexpr,   # B strides (B,C)
    stride_cb: tl.constexpr, stride_cc: tl.constexpr,   # dC strides (B,C)
    stride_dab: tl.constexpr, stride_dac: tl.constexpr, # dA strides (B,C)
    stride_dbb: tl.constexpr, stride_dbc: tl.constexpr, # dB strides (B,C)
    B: tl.constexpr, C: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)                 # batch row
    pid_c_blk = tl.program_id(1)             # block along C

    c_idx = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)   # (BLOCK_C,)
    mask = (pid_b < B) & (c_idx < C)                      # (BLOCK_C,)

    # Base offsets for each element (pid_b, c_idx), in elements
    base_a  = pid_b * stride_ab  + c_idx * stride_ac
    base_b  = pid_b * stride_bb  + c_idx * stride_bc
    base_dc = pid_b * stride_cb  + c_idx * stride_cc
    base_da = pid_b * stride_dab + c_idx * stride_dac
    base_db = pid_b * stride_dbb + c_idx * stride_dbc

    # ---- load A (BLOCK_C,) ----
    a0   = tl.load(A_ptr + base_a + 0, mask=mask, other=0.0).to(tl.float32)
    a1   = tl.load(A_ptr + base_a + 1, mask=mask, other=0.0).to(tl.float32)
    a2   = tl.load(A_ptr + base_a + 2, mask=mask, other=0.0).to(tl.float32)
    a12  = tl.load(A_ptr + base_a + 3, mask=mask, other=0.0).to(tl.float32)
    a3   = tl.load(A_ptr + base_a + 4, mask=mask, other=0.0).to(tl.float32)
    a13  = tl.load(A_ptr + base_a + 5, mask=mask, other=0.0).to(tl.float32)
    a23  = tl.load(A_ptr + base_a + 6, mask=mask, other=0.0).to(tl.float32)
    a123 = tl.load(A_ptr + base_a + 7, mask=mask, other=0.0).to(tl.float32)

    # ---- load B (BLOCK_C,) ----
    b0   = tl.load(B_ptr + base_b + 0, mask=mask, other=0.0).to(tl.float32)
    b1   = tl.load(B_ptr + base_b + 1, mask=mask, other=0.0).to(tl.float32)
    b2   = tl.load(B_ptr + base_b + 2, mask=mask, other=0.0).to(tl.float32)
    b12  = tl.load(B_ptr + base_b + 3, mask=mask, other=0.0).to(tl.float32)
    b3   = tl.load(B_ptr + base_b + 4, mask=mask, other=0.0).to(tl.float32)
    b13  = tl.load(B_ptr + base_b + 5, mask=mask, other=0.0).to(tl.float32)
    b23  = tl.load(B_ptr + base_b + 6, mask=mask, other=0.0).to(tl.float32)
    b123 = tl.load(B_ptr + base_b + 7, mask=mask, other=0.0).to(tl.float32)

    # ---- load dC (BLOCK_C,) ----
    dc0   = tl.load(dC_ptr + base_dc + 0, mask=mask, other=0.0).to(tl.float32)
    dc1   = tl.load(dC_ptr + base_dc + 1, mask=mask, other=0.0).to(tl.float32)
    dc2   = tl.load(dC_ptr + base_dc + 2, mask=mask, other=0.0).to(tl.float32)
    dc12  = tl.load(dC_ptr + base_dc + 3, mask=mask, other=0.0).to(tl.float32)
    dc3   = tl.load(dC_ptr + base_dc + 4, mask=mask, other=0.0).to(tl.float32)
    dc13  = tl.load(dC_ptr + base_dc + 5, mask=mask, other=0.0).to(tl.float32)
    dc23  = tl.load(dC_ptr + base_dc + 6, mask=mask, other=0.0).to(tl.float32)
    dc123 = tl.load(dC_ptr + base_dc + 7, mask=mask, other=0.0).to(tl.float32)

    # match your original convention:
    dc3v = dc3
    dc4, dc5, dc6, dc7 = dc12, dc23, dc13, dc123

    # reverse(b): grade-2 and grade-3 flip sign
    rb0, rb1, rb2, rb3 = b0, b1, b2, b3
    rb4, rb5, rb6, rb7 = -b12, -b23, -b13, -b123

    # reverse(a): grade-2 and grade-3 flip sign
    ra0, ra1, ra2, ra3 = a0, a1, a2, a3
    ra4, ra5, ra6, ra7 = -a12, -a23, -a13, -a123

    # ---- dA = dC * reverse(B) ----
    da0 = dc0*rb0 + dc1*rb1 + dc2*rb2 + dc3v*rb3 - dc4*rb4 - dc5*rb5 - dc6*rb6 - dc7*rb7
    da1 = dc0*rb1 + dc1*rb0 - dc2*rb4 - dc3v*rb6 + dc4*rb2 - dc5*rb7 + dc6*rb3 - dc7*rb5
    da2 = dc0*rb2 + dc1*rb4 + dc2*rb0 - dc3v*rb5 - dc4*rb1 + dc5*rb3 + dc6*rb7 + dc7*rb6
    da3 = dc0*rb3 + dc1*rb6 + dc2*rb5 + dc3v*rb0 - dc4*rb7 - dc5*rb2 - dc6*rb1 - dc7*rb4
    da4 = dc0*rb4 + dc1*rb2 - dc2*rb1 + dc3v*rb7 + dc4*rb0 + dc5*rb6 - dc6*rb5 + dc7*rb3
    da5 = dc0*rb5 + dc1*rb7 + dc2*rb3 - dc3v*rb2 - dc4*rb6 + dc5*rb0 + dc6*rb4 + dc7*rb1
    da6 = dc0*rb6 + dc1*rb3 - dc2*rb7 - dc3v*rb1 + dc4*rb5 - dc5*rb4 + dc6*rb0 - dc7*rb2
    da7 = dc0*rb7 + dc1*rb5 - dc2*rb6 + dc3v*rb4 + dc4*rb3 + dc5*rb1 - dc6*rb2 + dc7*rb0

    # ---- dB = reverse(A) * dC ----
    db0 = ra0*dc0 + ra1*dc1 + ra2*dc2 + ra3*dc3v - ra4*dc4 - ra5*dc5 - ra6*dc6 - ra7*dc7
    db1 = ra0*dc1 + ra1*dc0 - ra2*dc4 - ra3*dc6 + ra4*dc2 - ra5*dc7 + ra6*dc3v - ra7*dc5
    db2 = ra0*dc2 + ra1*dc4 + ra2*dc0 - ra3*dc5 - ra4*dc1 + ra5*dc3v + ra6*dc7 + ra7*dc6
    db3 = ra0*dc3v + ra1*dc6 + ra2*dc5 + ra3*dc0 - ra4*dc7 - ra5*dc2 - ra6*dc1 - ra7*dc4
    db4 = ra0*dc4 + ra1*dc2 - ra2*dc1 + ra3*dc7 + ra4*dc0 + ra5*dc6 - ra6*dc5 + ra7*dc3v
    db5 = ra0*dc5 + ra1*dc7 + ra2*dc3v - ra3*dc2 - ra4*dc6 + ra5*dc0 + ra6*dc4 + ra7*dc1
    db6 = ra0*dc6 + ra1*dc3v - ra2*dc7 - ra3*dc1 + ra4*dc5 - ra5*dc4 + ra6*dc0 - ra7*dc2
    db7 = ra0*dc7 + ra1*dc5 - ra2*dc6 + ra3*dc4 + ra4*dc3v + ra5*dc1 - ra6*dc2 + ra7*dc0

    # Store in your layout [1,e1,e2,e12,e3,e13,e23,e123]
    tl.store(dA_ptr + base_da + 0, da0.to(OUT_DTYPE), mask=mask)
    tl.store(dA_ptr + base_da + 1, da1.to(OUT_DTYPE), mask=mask)
    tl.store(dA_ptr + base_da + 2, da2.to(OUT_DTYPE), mask=mask)
    tl.store(dA_ptr + base_da + 3, da4.to(OUT_DTYPE), mask=mask)  # e12
    tl.store(dA_ptr + base_da + 4, da3.to(OUT_DTYPE), mask=mask)  # e3
    tl.store(dA_ptr + base_da + 5, da6.to(OUT_DTYPE), mask=mask)  # e13
    tl.store(dA_ptr + base_da + 6, da5.to(OUT_DTYPE), mask=mask)  # e23
    tl.store(dA_ptr + base_da + 7, da7.to(OUT_DTYPE), mask=mask)

    tl.store(dB_ptr + base_db + 0, db0.to(OUT_DTYPE), mask=mask)
    tl.store(dB_ptr + base_db + 1, db1.to(OUT_DTYPE), mask=mask)
    tl.store(dB_ptr + base_db + 2, db2.to(OUT_DTYPE), mask=mask)
    tl.store(dB_ptr + base_db + 3, db4.to(OUT_DTYPE), mask=mask)  # e12
    tl.store(dB_ptr + base_db + 4, db3.to(OUT_DTYPE), mask=mask)  # e3
    tl.store(dB_ptr + base_db + 5, db6.to(OUT_DTYPE), mask=mask)  # e13
    tl.store(dB_ptr + base_db + 6, db5.to(OUT_DTYPE), mask=mask)  # e23
    tl.store(dB_ptr + base_db + 7, db7.to(OUT_DTYPE), mask=mask)


def Cl3_Geometric_Product_Bwd(A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    """
    Gradients for C = A * B (Cl(3) GP), for tensors shaped (..., 8).

    We interpret the batch as 2D: [B_dim, C_dim] where
      B_dim = batch_shape[0] (or 1 if no batch dims),
      C_dim = prod(batch_shape[1:]) (or 1 if only one batch dim).
    """
    assert A.shape == B.shape == dC.shape, "A, B, dC must have the same shape."
    assert A.shape[-1] == 8 and B.shape[-1] == 8 and dC.shape[-1] == 8, \
        "Input tensors must have last dimension of size 8."
    assert A.is_cuda and B.is_cuda and dC.is_cuda, "Tensors must be CUDA."
    assert A.device == B.device, "Tensors must be on same device."
    assert A.dtype == B.dtype == dC.dtype, "A, B, dC must have the same dtype."

    batch_shape = A.shape[:-1]

    # Collapse batch dims into (B_dim, C_dim) for a 2D Triton grid.
    if len(batch_shape) == 0:
        B_dim, C_dim = 1, 1
        A2 = A.view(1, 1, 8)
        B2 = B.view(1, 1, 8)
        dC2 = dC.view(1, 1, 8)
    elif len(batch_shape) == 1:
        B_dim, C_dim = batch_shape[0], 1
        A2 = A.view(B_dim, 1, 8)
        B2 = B.view(B_dim, 1, 8)
        dC2 = dC.view(B_dim, 1, 8)
    else:
        B_dim = batch_shape[0]
        C_dim = int(torch.tensor(batch_shape[1:]).prod().item())  # safe for Py3.13
        A2 = A.reshape(B_dim, C_dim, 8)
        B2 = B.reshape(B_dim, C_dim, 8)
        dC2 = dC.reshape(B_dim, C_dim, 8)
    if A2.is_contiguous() is False:
        A2 = A2.contiguous()
    if B2.is_contiguous() is False:
        B2 = B2.contiguous()
    if dC2.is_contiguous() is False:
        dC2 = dC2.contiguous()

    # Allocate outputs with same shape/strides as inputs (so grads match layout expectations)
    dA2 = torch.empty_like(A2)
    dB2 = torch.empty_like(B2)

    # Strides in *elements* (Triton expects element strides, not bytes)
    stride_ab, stride_ac = A2.stride(0), A2.stride(1)
    stride_bb, stride_bc = B2.stride(0), B2.stride(1)
    stride_cb, stride_cc = dC2.stride(0), dC2.stride(1)
    stride_dab, stride_dac = dA2.stride(0), dA2.stride(1)
    stride_dbb, stride_dbc = dB2.stride(0), dB2.stride(1)

    # OUT_DTYPE
    if A.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif A.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32
    BLOCK_C = 128  # try 64/128/256
    grid = (B_dim, triton.cdiv(C_dim, BLOCK_C))
    cl3_gp_kernel_bwd[grid](
        A2, B2, dC2,
        dA2, dB2,
        stride_ab, stride_ac,
        stride_bb, stride_bc,
        stride_cb, stride_cc,
        stride_dab, stride_dac,
        stride_dbb, stride_dbc,
        B=B_dim, C=C_dim,
        OUT_DTYPE=OUT_DTYPE,
        BLOCK_C=BLOCK_C,
        num_warps=4,    # try 2/4/8
        num_stages=2,
    )

    # Reshape grads back to original batch shape
    dA = dA2.reshape(*batch_shape, 8)
    dB = dB2.reshape(*batch_shape, 8)
    return dA, dB


class Cl3_GP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        C = Cl3_Geometric_Product(A, B)
        return C

    @staticmethod
    def backward(ctx, dC):
        A, B = ctx.saved_tensors
        dA, dB = Cl3_Geometric_Product_Bwd(A, B, dC)
        return dA, dB