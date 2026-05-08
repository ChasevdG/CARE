import torch
import triton
import triton.language as tl


@triton.jit
def cl3_inverse_kernel(
    X_ptr, Y_ptr,
    stride_x0: tl.constexpr,
    stride_y0: tl.constexpr,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    inb = pid < N
    offs = tl.arange(0, 8)
    base_x = pid * stride_x0
    base_y = pid * stride_y0
    x0 = tl.load(X_ptr + base_x + 0, mask=inb, other=0.0)
    x1 = tl.load(X_ptr + base_x + 1, mask=inb, other=0.0)
    x2 = tl.load(X_ptr + base_x + 2, mask=inb, other=0.0)
    x3 = tl.load(X_ptr + base_x + 4, mask=inb, other=0.0)
    x12 = tl.load(X_ptr + base_x + 3, mask=inb, other=0.0)
    x13 = tl.load(X_ptr + base_x + 5, mask=inb, other=0.0)
    x23 = tl.load(X_ptr + base_x + 6, mask=inb, other=0.0)
    x123 = tl.load(X_ptr + base_x + 7, mask=inb, other=0.0)

    # inverse: grade-0 same, grade-1 neg, grade-2 neg, grade-3 same
    y0 = x0
    y1, y2, y3 = -x1, -x2, -x3
    y12, y13, y23 = -x12, -x13, -x23
    y123 = x123

    # Normalization factor
    norm = (x0**2 + x1**2 + x2**2 + x3**2
            + x12**2 + x13**2 + x23**2 + x123**2)
    norm = tl.where(norm == 0.0, 1.0, norm)
    inv_norm = 1.0 / norm

    y0 = y0 * inv_norm
    y1 = y1 * inv_norm
    y2 = y2 * inv_norm
    y3 = y3 * inv_norm
    y12 = y12 * inv_norm
    y13 = y13 * inv_norm
    y23 = y23 * inv_norm
    y123 = y123 * inv_norm

    tl.store(Y_ptr + base_y + 0, y0, mask=inb)
    tl.store(Y_ptr + base_y + 1, y1, mask=inb)
    tl.store(Y_ptr + base_y + 2, y2, mask=inb)
    tl.store(Y_ptr + base_y + 3, y12, mask=inb)
    tl.store(Y_ptr + base_y + 4, y3, mask=inb)
    tl.store(Y_ptr + base_y + 5, y13, mask=inb)
    tl.store(Y_ptr + base_y + 6, y23, mask=inb)
    tl.store(Y_ptr + base_y + 7, y123, mask=inb)


def Cl3_Inverse(X: torch.Tensor) -> torch.Tensor:
    """
    inverse(X) for Cl(3) in basis [1,e1,e2,e12,e3,e13,e23,e123]
    X: (...,8) -> (...,8)
    """
    assert X.shape[-1] == 8
    assert X.is_cuda, "CUDA only"
    X2 = X.contiguous()

    batch_shape = X2.shape[:-1]
    N = int(X2.numel() // 8)
    X_flat = X2.view(N, 8)
    Y_flat = torch.empty_like(X_flat)

    if X.dtype == torch.float16:
        out_dtype = tl.float16
    elif X.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif X.dtype == torch.float32:
        out_dtype = tl.float32

    grid = (triton.cdiv(N, 1024),)
    cl3_inverse_kernel[grid](
        X_flat,
        Y_flat,
        X_flat.stride(0),
        Y_flat.stride(0),
        N,
        out_dtype,
    )

    Y = Y_flat.view(*batch_shape, 8)
    return Y