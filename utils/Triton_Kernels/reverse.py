import torch
import triton
import triton.language as tl


@triton.jit
def cl3_reverse_kernel(
    X_ptr, Y_ptr,
    stride_x0: tl.constexpr,
    stride_y0: tl.constexpr,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Can probably be optimized more with atomic ops but this is a minor kernel
    pid = tl.program_id(0)
    inb = pid < N

    offs = tl.arange(0, 8)
    base_x = pid * stride_x0
    base_y = pid * stride_y0

    x12 = tl.load(X_ptr + base_x + 3, mask=inb, other=0.0)
    x13 = tl.load(X_ptr + base_x + 5, mask=inb, other=0.0)
    x23 = tl.load(X_ptr + base_x + 6, mask=inb, other=0.0)
    x123 = tl.load(X_ptr + base_x + 7, mask=inb, other=0.0)

    # reverse: grade-0 same, grade-1 same, grade-2 neg, grade-3 neg
    # y0, y1, y2, y3 = x0, x1, x2, x3
    y12, y13, y23, y123 = -x12, -x13, -x23, -x123

    tl.store(Y_ptr + base_y + 0, tl.load(X_ptr + base_x + 0, mask=inb, other=0.0), mask=inb)
    tl.store(Y_ptr + base_y + 1, tl.load(X_ptr + base_x + 1, mask=inb, other=0.0), mask=inb)
    tl.store(Y_ptr + base_y + 2, tl.load(X_ptr + base_x + 2, mask=inb, other=0.0), mask=inb)
    tl.store(Y_ptr + base_y + 3, y12, mask=inb)
    tl.store(Y_ptr + base_y + 4, tl.load(X_ptr + base_x + 4, mask=inb, other=0.0), mask=inb)
    tl.store(Y_ptr + base_y + 5, y13, mask=inb)
    tl.store(Y_ptr + base_y + 6, y23, mask=inb)
    tl.store(Y_ptr + base_y + 7, y123, mask=inb)


def Cl3_Reverse(X: torch.Tensor) -> torch.Tensor:
    """
    reverse(X) for Cl(3) in basis [1,e1,e2,e12,e3,e13,e23,e123]
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
    else:
        raise TypeError(f"Unsupported dtype: {X.dtype}")

    grid = (triton.cdiv(N, 1),) # Needs to be fixed later for multiple at once
    cl3_reverse_kernel[grid](
        X_flat, Y_flat,
        stride_x0=X_flat.stride(0),
        stride_y0=Y_flat.stride(0),
        N=N,
        OUT_DTYPE=out_dtype,
        num_warps=1,
        num_stages=1,
    )
    return Y_flat.view(*batch_shape, 8)