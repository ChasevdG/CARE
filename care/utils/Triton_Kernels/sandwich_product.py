import torch
import triton
import triton.language as tl
 
 
# ============================================================================
# Forward kernel
# ============================================================================
@triton.jit
def cl3_sandwich_block_kernel(
    R_ptr, X_ptr, Y_ptr,
    stride_r0: tl.constexpr,
    stride_x0: tl.constexpr,
    stride_y0: tl.constexpr,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_INVERSE: tl.constexpr,   # True  -> Y = R · X · R^{-1}
                                 # False -> Y = R · X · reverse(R)  (unnormalized)
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    inb = rows < N
 
    r_base = R_ptr + rows * stride_r0
    x_base = X_ptr + rows * stride_x0
    y_base = Y_ptr + rows * stride_y0
 
    # ---- load R ----
    r0   = tl.load(r_base + 0, mask=inb, other=0.0).to(tl.float32)
    r1   = tl.load(r_base + 1, mask=inb, other=0.0).to(tl.float32)
    r2   = tl.load(r_base + 2, mask=inb, other=0.0).to(tl.float32)
    r12  = tl.load(r_base + 3, mask=inb, other=0.0).to(tl.float32)
    r3   = tl.load(r_base + 4, mask=inb, other=0.0).to(tl.float32)
    r13  = tl.load(r_base + 5, mask=inb, other=0.0).to(tl.float32)
    r23  = tl.load(r_base + 6, mask=inb, other=0.0).to(tl.float32)
    r123 = tl.load(r_base + 7, mask=inb, other=0.0).to(tl.float32)
 
    # ---- load X ----
    x0   = tl.load(x_base + 0, mask=inb, other=0.0).to(tl.float32)
    x1   = tl.load(x_base + 1, mask=inb, other=0.0).to(tl.float32)
    x2   = tl.load(x_base + 2, mask=inb, other=0.0).to(tl.float32)
    x12  = tl.load(x_base + 3, mask=inb, other=0.0).to(tl.float32)
    x3   = tl.load(x_base + 4, mask=inb, other=0.0).to(tl.float32)
    x13  = tl.load(x_base + 5, mask=inb, other=0.0).to(tl.float32)
    x23  = tl.load(x_base + 6, mask=inb, other=0.0).to(tl.float32)
    x123 = tl.load(x_base + 7, mask=inb, other=0.0).to(tl.float32)
 
    # ---- first product: T = R · X ----
    t0   = r0*x0 + r1*x1 + r2*x2 + r3*x3 - r12*x12 - r23*x23 - r13*x13 - r123*x123
    t1   = r0*x1 + r1*x0 - r2*x12 - r3*x13 + r12*x2 - r23*x123 + r13*x3 - r123*x23
    t2   = r0*x2 + r1*x12 + r2*x0 - r3*x23 - r12*x1 + r23*x3 + r13*x123 + r123*x13
    t3   = r0*x3 + r1*x13 + r2*x23 + r3*x0 - r12*x123 - r23*x2 - r13*x1 - r123*x12
    t12  = r0*x12 + r1*x2 - r2*x1 + r3*x123 + r12*x0 + r23*x13 - r13*x23 + r123*x3
    t23  = r0*x23 + r1*x123 + r2*x3 - r3*x2 - r12*x13 + r23*x0 + r13*x12 + r123*x1
    t13  = r0*x13 + r1*x3 - r2*x123 - r3*x1 + r12*x23 - r23*x12 + r13*x0 - r123*x2
    t123 = r0*x123 + r1*x23 - r2*x13 + r3*x12 + r12*x3 + r23*x1 - r13*x2 + r123*x0
 
    # ---- right factor: reverse(R)  OR  R^{-1} ----
    if USE_INVERSE:
        # Cl(3,0) inverse:
        #   Bar(R) = Clifford conjugation (flip grades 1 and 2)
        #   N = R · Bar(R) is always in the centre  span{1, e123}:  N = n_s + n_p e123
        #   N^{-1} = (n_s - n_p e123) / (n_s^2 + n_p^2)        (since e123^2 = -1)
        #   R^{-1} = Bar(R) · N^{-1}
        n_s = r0*r0 - r1*r1 - r2*r2 - r3*r3 + r12*r12 + r13*r13 + r23*r23 - r123*r123
        n_p = 2.0 * (r0*r123 - r1*r23 + r2*r13 - r3*r12)
        denom = n_s*n_s + n_p*n_p
        denom = tl.where(denom == 0.0, 1.0, denom)
        inv_s =  n_s / denom
        inv_p = -n_p / denom
 
        rr0   =  inv_s*r0   - inv_p*r123
        rr1   = -inv_s*r1   + inv_p*r23
        rr2   = -inv_s*r2   - inv_p*r13
        rr12  = -inv_s*r12  - inv_p*r3
        rr3   = -inv_s*r3   + inv_p*r12
        rr13  = -inv_s*r13  + inv_p*r2
        rr23  = -inv_s*r23  - inv_p*r1
        rr123 =  inv_s*r123 + inv_p*r0
    else:
        # reverse(R): keep grades 0,1; flip grades 2,3.
        rr0,  rr1,  rr2,  rr3   =  r0,   r1,   r2,   r3
        rr12, rr13, rr23, rr123 = -r12, -r13, -r23, -r123
 
    # ---- second product: Y = T · (right factor) ----
    y0   = t0*rr0 + t1*rr1 + t2*rr2 + t3*rr3 - t12*rr12 - t23*rr23 - t13*rr13 - t123*rr123
    y1   = t0*rr1 + t1*rr0 - t2*rr12 - t3*rr13 + t12*rr2 - t23*rr123 + t13*rr3 - t123*rr23
    y2   = t0*rr2 + t1*rr12 + t2*rr0 - t3*rr23 - t12*rr1 + t23*rr3 + t13*rr123 + t123*rr13
    y3   = t0*rr3 + t1*rr13 + t2*rr23 + t3*rr0 - t12*rr123 - t23*rr2 - t13*rr1 - t123*rr12
    y12  = t0*rr12 + t1*rr2 - t2*rr1 + t3*rr123 + t12*rr0 + t23*rr13 - t13*rr23 + t123*rr3
    y23  = t0*rr23 + t1*rr123 + t2*rr3 - t3*rr2 - t12*rr13 + t23*rr0 + t13*rr12 + t123*rr1
    y13  = t0*rr13 + t1*rr3 - t2*rr123 - t3*rr1 + t12*rr23 - t23*rr12 + t13*rr0 - t123*rr2
    y123 = t0*rr123 + t1*rr23 - t2*rr13 + t3*rr12 + t12*rr3 + t23*rr1 - t13*rr2 + t123*rr0
 
    # ---- store ----
    tl.store(y_base + 0, y0.to(OUT_DTYPE),   mask=inb)
    tl.store(y_base + 1, y1.to(OUT_DTYPE),   mask=inb)
    tl.store(y_base + 2, y2.to(OUT_DTYPE),   mask=inb)
    tl.store(y_base + 3, y12.to(OUT_DTYPE),  mask=inb)
    tl.store(y_base + 4, y3.to(OUT_DTYPE),   mask=inb)
    tl.store(y_base + 5, y13.to(OUT_DTYPE),  mask=inb)
    tl.store(y_base + 6, y23.to(OUT_DTYPE),  mask=inb)
    tl.store(y_base + 7, y123.to(OUT_DTYPE), mask=inb)
 
 
def _resolve_out_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise TypeError(f"Unsupported dtype: {dtype}")
 
 
def Cl3_Sandwich_Product(R: torch.Tensor, X: torch.Tensor, mode: str = "reversion") -> torch.Tensor:
    """
    Cl(3,0) sandwich:
        mode == "reversion" : Y = R · X · reverse(R)   (unnormalized)
        mode == "inverse"   : Y = R · X · R^{-1}
    """
    assert mode in ("reversion", "inverse"), f"mode must be 'reversion' or 'inverse', got {mode!r}"
    assert R.shape == X.shape and R.shape[-1] == 8
    assert R.is_cuda and X.is_cuda
    if R.dtype != X.dtype:
        R = R.to(X.dtype)
 
    R2 = R.contiguous()
    X2 = X.contiguous()
 
    batch_shape = R2.shape[:-1]
    N = int(R2.numel() // 8)
 
    R_flat = R2.view(N, 8)
    X_flat = X2.view(N, 8)
    Y_flat = torch.empty((N, 8), device=R.device, dtype=R.dtype)
 
    BLOCK_M = 128
    grid = (triton.cdiv(N, BLOCK_M),)
    cl3_sandwich_block_kernel[grid](
        R_flat, X_flat, Y_flat,
        stride_r0=R_flat.stride(0),
        stride_x0=X_flat.stride(0),
        stride_y0=Y_flat.stride(0),
        N=N,
        OUT_DTYPE=_resolve_out_dtype(R.dtype),
        BLOCK_M=BLOCK_M,
        USE_INVERSE=(mode == "inverse"),
        num_warps=4,
        num_stages=2,
    )
    return Y_flat.view(*batch_shape, 8)
 
@triton.jit
def cl3_sandwich_bwd_block_kernel(
    R_ptr, X_ptr, dY_ptr,
    dR_ptr, dX_ptr,
    stride_r0: tl.constexpr,
    stride_x0: tl.constexpr,
    stride_dy0: tl.constexpr,
    stride_dr0: tl.constexpr,
    stride_dx0: tl.constexpr,
    N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_INVERSE: tl.constexpr,   # must match the forward call that produced Y
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    inb = rows < N
 
    r_base  = R_ptr  + rows * stride_r0
    x_base  = X_ptr  + rows * stride_x0
    dy_base = dY_ptr + rows * stride_dy0
    dr_base = dR_ptr + rows * stride_dr0
    dx_base = dX_ptr + rows * stride_dx0
 
    # ---- load R ----
    r0   = tl.load(r_base + 0, mask=inb, other=0.0).to(tl.float32)
    r1   = tl.load(r_base + 1, mask=inb, other=0.0).to(tl.float32)
    r2   = tl.load(r_base + 2, mask=inb, other=0.0).to(tl.float32)
    r12  = tl.load(r_base + 3, mask=inb, other=0.0).to(tl.float32)
    r3   = tl.load(r_base + 4, mask=inb, other=0.0).to(tl.float32)
    r13  = tl.load(r_base + 5, mask=inb, other=0.0).to(tl.float32)
    r23  = tl.load(r_base + 6, mask=inb, other=0.0).to(tl.float32)
    r123 = tl.load(r_base + 7, mask=inb, other=0.0).to(tl.float32)
 
    # ---- load X ----
    x0   = tl.load(x_base + 0, mask=inb, other=0.0).to(tl.float32)
    x1   = tl.load(x_base + 1, mask=inb, other=0.0).to(tl.float32)
    x2   = tl.load(x_base + 2, mask=inb, other=0.0).to(tl.float32)
    x12  = tl.load(x_base + 3, mask=inb, other=0.0).to(tl.float32)
    x3   = tl.load(x_base + 4, mask=inb, other=0.0).to(tl.float32)
    x13  = tl.load(x_base + 5, mask=inb, other=0.0).to(tl.float32)
    x23  = tl.load(x_base + 6, mask=inb, other=0.0).to(tl.float32)
    x123 = tl.load(x_base + 7, mask=inb, other=0.0).to(tl.float32)
 
    # ---- load dY ----
    dy0   = tl.load(dy_base + 0, mask=inb, other=0.0).to(tl.float32)
    dy1   = tl.load(dy_base + 1, mask=inb, other=0.0).to(tl.float32)
    dy2   = tl.load(dy_base + 2, mask=inb, other=0.0).to(tl.float32)
    dy12  = tl.load(dy_base + 3, mask=inb, other=0.0).to(tl.float32)
    dy3   = tl.load(dy_base + 4, mask=inb, other=0.0).to(tl.float32)
    dy13  = tl.load(dy_base + 5, mask=inb, other=0.0).to(tl.float32)
    dy23  = tl.load(dy_base + 6, mask=inb, other=0.0).to(tl.float32)
    dy123 = tl.load(dy_base + 7, mask=inb, other=0.0).to(tl.float32)
 
    # ----------------------------------------------------------------------
    # 1) Recompute T = R · X
    # ----------------------------------------------------------------------
    t0   = r0*x0 + r1*x1 + r2*x2 + r3*x3 - r12*x12 - r23*x23 - r13*x13 - r123*x123
    t1   = r0*x1 + r1*x0 - r2*x12 - r3*x13 + r12*x2 - r23*x123 + r13*x3 - r123*x23
    t2   = r0*x2 + r1*x12 + r2*x0 - r3*x23 - r12*x1 + r23*x3 + r13*x123 + r123*x13
    t3   = r0*x3 + r1*x13 + r2*x23 + r3*x0 - r12*x123 - r23*x2 - r13*x1 - r123*x12
    t12  = r0*x12 + r1*x2 - r2*x1 + r3*x123 + r12*x0 + r23*x13 - r13*x23 + r123*x3
    t23  = r0*x23 + r1*x123 + r2*x3 - r3*x2 - r12*x13 + r23*x0 + r13*x12 + r123*x1
    t13  = r0*x13 + r1*x3 - r2*x123 - r3*x1 + r12*x23 - r23*x12 + r13*x0 - r123*x2
    t123 = r0*x123 + r1*x23 - r2*x13 + r3*x12 + r12*x3 + r23*x1 - r13*x2 + r123*x0
 
    # ----------------------------------------------------------------------
    # 2) sr := reverse(S)
    #      reversion mode: S = reverse(R)        →  sr = R
    #      inverse   mode: S = R^{-1}            →  sr = reverse(R^{-1})
    # ----------------------------------------------------------------------
    if USE_INVERSE:
        n_s = r0*r0 - r1*r1 - r2*r2 - r3*r3 + r12*r12 + r13*r13 + r23*r23 - r123*r123
        n_p = 2.0 * (r0*r123 - r1*r23 + r2*r13 - r3*r12)
        denom = n_s*n_s + n_p*n_p
        denom = tl.where(denom == 0.0, 1.0, denom)
        inv_s =  n_s / denom
        inv_p = -n_p / denom
        # reverse(R^{-1}) = R^{-1} with grades 2 and 3 negated
        sr0   =  inv_s*r0   - inv_p*r123
        sr1   = -inv_s*r1   + inv_p*r23
        sr2   = -inv_s*r2   - inv_p*r13
        sr12  =  inv_s*r12  + inv_p*r3
        sr3   = -inv_s*r3   + inv_p*r12
        sr13  =  inv_s*r13  - inv_p*r2
        sr23  =  inv_s*r23  + inv_p*r1
        sr123 = -inv_s*r123 - inv_p*r0
    else:
        sr0,  sr1,  sr2,  sr3   = r0,  r1,  r2,  r3
        sr12, sr13, sr23, sr123 = r12, r13, r23, r123
 
    # ----------------------------------------------------------------------
    # 3) dT = dY · sr
    # ----------------------------------------------------------------------
    dt0   = dy0*sr0 + dy1*sr1 + dy2*sr2 + dy3*sr3 - dy12*sr12 - dy23*sr23 - dy13*sr13 - dy123*sr123
    dt1   = dy0*sr1 + dy1*sr0 - dy2*sr12 - dy3*sr13 + dy12*sr2 - dy23*sr123 + dy13*sr3 - dy123*sr23
    dt2   = dy0*sr2 + dy1*sr12 + dy2*sr0 - dy3*sr23 - dy12*sr1 + dy23*sr3 + dy13*sr123 + dy123*sr13
    dt3   = dy0*sr3 + dy1*sr13 + dy2*sr23 + dy3*sr0 - dy12*sr123 - dy23*sr2 - dy13*sr1 - dy123*sr12
    dt12  = dy0*sr12 + dy1*sr2 - dy2*sr1 + dy3*sr123 + dy12*sr0 + dy23*sr13 - dy13*sr23 + dy123*sr3
    dt23  = dy0*sr23 + dy1*sr123 + dy2*sr3 - dy3*sr2 - dy12*sr13 + dy23*sr0 + dy13*sr12 + dy123*sr1
    dt13  = dy0*sr13 + dy1*sr3 - dy2*sr123 - dy3*sr1 + dy12*sr23 - dy23*sr12 + dy13*sr0 - dy123*sr2
    dt123 = dy0*sr123 + dy1*sr23 - dy2*sr13 + dy3*sr12 + dy12*sr3 + dy23*sr1 - dy13*sr2 + dy123*sr0
 
    # ----------------------------------------------------------------------
    # 4) dX = reverse(R) · dT
    # ----------------------------------------------------------------------
    rr0,  rr1,  rr2,  rr3   =  r0,   r1,   r2,   r3
    rr12, rr13, rr23, rr123 = -r12, -r13, -r23, -r123
 
    dx0   = rr0*dt0 + rr1*dt1 + rr2*dt2 + rr3*dt3 - rr12*dt12 - rr23*dt23 - rr13*dt13 - rr123*dt123
    dx1   = rr0*dt1 + rr1*dt0 - rr2*dt12 - rr3*dt13 + rr12*dt2 - rr23*dt123 + rr13*dt3 - rr123*dt23
    dx2   = rr0*dt2 + rr1*dt12 + rr2*dt0 - rr3*dt23 - rr12*dt1 + rr23*dt3 + rr13*dt123 + rr123*dt13
    dx3   = rr0*dt3 + rr1*dt13 + rr2*dt23 + rr3*dt0 - rr12*dt123 - rr23*dt2 - rr13*dt1 - rr123*dt12
    dx12  = rr0*dt12 + rr1*dt2 - rr2*dt1 + rr3*dt123 + rr12*dt0 + rr23*dt13 - rr13*dt23 + rr123*dt3
    dx23  = rr0*dt23 + rr1*dt123 + rr2*dt3 - rr3*dt2 - rr12*dt13 + rr23*dt0 + rr13*dt12 + rr123*dt1
    dx13  = rr0*dt13 + rr1*dt3 - rr2*dt123 - rr3*dt1 + rr12*dt23 - rr23*dt12 + rr13*dt0 - rr123*dt2
    dx123 = rr0*dt123 + rr1*dt23 - rr2*dt13 + rr3*dt12 + rr12*dt3 + rr23*dt1 - rr13*dt2 + rr123*dt0
 
    # ----------------------------------------------------------------------
    # 5) dR_first = dT · reverse(X)
    # ----------------------------------------------------------------------
    rx0,  rx1,  rx2,  rx3   =  x0,   x1,   x2,   x3
    rx12, rx13, rx23, rx123 = -x12, -x13, -x23, -x123
 
    dr10   = dt0*rx0 + dt1*rx1 + dt2*rx2 + dt3*rx3 - dt12*rx12 - dt23*rx23 - dt13*rx13 - dt123*rx123
    dr11   = dt0*rx1 + dt1*rx0 - dt2*rx12 - dt3*rx13 + dt12*rx2 - dt23*rx123 + dt13*rx3 - dt123*rx23
    dr12_  = dt0*rx2 + dt1*rx12 + dt2*rx0 - dt3*rx23 - dt12*rx1 + dt23*rx3 + dt13*rx123 + dt123*rx13
    dr13_  = dt0*rx3 + dt1*rx13 + dt2*rx23 + dt3*rx0 - dt12*rx123 - dt23*rx2 - dt13*rx1 - dt123*rx12
    dr1_12 = dt0*rx12 + dt1*rx2 - dt2*rx1 + dt3*rx123 + dt12*rx0 + dt23*rx13 - dt13*rx23 + dt123*rx3
    dr1_23 = dt0*rx23 + dt1*rx123 + dt2*rx3 - dt3*rx2 - dt12*rx13 + dt23*rx0 + dt13*rx12 + dt123*rx1
    dr1_13 = dt0*rx13 + dt1*rx3 - dt2*rx123 - dt3*rx1 + dt12*rx23 - dt23*rx12 + dt13*rx0 - dt123*rx2
    dr1_123= dt0*rx123 + dt1*rx23 - dt2*rx13 + dt3*rx12 + dt12*rx3 + dt23*rx1 - dt13*rx2 + dt123*rx0
 
    # ----------------------------------------------------------------------
    # 6) dS = reverse(T) · dY
    # ----------------------------------------------------------------------
    rt0,  rt1,  rt2,  rt3   =  t0,   t1,   t2,   t3
    rt12, rt13, rt23, rt123 = -t12, -t13, -t23, -t123
 
    drr0   = rt0*dy0 + rt1*dy1 + rt2*dy2 + rt3*dy3 - rt12*dy12 - rt23*dy23 - rt13*dy13 - rt123*dy123
    drr1   = rt0*dy1 + rt1*dy0 - rt2*dy12 - rt3*dy13 + rt12*dy2 - rt23*dy123 + rt13*dy3 - rt123*dy23
    drr2   = rt0*dy2 + rt1*dy12 + rt2*dy0 - rt3*dy23 - rt12*dy1 + rt23*dy3 + rt13*dy123 + rt123*dy13
    drr3   = rt0*dy3 + rt1*dy13 + rt2*dy23 + rt3*dy0 - rt12*dy123 - rt23*dy2 - rt13*dy1 - rt123*dy12
    drr12  = rt0*dy12 + rt1*dy2 - rt2*dy1 + rt3*dy123 + rt12*dy0 + rt23*dy13 - rt13*dy23 + rt123*dy3
    drr23  = rt0*dy23 + rt1*dy123 + rt2*dy3 - rt3*dy2 - rt12*dy13 + rt23*dy0 + rt13*dy12 + rt123*dy1
    drr13  = rt0*dy13 + rt1*dy3 - rt2*dy123 - rt3*dy1 + rt12*dy23 - rt23*dy12 + rt13*dy0 - rt123*dy2
    drr123 = rt0*dy123 + rt1*dy23 - rt2*dy13 + rt3*dy12 + rt12*dy3 + rt23*dy1 - rt13*dy2 + rt123*dy0
 
    # ----------------------------------------------------------------------
    # 7) dR_second  (the only branch that actually depends on USE_INVERSE)
    # ----------------------------------------------------------------------
    if USE_INVERSE:
        # dR_second = - sr · dS · sr,  where sr = reverse(R^{-1})
        # First leg: tmp = sr · dS
        tmp0   = sr0*drr0 + sr1*drr1 + sr2*drr2 + sr3*drr3 - sr12*drr12 - sr23*drr23 - sr13*drr13 - sr123*drr123
        tmp1   = sr0*drr1 + sr1*drr0 - sr2*drr12 - sr3*drr13 + sr12*drr2 - sr23*drr123 + sr13*drr3 - sr123*drr23
        tmp2   = sr0*drr2 + sr1*drr12 + sr2*drr0 - sr3*drr23 - sr12*drr1 + sr23*drr3 + sr13*drr123 + sr123*drr13
        tmp3   = sr0*drr3 + sr1*drr13 + sr2*drr23 + sr3*drr0 - sr12*drr123 - sr23*drr2 - sr13*drr1 - sr123*drr12
        tmp12  = sr0*drr12 + sr1*drr2 - sr2*drr1 + sr3*drr123 + sr12*drr0 + sr23*drr13 - sr13*drr23 + sr123*drr3
        tmp23  = sr0*drr23 + sr1*drr123 + sr2*drr3 - sr3*drr2 - sr12*drr13 + sr23*drr0 + sr13*drr12 + sr123*drr1
        tmp13  = sr0*drr13 + sr1*drr3 - sr2*drr123 - sr3*drr1 + sr12*drr23 - sr23*drr12 + sr13*drr0 - sr123*drr2
        tmp123 = sr0*drr123 + sr1*drr23 - sr2*drr13 + sr3*drr12 + sr12*drr3 + sr23*drr1 - sr13*drr2 + sr123*drr0
 
        # Second leg: dR_second = -tmp · sr
        dr20    = -(tmp0*sr0 + tmp1*sr1 + tmp2*sr2 + tmp3*sr3 - tmp12*sr12 - tmp23*sr23 - tmp13*sr13 - tmp123*sr123)
        dr21    = -(tmp0*sr1 + tmp1*sr0 - tmp2*sr12 - tmp3*sr13 + tmp12*sr2 - tmp23*sr123 + tmp13*sr3 - tmp123*sr23)
        dr22    = -(tmp0*sr2 + tmp1*sr12 + tmp2*sr0 - tmp3*sr23 - tmp12*sr1 + tmp23*sr3 + tmp13*sr123 + tmp123*sr13)
        dr23_   = -(tmp0*sr3 + tmp1*sr13 + tmp2*sr23 + tmp3*sr0 - tmp12*sr123 - tmp23*sr2 - tmp13*sr1 - tmp123*sr12)
        dr2_12  = -(tmp0*sr12 + tmp1*sr2 - tmp2*sr1 + tmp3*sr123 + tmp12*sr0 + tmp23*sr13 - tmp13*sr23 + tmp123*sr3)
        dr2_23  = -(tmp0*sr23 + tmp1*sr123 + tmp2*sr3 - tmp3*sr2 - tmp12*sr13 + tmp23*sr0 + tmp13*sr12 + tmp123*sr1)
        dr2_13  = -(tmp0*sr13 + tmp1*sr3 - tmp2*sr123 - tmp3*sr1 + tmp12*sr23 - tmp23*sr12 + tmp13*sr0 - tmp123*sr2)
        dr2_123 = -(tmp0*sr123 + tmp1*sr23 - tmp2*sr13 + tmp3*sr12 + tmp12*sr3 + tmp23*sr1 - tmp13*sr2 + tmp123*sr0)
    else:
        # dR_second = reverse(dS)  (reversion is a self-adjoint involution)
        dr20,    dr21,    dr22,    dr23_   =  drr0,   drr1,   drr2,   drr3
        dr2_12,  dr2_13,  dr2_23,  dr2_123 = -drr12, -drr13, -drr23, -drr123
 
    # ----------------------------------------------------------------------
    # 8) total dR
    # ----------------------------------------------------------------------
    dR0   = dr10    + dr20
    dR1   = dr11    + dr21
    dR2   = dr12_   + dr22
    dR3   = dr13_   + dr23_
    dR12  = dr1_12  + dr2_12
    dR23  = dr1_23  + dr2_23
    dR13  = dr1_13  + dr2_13
    dR123 = dr1_123 + dr2_123
 
    # ---- store ----
    tl.store(dx_base + 0, dx0.to(OUT_DTYPE),   mask=inb)
    tl.store(dx_base + 1, dx1.to(OUT_DTYPE),   mask=inb)
    tl.store(dx_base + 2, dx2.to(OUT_DTYPE),   mask=inb)
    tl.store(dx_base + 3, dx12.to(OUT_DTYPE),  mask=inb)
    tl.store(dx_base + 4, dx3.to(OUT_DTYPE),   mask=inb)
    tl.store(dx_base + 5, dx13.to(OUT_DTYPE),  mask=inb)
    tl.store(dx_base + 6, dx23.to(OUT_DTYPE),  mask=inb)
    tl.store(dx_base + 7, dx123.to(OUT_DTYPE), mask=inb)
 
    tl.store(dr_base + 0, dR0.to(OUT_DTYPE),   mask=inb)
    tl.store(dr_base + 1, dR1.to(OUT_DTYPE),   mask=inb)
    tl.store(dr_base + 2, dR2.to(OUT_DTYPE),   mask=inb)
    tl.store(dr_base + 3, dR12.to(OUT_DTYPE),  mask=inb)
    tl.store(dr_base + 4, dR3.to(OUT_DTYPE),   mask=inb)
    tl.store(dr_base + 5, dR13.to(OUT_DTYPE),  mask=inb)
    tl.store(dr_base + 6, dR23.to(OUT_DTYPE),  mask=inb)
    tl.store(dr_base + 7, dR123.to(OUT_DTYPE), mask=inb)
 
 
def Cl3_Sandwich_Backward(R: torch.Tensor, X: torch.Tensor, dY: torch.Tensor,
                          mode: str = "reversion"):
    """
    Returns (dR, dX) for Y = R · X · S with S according to `mode`.
    `mode` MUST match the forward call that produced Y.
    """
    assert mode in ("reversion", "inverse"), f"mode must be 'reversion' or 'inverse', got {mode!r}"
    assert R.shape == X.shape == dY.shape and R.shape[-1] == 8
    assert R.is_cuda and X.is_cuda and dY.is_cuda
 
    R2  = R.contiguous()
    X2  = X.contiguous()
    dY2 = dY.contiguous()
 
    batch_shape = R2.shape[:-1]
    N = int(R2.numel() // 8)
 
    R_flat  = R2.view(N, 8)
    X_flat  = X2.view(N, 8)
    dY_flat = dY2.view(N, 8)
    dR_flat = torch.empty((N, 8), device=R.device, dtype=R.dtype)
    dX_flat = torch.empty((N, 8), device=R.device, dtype=R.dtype)
 
    BLOCK_M = 128
    grid = (triton.cdiv(N, BLOCK_M),)
    cl3_sandwich_bwd_block_kernel[grid](
        R_flat, X_flat, dY_flat,
        dR_flat, dX_flat,
        stride_r0=R_flat.stride(0),
        stride_x0=X_flat.stride(0),
        stride_dy0=dY_flat.stride(0),
        stride_dr0=dR_flat.stride(0),
        stride_dx0=dX_flat.stride(0),
        N=N,
        OUT_DTYPE=_resolve_out_dtype(R.dtype),
        BLOCK_M=BLOCK_M,
        USE_INVERSE=(mode == "inverse"),
        num_warps=4,
        num_stages=2,
    )
    return dR_flat.view(*batch_shape, 8), dX_flat.view(*batch_shape, 8)


class _Cl3SandwichFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, X, mode):
        ctx.save_for_backward(R, X)
        ctx.mode = mode
        return Cl3_Sandwich_Product(R, X, mode)
 
    @staticmethod
    def backward(ctx, dY):
        R, X = ctx.saved_tensors
        dR, dX = Cl3_Sandwich_Backward(R, X, dY.contiguous(), ctx.mode)
        return dR, dX, None
 
 
def cl3_sandwich(R: torch.Tensor, X: torch.Tensor, mode: str = "reversion") -> torch.Tensor:
    """Autograd-aware entry point: Y = R · X · S, S ∈ {reverse(R), R^{-1}}."""
    return _Cl3SandwichFn.apply(R, X, mode)