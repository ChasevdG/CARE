"""
Speed test: Cl3 PyTorch vs Triton.
Tests:
  - Geometric product (GP)
  - Sandwich with reversion:  A * B * rev(A)     (sign flip only, no norm)
  - Sandwich with inversion:  A * B * A^{-1}     (sign flip + norm division)

Run:
    python speed_test.py
    python speed_test.py --device cuda
    python speed_test.py --device cpu   # Triton sections shown as N/A
"""

import argparse
import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


# ── timing ─────────────────────────────────────────────────────────────────────

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1e3  # ms per call


# ── display ────────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Batch':<22} {'Torch (ms)':>12} {'Triton (ms)':>12} {'Speedup':>9}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*9}")


def row(name, t_torch, t_triton):
    if t_triton is None:
        print(f"  {name:<22} {t_torch:>12.4f} {'N/A':>12} {'N/A':>9}")
        return
    speedup = t_torch / t_triton
    tag = " ◄T" if speedup > 1.05 else (" ◄Py" if speedup < 0.95 else "")
    print(f"  {name:<22} {t_torch:>12.4f} {t_triton:>12.4f} {speedup:>8.2f}x{tag}")


# ── pure-function helpers (for torch SP variants) ─────────────────────────────

def _gp_fn(A, B, signs, indices):
    AB = torch.einsum("...i,...j->...ij", A, B)
    signed = AB * signs
    flat = signed.reshape(*signed.shape[:-2], 64)
    flat_idx = indices.reshape(64).expand_as(flat)
    out = torch.zeros(*flat.shape[:-1], 8, dtype=A.dtype, device=A.device)
    return out.scatter_add(-1, flat_idx, flat)


def _sp_rev_fn(A, B, signs, indices, rev_signs):
    """Sandwich with reversion: A * B * rev(A).  No norm division."""
    T = _gp_fn(A, B, signs, indices)
    return _gp_fn(T, A * rev_signs, signs, indices)


def _sp_inv_fn(A, B, signs, indices, rev_signs):
    """Sandwich with inverse: A * B * (rev(A) / ||A||²)."""
    norm = (A * A).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    A_inv = A * rev_signs / norm
    T = _gp_fn(A, B, signs, indices)
    return _gp_fn(T, A_inv, signs, indices)


# reversion sign pattern: grade-0,1 → +1; grade-2,3 → −1
_REV_SIGNS = torch.tensor([1, 1, 1, -1, 1, -1, -1, -1], dtype=torch.float32)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters",  type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    has_cuda = device.type == "cuda"

    print(f"\nDevice  : {device}")
    if has_cuda:
        print(f"GPU     : {torch.cuda.get_device_name(device)}")
    else:
        print("Note    : Triton kernels require CUDA — those columns will show N/A")

    from utils.Cl3_torch  import Cl3
    from utils.Cl3_triton import Cl3_triton
    if has_cuda:
        from utils.Triton_Kernels.sandwich_product import cl3_sandwich

    gp_signs   = Cl3.gp_signs.to(device=device, dtype=torch.float32)
    gp_indices = Cl3.gp_indices.to(device=device)
    rev_signs  = _REV_SIGNS.to(device=device)

    W, I = args.warmup, args.iters

    batch_shapes = [
        (64,),
        (512,),
        (4096,),
        (128, 64),
        (512, 64),
    ]

    # ── geometric product ──────────────────────────────────────────────────────
    header("Geometric Product  (A * B)")

    for bs in batch_shapes:
        A = torch.randn(*bs, 8, device=device)
        B = torch.randn(*bs, 8, device=device)
        a_t  = Cl3(A);         b_t  = Cl3(B)
        a_tr = Cl3_triton(A);  b_tr = Cl3_triton(B)

        t_torch = bench(lambda: a_t.__torch_gp__(b_t), W, I, device)
        if has_cuda and Cl3_triton.gp_func is Cl3_triton.__triton_mul__:
            t_triton = bench(lambda: a_tr.__triton_mul__(b_tr), W, I, device)
        elif has_cuda:
            t_triton = bench(lambda: a_tr.__torch_gp__(b_tr), W, I, device)
        else:
            t_triton = None

        row(str(list(bs)), t_torch, t_triton)

    # ── sandwich — reversion ───────────────────────────────────────────────────
    header("Sandwich — Reversion  (A * B * rev(A))")

    for bs in batch_shapes:
        A = torch.randn(*bs, 8, device=device)
        B = torch.randn(*bs, 8, device=device)

        t_torch  = bench(lambda: _sp_rev_fn(A, B, gp_signs, gp_indices, rev_signs), W, I, device)
        t_triton = bench(lambda: cl3_sandwich(A, B, mode="reversion"), W, I, device) if has_cuda else None

        row(str(list(bs)), t_torch, t_triton)

    # ── sandwich — inversion ───────────────────────────────────────────────────
    header("Sandwich — Inversion  (A * B * A⁻¹)")

    for bs in batch_shapes:
        A = torch.randn(*bs, 8, device=device)
        B = torch.randn(*bs, 8, device=device)
        a_t = Cl3(A); b_t = Cl3(B)

        t_torch  = bench(lambda: a_t.__matmul__(b_t), W, I, device)
        t_triton = bench(lambda: cl3_sandwich(A, B, mode="inverse"), W, I, device) if has_cuda else None

        row(str(list(bs)), t_torch, t_triton)

    print(f"\n  ◄T = Triton faster   ◄Py = Torch faster")

    # ── numerical sanity ───────────────────────────────────────────────────────
    print("\n--- Numerical sanity (max |torch - triton|, batch=256) ---")
    bs = (256,)

    # Unit rotors so rev(A) = A^{-1} and both sandwich formulas agree
    raw = torch.randn(*bs, 8, device=device)
    A_rotor = torch.zeros_like(raw)
    A_rotor[..., 0] = raw[..., 0]   # scalar
    A_rotor[..., 3] = raw[..., 3]   # e12
    A_rotor[..., 5] = raw[..., 5]   # e13
    A_rotor[..., 6] = raw[..., 6]   # e23
    A_rotor = A_rotor / (A_rotor * A_rotor).sum(-1, keepdim=True).clamp(min=1e-12).sqrt()

    B = torch.randn(*bs, 8, device=device)

    out_torch_gp = Cl3(A_rotor).__torch_gp__(Cl3(B)).data
    if has_cuda:
        out_triton_gp = Cl3_triton(A_rotor).__triton_mul__(Cl3_triton(B)).data
        print(f"  GP     : {(out_torch_gp - out_triton_gp).abs().max().item():.2e}")

        out_torch_rev  = _sp_rev_fn(A_rotor, B, gp_signs, gp_indices, rev_signs)
        out_triton_rev = cl3_sandwich(A_rotor, B, mode="reversion")
        print(f"  SP-rev : {(out_torch_rev - out_triton_rev).abs().max().item():.2e}")

        out_torch_inv  = _sp_inv_fn(A_rotor, B, gp_signs, gp_indices, rev_signs)
        out_triton_inv = cl3_sandwich(A_rotor, B, mode="inverse")
        print(f"  SP-inv : {(out_torch_inv - out_triton_inv).abs().max().item():.2e}")
    else:
        print("  (skipped — no CUDA)")


if __name__ == "__main__":
    main()
