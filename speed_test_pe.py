"""
Speed test for all Positional_Embeddings methods.

D=48 satisfies every method's constraint:
  D%2 == 0  (standard RoPE pairs)
  D%4 == 0  (axial with pos_dim=2 → D%(2*2))
  D%3 == 0  (spherical/quaternion 3-vectors)
  D%8 == 0  (CARE: Cl(3,0) 8-component multivectors)

Run:
    python speed_test_pe.py
    python speed_test_pe.py --device cuda
    python speed_test_pe.py --device cpu
"""

import argparse
import time
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(__file__))

from Positional_Embeddings.AxialRoPE       import Axial_RoPE
from Positional_Embeddings.Learned_Axial   import Learned_Axial_RoPE
from Positional_Embeddings.Mixed_RoPE      import Mixed_RoPE
from Positional_Embeddings.SphericalRoPE   import Spherical_RoPE
from Positional_Embeddings.Learned_Spherical import Learned_Spherical_RoPE
from Positional_Embeddings.QuatRo          import QuatRo, Mixed_QuatRo, Spherical_QuatRo
from Positional_Embeddings.CliffordRope    import CARE, Spherical_CARE, Mixed_CARE, Quaternion_CARE
from Positional_Embeddings.NoPE            import No_PE


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
    return (time.perf_counter() - t0) / iters * 1e3


# ── display ────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def table_header():
    print(f"  {'Method':<26} {'ms/call':>10} {'MEl/s':>10}  Group")
    print(f"  {'-'*26} {'-'*10} {'-'*10}  -----")


def table_row(name, t_ms, n_elements, group=""):
    tput = n_elements / (t_ms * 1e-3) / 1e6
    print(f"  {name:<26} {t_ms:>10.4f} {tput:>10.1f}  {group}")


def sweep_header(col_label):
    print(f"\n  {'Method':<26}", end="")
    for v in col_label:
        print(f" {str(v):>8}", end="")
    print()
    print(f"  {'-'*26}", end="")
    for _ in col_label:
        print(f" {'--------':>8}", end="")
    print()


def sweep_row(name, times):
    print(f"  {name:<26}", end="")
    for t in times:
        print(f" {t:>8.3f}", end="")
    print()


# ── model factory ──────────────────────────────────────────────────────────────

def make(cls, device, **kwargs):
    return cls(**kwargs).to(device).eval()


def run(model, x, pos, has_pos_arg):
    if has_pos_arg:
        return model(x, pos)
    else:
        return model(x)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters",  type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")

    W, I = args.warmup, args.iters

    # ── shared defaults ────────────────────────────────────────────────────────
    B, H, P, D = 4, 8, 256, 48
    pos = torch.randn(P, 2, device=device)

    # (cls, init_kwargs, has_pos_arg, group_label)
    methods = [
        (No_PE,               dict(embedding_dim=D, P_x=16, P_y=16),                      False, "baseline"),
        (Axial_RoPE,          dict(embedding_dim=D, pos_dim=2),                            True,  "axial"),
        (Learned_Axial_RoPE,  dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "axial"),
        (Mixed_RoPE,          dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "axial"),
        (Spherical_RoPE,      dict(embedding_dim=D, pos_dim=2),                            True,  "spherical"),
        (Learned_Spherical_RoPE, dict(embedding_dim=D, pos_dim=2, heads=H),               True,  "spherical"),
        (QuatRo,              dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "quaternion"),
        (Mixed_QuatRo,        dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "quaternion"),
        (Spherical_QuatRo,    dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "quaternion"),
        (CARE,                dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "clifford"),
        (Spherical_CARE,      dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "clifford"),
        (Mixed_CARE,          dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "clifford"),
        (Quaternion_CARE,     dict(embedding_dim=D, pos_dim=2, heads=H),                  True,  "clifford"),
    ]

    # ── Section 1: all-methods comparison ─────────────────────────────────────
    section(f"All methods  —  B={B} H={H} P={P} D={D}  (ms per call)")
    table_header()

    results = {}
    x = torch.randn(B, H, P, D, device=device)

    for cls, kwargs, has_pos, group in methods:
        name = cls.__name__
        model = make(cls, device, **kwargs)
        with torch.no_grad():
            t = bench(lambda m=model, hp=has_pos: run(m, x, pos, hp), W, I, device)
        results[name] = t
        table_row(name, t, B * H * P * D, group)

    # ── Section 2: sequence length sweep ──────────────────────────────────────
    Ps = [64, 128, 256, 512, 1024, 2048]
    sweep_methods = [
        (No_PE,               dict(embedding_dim=D, P_x=8,  P_y=8),   False),
        (Axial_RoPE,          dict(embedding_dim=D, pos_dim=2),        True),
        (Spherical_RoPE,      dict(embedding_dim=D, pos_dim=2),        True),
        (QuatRo,              dict(embedding_dim=D, pos_dim=2, heads=H), True),
        (CARE,                dict(embedding_dim=D, pos_dim=2, heads=H), True),
    ]

    section(f"Sequence length sweep  —  B={B} H={H} D={D}  (ms per call)")
    sweep_header(Ps)

    for cls, kwargs, has_pos in sweep_methods:
        model = make(cls, device, **kwargs)
        times = []
        for P_ in Ps:
            x_ = torch.randn(B, H, P_, D, device=device)
            p_ = torch.randn(P_, 2, device=device)
            with torch.no_grad():
                t = bench(lambda m=model, x=x_, p=p_, hp=has_pos: run(m, x, p, hp), W, I, device)
            times.append(t)
        sweep_row(cls.__name__, times)

    # ── Section 3: embedding dim sweep ────────────────────────────────────────
    # Only dims divisible by lcm(2,3,8)=24 so all methods stay valid
    Ds = [24, 48, 96, 192]

    section(f"Embedding dim sweep  —  B={B} H={H} P={P}  (ms per call)")
    sweep_header(Ds)

    for cls, kwargs_base, has_pos in sweep_methods:
        times = []
        for D_ in Ds:
            kw = dict(kwargs_base)
            kw["embedding_dim"] = D_
            # No_PE uses P_x/P_y, not embedding_dim only
            if cls is No_PE:
                kw["P_x"] = 16; kw["P_y"] = 16
            model = make(cls, device, **kw)
            x_ = torch.randn(B, H, P, D_, device=device)
            p_ = torch.randn(P, 2, device=device)
            with torch.no_grad():
                t = bench(lambda m=model, x=x_, p=p_, hp=has_pos: run(m, x, p, hp), W, I, device)
            times.append(t)
        sweep_row(cls.__name__, times)

    # ── Section 4: batch size sweep ───────────────────────────────────────────
    Bs = [1, 2, 4, 8, 16, 32]

    section(f"Batch size sweep  —  H={H} P={P} D={D}  (ms per call)")
    sweep_header(Bs)

    for cls, kwargs, has_pos in sweep_methods:
        model = make(cls, device, **kwargs)
        times = []
        for B_ in Bs:
            x_ = torch.randn(B_, H, P, D, device=device)
            with torch.no_grad():
                t = bench(lambda m=model, x=x_, p=pos, hp=has_pos: run(m, x, p, hp), W, I, device)
            times.append(t)
        sweep_row(cls.__name__, times)

    print()


if __name__ == "__main__":
    main()
