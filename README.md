# CARE — Clifford Algebra Rotary Embeddings

A collection of geometric positional embeddings for transformers from the 'Clifford Algebraic Rotor Embeddings: Maybe embeddings should start to CARE', including  axial RoPE, Mixed RoPE, Spherical RoPE. The QuatRo and CARE equivalents are provided as well.

All methods share the same interface: `forward(x, pos)` where `x` is `[B, H, P, D]` and `pos` is `[P, M]`.

---

## Installation

**Recommended: conda environment**

```bash
conda create -n care python=3.11
conda activate care

# PyTorch — pick the right CUDA version for your GPU
# See https://pytorch.org/get-started/locally/ for the correct command
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Triton (GPU kernels) and notebook dependencies
pip install -r requirements.txt
```

**CPU-only (Triton kernels disabled, pure PyTorch fallback used)**

```bash
conda create -n care-cpu python=3.11
conda activate care-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyter matplotlib
```

---

## Quick Start

```python
import Positional_Embeddings
from PE_registry import PE_REGISTRY as PEs

CARE = PEs['CARE']

B, H, P, D = 4, 8, 256, 48   # batch, heads, tokens, embedding dim

x   = torch.randn(B, H, P, D)
pos = torch.randn(P, 2)        # 2-D position per token

# CARE (D must be divisible by 8)
model = CARE(embedding_dim=D, pos_dim=2, heads=H)
out   = model(x, pos)          # [4, 8, 256, 48]

# Axial RoPE (D must be divisible by 2*pos_dim)
model = Axial_RoPE(embedding_dim=D, pos_dim=2, heads=None)
out   = model(x, pos)          # [4, 8, 256, 48]
```

To list the positional encoding options
```python
print(list(PE_REGISTRY.keys()))              # all registered names
```

---

## Methods

All embeddings accept `x: [B, H, P, D]` and `pos: [P, M]` or `[B, P, M]`.  
`D` must satisfy the divisibility constraint for the chosen method.

### Axial RoPE family

Each channel-pair is rotated by a single positional axis.

| Class | `D` constraint | Frequencies | `heads` arg |
|---|---|---|---|
| `Axial_RoPE` | `D % (2·M) == 0` | fixed `100^(−2d/D)` | no |
| `Learned_Axial_RoPE` | `D % (2·M) == 0` | learned per-head, per-axis | yes |
| `Mixed_RoPE` | `D % 2 == 0` | learned per-head | yes |

```python
Axial_RoPE = PEs['Axial RoPE']
Mixed_RoPE = PEs['Mixed RoPE']
Learned_Axial_RoPE = PEs['Learned Axial RoPE']


pos = torch.randn(P, 2)   # M=2

m1 = Axial_RoPE(embedding_dim=48, pos_dim=2, heads=None)
m2 = Learned_Axial_RoPE(embedding_dim=48, pos_dim=2, heads=H)
m3 = Mixed_RoPE(embedding_dim=48, pos_dim=2, heads=H)
```

### Spherical RoPE

Channels are grouped into 3-vectors and rotated in SO(3).

| Class | `D` constraint | Frequencies | `heads` arg |
|---|---|---|---|
| `Spherical RoPE` | `D % 3 == 0` | fixed `100^(−2d/D)` | no |
| `Learned Spherical RoPE` | `D % 3 == 0` | learned per-head | yes |

```python
Spherical_RoPE = PEs['Axial RoPE']
Learned_Spherical_RoPE = PEs['Learned Axial RoPE']

m1 = Spherical_RoPE(embedding_dim=48, pos_dim=2)
m2 = Learned_Spherical_RoPE(embedding_dim=48, pos_dim=2, heads=H)
```

### Quaternion RoPE family

Per-head learned rotation axes represented as quaternions / bivectors.

| Class | `D` constraint | Axes |
|---|---|---|
| `QuatRo` | `D % 3 == 0` | fully learned (magnitude + direction) |
| `Spherical QuatRo` | `D % 3 == 0` | fixed orthogonal axes (e12, e31) |
| `Mixed QuatRo` | `D % 3 == 0` | fixed co-planar axes (e12, e12) |

```python
QuatRo = PEs['QuatRo']
Spherical_QuatRo = PEs['Spherical QuatRo']
Mixed_QuatRo = PEs['Mixed QuatRo']

m1 = QuatRo(embedding_dim=48, pos_dim=2, heads=H)
m2 = Spherical_QuatRo(embedding_dim=48, pos_dim=2, heads=H)
m3 = Mixed_QuatRo(embedding_dim=48, pos_dim=2, heads=H)
```

### CARE — Clifford Algebra Rotary Embeddings

Channels are grouped into full Cl(3,0) multivectors (8-D) and rotated via the sandwich product `R·x·R⁻¹`.

| Class | `D` constraint | Axes |
|---|---|---|
| `CARE` | `D % 8 == 0` | learned 3-vector axes |
| `Spherical CARE` | `D % 3 == 0` | fixed orthogonal bivectors (e12, e31) |
| `Mixed CARE` | `D % 3 == 0` | fixed co-planar bivectors (e12, e12) |
| `Quaternion CARE` | `D % 3 == 0` | jointly learned magnitude + axis |

```python

CARE = PEs['CARE']
Quaternion_CARE = PEs['Quaternion CARE']
Spherical_CARE = PEs['Spherical CARE']
Mixed_CARE = PEs['Mixed CARE']

# Full multivector CARE — D must be divisible by 8
m1 = CARE(embedding_dim=64, pos_dim=2, heads=H)

# 3-vector variants — D must be divisible by 3
m2 = Spherical_CARE(embedding_dim=48, pos_dim=2, heads=H)
m3 = Mixed_CARE(embedding_dim=48, pos_dim=2, heads=H)
m4 = Quaternion_CARE(embedding_dim=48, pos_dim=2, heads=H)

out = m1(x, pos)   # [B, H, P, D]
```

### No PE

Identity pass-through — useful as a baseline.

```python
No_PE = 

m = No_PE(embedding_dim=D, P_x=16, P_y=16)
out = m(x)   # returns x unchanged
```

---

## Registry

All classes auto-register into `PE_REGISTRY` at import time.
Import `Positional_Embeddings` to populate the registry, then look up classes by name:

```python
import Positional_Embeddings                 # triggers auto-import of all modules
from PE_registry import PE_REGISTRY

print(list(PE_REGISTRY.keys()))              # all registered names

model = cls(embedding_dim=48, pos_dim=2, heads=8)
```

---

## Pre-caching positions

Pass positions at construction time to avoid recomputing rotations every forward call:

```python
pos = torch.randn(256, 2)
model = CARE(embedding_dim=48, pos_dim=2, heads=8, positions=pos)

x = torch.randn(4, 8, 256, 48)
out = model(x)          # pos already cached; no need to pass it again
out = model(x, pos)     # also fine — same result
```

---

## Speed tests

```bash
# Benchmark all PE methods (latency + sequence/dim/batch sweeps)
python speed_test_pe.py

# Benchmark Cl(3) geometric product and sandwich product
python speed_test.py
```

---

## Visualisation

`rope_visualization.ipynb` contains interactive plots showing how each method rotates feature vectors as a function of 2-D position, broken down by rotation dimension (2-D pairs, 3-vectors, 8-D multivectors).

```bash
jupyter notebook rope_visualization.ipynb
```

---

## Citation

If you use this code, please cite the relevant papers.

For **CARE** and **QuatRo**, please cite : 
```bibtex
@inproceedings{sriram2025care,
  title     = {Clifford Algebraic Rotor Embeddings: Maybe embeddings should start to {CARE}},
  author    = {Sriram, Sameeksha and Paliwal, Ayush and Ecker, Alexander S. and van de Geijn, Chase},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
}
```

For **Spherical RoPE**, please cite :
```bibtex
@inproceedings{vandegeijn2025circular,
  title     = {A Circular Argument: Does {RoPE} need to be Equivariant for Vision?},
  author    = {van de Geijn, Chase and L\"{u}ddecke, Timo and Turishcheva, Polina and Ecker, Alexander S.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
}
```

For **Mixed RoPE** and **Axial RoPE**, please cite:
```bibtex
@inproceedings{heo2024ropevit,
  title     = {Rotary Position Embedding for Vision Transformer},
  author    = {Heo, Byeongho and Park, Song and Han, Dongyoon and Yun, Sangdoo},
  booktitle = {European Conference on Computer Vision},
  year      = {2024},
}
```

