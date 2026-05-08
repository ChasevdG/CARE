import torch
from ..PE_registry import PE_REGISTRY, register_PE

class No_PE(torch.nn.Module):
  def __init__(self, embedding_dim=None, positions=None, pos_dim=None, heads=12, max_freq=100.0, initialization="random", init_scale=1.0):
    super(No_PE, self).__init__()

  def forward(self, x, pos=None):
    return x
register_PE("NoPE", No_PE)