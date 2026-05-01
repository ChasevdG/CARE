import torch
from PE_registry import PE_REGISTRY, register_PE

class No_PE(torch.nn.Module):
  def __init__(self, embedding_dim, P_x, P_y):
    super(No_PE, self).__init__()

  def forward(self, x):
    return x
  
  def _set_device_(self,device):
    pass
  def extrapolate_mode(self):
    pass
  def train_mode(self):
    pass
  def set_Patches(self, P_x, P_y):
    pass
  
register_PE(
      "NoPE",
      {"PE_method" : No_PE, 
        "stem_only" : False, 
        "shared_pe" : False, 
        "rot_x" : False,
        "rot_value" : False
        }
  )