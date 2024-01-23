import torch
import random 
import numpy as np
 
def input_mapping(x1:torch.Tensor, x2:torch.Tensor):

    # initialy no bias relevance, no x2 relevance in x1 and no x2 relevance in x1
    x1 = torch.stack([torch.zeros_like(x1),     # bias relevance
                      x1,                       # Source 1 relevance
                      torch.zeros_like(x1)      # Source 2 relevance
                      ], 0)
    x2 = torch.stack([torch.zeros_like(x2),     # bias relevance
                      torch.zeros_like(x2),     # Source 1 relevance
                      x2                        # Source 2 relevance
                      ], 0)

    x = [x1,x2]
    return x


def set_seed(seed:int):
  np.random.seed(seed=seed)
  torch.manual_seed(seed=seed)
  random.seed(a=seed)