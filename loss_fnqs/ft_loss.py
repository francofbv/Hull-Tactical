import torch
from typing import Optional

def vol_penalized_sharpe(
    a: torch.Tensor, # allocations in [0, 2], shape [..., T]
    r_m: torch.tensor, # excess market returns aligned to a
    sigma_m: torch.Tensor, # market daily realized vol
    *,
    mask: Optional[torch.Tensor] = None
):

    if mask is None: mask = torch.ones_like(r_m, dtype=torch.book) # set it to vector of ones (no masking)
    mask = mask.bool()

    # excess returns of model strategy
    s = torch.where(mask, a * r_m, torch.zeros_like(r_m))

    denom = mask.sum(dim=-1).clamp_min(1) # num of valid timesteps (for normalization)
    
    if use_geom_mean:
        