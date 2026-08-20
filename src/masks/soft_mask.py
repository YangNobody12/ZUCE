"""
Phase 5A: Soft Capability Mask
Maintains continuous differentiable masks M_i in [0, 1] across layers, heads, and neurons.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class SoftCapabilityMask(nn.Module):
    def __init__(self, num_layers: int, intermediate_size: int, init_val: float = 0.8):
        super().__init__()
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size

        # Unconstrained parameter mapped to [0, 1] via sigmoid
        init_logit = torch.logit(torch.tensor(init_val, dtype=torch.float32))
        self.mask_logits = nn.Parameter(torch.full((num_layers, intermediate_size), init_logit))

    def forward(self) -> torch.Tensor:
        """Returns continuous mask M in [0, 1]."""
        return torch.sigmoid(self.mask_logits)

    def binarize(self, threshold: float = 0.5) -> torch.Tensor:
        """Converts soft mask to binary mask: 1(M_i > threshold)."""
        with torch.no_grad():
            return (self.forward() > threshold).float()
