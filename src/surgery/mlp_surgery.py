"""
Phase 7B: SwiGLU MLP Physical Tensor Surgery
Slices gate_proj, up_proj, and down_proj along intermediate dimension according to subset S.
Teacher: W_gate, W_up in R^{8960 x d}, W_down in R^{d x 8960}
Student: W'_gate, W'_up in R^{K x d}, W'_down in R^{d x K}
"""

import torch
from typing import Tuple, List

def slice_swiglu_mlp(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    retained_indices: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Physically extracts subset S of neurons from SwiGLU projection matrices.
    """
    device = gate_weight.device
    idx_tensor = torch.tensor(retained_indices, dtype=torch.long, device=device)

    # Slice gate_proj (dim 0: intermediate_size)
    new_gate = torch.index_select(gate_weight, dim=0, index=idx_tensor)

    # Slice up_proj (dim 0: intermediate_size)
    new_up = torch.index_select(up_weight, dim=0, index=idx_tensor)

    # Slice down_proj (dim 1: intermediate_size)
    new_down = torch.index_select(down_weight, dim=1, index=idx_tensor)

    return new_gate, new_up, new_down
