"""
Weight Indexing and Slicing Utilities for Model Surgery.
Handles tensor dimension slicing for MLP gate/up/down projections and multi-head attention.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple

def slice_mlp_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    retained_neuron_indices: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slices MLP projection weight matrices according to retained neuron indices.

    Shapes:
        gate_proj.weight: [intermediate_size, hidden_size] -> sliced on dim 0
        up_proj.weight:   [intermediate_size, hidden_size] -> sliced on dim 0
        down_proj.weight: [hidden_size, intermediate_size] -> sliced on dim 1
    """
    indices = torch.tensor(retained_neuron_indices, dtype=torch.long, device=gate_weight.device)
    
    new_gate = torch.index_select(gate_weight, dim=0, index=indices)
    new_up = torch.index_select(up_weight, dim=0, index=indices)
    new_down = torch.index_select(down_weight, dim=1, index=indices)
    
    return new_gate, new_up, new_down

def slice_attention_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    retained_head_indices: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slices Self-Attention projection weights when reducing number of attention heads.
    """
    device = q_weight.device
    hidden_size = q_weight.shape[1]

    # Reshape Q: [num_heads, head_dim, hidden_size]
    q_reshaped = q_weight.view(num_heads, head_dim, hidden_size)
    head_idx_tensor = torch.tensor(retained_head_indices, dtype=torch.long, device=device)
    new_q = torch.index_select(q_reshaped, dim=0, index=head_idx_tensor)
    new_q = new_q.view(-1, hidden_size)

    # Handle KV heads (respecting GQA group ratio)
    kv_group_ratio = max(1, num_heads // num_kv_heads)
    retained_kv_indices = sorted(list(set([h // kv_group_ratio for h in retained_head_indices])))
    kv_idx_tensor = torch.tensor(retained_kv_indices, dtype=torch.long, device=device)

    k_reshaped = k_weight.view(num_kv_heads, head_dim, hidden_size)
    v_reshaped = v_weight.view(num_kv_heads, head_dim, hidden_size)
    new_k = torch.index_select(k_reshaped, dim=0, index=kv_idx_tensor).view(-1, hidden_size)
    new_v = torch.index_select(v_reshaped, dim=0, index=kv_idx_tensor).view(-1, hidden_size)

    # Reshape O: [hidden_size, num_heads, head_dim]
    o_reshaped = o_weight.view(hidden_size, num_heads, head_dim)
    new_o = torch.index_select(o_reshaped, dim=1, index=head_idx_tensor)
    new_o = new_o.view(hidden_size, -1)

    return new_q, new_k, new_v, new_o
