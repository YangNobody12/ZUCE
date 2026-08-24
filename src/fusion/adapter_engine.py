"""
ZUCE-Fusion: Capability Adapters & Fused Backbone Integration
Maintains low-rank capability adapters (Coding, Reasoning, Language/Thai)
quantized with AMPQ mixed precision and blended dynamically via Router weights:
h_fused = h_backbone + sum_{e in TopK(r(x))} r_e(x) * A_e(h_backbone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional

from ..quantization.ampq_engine import GroupQuantizer
from .capability_router import DynamicCapabilityRouter

class CapabilityAdapter(nn.Module):
    """
    Compact Bottleneck Capability Adapter:
    A_e(x) = W_up * SiLU(W_down * x) * scaling
    Supports AMPQ mixed-precision quantization.
    """
    def __init__(self, hidden_dim: int = 1536, bottleneck_dim: int = 128, name: str = "adapter"):
        super().__init__()
        self.name = name
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.scaling = 1.0 / math.sqrt(bottleneck_dim) if bottleneck_dim > 0 else 1.0
        self.precision_bits = 8 # Default INT8

        # Initialize with near-zero up projection for identity initial state
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.down_proj(x))
        return self.up_proj(h) * self.scaling

    def quantize_adapter(self, target_bits: int = 4, group_size: int = 64) -> Dict[str, Any]:
        """Quantizes adapter weights using GroupQuantizer."""
        self.precision_bits = target_bits
        q_down, meta_d = GroupQuantizer.quantize_tensor_with_group_map(
            self.down_proj.weight.data,
            [target_bits] * 1000,
            group_size=group_size
        )
        q_up, meta_u = GroupQuantizer.quantize_tensor_with_group_map(
            self.up_proj.weight.data,
            [target_bits] * 1000,
            group_size=group_size
        )
        self.down_proj.weight.data.copy_(q_down)
        self.up_proj.weight.data.copy_(q_up)
        return {
            "name": self.name,
            "precision_bits": target_bits,
            "down_compression": meta_d["compression_ratio"],
            "up_compression": meta_u["compression_ratio"]
        }

import math

class ZUCEFusionModel(nn.Module):
    """
    Complete ZUCE-Fusion System:
    Combines Compact Shared Backbone (INT4/AMPQ) + Dynamic Router (BF16)
    + Capability Adapters (Coding, Reasoning, Language/Thai).
    """
    def __init__(
        self,
        backbone_model: nn.Module,
        hidden_dim: int = 1536,
        adapter_rank: int = 128,
        top_k: int = 2
    ):
        super().__init__()
        self.backbone = backbone_model
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.router = DynamicCapabilityRouter(hidden_dim=hidden_dim, top_k=top_k)
        
        # Capability Adapter Registry
        self.adapters = nn.ModuleDict({
            "coding_expert": CapabilityAdapter(hidden_dim, adapter_rank, name="qwen_coding"),
            "reasoning_expert": CapabilityAdapter(hidden_dim, adapter_rank, name="qwen3_reasoning"),
            "language_thai_expert": CapabilityAdapter(hidden_dim, adapter_rank, name="thai_language"),
            "general_instruction_expert": CapabilityAdapter(hidden_dim, adapter_rank, name="general_instruct")
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes unified forward pass with dynamic capability adapter blending.
        """
        # 1. Forward through backbone to get intermediate hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1] # [B, S, D]
        
        # 2. Dynamic Capability Routing
        route_info = self.router(hidden_states, top_k=self.top_k)
        top_k_indices = route_info["top_k_indices"] # [B, K]
        top_k_weights = route_info["top_k_weights"] # [B, K]
        
        # 3. Blend active adapters
        adapter_output = torch.zeros_like(hidden_states)
        batch_size = hidden_states.shape[0]
        
        for b in range(batch_size):
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[b, k_idx].item()
                expert_weight = top_k_weights[b, k_idx]
                expert_name = self.router.expert_names[expert_idx]
                
                if expert_name in self.adapters:
                    act_out = self.adapters[expert_name](hidden_states[b:b+1])
                    adapter_output[b:b+1] += expert_weight * act_out

        # 4. Final output projection via LM Head
        fused_hidden = hidden_states + adapter_output
        normed_hidden = self.backbone.model.norm(fused_hidden) if hasattr(self.backbone.model, "norm") else fused_hidden
        logits = self.backbone.lm_head(normed_hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "routing_info": route_info["routing_summary"]
        }
