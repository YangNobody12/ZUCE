"""
ZUCE-Fusion: Dynamic Top-1 / Top-2 Capability Router
Computes routing coefficients r(x) = softmax(W_r * h_x)
and activates only the top-k relevant capability adapters per prompt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional

EXPERT_NAMES = ["coding_expert", "reasoning_expert", "language_thai_expert", "general_instruction_expert"]

class DynamicCapabilityRouter(nn.Module):
    def __init__(self, hidden_dim: int = 1536, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_names = EXPERT_NAMES[:num_experts]
        
        # Router linear gate
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.orthogonal_(self.gate.weight)

    def calibrate_with_domain_anchors(self, domain_embeddings: Dict[str, torch.Tensor]):
        """
        Calibrates router projection weights with domain prototypical representations:
        W_r[i, :] = E[h_{domain_i}] / ||E[h_{domain_i}]||
        """
        with torch.no_grad():
            for i, name in enumerate(self.expert_names):
                if name in domain_embeddings:
                    proto = domain_embeddings[name].to(self.gate.weight.device, dtype=self.gate.weight.dtype)
                    proto_norm = proto / (torch.norm(proto) + 1e-8)
                    self.gate.weight.data[i, :] = proto_norm * 3.5

    def forward(self, h_x: torch.Tensor, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Computes dynamic routing probabilities and active expert weights.
        h_x: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        """
        k = top_k if top_k is not None else self.top_k
        
        # Aggregate token representations if sequence given
        if h_x.ndim == 3:
            h_pooled = h_x.mean(dim=1)
        else:
            h_pooled = h_x

        # Auto-align device & dtype with incoming tensor
        if self.gate.weight.device != h_pooled.device or self.gate.weight.dtype != h_pooled.dtype:
            self.gate = self.gate.to(device=h_pooled.device, dtype=h_pooled.dtype)

        logits = self.gate(h_pooled) # [batch_size, num_experts]
        probs = F.softmax(logits, dim=-1) # [batch_size, num_experts]

        # Top-K selection
        top_k_vals, top_k_indices = torch.topk(probs, k, dim=-1)
        # Normalize top-k weights so sum = 1.0
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        batch_size = h_pooled.shape[0]
        routing_results = []

        for b in range(batch_size):
            active_experts = {}
            for idx, w in zip(top_k_indices[b].tolist(), top_k_weights[b].tolist()):
                name = self.expert_names[idx]
                active_experts[name] = round(float(w), 4)

            full_distribution = {
                self.expert_names[i]: round(float(probs[b, i].item()), 4)
                for i in range(self.num_experts)
            }

            routing_results.append({
                "active_experts": active_experts,
                "full_distribution": full_distribution,
                "primary_expert": self.expert_names[top_k_indices[b, 0].item()]
            })

        return {
            "probs": probs,
            "top_k_weights": top_k_weights,
            "top_k_indices": top_k_indices,
            "routing_summary": routing_results[0] if batch_size == 1 else routing_results
        }

    def route_text_prompt(self, prompt: str, tokenizer: Any, backbone_model: nn.Module) -> Dict[str, Any]:
        """Classifies a textual prompt and returns expert routing decisions."""
        device = next(backbone_model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = backbone_model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            return self.forward(last_hidden.to(self.gate.weight.device))
