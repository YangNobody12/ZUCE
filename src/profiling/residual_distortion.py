"""
Residual Distortion & Directional Alignment Profiler
Calculates:
1. Energy Retention: E_l = ||m'_l|| / ||m_l||
2. Direction Retention: C_l = cos(m_l, m'_l) = (m_l^T m'_l) / (||m_l|| * ||m'_l||)
3. Residual Drift: D_l = ||h_l - \hat{h}_l|| / ||h_l||
Across all decoder layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple

class ResidualDistortionProfiler:
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.teacher.eval()
        self.student.eval()

    def profile_residual_distortion(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Profiles Energy Retention, Cosine Similarity, and Residual Drift layer-by-layer.
        """
        t_layers = self.teacher.model.layers
        s_layers = self.student.model.layers
        num_s_layers = len(s_layers)

        # Collect activations across layers
        energy_retentions = [0.0] * num_s_layers
        cosine_directions = [0.0] * num_s_layers
        residual_drifts = [0.0] * num_s_layers
        n_samples = 0

        for p in prompts[:10]:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                t_out = self.teacher(input_ids=input_ids, output_hidden_states=True)
                s_out = self.student(input_ids=input_ids, output_hidden_states=True)

            t_hidden = t_out.hidden_states # Tuple of (num_layers + 1)
            s_hidden = s_out.hidden_states

            for l_idx in range(num_s_layers):
                # Hidden states at layer l
                h_t = t_hidden[l_idx + 1].float()
                h_s = s_hidden[l_idx + 1].float()

                # Residual Drift D_l = ||h_l - \hat{h}_l|| / ||h_l||
                drift = torch.norm(h_t - h_s, dim=-1) / (torch.norm(h_t, dim=-1) + 1e-8)
                residual_drifts[l_idx] += float(drift.mean().item())

                # MLP output contributions m_l and m'_l
                m_t = t_layers[l_idx].mlp(t_layers[l_idx].post_attention_layernorm(t_hidden[l_idx])).float()
                m_s = s_layers[l_idx].mlp(s_layers[l_idx].post_attention_layernorm(s_hidden[l_idx])).float()

                norm_t = torch.norm(m_t, dim=-1)
                norm_s = torch.norm(m_s, dim=-1)

                # Energy Retention E_l = ||m'_l|| / ||m_l||
                energy = norm_s / (norm_t + 1e-8)
                energy_retentions[l_idx] += float(energy.mean().item())

                # Direction Retention C_l = cos(m_l, m'_l)
                dot = (m_t * m_s).sum(dim=-1)
                cos = dot / (norm_t * norm_s + 1e-8)
                cosine_directions[l_idx] += float(cos.mean().item())

            n_samples += 1

        results_table = []
        for l_idx in range(num_s_layers):
            results_table.append({
                "layer": l_idx,
                "energy_retention": round(energy_retentions[l_idx] / max(n_samples, 1), 3),
                "cosine_direction": round(cosine_directions[l_idx] / max(n_samples, 1), 3),
                "residual_drift": round(residual_drifts[l_idx] / max(n_samples, 1), 3)
            })

        return results_table
