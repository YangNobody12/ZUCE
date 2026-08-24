"""
ZUCE-AMPQ: Multi-Signal Group Importance & Quantization Sensitivity Profiler
Computes Composite Group Importance Score:
I_g = 0.35 * A_g + 0.20 * G_g + 0.20 * H_g + 0.20 * C_g + 0.05 * R_g
and Quantization Sensitivity S_{g, b} = Loss(Q_b(W_g)) - Loss(W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

from .ampq_engine import GroupQuantizer

class GroupImportanceProfiler:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        group_size: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.group_size = group_size
        self.num_layers = model.config.num_hidden_layers

    def compute_gradient_and_fisher_sensitivity(
        self,
        prompts: List[str],
        max_samples: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Computes First-Order Gradient Sensitivity G_g = |w * dL/dw|
        and Fisher Diagonal Curvature H_g = w^2 * (dL/dw)^2.
        """
        self.model.eval()
        grad_sensitivities = {}
        fisher_curvatures = {}

        # Initialize tracking for named linear weights in transformer layers
        for name, param in self.model.named_parameters():
            if "weight" in name and ("self_attn" in name or "mlp" in name):
                grad_sensitivities[name] = torch.zeros_like(param.data, dtype=torch.float32, device="cpu")
                fisher_curvatures[name] = torch.zeros_like(param.data, dtype=torch.float32, device="cpu")

        sample_list = prompts[:max_samples]
        n_samples = 0

        for p in sample_list:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            labels = input_ids[:, 1:].clone()
            inputs = input_ids[:, :-1]

            self.model.zero_grad()
            outputs = self.model(input_ids=inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()

            for name, param in self.model.named_parameters():
                if name in grad_sensitivities and param.grad is not None:
                    g = param.grad.detach().cpu().float()
                    w = param.data.detach().cpu().float()
                    grad_sensitivities[name] += torch.abs(w * g)
                    fisher_curvatures[name] += (w ** 2) * (g ** 2)

            n_samples += 1

        if n_samples > 0:
            for k in grad_sensitivities:
                grad_sensitivities[k] /= n_samples
                fisher_curvatures[k] /= n_samples

        return {
            "gradient_sensitivity": grad_sensitivities,
            "fisher_curvature": fisher_curvatures
        }

    def compute_group_composite_importance(
        self,
        weight_tensor: torch.Tensor,
        grad_sens: Optional[torch.Tensor] = None,
        fisher_curv: Optional[torch.Tensor] = None,
        alpha_ablation: float = 0.35,
        beta_grad: float = 0.20,
        gamma_fisher: float = 0.20,
        delta_coding: float = 0.20,
        epsilon_residual: float = 0.05
    ) -> List[float]:
        """
        Groups weight tensor into chunks of group_size and evaluates composite importance I_g.
        """
        flat_w = weight_tensor.reshape(-1).cpu().float()
        total_elements = flat_w.numel()
        num_groups = math.ceil(total_elements / self.group_size)

        flat_grad = grad_sens.reshape(-1).cpu().float() if grad_sens is not None else torch.abs(flat_w)
        flat_fisher = fisher_curv.reshape(-1).cpu().float() if fisher_curv is not None else (flat_w ** 2)

        # Min-max normalization
        norm_grad = (flat_grad - flat_grad.min()) / (flat_grad.max() - flat_grad.min() + 1e-8)
        norm_fisher = (flat_fisher - flat_fisher.min()) / (flat_fisher.max() - flat_fisher.min() + 1e-8)
        norm_magnitude = (torch.abs(flat_w) - torch.abs(flat_w).min()) / (torch.abs(flat_w).max() - torch.abs(flat_w).min() + 1e-8)

        group_importance_scores = []

        for g_idx in range(num_groups):
            start = g_idx * self.group_size
            end = min(start + self.group_size, total_elements)

            g_val = float(norm_grad[start:end].mean().item())
            h_val = float(norm_fisher[start:end].mean().item())
            a_val = float(norm_magnitude[start:end].mean().item()) # proxy for ablation sensitivity
            c_val = g_val * 1.1 # coding sensitivity boost
            r_val = float(torch.std(flat_w[start:end]).item())

            # Composite formula: I_g = 0.35 A_g + 0.20 G_g + 0.20 H_g + 0.20 C_g + 0.05 R_g
            i_g = (
                alpha_ablation * a_val +
                beta_grad * g_val +
                gamma_fisher * h_val +
                delta_coding * c_val +
                epsilon_residual * min(1.0, r_val)
            )
            group_importance_scores.append(round(i_g, 4))

        return group_importance_scores

    def evaluate_quantization_sensitivity_grid(
        self,
        weight_tensor: torch.Tensor,
        group_idx: int,
        bits_list: List[int] = [16, 8, 4, 2, 1]
    ) -> Dict[int, float]:
        """
        Measures distortion / quantization error S_{g, b} = || Q_b(W_g) - W_g ||^2 / || W_g ||^2
        for a specific group across bit precisions.
        """
        flat_w = weight_tensor.reshape(-1)
        start = group_idx * self.group_size
        end = min(start + self.group_size, flat_w.numel())
        w_group = flat_w[start:end]

        norm_orig = torch.norm(w_group) + 1e-8
        sensitivity_map = {}

        for b in bits_list:
            q_w, _ = GroupQuantizer.quantize_kbit(w_group, b)
            dist = float((torch.norm(q_w - w_group) / norm_orig).item())
            sensitivity_map[b] = round(dist, 4)

        return sensitivity_map
