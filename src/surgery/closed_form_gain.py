"""
Closed-Form Gain Calculator & Calibrator
Computes exact analytical gains without SGD, backpropagation, or optimizer steps:
1. Layer-wise Scalar Gain: g_l^* = E[y_P^T y_T] / E[||y_P||^2]
2. Channel-wise Diagonal Gain: g_{l, j}^* = (sum y_{P, j} y_{T, j}) / (sum y_{P, j}^2 + eps)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple

class ClosedFormGainCalibrator:
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.teacher.eval()
        self.student.eval()

    def compute_layerwise_scalar_gains(self, prompts: List[str]) -> List[float]:
        """
        Calculates g_l^* = E[y_P^T y_T] / E[||y_P||^2] for each layer.
        """
        t_layers = self.teacher.model.layers
        s_layers = self.student.model.layers
        num_s_layers = len(s_layers)

        numerator = [0.0] * num_s_layers
        denominator = [0.0] * num_s_layers

        for p in prompts[:15]:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                t_out = self.teacher(input_ids=input_ids, output_hidden_states=True)
                s_out = self.student(input_ids=input_ids, output_hidden_states=True)

            t_hidden = t_out.hidden_states
            s_hidden = s_out.hidden_states

            for l_idx in range(num_s_layers):
                y_T = t_layers[l_idx].mlp(t_layers[l_idx].post_attention_layernorm(t_hidden[l_idx])).float()
                y_P = s_layers[l_idx].mlp(s_layers[l_idx].post_attention_layernorm(s_hidden[l_idx])).float()

                # Dot product y_P^T y_T summed over all tokens
                dot = (y_P * y_T).sum().item()
                # Norm squared ||y_P||^2 summed over all tokens
                norm_sq = (y_P * y_P).sum().item()

                numerator[l_idx] += dot
                denominator[l_idx] += norm_sq

        scalar_gains = []
        for l_idx in range(num_s_layers):
            g = numerator[l_idx] / max(denominator[l_idx], 1e-8)
            scalar_gains.append(float(g))

        return scalar_gains

    def compute_channelwise_diagonal_gains(self, prompts: List[str]) -> List[torch.Tensor]:
        """
        Calculates diagonal vector g_{l, j}^* = sum(y_{P, j} y_{T, j}) / sum(y_{P, j}^2 + eps).
        """
        t_layers = self.teacher.model.layers
        s_layers = self.student.model.layers
        num_s_layers = len(s_layers)
        d_model = self.student.config.hidden_size

        numerators = [torch.zeros(d_model, device=self.device) for _ in range(num_s_layers)]
        denominators = [torch.zeros(d_model, device=self.device) for _ in range(num_s_layers)]

        for p in prompts[:15]:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                t_out = self.teacher(input_ids=input_ids, output_hidden_states=True)
                s_out = self.student(input_ids=input_ids, output_hidden_states=True)

            t_hidden = t_out.hidden_states
            s_hidden = s_out.hidden_states

            for l_idx in range(num_s_layers):
                y_T = t_layers[l_idx].mlp(t_layers[l_idx].post_attention_layernorm(t_hidden[l_idx])).float() # [B, S, D]
                y_P = s_layers[l_idx].mlp(s_layers[l_idx].post_attention_layernorm(s_hidden[l_idx])).float()

                # Sum along batch and sequence dimensions -> [D]
                dot_d = (y_P * y_T).sum(dim=(0, 1))
                sq_d = (y_P * y_P).sum(dim=(0, 1))

                numerators[l_idx] += dot_d
                denominators[l_idx] += sq_d

        channel_gains = []
        for l_idx in range(num_s_layers):
            g_vec = numerators[l_idx] / (denominators[l_idx] + 1e-8)
            channel_gains.append(g_vec.detach().cpu())

        return channel_gains

    @staticmethod
    def apply_scalar_gains_to_model(model: nn.Module, gains: List[float]) -> nn.Module:
        """Applies scalar gain g_l to W_down in each layer."""
        calib_model = copy.deepcopy(model)
        for l_idx, g in enumerate(gains):
            calib_model.model.layers[l_idx].mlp.down_proj.weight.data *= float(g)
        return calib_model

    @staticmethod
    def apply_channel_gains_to_model(model: nn.Module, channel_gains: List[torch.Tensor]) -> nn.Module:
        """Applies diagonal vector g_{l, j} to row j of W_down in each layer."""
        calib_model = copy.deepcopy(model)
        for l_idx, g_vec in enumerate(channel_gains):
            # W_down is [hidden_size, intermediate_size]
            # multiplying each output row j by g_vec[j]
            weight = calib_model.model.layers[l_idx].mlp.down_proj.weight.data
            g_tensor = g_vec.to(device=weight.device, dtype=weight.dtype).unsqueeze(1) # [D, 1]
            calib_model.model.layers[l_idx].mlp.down_proj.weight.data = weight * g_tensor
        return calib_model
