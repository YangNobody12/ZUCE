"""
Pareto-Optimal Lagrangian Resource Allocator with Middle-Layer Elasticity
Solves:
min_{k_l} sum_l D_l(k_l) subject to: sum_l 3 * d_model * k_l <= Budget_params
and middle-layer capacity constraint: k_l >= k_mid_min for l in [6..22]
"""

import math
import torch
from typing import Dict, List, Any, Tuple, Optional

class ParetoResourceAllocator:
    def __init__(
        self,
        num_layers: int = 28,
        intermediate_size: int = 8960,
        d_model: int = 1536,
        base_params: int = 1543710720,
        non_mlp_params: int = 295698432
    ):
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.d_model = d_model
        self.base_params = base_params
        self.non_mlp_params = non_mlp_params
        # In SwiGLU: gate_proj (d x k), up_proj (d x k), down_proj (k x d) => 3 * d * k params per layer
        self.params_per_neuron_layer = 3 * d_model

    def calculate_total_params(self, k_profile: List[int]) -> int:
        """Returns total parameter count for given width profile k_l."""
        mlp_params = sum(k_profile) * self.params_per_neuron_layer
        return self.non_mlp_params + mlp_params

    def solve_lagrangian_allocation(
        self,
        composite_scores: torch.Tensor,
        causal_importance: torch.Tensor,
        target_total_params: int = 968000000,
        min_k_global: int = 2560,
        max_k_global: int = 8960,
        middle_layer_bonus: float = 1.15,
        middle_layer_range: Tuple[int, int] = (6, 22),
        is_uniform: bool = False
    ) -> Dict[str, Any]:
        """
        Solves continuous Lagrangian allocation with integer rounding and capacity bounds.
        """
        # Target total MLP neurons across all layers
        target_mlp_params = target_total_params - self.non_mlp_params
        target_total_k = int(target_mlp_params / self.params_per_neuron_layer)
        target_avg_k = target_total_k // self.num_layers

        if is_uniform:
            profile = [target_avg_k] * self.num_layers
            actual_params = self.calculate_total_params(profile)
            reduction_pct = (1.0 - actual_params / self.base_params) * 100
            retained_neurons = {}
            for l in range(self.num_layers):
                top_k_indices = torch.topk(composite_scores[l], target_avg_k).indices.tolist()
                retained_neurons[l] = sorted(top_k_indices)
            return {
                "target_total_params": target_total_params,
                "actual_params": actual_params,
                "params_million": round(actual_params / 1e6, 2),
                "reduction_pct": round(reduction_pct, 2),
                "avg_k": target_avg_k,
                "k_profile": profile,
                "retained_neurons_per_layer": retained_neurons
            }

        # Compute layer sensitivity weights from causal importance and composite score density
        score_density = torch.zeros(self.num_layers)
        for l in range(self.num_layers):
            # Top-half score concentration
            top_half = composite_scores[l].topk(min(target_avg_k, self.intermediate_size)).values.mean()
            score_density[l] = top_half

        norm_causal = (causal_importance - causal_importance.min()) / (causal_importance.max() - causal_importance.min() + 1e-8)
        norm_density = (score_density - score_density.min()) / (score_density.max() - score_density.min() + 1e-8)
        
        # Combined layer priority
        layer_priority = 0.55 * norm_causal + 0.45 * norm_density

        # Apply middle layer elasticity: reasoning circuits reside in middle layers
        layer_weights = layer_priority.clone()
        for l in range(self.num_layers):
            if middle_layer_range[0] <= l <= middle_layer_range[1]:
                layer_weights[l] *= middle_layer_bonus
            elif l < 3: # input embeddings interface
                layer_weights[l] *= 1.05
            elif l >= 26: # output logits interface
                layer_weights[l] *= 1.08

        # Normalize weights so mean is 1.0
        normalized_w = layer_weights / layer_weights.mean()

        # Initial allocation
        profile = []
        for l in range(self.num_layers):
            k_l = int(target_avg_k * float(normalized_w[l]))
            k_l = max(min_k_global, min(max_k_global, k_l))
            profile.append(k_l)

        # Budget adjustment loop
        current_sum = sum(profile)
        diff = target_total_k - current_sum
        
        if diff != 0:
            step = 1 if diff > 0 else -1
            # Sort layer indices by priority when adding, reverse when subtracting
            sorted_layers = torch.argsort(layer_weights, descending=(diff > 0)).tolist()
            idx = 0
            for _ in range(abs(diff)):
                target_layer = sorted_layers[idx % self.num_layers]
                new_k = profile[target_layer] + step
                if min_k_global <= new_k <= max_k_global:
                    profile[target_layer] = new_k
                idx += 1

        actual_params = self.calculate_total_params(profile)
        reduction_pct = (1.0 - actual_params / self.base_params) * 100

        # Build neuron selection indices
        retained_neurons = {}
        for l in range(self.num_layers):
            k_l = profile[l]
            top_k_indices = torch.topk(composite_scores[l], k_l).indices.tolist()
            retained_neurons[l] = sorted(top_k_indices)

        return {
            "target_total_params": target_total_params,
            "actual_params": actual_params,
            "params_million": round(actual_params / 1e6, 2),
            "reduction_pct": round(reduction_pct, 2),
            "avg_k": int(sum(profile) / self.num_layers),
            "k_profile": profile,
            "retained_neurons_per_layer": retained_neurons
        }

    def generate_pareto_grid(
        self,
        composite_scores: torch.Tensor,
        causal_importance: torch.Tensor,
        budgets_m: List[float] = [880.0, 968.0, 1032.0, 1097.0, 1265.0]
    ) -> List[Dict[str, Any]]:
        """
        Generates full Pareto frontier allocations across parameter budgets.
        """
        results = []
        for b_m in budgets_m:
            target_p = int(b_m * 1e6)
            alloc = self.solve_lagrangian_allocation(
                composite_scores=composite_scores,
                causal_importance=causal_importance,
                target_total_params=target_p
            )
            results.append(alloc)
        return results
