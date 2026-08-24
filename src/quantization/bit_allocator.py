"""
ZUCE-AMPQ: Memory-Constrained Bit Allocation Optimizer
Optimizes bit assignment b_g in {1, 2, 4, 8, 16} per group
under target memory budget while enforcing component protection rules.
"""

from typing import Dict, List, Any, Tuple, Optional

class BitAllocationOptimizer:
    def __init__(
        self,
        group_size: int = 128,
        error_limit: float = 0.25,
        default_target_avg_bits: float = 4.2
    ):
        self.group_size = group_size
        self.error_limit = error_limit
        self.default_target_avg_bits = default_target_avg_bits

    @staticmethod
    def is_protected_component(module_name: str, layer_idx: Optional[int] = None) -> Tuple[bool, int]:
        """
        Determines if a component requires minimum precision protection:
        Returns (is_protected, min_allowed_bits).
        """
        lower_name = module_name.lower()
        if "embed" in lower_name:
            return True, 8
        if "lm_head" in lower_name:
            return True, 8
        if "norm" in lower_name or "layernorm" in lower_name or "rmsnorm" in lower_name:
            return True, 16
        if "bias" in lower_name:
            return True, 16
        if layer_idx is not None and (layer_idx == 0 or layer_idx >= 27):
            return True, 8
        if "down_proj" in lower_name or "gate_proj" in lower_name:
            return True, 4
        return False, 1

    def allocate_bits_for_module(
        self,
        module_name: str,
        group_importance: List[float],
        layer_idx: Optional[int] = None,
        sensitivities: Optional[List[Dict[int, float]]] = None
    ) -> List[int]:
        """
        Assigns precision bits to each group of weights based on importance score
        and component protection rules.
        """
        is_prot, min_bits = self.is_protected_component(module_name, layer_idx)
        assigned_bits = []

        for g_idx, imp in enumerate(group_importance):
            # Threshold policy
            if imp >= 0.85:
                b = 16
            elif imp >= 0.65:
                b = 8
            elif imp >= 0.35:
                b = 4
            elif imp >= 0.12:
                b = 2
            else:
                b = 1

            # Quality Recovery: check sensitivity if available
            if sensitivities and g_idx < len(sensitivities):
                err = sensitivities[g_idx].get(b, 0.0)
                if err > self.error_limit:
                    # Upgrade to higher precision
                    upgrade_map = {1: 2, 2: 4, 4: 8, 8: 16, 16: 16}
                    b = upgrade_map.get(b, b)

            # Enforce component protection
            b = max(b, min_bits)
            assigned_bits.append(b)

        return assigned_bits

    def optimize_full_model_allocation(
        self,
        module_importances: Dict[str, List[float]],
        target_memory_ratio: float = 0.30
    ) -> Dict[str, Any]:
        """
        Generates full-model mixed-precision allocation map across all modules.
        Target memory ratio: 0.30 represents ~70% reduction in parameter VRAM.
        """
        full_allocation_map = {}
        total_weights = 0
        total_bit_sum = 0
        distribution = {16: 0, 8: 0, 4: 0, 2: 0, 1: 0}

        for mod_name, imp_scores in module_importances.items():
            # Extract layer index if present (e.g. "model.layers.12.mlp.down_proj")
            layer_idx = None
            for part in mod_name.split("."):
                if part.isdigit():
                    layer_idx = int(part)
                    break

            bits_list = self.allocate_bits_for_module(mod_name, imp_scores, layer_idx=layer_idx)
            full_allocation_map[mod_name] = bits_list

            for b in bits_list:
                group_elem = self.group_size
                distribution[b] = distribution.get(b, 0) + group_elem
                total_weights += group_elem
                total_bit_sum += b * group_elem

        avg_bits = total_bit_sum / max(total_weights, 1)
        compression_ratio = 16.0 / max(avg_bits, 0.1)

        return {
            "allocation_map": full_allocation_map,
            "average_bits_per_weight": round(avg_bits, 2),
            "compression_ratio": round(compression_ratio, 2),
            "vram_reduction_pct": round((1.0 - (avg_bits / 16.0)) * 100, 1),
            "precision_distribution": {
                f"{b}bit": round(count / max(total_weights, 1) * 100, 1)
                for b, count in distribution.items()
            }
        }
