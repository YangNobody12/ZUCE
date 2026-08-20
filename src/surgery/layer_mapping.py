"""
Phase 7A: Layer Mapping Optimizer
Selects strictly monotonic layer sequence l_1 < l_2 < ... < l_k that maximizes
capability score while respecting target parameter budget P_target.
"""

from typing import List, Dict, Any

class LayerMappingOptimizer:
    def __init__(self, num_total_layers: int = 28):
        self.num_total_layers = num_total_layers

    def select_monotonic_layers(
        self,
        layer_importance_scores: List[float],
        target_num_layers: int = 16
    ) -> List[int]:
        """
        Selects target_num_layers strictly ordered layers:
        - Keeps embedding boundary layer 0
        - Keeps final projection layer num_total_layers - 1
        - Selects top middle layers sorted monotonically.
        """
        if target_num_layers >= self.num_total_layers:
            return list(range(self.num_total_layers))

        fixed_early = [0]
        fixed_late = [self.num_total_layers - 1]
        n_middle_needed = target_num_layers - len(fixed_early) - len(fixed_late)

        # Score middle layers
        middle_indices = list(range(1, self.num_total_layers - 1))
        indexed_scores = [(idx, layer_importance_scores[idx]) for idx in middle_indices]
        
        # Sort by importance
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_middle = [idx for idx, score in indexed_scores[:n_middle_needed]]

        # Combine and enforce strict monotonicity l_1 < l_2 < ... < l_k
        all_retained = sorted(list(set(fixed_early + top_middle + fixed_late)))
        return all_retained
