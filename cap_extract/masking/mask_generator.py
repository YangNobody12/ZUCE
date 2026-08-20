"""
Phase 4: Expert Mask / Capability Mask Generator
Translates importance scores and circuit graphs into binary and continuous
capability masks M for layers, intermediate neurons, and attention heads.
"""

import os
import torch
from typing import Dict, Any, Optional, Tuple
from ..configs.base_config import ExtractionConfig

class MaskGenerator:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    def generate_capability_mask(
        self,
        neuron_importance_data: Dict[str, Any],
        target_domain: str = "coding",
        neuron_retention_ratio: Optional[float] = None,
        head_retention_ratio: Optional[float] = None,
        layer_retention_ratio: Optional[float] = None,
        use_selectivity: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generates binary masks M_neuron and M_head for the specified domain.

        Args:
            neuron_importance_data: output from Phase 2
            target_domain: "coding", "math", or "translation"
            neuron_retention_ratio: e.g. 0.40 to keep top 40% neurons
            head_retention_ratio: e.g. 0.50 to keep top 50% heads
            use_selectivity: if True, combines raw importance with domain selectivity
        """
        print("\n" + "="*70)
        print(f"PHASE 4: CAPABILITY MASK GENERATION ({target_domain.upper()})")
        print("="*70)

        n_ratio = neuron_retention_ratio or self.config.target_neuron_retention_ratio
        h_ratio = head_retention_ratio or self.config.target_head_retention_ratio

        meta = neuron_importance_data["meta"]
        num_layers = meta["num_layers"]
        intermediate_size = meta["intermediate_size"]
        num_heads = meta["num_heads"]

        # Base importance score
        raw_n_imp = neuron_importance_data["neuron_importance"][target_domain] # [L, D]
        raw_h_imp = neuron_importance_data["head_importance"][target_domain]   # [L, H]

        if use_selectivity and "selectivity" in neuron_importance_data and target_domain in neuron_importance_data["selectivity"]:
            selectivity = neuron_importance_data["selectivity"][target_domain]
            # Normalizing and combining: 70% domain importance + 30% domain selectivity
            norm_imp = (raw_n_imp - raw_n_imp.min()) / (raw_n_imp.max() - raw_n_imp.min() + 1e-8)
            norm_sel = (selectivity - selectivity.min()) / (selectivity.max() - selectivity.min() + 1e-8)
            final_n_score = 0.7 * norm_imp + 0.3 * norm_sel
        else:
            final_n_score = raw_n_imp

        final_h_score = raw_h_imp

        # 1. Generate Neuron Mask (Shape: [num_layers, intermediate_size])
        k_neurons = max(1, int(intermediate_size * n_ratio))
        neuron_mask = torch.zeros_like(final_n_score, dtype=torch.float32)
        retained_neuron_indices = {}

        for l in range(num_layers):
            top_indices = torch.topk(final_n_score[l], k_neurons).indices
            neuron_mask[l, top_indices] = 1.0
            retained_neuron_indices[l] = top_indices.tolist()

        # 2. Generate Head Mask (Shape: [num_layers, num_heads])
        k_heads = max(1, int(num_heads * h_ratio))
        head_mask = torch.zeros_like(final_h_score, dtype=torch.float32)
        retained_head_indices = {}

        for l in range(num_layers):
            top_h_indices = torch.topk(final_h_score[l], k_heads).indices
            head_mask[l, top_h_indices] = 1.0
            retained_head_indices[l] = top_h_indices.tolist()

        # 3. Layer Selection (if layer_retention_ratio < 1.0)
        l_ratio = layer_retention_ratio if layer_retention_ratio is not None else self.config.target_layer_retention_ratio
        if l_ratio < 1.0:
            k_layers = max(2, int(num_layers * l_ratio))
            layer_scores = final_n_score.sum(dim=-1)
            if num_layers > 2 and k_layers < num_layers:
                middle_scores = layer_scores[1:-1]
                top_middle = torch.topk(middle_scores, min(k_layers - 2, len(middle_scores))).indices + 1
                retained_layer_indices = sorted(list(set([0] + top_middle.tolist() + [num_layers - 1])))
            else:
                retained_layer_indices = list(range(num_layers))
        else:
            retained_layer_indices = list(range(num_layers))

        # Total retained parameters stats
        total_neurons = num_layers * intermediate_size
        active_neurons = int(neuron_mask.sum().item())
        total_heads = num_layers * num_heads
        active_heads = int(head_mask.sum().item())

        print(f"Target Capability: {target_domain}")
        print(f"Retained Layers  : {len(retained_layer_indices)}/{num_layers} ({len(retained_layer_indices)/num_layers*100:.1f}%)")
        print(f"Retained Neurons : {active_neurons}/{total_neurons} ({active_neurons/total_neurons*100:.1f}%)")
        print(f"Retained Heads   : {active_heads}/{total_heads} ({active_heads/total_heads*100:.1f}%)")

        mask_dict = {
            "domain": target_domain,
            "neuron_mask": neuron_mask,
            "head_mask": head_mask,
            "retained_layer_indices": retained_layer_indices,
            "retained_neuron_indices": retained_neuron_indices,
            "retained_head_indices": retained_head_indices,
            "layer_retention_ratio": l_ratio,
            "neuron_retention_ratio": n_ratio,
            "head_retention_ratio": h_ratio,
            "meta": meta
        }

        # Save mask
        out_file = self.config.mask_output_path
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save(mask_dict, out_file)
        print(f"\n[Phase 4 Complete] Capability Mask saved to: {out_file}")

        return mask_dict
