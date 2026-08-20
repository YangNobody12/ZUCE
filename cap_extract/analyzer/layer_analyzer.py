"""
Phase 1: Layer Analysis Engine
Evaluates layer-wise, MLP-wise, and Attention-wise importance for target capabilities
using Activation, Gradient, Fisher Information, and KL Divergence metrics.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .hooks import HookManager
from ..configs.base_config import ExtractionConfig
from ..datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS, TRANSLATION_PROMPTS, GENERAL_PROMPTS

from ..utils import prepare_inputs

class LayerAnalyzer:
    def __init__(self, model: nn.Module, tokenizer: Any, config: Optional[ExtractionConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()
        self.hook_mgr = HookManager(self.model)
        self.num_layers = self.hook_mgr.num_layers()
        
    def _get_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute logits for the last token position."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, -1, :].float()

    def compute_layer_kl_importance(
        self,
        prompts: List[str],
        target_component: str = "mlp", # "layer", "mlp", "attn"
        alphas: Optional[List[float]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculates output KL Divergence shift when scaling down each layer/submodule.
        Higher KL Divergence implies the layer/submodule is critical for that capability.
        """
        if alphas is None:
            alphas = self.config.layer_alphas

        results: Dict[int, Dict[str, float]] = {l: {} for l in range(self.num_layers)}
        self.model.eval()
        
        for prompt in tqdm(prompts[:self.config.num_calibration_samples], desc=f"Evaluating KL ({target_component})"):
            raw_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len
            )
            inputs = prepare_inputs(raw_inputs, self.model.device)
            
            with torch.no_grad():
                base_logits = self._get_logits(inputs["input_ids"], inputs.get("attention_mask"))
                base_log_probs = F.log_softmax(base_logits, dim=-1)

            for layer_idx in range(self.num_layers):
                for alpha in alphas:
                    if alpha == 1.0:
                        kl_val = 0.0
                    else:
                        if target_component == "layer":
                            self.hook_mgr.register_layer_scale_hook(layer_idx, alpha)
                        elif target_component == "mlp":
                            self.hook_mgr.register_mlp_scale_hook(layer_idx, alpha)
                        elif target_component == "attn":
                            self.hook_mgr.register_attn_scale_hook(layer_idx, alpha)

                        with torch.no_grad():
                            scaled_logits = self._get_logits(inputs["input_ids"], inputs.get("attention_mask"))
                            scaled_probs = F.softmax(scaled_logits, dim=-1)
                            kl_val = F.kl_div(base_log_probs, scaled_probs, reduction="batchmean").item()

                        self.hook_mgr.clear()

                    key = f"alpha_{alpha:.1f}"
                    results[layer_idx][key] = results[layer_idx].get(key, 0.0) + (kl_val / len(prompts))

        return results

    def compute_gradient_fisher_importance(
        self,
        prompts: List[str]
    ) -> Dict[str, Dict[int, float]]:
        """
        Computes First-Order Gradient Sensitivity and Diagonal Empirical Fisher Information
        per layer across domain prompts.
        """
        self.model.eval()
        grad_sensitivities = {l: 0.0 for l in range(self.num_layers)}
        fisher_info = {l: 0.0 for l in range(self.num_layers)}
        activation_norms = {l: 0.0 for l in range(self.num_layers)}

        for prompt in tqdm(prompts[:self.config.num_calibration_samples], desc="Computing Grad & Fisher"):
            raw_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len
            )
            inputs = prepare_inputs(raw_inputs, self.model.device)
            
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            labels = input_ids[:, 1:].clone()
            input_tokens = input_ids[:, :-1]

            self.hook_mgr.clear()
            self.hook_mgr.register_activation_capture("layer")
            self.hook_mgr.register_gradient_capture("layer")

            outputs = self.model(input_ids=input_tokens)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            self.model.zero_grad()
            loss.backward()

            for l_idx in range(self.num_layers):
                act_key = f"layer_{l_idx}"
                if act_key in self.hook_mgr.activations and act_key in self.hook_mgr.gradients:
                    act = self.hook_mgr.activations[act_key]
                    grad = self.hook_mgr.gradients[act_key]

                    # First-order sensitivity: |grad * act|
                    sens = torch.abs(grad * act).sum().item()
                    grad_sensitivities[l_idx] += sens

                    # Empirical Fisher info: (grad)^2
                    fish = (grad ** 2).sum().item()
                    fisher_info[l_idx] += fish

                    # Activation norm
                    act_norms = torch.norm(act, p=2).item()
                    activation_norms[l_idx] += act_norms

            self.hook_mgr.clear()

        n_samples = max(1, len(prompts[:self.config.num_calibration_samples]))
        return {
            "grad_sensitivity": {l: v / n_samples for l, v in grad_sensitivities.items()},
            "fisher_info": {l: v / n_samples for l, v in fisher_info.items()},
            "activation_norm": {l: v / n_samples for l, v in activation_norms.items()}
        }

    def generate_layer_importance_matrix(self) -> Dict[str, Any]:
        """
        Executes full Phase 1 Layer Analysis across Coding, Math, and Translation domains.
        Returns and saves the complete Layer Importance Matrix.
        """
        domains = {
            "coding": CODING_PROMPTS,
            "math": MATH_PROMPTS,
            "translation": TRANSLATION_PROMPTS
        }

        matrix: Dict[str, Any] = {
            "num_layers": self.num_layers,
            "domains": {}
        }

        print("\n" + "="*70)
        print("PHASE 1: LAYER IMPORTANCE MATRIX EXTRACTION")
        print("="*70)

        for domain_name, prompts in domains.items():
            print(f"\n--- Analyzing Domain: {domain_name.upper()} ---")
            kl_layer = self.compute_layer_kl_importance(prompts, target_component="layer", alphas=[0.0, 0.5])
            kl_mlp = self.compute_layer_kl_importance(prompts, target_component="mlp", alphas=[0.0, 0.5])
            kl_attn = self.compute_layer_kl_importance(prompts, target_component="attn", alphas=[0.0, 0.5])
            grad_fisher = self.compute_gradient_fisher_importance(prompts)

            # Combined domain score per layer: Normalized weighted sum
            scores = {}
            for l in range(self.num_layers):
                kl_drop_score = kl_layer[l].get("alpha_0.0", 0.0)
                mlp_drop_score = kl_mlp[l].get("alpha_0.0", 0.0)
                attn_drop_score = kl_attn[l].get("alpha_0.0", 0.0)
                fisher_val = grad_fisher["fisher_info"][l]
                grad_val = grad_fisher["grad_sensitivity"][l]

                scores[l] = {
                    "layer_kl_drop": kl_drop_score,
                    "mlp_kl_drop": mlp_drop_score,
                    "attn_kl_drop": attn_drop_score,
                    "fisher_info": fisher_val,
                    "grad_sensitivity": grad_val,
                    "composite_importance": (kl_drop_score * 0.4 + mlp_drop_score * 0.3 + attn_drop_score * 0.3)
                }

            matrix["domains"][domain_name] = scores

        # Save to output path
        out_file = self.config.matrix_output_path
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2)

        print(f"\n[Phase 1 Complete] Layer Importance Matrix saved to: {out_file}")
        return matrix
