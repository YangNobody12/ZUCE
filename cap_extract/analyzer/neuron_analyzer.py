"""
Phase 2: Neuron & Attention Head Analysis Engine
Evaluates fine-grained importance of intermediate MLP neurons (e.g. 4096 / 8960 per layer)
and Attention heads for target capabilities using gradient-activation attribution.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from .hooks import HookManager
from ..configs.base_config import ExtractionConfig
from ..datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS, TRANSLATION_PROMPTS, GENERAL_PROMPTS
from ..utils import prepare_inputs

class NeuronAnalyzer:
    def __init__(self, model: nn.Module, tokenizer: Any, config: Optional[ExtractionConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()
        self.hook_mgr = HookManager(self.model)
        self.num_layers = self.hook_mgr.num_layers()
        self._infer_model_dimensions()

    def _infer_model_dimensions(self):
        """Extract intermediate_size, hidden_size, and num_attention_heads from model config."""
        cfg = getattr(self.model, "config", None)
        if cfg:
            self.intermediate_size = getattr(cfg, "intermediate_size", 4096)
            self.hidden_size = getattr(cfg, "hidden_size", 2048)
            self.num_heads = getattr(cfg, "num_attention_heads", 16)
            self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.intermediate_size = 4096
            self.hidden_size = 2048
            self.num_heads = 16
            self.num_kv_heads = 16
            self.head_dim = 128

    def compute_neuron_head_importance(
        self,
        prompts: List[str],
        domain_name: str = "coding"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes attribution importance for each intermediate neuron:
            Score_i = E_{x} [ | grad_i * act_i | ]
        and for each attention head:
            HeadScore_h = E_{x} [ || grad_h * act_h ||_2 ]

        Returns:
            neuron_importance: Tensor of shape [num_layers, intermediate_size]
            head_importance: Tensor of shape [num_layers, num_heads]
        """
        device = self.model.device
        neuron_importance = torch.zeros((self.num_layers, self.intermediate_size), device=device, dtype=torch.float32)
        head_importance = torch.zeros((self.num_layers, self.num_heads), device=device, dtype=torch.float32)

        self.model.eval()
        samples_processed = 0

        for prompt in tqdm(prompts[:self.config.num_calibration_samples], desc=f"Analyzing Neurons ({domain_name})"):
            raw_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len
            )
            inputs = prepare_inputs(raw_inputs, device)

            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            labels = input_ids[:, 1:].clone()
            input_tokens = input_ids[:, :-1]

            self.hook_mgr.clear()
            self.hook_mgr.register_activation_capture("mlp_intermediate")
            self.hook_mgr.register_gradient_capture("mlp_intermediate")
            self.hook_mgr.register_activation_capture("attn")
            self.hook_mgr.register_gradient_capture("attn")

            outputs = self.model(input_ids=input_tokens)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            self.model.zero_grad()
            loss.backward()

            for l_idx in range(self.num_layers):
                # 1. MLP Neuron Attribution
                mlp_key = f"mlp_neuron_{l_idx}"
                if mlp_key in self.hook_mgr.activations and mlp_key in self.hook_mgr.gradients:
                    act = self.hook_mgr.activations[mlp_key]     # Shape: [batch, seq_len, intermediate_size]
                    grad = self.hook_mgr.gradients[mlp_key]      # Shape: [batch, seq_len, intermediate_size]
                    # Absolute Taylor first-order attribution per neuron across batch and tokens
                    attribution = torch.abs(grad * act).sum(dim=(0, 1)) # Shape: [intermediate_size]
                    neuron_importance[l_idx] += attribution.float()

                # 2. Attention Head Attribution
                attn_key = f"attn_{l_idx}"
                if attn_key in self.hook_mgr.activations and attn_key in self.hook_mgr.gradients:
                    act = self.hook_mgr.activations[attn_key]   # Shape: [batch, seq_len, hidden_size]
                    grad = self.hook_mgr.gradients[attn_key]    # Shape: [batch, seq_len, hidden_size]
                    
                    # Reshape hidden_size -> (num_heads, head_dim)
                    act_heads = act.view(act.size(0), act.size(1), self.num_heads, self.head_dim)
                    grad_heads = grad.view(grad.size(0), grad.size(1), self.num_heads, self.head_dim)
                    
                    head_attr = torch.abs(grad_heads * act_heads).sum(dim=(0, 1, 3)) # Shape: [num_heads]
                    head_importance[l_idx] += head_attr.float()

            self.hook_mgr.clear()
            samples_processed += 1

        if samples_processed > 0:
            neuron_importance /= samples_processed
            head_importance /= samples_processed

        return neuron_importance.cpu(), head_importance.cpu()

    def run_full_neuron_analysis(self) -> Dict[str, Any]:
        """
        Executes Phase 2 fine-grained neuron & attention head profiling across all domains
        and calculates capability-selectivity indices.
        """
        print("\n" + "="*70)
        print("PHASE 2: NEURON & ATTENTION HEAD IMPORTANCE PROFILING")
        print(f"Dimensions: {self.num_layers} Layers | {self.intermediate_size} Neurons/Layer | {self.num_heads} Heads/Layer")
        print("="*70)

        domain_datasets = {
            "coding": CODING_PROMPTS,
            "math": MATH_PROMPTS,
            "translation": TRANSLATION_PROMPTS,
            "general": GENERAL_PROMPTS
        }

        results = {
            "meta": {
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "num_heads": self.num_heads,
                "hidden_size": self.hidden_size
            },
            "neuron_importance": {},
            "head_importance": {},
            "selectivity": {}
        }

        for domain, prompts in domain_datasets.items():
            n_imp, h_imp = self.compute_neuron_head_importance(prompts, domain_name=domain)
            results["neuron_importance"][domain] = n_imp
            results["head_importance"][domain] = h_imp

        # Compute Domain Selectivity Index: Importance(domain) / (Importance(general) + eps)
        eps = 1e-8
        gen_n_imp = results["neuron_importance"]["general"]
        for domain in ["coding", "math", "translation"]:
            d_n_imp = results["neuron_importance"][domain]
            selectivity = (d_n_imp - gen_n_imp) / (gen_n_imp + eps)
            results["selectivity"][domain] = selectivity

        # Save to PyTorch tensor output
        out_file = self.config.neuron_output_path
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        torch.save(results, out_file)

        print(f"\n[Phase 2 Complete] Neuron & Head Importance Tensors saved to: {out_file}")
        return results
