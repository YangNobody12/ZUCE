"""
Phase 3: Neuron-Level Attribution & Domain Selectivity
Computes First-Order Taylor Attribution A_i = E[|z_i * dL/dz_i|] for each SwiGLU intermediate neuron,
and derives Z-score Domain Selectivity S_i = (A_i - mu(A_other)) / (sigma(A_other) + eps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from .gradient_hooks import HookController

class NeuronAttributionProfiler:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_controller = HookController(self.model)
        self.num_layers = self.hook_controller.num_layers()
        self.intermediate_size = getattr(self.model.config, "intermediate_size", 8960)

    def compute_task_attribution(self, prompts: List[str], task_name: str) -> torch.Tensor:
        """
        Computes First-Order Taylor Attribution matrix [num_layers, intermediate_size]
        for the given task domain.
        """
        self.model.eval()
        attributions = torch.zeros((self.num_layers, self.intermediate_size), device=self.device, dtype=torch.float32)
        n_samples = 0

        for p in tqdm(prompts, desc=f"Attribution ({task_name})"):
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=384).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            labels = input_ids[:, 1:].clone()
            inputs = input_ids[:, :-1]

            self.hook_controller.clear()
            self.hook_controller.register_neuron_activation_and_gradient()

            outputs = self.model(input_ids=inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            self.model.zero_grad()
            loss.backward()

            for l_idx in range(self.num_layers):
                key = f"mlp_{l_idx}"
                if key in self.hook_controller.activations and key in self.hook_controller.gradients:
                    act = self.hook_controller.activations[key]
                    grad = self.hook_controller.gradients[key]
                    # Absolute Taylor attribution across batch and sequence tokens
                    taylor = torch.abs(grad * act).sum(dim=(0, 1))
                    attributions[l_idx] += taylor.float()

            self.hook_controller.clear()
            n_samples += 1

        if n_samples > 0:
            attributions /= n_samples

        return attributions.cpu()

    def profile_all_domains_with_selectivity(
        self,
        task_prompts_dict: Dict[str, List[str]],
        target_domain: str = "coding"
    ) -> Dict[str, Any]:
        """
        Profiles attributions for all domains and calculates Z-score Domain Selectivity.
        """
        raw_attributions = {}
        for domain, prompts in task_prompts_dict.items():
            attr = self.compute_task_attribution(prompts, domain)
            raw_attributions[domain] = attr

        # Compute Z-score selectivity for target domain against contrast domains
        contrast_domains = [d for d in raw_attributions if d != target_domain]
        eps = 1e-8

        target_attr = raw_attributions[target_domain] # [L, D]
        contrast_stack = torch.stack([raw_attributions[d] for d in contrast_domains], dim=0) # [NumContrast, L, D]

        mu_contrast = contrast_stack.mean(dim=0)
        sigma_contrast = contrast_stack.std(dim=0)

        # Z-score Domain Selectivity
        z_selectivity = (target_attr - mu_contrast) / (sigma_contrast + eps)

        return {
            "target_domain": target_domain,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "attributions": raw_attributions,
            "z_selectivity": z_selectivity
        }
