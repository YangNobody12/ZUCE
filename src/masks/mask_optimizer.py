"""
Phase 5B: Soft Mask Optimizer
Optimizes continuous mask: min_M L_task(M) + lambda ||M||_1
Freezes model weights and trains only mask logits on the target capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Any
from tqdm import tqdm

from .soft_mask import SoftCapabilityMask
from ..profiling.gradient_hooks import HookController

class MaskOptimizer:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.intermediate_size = getattr(model.config, "intermediate_size", 8960)

    def optimize_capability_mask(
        self,
        prompts: List[str],
        epochs: int = 5,
        lr: float = 0.02,
        l1_lambda: float = 0.005,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Optimizes soft mask M to minimize task loss while penalizing non-sparse activations.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        soft_mask = SoftCapabilityMask(self.num_layers, self.intermediate_size).to(self.device)
        optimizer = Adam(soft_mask.parameters(), lr=lr)
        hook_ctrl = HookController(self.model)

        print(f"\n[Phase 5] Optimizing Soft Capability Mask ({epochs} epochs, lambda={l1_lambda})...")

        for epoch in range(epochs):
            total_loss = 0.0
            total_l1 = 0.0

            for p in prompts:
                enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                input_ids = enc["input_ids"]
                if input_ids.shape[1] < 2:
                    continue

                labels = input_ids[:, 1:].clone()
                inputs = input_ids[:, :-1]

                current_mask = soft_mask() # [L, D]

                # Register soft mask on all layers
                hook_ctrl.clear()
                for l_idx in range(self.num_layers):
                    hook_ctrl.register_soft_mask_hook(l_idx, current_mask[l_idx])

                outputs = self.model(input_ids=inputs)
                logits = outputs.logits
                task_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                # L1 Sparsity Penalty
                l1_penalty = l1_lambda * current_mask.mean()
                loss = task_loss + l1_penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hook_ctrl.clear()
                total_loss += task_loss.item()
                total_l1 += l1_penalty.item()

            avg_loss = total_loss / len(prompts)
            avg_l1 = total_l1 / len(prompts)
            sparsity = (soft_mask.binarize(threshold) == 0).float().mean().item() * 100
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Task Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | Sparsity: {sparsity:.1f}%")

        binary_mask = soft_mask.binarize(threshold).cpu()
        continuous_mask = soft_mask().detach().cpu()

        # Enforce budget constraint if mask is unpruned
        if binary_mask.mean() > 0.6:
            # Rank neurons by attribution & selectivity to enforce target intermediate size (e.g. 2304 / 8960 = 25.7%)
            target_k = 2304
            budget_mask = torch.zeros_like(binary_mask)
            for l_idx in range(self.num_layers):
                top_indices = torch.topk(continuous_mask[l_idx], target_k).indices
                budget_mask[l_idx, top_indices] = 1.0
            binary_mask = budget_mask

        return {
            "binary_mask": binary_mask,
            "continuous_mask": continuous_mask,
            "active_neuron_ratio": float(binary_mask.mean().item()),
            "sparsity_pct": round((1.0 - binary_mask.mean().item()) * 100, 2)
        }
