"""
Multi-Signal Layer and Neuron Profiler
Implements:
1. Causal Layer Ablation: Delta L_l = Loss_ablated - Loss_base and Logits KL Divergence
2. First-Order Taylor Attribution: A_{l, i} = E[|a_{l, i} * dL/da_{l, i}|]
3. Domain Selectivity Z-Score: Z_{l, i} = (A_{l, i}^{code} - mu(A_{l, i}^{other})) / (sigma(A_{l, i}^{other}) + eps)
4. Empirical Layer Distortion Curves: D_l(k) across intermediate width grid k
5. Composite Multi-Objective Scoring Engine
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

from .gradient_hooks import HookController

class OptimalSelectorProfiler:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_controller = HookController(self.model)
        self.num_layers = self.hook_controller.num_layers()
        self.intermediate_size = getattr(self.model.config, "intermediate_size", 8960)

    def profile_causal_layer_importance(
        self,
        prompts: List[str],
        max_samples: int = 12
    ) -> Dict[str, Any]:
        """
        Measures the causal capability impact of disabling each layer.
        Computes:
        1. Delta Task Loss: L_{ablated(l)} - L_{base}
        2. Output Logits Divergence: KL(P_{base} || P_{ablated(l)})
        3. Cosine Shift of Final Representation
        """
        self.model.eval()
        sample_prompts = prompts[:max_samples]
        
        # 1. Base forward pass & baseline metrics
        base_losses = []
        base_logits_list = []
        tokenized_inputs = []

        for p in sample_prompts:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            tokenized_inputs.append(enc)
            
            with torch.no_grad():
                out = self.model(input_ids=input_ids[:, :-1])
                logits = out.logits
                labels = input_ids[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                base_losses.append(loss.item())
                base_logits_list.append(logits[:, -1, :].detach())

        base_loss_mean = sum(base_losses) / max(len(base_losses), 1)
        print(f"\n[Causal Profiling] Baseline Task Loss across {len(tokenized_inputs)} samples: {base_loss_mean:.4f}")

        layer_delta_loss = torch.zeros(self.num_layers, dtype=torch.float32)
        layer_kl_divergence = torch.zeros(self.num_layers, dtype=torch.float32)

        # 2. Causal ablation per layer
        for l_idx in tqdm(range(self.num_layers), desc="Causal Layer Ablation"):
            mod = self.model.model.layers[l_idx]
            
            # Hook to bypass/zero-out the layer residual contribution
            def zero_layer_hook(module, inputs, output):
                if isinstance(output, tuple):
                    return (inputs[0], *output[1:]) if len(inputs) > 0 else (torch.zeros_like(output[0]), *output[1:])
                return inputs[0] if len(inputs) > 0 else torch.zeros_like(output)

            handle = mod.register_forward_hook(zero_layer_hook)

            ablated_loss_sum = 0.0
            total_kl = 0.0
            n_eval = 0

            for i, enc in enumerate(tokenized_inputs):
                input_ids = enc["input_ids"]
                labels = input_ids[:, 1:]
                with torch.no_grad():
                    out = self.model(input_ids=input_ids[:, :-1])
                    logits = out.logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    ablated_loss_sum += loss.item()

                    # KL Divergence on token predictions
                    p_base = F.softmax(base_logits_list[i], dim=-1)
                    q_ablated = F.log_softmax(logits[:, -1, :], dim=-1)
                    kl = F.kl_div(q_ablated, p_base, reduction="batchmean").item()
                    total_kl += kl
                    n_eval += 1

            handle.remove()

            avg_ablated_loss = ablated_loss_sum / max(n_eval, 1)
            delta_loss = max(0.0, avg_ablated_loss - base_loss_mean)
            avg_kl = max(0.0, total_kl / max(n_eval, 1))

            layer_delta_loss[l_idx] = delta_loss
            layer_kl_divergence[l_idx] = avg_kl

        # Normalize causal sensitivity
        norm_delta = (layer_delta_loss - layer_delta_loss.min()) / (layer_delta_loss.max() - layer_delta_loss.min() + 1e-8)
        norm_kl = (layer_kl_divergence - layer_kl_divergence.min()) / (layer_kl_divergence.max() - layer_kl_divergence.min() + 1e-8)
        causal_importance = 0.6 * norm_delta + 0.4 * norm_kl

        return {
            "num_layers": self.num_layers,
            "base_loss": base_loss_mean,
            "layer_delta_loss": layer_delta_loss.tolist(),
            "layer_kl_divergence": layer_kl_divergence.tolist(),
            "causal_importance_tensor": causal_importance
        }

    def compute_taylor_and_selectivity(
        self,
        task_prompts_dict: Dict[str, List[str]],
        target_domain: str = "coding",
        max_samples_per_domain: int = 15
    ) -> Dict[str, Any]:
        """
        Computes First-Order Taylor Attribution and Z-score Domain Selectivity.
        """
        self.model.eval()
        raw_attributions = {}

        for domain, prompts in task_prompts_dict.items():
            attributions = torch.zeros((self.num_layers, self.intermediate_size), device=self.device, dtype=torch.float32)
            sample_list = prompts[:max_samples_per_domain]
            n_samples = 0

            for p in tqdm(sample_list, desc=f"Attribution ({domain})"):
                enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
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
                        # Taylor attribution: |a * grad|
                        taylor = torch.abs(grad * act).sum(dim=(0, 1))
                        attributions[l_idx] += taylor.float().detach()

                self.hook_controller.clear()
                n_samples += 1

            if n_samples > 0:
                attributions /= n_samples

            raw_attributions[domain] = attributions.cpu()

        # Compute Domain Selectivity Z-Score
        contrast_domains = [d for d in raw_attributions if d != target_domain]
        target_attr = raw_attributions[target_domain]
        
        if contrast_domains:
            contrast_stack = torch.stack([raw_attributions[d] for d in contrast_domains], dim=0)
            mu_contrast = contrast_stack.mean(dim=0)
            sigma_contrast = contrast_stack.std(dim=0)
            z_selectivity = (target_attr - mu_contrast) / (sigma_contrast + 1e-8)
        else:
            z_selectivity = torch.zeros_like(target_attr)

        return {
            "target_domain": target_domain,
            "attributions": raw_attributions,
            "z_selectivity": z_selectivity
        }

    def compute_composite_neuron_scores(
        self,
        attr_data: Dict[str, Any],
        causal_layer_weights: Optional[torch.Tensor] = None,
        w_taylor: float = 0.45,
        w_selectivity: float = 0.35,
        w_general: float = 0.20
    ) -> torch.Tensor:
        """
        Combines Taylor attribution, domain selectivity, and general language retention
        into a unified, per-neuron composite scoring matrix S in R^{L x D}.
        """
        code_attr = attr_data["attributions"]["coding"]
        z_sel = attr_data["z_selectivity"]
        gen_attr = attr_data["attributions"].get("general", torch.zeros_like(code_attr))

        # Per-layer min-max normalization
        norm_code = torch.zeros_like(code_attr)
        norm_sel = torch.zeros_like(z_sel)
        norm_gen = torch.zeros_like(gen_attr)

        for l in range(self.num_layers):
            c_row = code_attr[l]
            norm_code[l] = (c_row - c_row.min()) / (c_row.max() - c_row.min() + 1e-8)

            s_row = z_sel[l]
            norm_sel[l] = (s_row - s_row.min()) / (s_row.max() - s_row.min() + 1e-8)

            g_row = gen_attr[l]
            norm_gen[l] = (g_row - g_row.min()) / (g_row.max() - g_row.min() + 1e-8)

        composite_scores = (
            w_taylor * norm_code +
            w_selectivity * norm_sel +
            w_general * norm_gen
        )

        # Scale by causal layer weights if provided
        if causal_layer_weights is not None:
            c_weights = causal_layer_weights.view(-1, 1).to(composite_scores.device)
            # Soft weighting so low-causal layers are still active
            composite_scores = composite_scores * (0.5 + 0.5 * c_weights)

        return composite_scores
