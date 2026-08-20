"""
Phase 2: Component & Layer Sensitivity Profiling Engine
Measures task-conditioned sensitivity for each layer and sub-component
(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
producing the 3D impact tensor I in R^{L x C x T}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from .gradient_hooks import HookController

COMPONENTS = ["layer", "self_attn", "mlp", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class ComponentSensitivityProfiler:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_controller = HookController(self.model)
        self.num_layers = self.hook_controller.num_layers()

    def compute_baseline_metrics(self, prompts: List[str]) -> Tuple[float, List[torch.Tensor]]:
        """Computes baseline cross-entropy loss and logits on prompt list."""
        self.model.eval()
        total_loss = 0.0
        logits_list = []
        n_samples = 0

        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            labels = input_ids[:, 1:]
            inputs = input_ids[:, :-1]

            with torch.no_grad():
                outputs = self.model(input_ids=inputs)
                logits = outputs.logits
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                total_loss += loss.item()
                logits_list.append(logits[:, -1, :].detach())
                n_samples += 1

        avg_loss = total_loss / max(n_samples, 1)
        return avg_loss, logits_list

    def profile_component_sensitivity(
        self,
        task_prompts_dict: Dict[str, List[str]],
        alphas: List[float] = [1.0, 0.5, 0.0]
    ) -> Dict[str, Any]:
        """
        Profiles sensitivity across layers (L), components (C), and tasks (T).
        Returns sensitivity tensor I [num_layers, num_components, num_tasks].
        """
        tasks = list(task_prompts_dict.keys())
        num_tasks = len(tasks)
        num_comps = len(COMPONENTS)

        impact_tensor = torch.zeros((self.num_layers, num_comps, num_tasks), dtype=torch.float32)
        kl_tensor = torch.zeros((self.num_layers, num_comps, num_tasks), dtype=torch.float32)

        print(f"\n[Phase 2] Profiling Sensitivity: {self.num_layers} Layers x {num_comps} Components x {num_tasks} Tasks...")

        for t_idx, task_name in enumerate(tasks):
            prompts = task_prompts_dict[task_name]
            base_loss, base_logits = self.compute_baseline_metrics(prompts)
            print(f"  Task: {task_name.upper():<10} | Baseline Loss: {base_loss:.4f}")

            for l_idx in tqdm(range(self.num_layers), desc=f"Layers ({task_name})"):
                for c_idx, comp_name in enumerate(COMPONENTS):
                    # Test complete perturbation (alpha = 0.0)
                    self.hook_controller.clear()
                    self.hook_controller.register_component_scale_hook(l_idx, comp_name, alpha=0.0)

                    pert_loss = 0.0
                    total_kl = 0.0
                    n_eval = 0

                    for p_i, p in enumerate(prompts[:len(base_logits)]):
                        enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                        input_ids = enc["input_ids"]
                        if input_ids.shape[1] < 2:
                            continue
                        labels = input_ids[:, 1:]
                        inputs = input_ids[:, :-1]

                        with torch.no_grad():
                            outputs = self.model(input_ids=inputs)
                            logits = outputs.logits
                            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                            pert_loss += loss.item()

                            # Compute KL divergence on last token
                            p_log = F.log_softmax(base_logits[p_i], dim=-1)
                            q_prob = F.softmax(logits[:, -1, :], dim=-1)
                            kl = F.kl_div(p_log, q_prob, reduction="batchmean").item()
                            total_kl += kl
                            n_eval += 1

                    self.hook_controller.clear()
                    avg_pert_loss = pert_loss / max(n_eval, 1)
                    avg_kl = total_kl / max(n_eval, 1)

                    delta_loss = avg_pert_loss - base_loss
                    impact_tensor[l_idx, c_idx, t_idx] = delta_loss
                    kl_tensor[l_idx, c_idx, t_idx] = avg_kl

        return {
            "components": COMPONENTS,
            "tasks": tasks,
            "num_layers": self.num_layers,
            "impact_loss_tensor": impact_tensor,
            "kl_divergence_tensor": kl_tensor
        }
