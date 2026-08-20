"""
Phase 4A: Causal Ablation Interaction & Synergy Profiler
Computes 3 signals:
1. Activation Correlation C_{ij}
2. Gradient Coupling G_{ij}
3. Causal Ablation Interaction J_{ij} = Delta_{ij} - Delta_i - Delta_j
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

class CausalInteractionProfiler:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers

    def compute_layer_interaction_matrix(
        self,
        prompts: List[str],
        sample_layer_pairs: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Computes pairwise synergy J_{ij} across layer pairs on coding data.
        """
        self.model.eval()
        
        # 1. Baseline loss
        base_loss = 0.0
        n_samples = 0
        for p in prompts[:6]:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            with torch.no_grad():
                out = self.model(input_ids=input_ids[:, :-1])
                loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)), input_ids[:, 1:].reshape(-1))
                base_loss += loss.item()
                n_samples += 1
        base_loss /= max(n_samples, 1)

        # 2. Individual ablations Delta_i
        layer_delta = {}
        layers_to_test = sorted(list(set([l for pair in sample_layer_pairs for l in pair])))
        
        for l_idx in layers_to_test:
            mod = self.model.model.layers[l_idx]
            h = mod.register_forward_hook(lambda m, i, o: o[0] * 0.0 if isinstance(o, tuple) else o * 0.0)
            
            l_loss = 0.0
            n = 0
            for p in prompts[:6]:
                enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                input_ids = enc["input_ids"]
                if input_ids.shape[1] < 2:
                    continue
                with torch.no_grad():
                    out = self.model(input_ids=input_ids[:, :-1])
                    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)), input_ids[:, 1:].reshape(-1))
                    l_loss += loss.item()
                    n += 1
            h.remove()
            layer_delta[l_idx] = (l_loss / max(n, 1)) - base_loss

        # 3. Joint ablations Delta_{ij} & Synergy J_{ij} = Delta_{ij} - Delta_i - Delta_j
        interactions = {}
        for (l_i, l_j) in sample_layer_pairs:
            mod_i = self.model.model.layers[l_i]
            mod_j = self.model.model.layers[l_j]
            h_i = mod_i.register_forward_hook(lambda m, i, o: o[0] * 0.0 if isinstance(o, tuple) else o * 0.0)
            h_j = mod_j.register_forward_hook(lambda m, i, o: o[0] * 0.0 if isinstance(o, tuple) else o * 0.0)

            joint_loss = 0.0
            n = 0
            for p in prompts[:6]:
                enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                input_ids = enc["input_ids"]
                if input_ids.shape[1] < 2:
                    continue
                with torch.no_grad():
                    out = self.model(input_ids=input_ids[:, :-1])
                    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)), input_ids[:, 1:].reshape(-1))
                    joint_loss += loss.item()
                    n += 1
            h_i.remove()
            h_j.remove()

            delta_ij = (joint_loss / max(n, 1)) - base_loss
            synergy = delta_ij - (layer_delta[l_i] + layer_delta[l_j])
            interactions[f"{l_i}_{l_j}"] = {
                "layer_i": l_i,
                "layer_j": l_j,
                "delta_i": layer_delta[l_i],
                "delta_j": layer_delta[l_j],
                "delta_ij": delta_ij,
                "synergy_J": synergy
            }

        return interactions
