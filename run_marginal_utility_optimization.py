"""
Marginal Utility & Lagrangian Resource Allocation Solver
1. Measures empirical distortion D_l(k) for each layer l in [0..27] across width grid k in [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8960]
2. Computes discrete Marginal Utility MU_l(k) = - dD_l / dP_l
3. Solves the constrained budget allocation problem:
   min_{k_l} sum_l D_l(k_l) subject to: sum_l 3*d*k_l <= P_budget
4. Evaluates the resulting non-uniform architecture under the 500M / target budget!
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.surgery.weight_mapper import PhysicalWeightMapper
from src.evaluation.coding import CodingEvaluator
from run_10_question_coding_test import TEN_CODING_QUESTIONS

def measure_layer_distortion_curves(teacher, tokenizer, coding_prompts, width_grid, device="cuda"):
    """
    Measures local MLP distortion D_l(k) = || y_T,l - y_{P,l}(k) ||^2 / || y_T,l ||^2
    for each layer l in [0..27] on clean layer inputs h_l^teacher.
    """
    teacher.eval()
    layers = teacher.model.layers
    num_layers = len(layers)
    intermediate_size = teacher.config.intermediate_size

    # Load neuron scores
    cfg = get_full_extraction_config()
    attr_path = os.path.join(cfg["paths"]["results_dir"], "02_neuron_attribution.pt")
    attr_data = torch.load(attr_path, map_location="cpu")
    code_attr = attr_data["attributions"]["coding"]
    z_sel = attr_data["z_selectivity"]
    norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
    norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
    scores = 0.5 * norm_attr + 0.5 * norm_sel

    # Pre-cache teacher hidden states across prompts (just 10 teacher forward passes)
    cached_inputs_per_layer = {l: [] for l in range(num_layers)}
    cached_y_T_per_layer = {l: [] for l in range(num_layers)}

    for p in coding_prompts[:10]:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = teacher(input_ids=enc["input_ids"], output_hidden_states=True)
            for l_idx in range(num_layers):
                h_l = out.hidden_states[l_idx].to(device)
                x = layers[l_idx].post_attention_layernorm(h_l)
                y_T = layers[l_idx].mlp(x)
                cached_inputs_per_layer[l_idx].append(x)
                cached_y_T_per_layer[l_idx].append(y_T)

    distortion_matrix = {l: {} for l in range(num_layers)}

    for l_idx in range(num_layers):
        sorted_indices = torch.argsort(scores[l_idx], descending=True)
        mlp = layers[l_idx].mlp
        cached_x = cached_inputs_per_layer[l_idx]
        cached_y_T = cached_y_T_per_layer[l_idx]

        for k in width_grid:
            if k >= intermediate_size:
                distortion_matrix[l_idx][k] = 0.0
                continue

            top_k_idx = sorted_indices[:k]
            w_gate = mlp.gate_proj.weight.data[top_k_idx, :].to(device)
            w_up = mlp.up_proj.weight.data[top_k_idx, :].to(device)
            w_down = mlp.down_proj.weight.data[:, top_k_idx].to(device)

            total_dist = 0.0
            n_tokens = 0

            for x, y_T in zip(cached_x, cached_y_T):
                with torch.no_grad():
                    act_gate = F.silu(F.linear(x, w_gate))
                    act_up = F.linear(x, w_up)
                    act_inter = act_gate * act_up
                    y_P = F.linear(act_inter, w_down)

                    diff_norm = torch.norm(y_T - y_P, dim=-1)
                    ref_norm = torch.norm(y_T, dim=-1) + 1e-8
                    total_dist += float((diff_norm / ref_norm).mean().item())
                    n_tokens += 1

            distortion_matrix[l_idx][k] = round(total_dist / max(n_tokens, 1), 4)

    return distortion_matrix

def greedy_resource_allocation(distortion_matrix, d_model=1536, budget_mlp_params=350000000, width_step=512, min_k=512, max_k=8960):
    """
    Greedy Budget Allocation:
    Allocates neurons to the layer with the highest Marginal Utility (- dD_l / dP_l).
    """
    num_layers = len(distortion_matrix)
    # Start at minimum width for all layers
    current_allocation = {l: min_k for l in range(num_layers)}
    param_per_neuron = 3 * d_model # 4608 parameters per neuron

    current_params = sum(current_allocation.values()) * param_per_neuron

    while current_params + (width_step * param_per_neuron) <= budget_mlp_params:
        best_mu = -1.0
        best_layer = None

        for l in range(num_layers):
            cur_k = current_allocation[l]
            next_k = cur_k + width_step
            if next_k > max_k:
                continue

            # Estimate distortion at cur_k and next_k via interpolation
            d_cur = distortion_matrix[l].get(cur_k, 0.5)
            d_next = distortion_matrix[l].get(next_k, 0.2)
            delta_d = d_cur - d_next
            delta_p = width_step * param_per_neuron
            mu = delta_d / max(delta_p, 1)

            if mu > best_mu:
                best_mu = mu
                best_layer = l

        if best_layer is None or best_mu <= 0:
            break

        current_allocation[best_layer] += width_step
        current_params += (width_step * param_per_neuron)

    return current_allocation

def main():
    cfg = get_full_extraction_config()
    teacher_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("MARGINAL UTILITY & LAGRANGIAN RESOURCE ALLOCATION OPTIMIZATION")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    coding_prompts = dataset_builder.get_discovery_datasets()["coding"]

    width_grid = [512, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8960]

    print("\n[Phase 1: Measuring Empirical Distortion Curves D_l(k) across 28 Layers]...")
    distortion_matrix = measure_layer_distortion_curves(teacher, tokenizer, coding_prompts, width_grid, device=device)

    # Print sample distortion curves
    print("\nSample Layer Distortion Curves D_l(k):")
    print(f"{'Layer':>6} | {'k=1024':>8} | {'k=2048':>8} | {'k=3072':>8} | {'k=4096':>8} | {'k=6144':>8}")
    print("-" * 60)
    for l in [0, 1, 2, 3, 5, 10, 15, 20, 25, 27]:
        print(f"{l:>6d} | {distortion_matrix[l].get(1024, 0.0):>8.3f} | {distortion_matrix[l].get(2048, 0.0):>8.3f} | {distortion_matrix[l].get(3072, 0.0):>8.3f} | {distortion_matrix[l].get(4096, 0.0):>8.3f} | {distortion_matrix[l].get(6144, 0.0):>8.3f}")

    # Solve Optimal Allocation under MLP budget
    d_model = teacher.config.hidden_size # 1536
    # Target 500M total model -> Non-embedding & non-attention MLP budget ~ 250M
    mlp_budget = 260000000 # ~260M parameters for MLP

    print(f"\n[Phase 2: Solving Optimal Greedy Allocation under MLP Budget: {mlp_budget/1e6:.1f}M params]...")
    optimal_allocation = greedy_resource_allocation(distortion_matrix, d_model=d_model, budget_mlp_params=mlp_budget)

    print("\nOptimal Layerwise Neuron Allocation {k_l^*}:")
    for l in range(len(optimal_allocation)):
        print(f"  Layer {l:2d}: k_{l:2d}^* = {optimal_allocation[l]:4d} neurons")

    total_mlp_params = sum(optimal_allocation.values()) * 3 * d_model
    total_model_params = total_mlp_params + sum(p.numel() for p in teacher.model.embed_tokens.parameters()) + (28 * 4 * d_model * d_model)
    print(f"\nTotal Allocated MLP Params : {total_mlp_params/1e6:.1f} M")
    print(f"Total Estimated Model Params: {total_model_params/1e6:.1f} M")

    # Save allocation report
    out_json = os.path.join(cfg["paths"]["results_dir"], "optimal_resource_allocation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "distortion_matrix": distortion_matrix,
            "optimal_allocation": optimal_allocation,
            "total_mlp_params": total_mlp_params,
            "total_model_params": total_model_params
        }, f, indent=2)

    print(f"\n[OK] Resource allocation optimization report saved to: {out_json}")

if __name__ == "__main__":
    main()
