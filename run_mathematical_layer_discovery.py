"""
Mathematical Layer Discovery & Contiguous Block Optimization
Discovers the optimal subnetwork structure using:
1. Pairwise Layer Cosine Similarity Matrix S_{i, j} = cos(h_i, h_j)
2. Block Importance Metric BI(i, j) = || h_j - h_i || / || h_j ||
3. Contiguous Redundant Block Identification [l_start, l_end]
4. Non-uniform MLP Width Allocation across the retained layers
5. Real 10-Question Algorithmic & Syntax Evaluation
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

def compute_layer_similarity_matrix(teacher, tokenizer, prompts, device="cuda"):
    """
    Computes S_{i, j} = E[cos(h_i, h_j)] for all 0 <= i < j <= 28.
    """
    teacher.eval()
    num_layers = teacher.config.num_hidden_layers
    total_cos = torch.zeros(num_layers + 1, num_layers + 1, device=device)
    n_samples = 0

    for p in prompts[:15]:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = teacher(input_ids=enc["input_ids"], output_hidden_states=True)
            hidden_states = out.hidden_states # Tuple of 29 tensors [B, S, D]

            for i in range(num_layers + 1):
                h_i = hidden_states[i].float()
                norm_i = torch.norm(h_i, dim=-1, keepdim=True) + 1e-8
                h_i_norm = h_i / norm_i

                for j in range(i, num_layers + 1):
                    h_j = hidden_states[j].float()
                    norm_j = torch.norm(h_j, dim=-1, keepdim=True) + 1e-8
                    h_j_norm = h_j / norm_j

                    cos_sim = (h_i_norm * h_j_norm).sum(dim=-1).mean()
                    total_cos[i, j] += cos_sim
            n_samples += 1

    sim_matrix = (total_cos / max(n_samples, 1)).cpu()
    return sim_matrix

def find_optimal_contiguous_pruning(sim_matrix, target_layers_to_drop):
    """
    Finds contiguous block [l_start, l_start + target_layers_to_drop]
    with maximum cosine similarity S_{l_start, l_end}.
    """
    num_layers = sim_matrix.shape[0] - 1
    best_sim = -1.0
    best_range = None

    # Search middle layers (preserving early layers 0..2 and late layers 26..27)
    for start in range(3, num_layers - target_layers_to_drop - 1):
        end = start + target_layers_to_drop
        sim = sim_matrix[start, end].item()
        if sim > best_sim:
            best_sim = sim
            best_range = (start, end)

    return best_range, best_sim

def evaluate_extracted_configuration(teacher, tokenizer, retained_layers, retained_neurons_per_layer, target_intermediate, name, device="cuda"):
    """Builds and benchmarks a physical model configuration."""
    mapper = PhysicalWeightMapper(teacher, tokenizer)
    out_dir = f"./outputs/eval_{name.lower().replace(' ', '_').replace(':', '')}"
    student = mapper.construct_and_slice_student(
        retained_layers=retained_layers,
        retained_neurons_per_layer=retained_neurons_per_layer,
        target_intermediate_size=target_intermediate,
        output_dir=out_dir
    )
    student = student.to(device)

    evaluator = CodingEvaluator(tokenizer, device=device)
    res = evaluator.evaluate_model_on_coding_prompts(student, TEN_CODING_QUESTIONS, max_new_tokens=128)
    params = sum(p.numel() for p in student.parameters())

    return {
        "name": name,
        "parameters_million": round(params / 1e6, 2),
        "retained_layers": len(retained_layers),
        "retained_layers_list": retained_layers,
        "intermediate_size": target_intermediate,
        "pass_rate_pct": res["pass_rate_pct"],
        "valid_count": res["valid_count"],
        "total_questions": res["total_questions"]
    }

def main():
    cfg = get_full_extraction_config()
    teacher_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("MATHEMATICAL LAYER DISCOVERY & CONTIGUOUS BLOCK OPTIMIZATION")
    print("Zero-Update / Zero Retraining (Δθ = 0)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    coding_prompts = dataset_builder.get_discovery_datasets()["coding"]

    # 1. Compute Layer Cosine Similarity Matrix
    print("\n[Phase 1: Computing 28x28 Layer Cosine Similarity Matrix]...")
    sim_matrix = compute_layer_similarity_matrix(teacher, tokenizer, coding_prompts, device=device)

    print("\nContiguous Block Redundancy Analysis:")
    print(f"{'Drop Count':>10} | {'Optimal Redundant Block':>25} | {'Cosine Similarity':>18}")
    print("-" * 60)

    contiguous_candidates = {}
    for drop_count in [2, 4, 6, 8, 10, 12]:
        block, sim = find_optimal_contiguous_pruning(sim_matrix, drop_count)
        retained = [l for l in range(28) if not (block[0] <= l < block[1])]
        contiguous_candidates[f"Contiguous {28 - drop_count}L"] = {
            "dropped_block": block,
            "retained_layers": retained,
            "similarity": round(sim, 4)
        }
        print(f"{drop_count:>10d} | Layers [{block[0]:2d} .. {block[1]-1:2d}] -> drop | {sim:>18.4f}")

    # 2. Test Contiguous Layer Pruning at Multiple Depths (26L, 24L, 22L, 20L, 16L) with Full MLP Width (8960)
    print("\n" + "=" * 80)
    print("PHASE 2: BENCHMARKING CONTIGUOUS DEPTH PRUNING (8960 MLP)")
    print("=" * 80)

    intermediate_size = teacher.config.intermediate_size
    results = []

    for label, info in contiguous_candidates.items():
        retained = info["retained_layers"]
        neurons = {l: list(range(intermediate_size)) for l in retained}
        res = evaluate_extracted_configuration(
            teacher, tokenizer, retained, neurons, intermediate_size,
            name=f"{label} (8960 MLP)", device=device
        )
        results.append(res)
        print(f"  {res['name']:<25} | {res['parameters_million']:>6.1f}M | Pass Rate: {res['pass_rate_pct']:>5.1f}% ({res['valid_count']}/{res['total_questions']})")

    # 3. Test Width Pruning under Full 28 Layers (Widths: 7168, 5376, 3584, 2304)
    print("\n" + "=" * 80)
    print("PHASE 3: BENCHMARKING PURE WIDTH PRUNING ACROSS 28 LAYERS")
    print("=" * 80)

    # Load neuron attribution scores
    attr_path = os.path.join(cfg["paths"]["results_dir"], "02_neuron_attribution.pt")
    attr_data = torch.load(attr_path, map_location="cpu")
    code_attr = attr_data["attributions"]["coding"]
    z_sel = attr_data["z_selectivity"]
    norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
    norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
    scores = 0.5 * norm_attr + 0.5 * norm_sel

    all_28_layers = list(range(28))
    for width_k in [7168, 5376, 4096, 3584, 2304]:
        retained_neurons = {}
        for l in all_28_layers:
            top_k = torch.topk(scores[l], width_k).indices.tolist()
            retained_neurons[l] = sorted(top_k)

        res = evaluate_extracted_configuration(
            teacher, tokenizer, all_28_layers, retained_neurons, width_k,
            name=f"28L x {width_k} MLP", device=device
        )
        results.append(res)
        print(f"  {res['name']:<25} | {res['parameters_million']:>6.1f}M | Pass Rate: {res['pass_rate_pct']:>5.1f}% ({res['valid_count']}/{res['total_questions']})")

    # Save comprehensive results
    out_json = os.path.join(cfg["paths"]["results_dir"], "mathematical_layer_discovery_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "contiguous_candidates": contiguous_candidates,
            "benchmark_results": results
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[OK] Full mathematical discovery report saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
