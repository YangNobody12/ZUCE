"""
Scientific Ablation: Disentangling Depth Damage vs Width Damage
Tests 4 controlled architectures to isolate whether failure is caused by:
1. Depth Pruning (Layer Dropping: 28 -> 16 layers)
2. Width Pruning (MLP Slicing: 8960 -> 2304 neurons)
3. Depth x Width Interaction

Architectures evaluated:
- A: Teacher (28 Layers, 8960 MLP) - 1.54B Baseline
- B: Depth-Only Pruning (16 Layers, 8960 MLP) - ~0.95B
- C: Width-Only Pruning (28 Layers, 2304 MLP) - ~0.78B
- D: Combined Pruning (16 Layers, 2304 MLP) - ~0.49B

Metrics evaluated:
- Coding Syntax & Pass Rate (10 Questions)
- Perplexity / Cross-Entropy Loss on Coding Validation Split
- Language Coherence / Repetition Check
"""

import os
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.surgery.weight_mapper import PhysicalWeightMapper
from src.evaluation.coding import CodingEvaluator
from run_10_question_coding_test import TEN_CODING_QUESTIONS

def build_depth_only_model(teacher, tokenizer, retained_layers, output_dir="./outputs/ablation_depth_only_16L"):
    """Constructs 16 Layers x 8960 MLP (Zero width pruning)."""
    mapper = PhysicalWeightMapper(teacher, tokenizer)
    intermediate_size = teacher.config.intermediate_size
    retained_neurons = {l: list(range(intermediate_size)) for l in retained_layers}
    return mapper.construct_and_slice_student(
        retained_layers=retained_layers,
        retained_neurons_per_layer=retained_neurons,
        target_intermediate_size=intermediate_size,
        output_dir=output_dir
    )

def build_width_only_model(teacher, tokenizer, neuron_attribution, target_intermediate=2304, output_dir="./outputs/ablation_width_only_28L"):
    """Constructs 28 Layers x 2304 MLP (Zero layer dropping)."""
    mapper = PhysicalWeightMapper(teacher, tokenizer)
    num_layers = teacher.config.num_hidden_layers
    all_layers = list(range(num_layers))

    retained_neurons = {}
    for l in all_layers:
        top_k = torch.topk(neuron_attribution[l], target_intermediate).indices.tolist()
        retained_neurons[l] = sorted(top_k)

    return mapper.construct_and_slice_student(
        retained_layers=all_layers,
        retained_neurons_per_layer=retained_neurons,
        target_intermediate_size=target_intermediate,
        output_dir=output_dir
    )

def main():
    cfg = get_full_extraction_config()
    teacher_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("SCIENTIFIC ABLATION: DISENTANGLING DEPTH DAMAGE VS WIDTH DAMAGE")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)

    # Retained 16 layers from Phase 6 mapping
    retained_16_layers = [0, 1, 2, 3, 4, 5, 6, 7, 12, 14, 15, 16, 18, 20, 26, 27]

    # Load Phase 2 neuron scores
    attr_path = os.path.join(cfg["paths"]["results_dir"], "02_neuron_attribution.pt")
    attr_data = torch.load(attr_path, map_location="cpu")
    code_attr = attr_data["attributions"]["coding"]
    z_sel = attr_data["z_selectivity"]
    norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
    norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
    composite_neuron_scores = 0.5 * norm_attr + 0.5 * norm_sel

    # 1. Build Model B: Depth-Only (16L x 8960 MLP)
    print("\n[Building Model B: Depth-Only Pruning (16 Layers x 8960 MLP)]...")
    model_depth_only = build_depth_only_model(teacher, tokenizer, retained_16_layers)
    model_depth_only = model_depth_only.to(device)

    # 2. Build Model C: Width-Only (28 Layers x 2304 MLP)
    print("\n[Building Model C: Width-Only Pruning (28 Layers x 2304 MLP)]...")
    model_width_only = build_width_only_model(teacher, tokenizer, composite_neuron_scores, target_intermediate=2304)
    model_width_only = model_width_only.to(device)

    # 3. Load Model D: Combined (16 Layers x 2304 MLP)
    student_dir = cfg["paths"]["student_model_dir"]
    model_combined = AutoModelForCausalLM.from_pretrained(student_dir, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)

    # 4. Evaluate all 4 models on Coding Benchmark and Validation Loss
    dataset_builder = TaskDatasetBuilder(tokenizer)
    val_coding_prompts = dataset_builder.get_validation_datasets()["coding"]

    evaluator = CodingEvaluator(tokenizer, device=device)

    models = {
        "A: Teacher Baseline (28L, 8960 MLP)": teacher,
        "B: Depth-Only Pruning (16L, 8960 MLP)": model_depth_only,
        "C: Width-Only Pruning (28L, 2304 MLP)": model_width_only,
        "D: Combined Pruning (16L, 2304 MLP)": model_combined
    }

    ablation_results = {}
    print("\n" + "=" * 80)
    print(f"{'Condition':<40} | {'Params':<8} | {'Val CE Loss':<12} | {'Pass Rate (10Q)':<16}")
    print("=" * 80)

    for cond_name, m in models.items():
        m.eval()
        params = sum(p.numel() for p in m.parameters())

        # Compute Validation Cross-Entropy Loss
        total_loss = 0.0
        n_val = 0
        for vp in val_coding_prompts:
            enc = tokenizer(vp, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = m(input_ids=enc["input_ids"], labels=enc["input_ids"])
                total_loss += float(out.loss.item())
                n_val += 1
        avg_loss = total_loss / max(n_val, 1)

        # Run 10-Question Coding Benchmark
        res = evaluator.evaluate_model_on_coding_prompts(m, TEN_CODING_QUESTIONS, max_new_tokens=128)

        ablation_results[cond_name] = {
            "parameters_million": round(params / 1e6, 2),
            "val_loss": round(avg_loss, 4),
            "pass_rate_pct": res["pass_rate_pct"],
            "valid_count": res["valid_count"],
            "total_questions": res["total_questions"]
        }

        print(f"{cond_name:<40} | {params/1e6:>6.1f}M | {avg_loss:>12.4f} | {res['pass_rate_pct']:>6.1f}% ({res['valid_count']}/{res['total_questions']})")

    out_json = os.path.join(cfg["paths"]["results_dir"], "depth_vs_width_ablation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[OK] Disentanglement Ablation Report saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
