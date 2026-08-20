"""
Fine-Grained Sub-Billion Boundary Sweep (20-Question Coding Benchmark)
Tests MLP intermediate sizes:
k in [5500, 5200, 5000, 4800, 4500, 4200, 4000]
Measures exact parameter count, pass rate on 20 questions, and latency!
"""

import os
import sys
import copy
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_advanced_specialist_suite import EXTENDED_20_CODING_QUESTIONS
from src.evaluation.coding import CodingEvaluator
from src.surgery.weight_mapper import PhysicalWeightMapper

def main():
    base_model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 95)
    print("FINE-GRAINED SUB-BILLION BOUNDARY SWEEP (20 ALGORITHMIC PROBLEMS)")
    print("Testing the exact minimum parameter threshold where capability is preserved")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    teacher = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    teacher.eval()

    # Load neuron scores
    attr_path = "./outputs/scientific_reports/02_neuron_attribution.pt"
    attr_data = torch.load(attr_path, map_location="cpu")
    code_attr = attr_data["attributions"]["coding"]
    z_sel = attr_data["z_selectivity"]
    norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
    norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
    composite_scores = 0.5 * norm_attr + 0.5 * norm_sel

    evaluator = CodingEvaluator(tokenizer, device=device)
    base_params = sum(p.numel() for p in teacher.parameters())

    # Evaluate Teacher Baseline
    print("\n[Evaluating Base Model: Qwen2.5-1.5B (1,543.7M params)]...")
    teacher_res = evaluator.evaluate_model_on_coding_prompts(teacher, EXTENDED_20_CODING_QUESTIONS, max_new_tokens=64)

    results_table = []
    results_table.append({
        "name": "Base Teacher (1.54B)",
        "k": 8960,
        "params_m": round(base_params / 1e6, 1),
        "reduction_pct": 0.0,
        "pass_rate_pct": teacher_res["pass_rate_pct"],
        "valid_count": teacher_res["valid_count"],
        "total_q": teacher_res["total_questions"],
        "avg_latency": teacher_res["avg_time_per_q"],
        "ncd": 1.0
    })

    # Sweep widths
    k_values = [5500, 5200, 5000, 4800, 4500, 4200, 4000]

    for k in k_values:
        export_dir = f"./outputs/sweep_k_{k}"
        mapper = PhysicalWeightMapper(teacher, tokenizer)
        
        retained_neurons = {}
        for l in range(28):
            top_k = torch.topk(composite_scores[l], k).indices.tolist()
            retained_neurons[l] = sorted(top_k)

        student = mapper.construct_and_slice_student(
            retained_layers=list(range(28)),
            retained_neurons_per_layer=retained_neurons,
            target_intermediate_size=k,
            output_dir=export_dir
        )
        student = student.to(device)
        student.eval()

        res = evaluator.evaluate_model_on_coding_prompts(student, EXTENDED_20_CODING_QUESTIONS, max_new_tokens=64)
        params = sum(p.numel() for p in student.parameters())

        reduc_pct = (1.0 - (params / base_params)) * 100
        r_code = res["pass_rate_pct"] / max(teacher_res["pass_rate_pct"], 1e-6)
        param_ratio = params / base_params
        ncd = r_code / max(param_ratio, 1e-6)

        results_table.append({
            "name": f"Specialist (k={k})",
            "k": k,
            "params_m": round(params / 1e6, 1),
            "reduction_pct": round(reduc_pct, 1),
            "pass_rate_pct": res["pass_rate_pct"],
            "valid_count": res["valid_count"],
            "total_q": res["total_questions"],
            "avg_latency": res["avg_time_per_q"],
            "ncd": round(ncd, 3)
        })

        print(f"\n[Result] k={k:<4d} | Params: {params/1e6:6.1f}M (-{reduc_pct:4.1f}%) | Pass Rate: {res['pass_rate_pct']:4.1f}% ({res['valid_count']:2d}/{res['total_questions']:2d}) | Latency: {res['avg_time_per_q']:.2f}s | NCD: {ncd:.2f}x")

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print("SUB-BILLION FINE-GRAINED SWEEP SUMMARY TABLE")
    print("=" * 95)
    print(f"{'Model Architecture':<28} | {'Params':<8} | {'Reduction':<10} | {'Pass Rate (20Q)':<18} | {'Latency':<10} | {'NCD':<6}")
    print("-" * 95)
    for row in results_table:
        print(f"{row['name']:<28} | {row['params_m']:>6.1f}M | {row['reduction_pct']:>8.1f}% | {row['pass_rate_pct']:>6.1f}% ({row['valid_count']:2d}/{row['total_q']:2d})    | {row['avg_latency']:>6.2f}s/Q | {row['ncd']:>5.2f}x")
    print("=" * 95)

    # Save summary
    out_json = "./outputs/fine_grained_sub_billion_sweep_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "sweep_results": results_table
        }, f, indent=2)

    print(f"\n[OK] Fine-Grained Sweep Report saved to: {out_json}")

if __name__ == "__main__":
    main()
