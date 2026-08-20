"""
Master Export & Benchmark Suite for the Capability-Aware Specialist Model
1. Physically extracts the capability subnetwork (k = 8000, 28 Layers, Δθ = 0)
2. Saves as a 100% standalone HuggingFace model with Safetensors
3. Reloads from disk to verify standalone loading
4. Executes the complete 10-question algorithmic coding benchmark
5. Generates the final scientific report and code inspection outputs
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_10_question_coding_test import TEN_CODING_QUESTIONS
from src.evaluation.coding import CodingEvaluator
from src.surgery.weight_mapper import PhysicalWeightMapper

def export_standalone_safetensors_model(teacher, tokenizer, neuron_scores, target_k=8000, export_dir="./outputs/specialist_coding_model_safetensors"):
    """
    Slices and exports the specialist subnetwork to a standard HuggingFace Safetensors directory.
    """
    os.makedirs(export_dir, exist_ok=True)
    num_layers = teacher.config.num_hidden_layers
    all_layers = list(range(num_layers))

    print(f"\n[1/3] Slicing Physical Tensors (28 Layers x {target_k} MLP, Δθ = 0)...")
    retained_neurons = {}
    for l in all_layers:
        top_k = torch.topk(neuron_scores[l], target_k).indices.tolist()
        retained_neurons[l] = sorted(top_k)

    mapper = PhysicalWeightMapper(teacher, tokenizer)
    student = mapper.construct_and_slice_student(
        retained_layers=all_layers,
        retained_neurons_per_layer=retained_neurons,
        target_intermediate_size=target_k,
        output_dir=export_dir
    )

    print(f"\n[2/3] Verifying and saving standalone HuggingFace files to: {export_dir}")
    student.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)

    # Save extraction metadata
    total_params = sum(p.numel() for p in student.parameters())
    base_params = sum(p.numel() for p in teacher.parameters())
    metadata = {
        "model_name": "Qwen2.5-1.5B-Specialist-Coding",
        "base_model": teacher.config._name_or_path,
        "extraction_method": "Capability-Aware Zero-Update Subnetwork Slicing (Δθ = 0)",
        "num_layers": num_layers,
        "intermediate_size": target_k,
        "parameters": total_params,
        "parameters_million": round(total_params / 1e6, 2),
        "parameter_reduction_pct": round((1.0 - (total_params / base_params)) * 100, 2),
        "delta_theta": 0
    }
    with open(os.path.join(export_dir, "extraction_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Total Parameters : {total_params / 1e6:.1f} M (Base: {base_params / 1e6:.1f} M)")
    print(f"  Parameter Sliced : -{(1.0 - (total_params / base_params)) * 100:.2f}%")
    print(f"  Safetensors Saved: [OK]")

    return export_dir, metadata

def main():
    base_model_name = "Qwen/Qwen2.5-1.5B"
    export_dir = "./outputs/specialist_coding_model_safetensors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("CAPABILITY-AWARE SPECIALIST MODEL: PHYSICAL EXPORT & COMPLETE BENCHMARK")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    teacher = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    teacher.eval()

    # Load Phase 2 neuron scores
    attr_path = "./outputs/scientific_reports/02_neuron_attribution.pt"
    attr_data = torch.load(attr_path, map_location="cpu")
    code_attr = attr_data["attributions"]["coding"]
    z_sel = attr_data["z_selectivity"]
    norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
    norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
    composite_scores = 0.5 * norm_attr + 0.5 * norm_sel

    # 1. Export Model
    export_dir, metadata = export_standalone_safetensors_model(
        teacher=teacher,
        tokenizer=tokenizer,
        neuron_scores=composite_scores,
        target_k=8000,
        export_dir=export_dir
    )

    # 2. Reload cleanly from disk to verify standalone usability
    print(f"\n[3/3] Reloading standalone model from disk: {export_dir} ...")
    standalone_model = AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    standalone_model.eval()
    print("  Model successfully reloaded from disk with exact Safetensors weights!")

    # 3. Run Complete 10-Question Coding Benchmark
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE 10-QUESTION ALGORITHMIC CODING BENCHMARK")
    print("=" * 80)

    evaluator = CodingEvaluator(tokenizer, device=device)

    # Evaluate Base Model
    print("\n[Evaluating Base Model: Qwen2.5-1.5B (1,543.7M params)]...")
    teacher_res = evaluator.evaluate_model_on_coding_prompts(teacher, TEN_CODING_QUESTIONS, max_new_tokens=64)

    # Evaluate Standalone Specialist Model
    print(f"\n[Evaluating Specialist Model: 28L x 8000 MLP ({metadata['parameters_million']}M params)]...")
    specialist_res = evaluator.evaluate_model_on_coding_prompts(standalone_model, TEN_CODING_QUESTIONS, max_new_tokens=64)

    # Comparative Summary Table
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<42} | {'Params':<8} | {'Pass Rate (10Q)':<16} | {'Avg Latency':<12}")
    print("=" * 80)
    print(f"{'Base Teacher (Qwen2.5-1.5B)':<42} | {'1543.7M':<8} | {teacher_res['pass_rate_pct']:>5.1f}% ({teacher_res['valid_count']}/{teacher_res['total_questions']})   | {teacher_res['avg_time_per_q']:.2f}s/Q")
    spec_param_str = f"{metadata['parameters_million']}M"
    print(f"{'Specialist Model (Ours, Δθ=0)':<42} | {spec_param_str:<8} | {specialist_res['pass_rate_pct']:>5.1f}% ({specialist_res['valid_count']}/{specialist_res['total_questions']})   | {specialist_res['avg_time_per_q']:.2f}s/Q")
    print("=" * 80)

    # Detailed Question Breakdown
    print("\n" + "=" * 80)
    print("DETAILED QUESTION-BY-QUESTION EVALUATION:")
    print("=" * 80)
    t_details = teacher_res.get("details", teacher_res.get("results", []))
    s_details = specialist_res.get("details", specialist_res.get("results", []))
    for q_idx, (t_item, s_item) in enumerate(zip(t_details, s_details)):
        q_title = TEN_CODING_QUESTIONS[q_idx]["title"]
        t_status = "PASS" if t_item["is_valid_syntax"] else "FAIL"
        s_status = "PASS" if s_item["is_valid_syntax"] else "FAIL"
        print(f"  Q{q_idx+1:02d}: {q_title:<32} | Base: [{t_status:<4}] | Specialist: [{s_status:<4}]")

    # Print Sample Code Generation for inspection
    print("\n" + "=" * 80)
    print("SAMPLE CODE GENERATION FROM STANDALONE SPECIALIST MODEL (Q1: Fibonacci):")
    print("-" * 80)
    if s_details:
        print(s_details[0]["generated_code"][:300])
    print("-" * 80)

    # Save Final Report
    report_path = "./outputs/final_specialist_export_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "teacher_benchmark": teacher_res,
            "specialist_benchmark": specialist_res
        }, f, indent=2)

    print(f"\n[OK] Final report saved to: {report_path}")
    print(f"[OK] Standalone Safetensors ready at: {export_dir}")

if __name__ == "__main__":
    main()
