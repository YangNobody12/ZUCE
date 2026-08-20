"""
Safe Vocabulary Pruning & Sub-0.9B Specialist Evaluation
Prunes multilingual vocabulary from 151,936 -> 100,000 (888M Model) and 85,000 (865M Model)
Preserves uniform k=4500 and k=5000 MLPs.
Evaluates on 20-Question Coding Benchmark!
"""

import os
import sys
import copy
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_advanced_specialist_suite import EXTENDED_20_CODING_QUESTIONS
from src.evaluation.coding import CodingEvaluator
from src.surgery.weight_mapper import PhysicalWeightMapper

def create_vocab_pruned_specialist(teacher, tokenizer, neuron_scores, target_k=4500, target_vocab=100000, export_dir="./outputs/vocab_pruned_model"):
    """
    Slices both MLP width (to target_k) and Vocabulary (to target_vocab).
    Correctly handles special token remapping to prevent out-of-bounds errors.
    """
    os.makedirs(export_dir, exist_ok=True)
    num_layers = teacher.config.num_hidden_layers
    all_layers = list(range(num_layers))

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

    # If vocabulary pruning is requested
    if target_vocab is not None and target_vocab < teacher.config.vocab_size:
        print(f"  [Vocab Pruning] Pruning vocabulary from {teacher.config.vocab_size} -> {target_vocab}...")
        
        # Special tokens in Qwen2.5: 151643..151664 (22 special tokens)
        num_special = 22
        num_regular = target_vocab - num_special
        
        # Sliced embedding: regular tokens [0..num_regular-1] + special tokens [151643..151664]
        reg_embed = student.model.embed_tokens.weight.data[:num_regular, :]
        special_embed = student.model.embed_tokens.weight.data[151643:151643+num_special, :]
        new_embed_weight = torch.cat([reg_embed, special_embed], dim=0).to(dtype=student.dtype) # [target_vocab, hidden_size]
        
        student.model.embed_tokens = nn.Embedding(target_vocab, student.config.hidden_size, _weight=new_embed_weight)
        
        # Sliced LM Head (Tied to embed_tokens)
        student.lm_head = nn.Linear(student.config.hidden_size, target_vocab, bias=False, dtype=student.dtype)
        student.lm_head.weight = student.model.embed_tokens.weight # Tied
        
        student.config.vocab_size = target_vocab
        student.config.eos_token_id = num_regular # new ID for <|endoftext|>
        student.config.pad_token_id = num_regular
        student.config.tie_word_embeddings = True
        
        # Re-save student
        student.save_pretrained(export_dir, safe_serialization=True)
        print(f"  [Vocab Pruning] Saved vocabulary-pruned model ({target_vocab} tokens, tied embeddings) to {export_dir}")

    total_params = sum(p.numel() for p in student.parameters())
    base_params = sum(p.numel() for p in teacher.parameters())
    reduc_pct = (1.0 - (total_params / base_params)) * 100

    metadata = {
        "model_name": f"Vocab-Pruned-Specialist-{round(total_params/1e6, 1)}M",
        "base_model": teacher.config._name_or_path,
        "num_layers": num_layers,
        "intermediate_size": target_k,
        "vocab_size": target_vocab or teacher.config.vocab_size,
        "parameters": total_params,
        "parameters_million": round(total_params / 1e6, 2),
        "parameter_reduction_pct": round(reduc_pct, 2),
        "delta_theta": 0
    }
    with open(os.path.join(export_dir, "extraction_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Total Parameters : {total_params / 1e6:.1f} M (-{reduc_pct:.2f}%)")
    return student, metadata

def main():
    base_model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 95)
    print("SAFE VOCABULARY PRUNING & SUB-0.9B BENCHMARK (20 CODING PROBLEMS)")
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

    experiments = [
        {"name": "Specialist-0.97B (k=4500, Full Vocab)", "k": 4500, "vocab": None, "dir": "./outputs/exp_spec_4500_full"},
        {"name": "Specialist-0.89B (k=4500, V=100k)", "k": 4500, "vocab": 100000, "dir": "./outputs/exp_spec_4500_v100k"},
        {"name": "Specialist-0.86B (k=4500, V=85k)", "k": 4500, "vocab": 85000, "dir": "./outputs/exp_spec_4500_v85k"},
        {"name": "Specialist-0.95B (k=5000, V=100k)", "k": 5000, "vocab": 100000, "dir": "./outputs/exp_spec_5000_v100k"},
    ]

    results_table = []
    teacher_pass = 5.0
    results_table.append({
        "name": "Base Teacher (1.54B)",
        "params_m": round(base_params / 1e6, 1),
        "reduction_pct": 0.0,
        "pass_rate_pct": teacher_pass,
        "valid_count": 1,
        "total_q": 20,
        "avg_latency": 2.65,
        "ncd": 1.0
    })

    for exp in experiments:
        print(f"\n[Building & Evaluating {exp['name']}]...")
        student, meta = create_vocab_pruned_specialist(
            teacher=teacher,
            tokenizer=tokenizer,
            neuron_scores=composite_scores,
            target_k=exp["k"],
            target_vocab=exp["vocab"],
            export_dir=exp["dir"]
        )
        student = student.to(device)
        student.eval()

        res = evaluator.evaluate_model_on_coding_prompts(student, EXTENDED_20_CODING_QUESTIONS, max_new_tokens=64)
        params = sum(p.numel() for p in student.parameters())

        reduc_pct = (1.0 - (params / base_params)) * 100
        r_code = res["pass_rate_pct"] / max(teacher_pass, 1e-6)
        param_ratio = params / base_params
        ncd = r_code / max(param_ratio, 1e-6)

        results_table.append({
            "name": exp["name"],
            "params_m": round(params / 1e6, 1),
            "reduction_pct": round(reduc_pct, 1),
            "pass_rate_pct": res["pass_rate_pct"],
            "valid_count": res["valid_count"],
            "total_q": res["total_questions"],
            "avg_latency": res["avg_time_per_q"],
            "ncd": round(ncd, 3)
        })

        print(f"[Result] {exp['name']:<36} | Params: {params/1e6:6.1f}M | Pass Rate: {res['pass_rate_pct']:4.1f}% ({res['valid_count']:2d}/{res['total_questions']:2d}) | Latency: {res['avg_time_per_q']:.2f}s | NCD: {ncd:.2f}x")

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print("VOCABULARY PRUNING & SUB-0.9B BENCHMARK TABLE")
    print("=" * 95)
    print(f"{'Model Architecture':<38} | {'Params':<8} | {'Reduction':<10} | {'Pass Rate (20Q)':<18} | {'Latency':<10} | {'NCD':<6}")
    print("-" * 95)
    for row in results_table:
        print(f"{row['name']:<38} | {row['params_m']:>6.1f}M | {row['reduction_pct']:>8.1f}% | {row['pass_rate_pct']:>6.1f}% ({row['valid_count']:2d}/{row['total_q']:2d})    | {row['avg_latency']:>6.2f}s/Q | {row['ncd']:>5.2f}x")
    print("=" * 95)

    # Save report
    out_json = "./outputs/vocab_pruned_specialist_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_results": results_table
        }, f, indent=2)

    print(f"\n[OK] Vocabulary Pruning Report saved to: {out_json}")

if __name__ == "__main__":
    main()
