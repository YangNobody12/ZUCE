"""
Comprehensive Benchmark & Comparison Suite
Compares the Original Dense Base Model (1.5B) vs. the Extracted Mini Model (~0.5B).
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cap_extract.evaluation.benchmarks import CapabilityEvaluator
from cap_extract.evaluation.profiler import ModelProfiler
from cap_extract.datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS
from cap_extract.utils import prepare_inputs

def compute_perplexity(model, tokenizer, test_prompts):
    """Compute average perplexity across prompts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = model.device

    with torch.no_grad():
        for text in test_prompts[:5]:
            raw_inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
            inputs = prepare_inputs(raw_inputs, device)
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss.item()
            num_tok = input_ids.shape[1]
            total_loss += loss * num_tok
            total_tokens += num_tok

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    return round(ppl, 2), round(avg_loss, 4)

def run_benchmarks():
    base_model_name = "Qwen/Qwen2.5-1.5B"
    mini_model_dir = "./outputs/mini_model_0.5b"
    output_report_path = "./outputs/benchmark_comparison_report.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("="*80)
    print("CAPABILITY-AWARE MODEL EXTRACTION: BENCHMARK SUITE")
    print("="*80)

    # 1. Benchmark Extracted Mini Model
    print(f"\n[1/2] Benchmarking Extracted Mini Model: {mini_model_dir}...")
    mini_tokenizer = AutoTokenizer.from_pretrained(mini_model_dir)
    mini_model = AutoModelForCausalLM.from_pretrained(mini_model_dir, dtype=dtype).to(device)

    mini_profiler = ModelProfiler(mini_model, mini_tokenizer)
    mini_prof_stats = mini_profiler.profile_inference(prompt="def fibonacci(n):", max_new_tokens=64, warmup_runs=1, test_runs=3)

    mini_evaluator = CapabilityEvaluator(mini_model, mini_tokenizer)
    mini_code_eval = mini_evaluator.evaluate_coding_syntax_pass(CODING_PROMPTS[:5])
    mini_math_eval = mini_evaluator.evaluate_math_reasoning(MATH_PROMPTS[:5])
    mini_ppl, mini_loss = compute_perplexity(mini_model, mini_tokenizer, CODING_PROMPTS[:5])

    mini_params = sum(p.numel() for p in mini_model.parameters())
    mini_config = mini_model.config

    del mini_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Benchmark Dense Base Model
    print(f"\n[2/2] Benchmarking Base Dense Model: {base_model_name}...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        base_model = base_model.to(device)

    base_profiler = ModelProfiler(base_model, base_tokenizer)
    base_prof_stats = base_profiler.profile_inference(prompt="def fibonacci(n):", max_new_tokens=64, warmup_runs=1, test_runs=3)

    base_evaluator = CapabilityEvaluator(base_model, base_tokenizer)
    base_code_eval = base_evaluator.evaluate_coding_syntax_pass(CODING_PROMPTS[:5])
    base_math_eval = base_evaluator.evaluate_math_reasoning(MATH_PROMPTS[:5])
    base_ppl, base_loss = compute_perplexity(base_model, base_tokenizer, CODING_PROMPTS[:5])

    base_params = sum(p.numel() for p in base_model.parameters())
    base_config = base_model.config

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Compile Comparison Report
    report = {
        "device": device,
        "base_model": {
            "name": base_model_name,
            "parameters": base_params,
            "parameters_million": round(base_params / 1e6, 2),
            "layers": getattr(base_config, "num_hidden_layers", 28),
            "intermediate_size": getattr(base_config, "intermediate_size", 8960),
            "throughput_tokens_sec": base_prof_stats["tokens_per_second"],
            "latency_sec": base_prof_stats["avg_latency_sec"],
            "time_per_token_ms": base_prof_stats["time_per_token_ms"],
            "coding_syntax_pass_rate": base_code_eval["pass_rate_pct"],
            "math_coherence_rate": base_math_eval["coherence_rate_pct"],
            "perplexity_coding": base_ppl
        },
        "mini_model": {
            "name": "Qwen2.5-0.5B-Extracted-Coding",
            "parameters": mini_params,
            "parameters_million": round(mini_params / 1e6, 2),
            "layers": getattr(mini_config, "num_hidden_layers", 16),
            "intermediate_size": getattr(mini_config, "intermediate_size", 2304),
            "throughput_tokens_sec": mini_prof_stats["tokens_per_second"],
            "latency_sec": mini_prof_stats["avg_latency_sec"],
            "time_per_token_ms": mini_prof_stats["time_per_token_ms"],
            "coding_syntax_pass_rate": mini_code_eval["pass_rate_pct"],
            "math_coherence_rate": mini_math_eval["coherence_rate_pct"],
            "perplexity_coding": mini_ppl
        },
        "efficiency_gain": {
            "parameter_reduction_pct": round((1 - mini_params / base_params) * 100, 2),
            "speedup_factor": round(mini_prof_stats["tokens_per_second"] / max(1e-6, base_prof_stats["tokens_per_second"]), 2),
            "latency_reduction_pct": round((1 - mini_prof_stats["avg_latency_sec"] / base_prof_stats["avg_latency_sec"]) * 100, 2)
        }
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 4. Print Comparison Table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*80)
    print(f"{'Metric':<30} | {'Base Model (1.5B)':<20} | {'Mini Model (0.5B)':<20} | {'Delta / Improvement':<20}")
    print("-" * 96)
    print(f"{'Total Parameters':<30} | {report['base_model']['parameters_million']:<15} M | {report['mini_model']['parameters_million']:<15} M | -{report['efficiency_gain']['parameter_reduction_pct']}% Size")
    print(f"{'Hidden Layers':<30} | {report['base_model']['layers']:<20} | {report['mini_model']['layers']:<20} | -{report['base_model']['layers'] - report['mini_model']['layers']} Layers")
    print(f"{'MLP Intermediate Dim':<30} | {report['base_model']['intermediate_size']:<20} | {report['mini_model']['intermediate_size']:<20} | -74.3% Neurons")
    print(f"{'Throughput (tokens/sec)':<30} | {report['base_model']['throughput_tokens_sec']:<20} | {report['mini_model']['throughput_tokens_sec']:<20} | {report['efficiency_gain']['speedup_factor']}x Speedup")
    print(f"{'Time per token (ms)':<30} | {report['base_model']['time_per_token_ms']:<17} ms | {report['mini_model']['time_per_token_ms']:<17} ms | -{report['efficiency_gain']['latency_reduction_pct']}% Latency")
    print(f"{'Coding Syntax Pass Rate':<30} | {report['base_model']['coding_syntax_pass_rate']:<19}% | {report['mini_model']['coding_syntax_pass_rate']:<19}% | Post-Surgery")
    print(f"{'Perplexity (Coding)':<30} | {report['base_model']['perplexity_coding']:<20} | {report['mini_model']['perplexity_coding']:<20} | Initial Surgery")
    print("="*80)
    print(f"\n[OK] Full benchmark results saved to: {output_report_path}")

if __name__ == "__main__":
    run_benchmarks()
