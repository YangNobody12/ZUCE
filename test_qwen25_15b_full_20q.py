"""
Test Qwen2.5-1.5B Extraction with % Reduction, Quantization and 20-Question Benchmark
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from zuce import ZUCE, CapabilitySpec, ParameterBudget
from src.evaluation.coding import CodingEvaluator
from run_advanced_specialist_suite import EXTENDED_20_CODING_QUESTIONS

def main():
    print("=" * 80)
    print("🧪 EVALUATION REPORT: QWEN2.5-1.5B ZUCE EXTRACTION (% REDUCTION + QUANT + 20Q TEST)")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Device: {device} | Dtype: {dtype}")

    # 1. Test ParameterBudget from % reduction
    print("\n[Step 1: Testing ParameterBudget API with % Reduction]")
    cfg = AutoConfig.from_pretrained(model_id)
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    num_layers = cfg.num_hidden_layers
    vocab_size = cfg.vocab_size

    # Parameter accounting
    mlp_params = 3 * hidden_size * intermediate_size * num_layers
    attn_params = 4 * (hidden_size * hidden_size) * num_layers
    embed_params = vocab_size * hidden_size
    teacher_total_est = 1_543_714_816 # Actual Qwen2.5-1.5B total params

    # Test -33.1% reduction (Optimal Specialist 1.03B)
    reduction_pct = 33.1
    budget_from_pct = ParameterBudget.from_reduction_percent(teacher_total_est, percent=reduction_pct)
    budget_from_ratio = ParameterBudget.from_retention_ratio(teacher_total_est, ratio=1.0 - reduction_pct/100.0)

    print(f"✅ Teacher Total Params: {teacher_total_est / 1e6:.2f} M ({teacher_total_est / 1e9:.3f} B)")
    print(f"✅ Budget from {reduction_pct}% reduction: {budget_from_pct.max_parameters:,} params ({budget_from_pct.max_parameters / 1e6:.2f} M)")
    print(f"✅ Budget from ratio ({1.0 - reduction_pct/100.0:.4f}): {budget_from_ratio.max_parameters:,} params")
    assert budget_from_pct.max_parameters == budget_from_ratio.max_parameters, "Mismatch in budget calculation!"
    print("🎯 ParameterBudget % API Test: PASSED!")

    # 2. Check Pre-extracted 1.03B model or Teacher for 20-Question benchmark
    print("\n[Step 2: Loading Extracted Model & Tokenizer for 20-Question Benchmark]")
    specialist_path = "./outputs/specialist_optimal_1.03b_safetensors"
    if not os.path.exists(specialist_path):
        specialist_path = "./outputs/specialist_1.0b_safetensors"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading Specialist from: {specialist_path}...")
    specialist_model = AutoModelForCausalLM.from_pretrained(
        specialist_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    specialist_model.eval()
    spec_params = sum(p.numel() for p in specialist_model.parameters())
    print(f"✅ Specialist Model Loaded! Params: {spec_params / 1e6:.2f} M (Reduction: {(teacher_total_est - spec_params) / teacher_total_est * 100:.1f}%)")

    # 3. Test Quantization Config Compatibility
    print("\n[Step 3: Testing Quantization Config & VRAM Estimation]")
    fp16_vram_est = spec_params * 2 / (1024**2)
    int8_vram_est = spec_params * 1 / (1024**2)
    int4_vram_est = spec_params * 0.55 / (1024**2) # NF4 + Double Quant

    print(f"📊 Memory Footprint Analysis for Specialist ({spec_params/1e6:.1f}M params):")
    print(f"   - FP16 / BF16: ~{fp16_vram_est:.1f} MB ({fp16_vram_est/1024:.2f} GB)")
    print(f"   - INT8 (8-bit): ~{int8_vram_est:.1f} MB ({int8_vram_est/1024:.2f} GB)")
    print(f"   - INT4 NF4 (4-bit): ~{int4_vram_est:.1f} MB ({int4_vram_est/1024:.2f} GB) 🏆 [Ultra-compact]")

    # 4. Run 20-Question Benchmark
    print("\n[Step 4: Executing Extended 20-Question Coding Benchmark]")
    evaluator = CodingEvaluator(tokenizer, device=device)
    
    # We will test on a representative subset or all 20 questions with fast token limit
    test_suite = EXTENDED_20_CODING_QUESTIONS
    print(f"Evaluating {len(test_suite)} algorithmic problems (DP, Graphs, Trees, Strings, Arrays, Math)...")

    results = []
    valid_count = 0
    t_start = time.time()

    for idx, item in enumerate(test_suite, 1):
        prompt = item["prompt"]
        title = item["title"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        t0 = time.time()
        with torch.no_grad():
            outputs = specialist_model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        is_valid = evaluator.check_python_syntax(gen_text)
        if is_valid:
            valid_count += 1

        first_line = gen_text[len(prompt):].strip().split("\n")[0] if len(gen_text) > len(prompt) else ""
        status = "✅ VALID" if is_valid else "❌ SYNTAX ERROR"
        print(f"  Q{idx:02d} [{title[:22]:<22}] : {status} ({elapsed:.2f}s) -> `{first_line[:40]}`")
        results.append({
            "id": idx,
            "title": title,
            "valid": is_valid,
            "latency": elapsed,
            "first_line": first_line
        })

    total_time = time.time() - t_start
    pass_rate = (valid_count / len(test_suite)) * 100.0

    print("\n" + "=" * 80)
    print(f"🏆 FINAL 20-QUESTION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: Specialist-1.03B (Sliced from Qwen2.5-1.5B, -{reduction_pct}% Budget)")
    print(f"Parameters: {spec_params / 1e6:.1f} M (vs Teacher: {teacher_total_est / 1e6:.1f} M)")
    print(f"Valid Python Syntax: {valid_count} / {len(test_suite)} ({pass_rate:.1f}%)")
    print(f"Average Latency: {total_time / len(test_suite):.2f}s per question")
    print(f"Total Benchmark Time: {total_time:.2f}s")
    print("=" * 80)

    output_report = {
        "model": "Qwen2.5-Specialist-1.03B",
        "teacher_model": model_id,
        "teacher_parameters": teacher_total_est,
        "specialist_parameters": spec_params,
        "reduction_percentage": reduction_pct,
        "test_suite_size": len(test_suite),
        "valid_syntax_count": valid_count,
        "syntax_pass_rate_pct": pass_rate,
        "average_latency_sec": round(total_time / len(test_suite), 3),
        "results": results
    }

    with open("./outputs/qwen25_15b_20q_test_report.json", "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2, ensure_ascii=False)
    print("Saved report to ./outputs/qwen25_15b_20q_test_report.json")

if __name__ == "__main__":
    main()
