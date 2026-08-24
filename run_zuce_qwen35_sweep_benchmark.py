"""
ZUCE Capability Extraction & Multi-Level Benchmark Suite for Qwen3.5-0.8B
Performs:
1. ZUCE Zero-Update Capability Extraction at 10%, 30%, and 50% MLP Reduction Levels.
2. Comprehensive Benchmark Comparison across Original 0.8B vs ZUCE 10%, 30%, 50%.
"""

import os
import sys
import re
import time
import json
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from zuce import ZUCE
from zuce.types import CapabilitySpec, ParameterBudget
from zuce.presets import PRESETS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# EVALUATION DATASET
# ============================================================

CODING_PROMPTS = [
    {
        "id": "code_1",
        "title": "Fibonacci DP",
        "prompt": "Write a Python function `fibonacci(n)` using dynamic programming with O(n) time and O(1) space.\n```python\ndef fibonacci(n):",
    },
    {
        "id": "code_2",
        "title": "Two Sum Hash Map",
        "prompt": "Write a Python function `two_sum(nums, target)` that returns the 0-based indices of two numbers that add up to target in O(n) time.\n```python\ndef two_sum(nums, target):",
    },
    {
        "id": "code_3",
        "title": "Binary Search",
        "prompt": "Write a Python function `binary_search(arr, target)` that returns index of target in sorted arr, or -1 if not found.\n```python\ndef binary_search(arr, target):",
    },
    {
        "id": "code_4",
        "title": "Valid Parentheses Stack",
        "prompt": "Write a Python function `is_valid(s)` using a stack to verify if brackets '()[]{}' in string s are closed in correct order.\n```python\ndef is_valid(s):",
    }
]

INSTRUCTION_PROMPTS = [
    {
        "id": "inst_1",
        "title": "Strict JSON Output",
        "prompt": 'Extract the name, age, and occupation from this text into valid JSON with keys "name", "age", "occupation". Output ONLY the JSON object, nothing else.\nText: "Alex is a 29-year-old software engineer living in Tokyo."\nJSON:\n{',
        "type": "json"
    },
    {
        "id": "inst_2",
        "title": "3-Bullet Summary",
        "prompt": "Summarize the key advantages of open-source AI models in exactly 3 bullet points. Each bullet must start with a dash (-).\nSummary:\n-",
        "type": "bullets"
    }
]

MATH_PROMPTS = [
    {
        "id": "math_1",
        "title": "Multi-step Arithmetic",
        "prompt": "Solve this step-by-step: A store sells apples for $3 each and oranges for $2 each. Sarah buys 4 apples and 6 oranges. She pays with a $50 bill. How much change does she receive?\nStep-by-step calculation:",
        "expected": "26"
    }
]

THAI_PROMPTS = [
    {
        "id": "thai_1",
        "title": "Thai Translation",
        "prompt": "Translate this sentence to natural Thai:\n\"Artificial Intelligence and compact language models enable efficient edge computing on personal devices.\"\nคำแปลภาษาไทย:",
    },
    {
        "id": "thai_2",
        "title": "Thai Question Answering",
        "prompt": "คำถาม: จงอธิบายความแตกต่างระหว่าง RAM และ ROM ในคอมพิวเตอร์อย่างกระชับ\nคำตอบ:",
    }
]

def check_python_syntax(code_str: str) -> bool:
    match = re.search(r"```(?:python)?\s*(.*?)(?:```|$)", code_str, re.DOTALL)
    candidate = match.group(1) if match else code_str
    try:
        compile(candidate, "<string>", "exec")
        return True
    except Exception:
        func_match = re.search(r"(def\s+\w+\(.*?\):(?:\n\s+.*)+)", candidate)
        if func_match:
            try:
                compile(func_match.group(1), "<string>", "exec")
                return True
            except Exception:
                pass
        return False

def check_json_validity(text: str) -> bool:
    clean = text.strip()
    if not clean.startswith("{"):
        clean = "{" + clean
    try:
        json.loads(clean)
        return True
    except Exception:
        m = re.search(r"\{.*?\}", clean, re.DOTALL)
        if m:
            try:
                json.loads(m.group(0))
                return True
            except Exception:
                pass
    return False

# ============================================================
# PHASE 1: ZUCE EXTRACTION
# ============================================================

def run_zuce_extractions(base_model_id: str):
    print("=" * 90)
    print("PHASE 1: ZUCE ZERO-UPDATE EXTRACTIONS (10%, 30%, 50% MLP Reduction)")
    print("=" * 90)

    # In Qwen3.5-0.8B:
    # constant_parameters = 488,151,872
    # parameters_per_width = 73,728 (24 layers * 1024 hidden * 3 matrices)
    # original intermediate_size = 3584
    # Total teacher parameters = 752,393,024

    levels = [
        {"name": "zuce_10pct", "reduction_pct": 10, "retained_width": 3225, "dir": "./outputs/zuce_qwen35_10pct"},
        {"name": "zuce_30pct", "reduction_pct": 30, "retained_width": 2508, "dir": "./outputs/zuce_qwen35_30pct"},
        {"name": "zuce_50pct", "reduction_pct": 50, "retained_width": 1792, "dir": "./outputs/zuce_qwen35_50pct"},
    ]

    extraction_results = {}
    spec = CapabilitySpec(
        target=PRESETS["coding"] + PRESETS["translation"][:2],
        contrasts={"math": PRESETS["math"]},
        name="coding_multilingual_spec"
    )

    for item in levels:
        out_dir = Path(item["dir"])
        target_width = item["retained_width"]
        max_params = 488151872 + target_width * 73728

        print(f"\n>>> Extracting {item['name']} (MLP Width: {target_width} / 3584, Max Params: {max_params:,})...")
        
        if out_dir.exists() and (out_dir / "model.safetensors").exists():
            print(f"    Found existing artifact at {out_dir}, skipping extraction.")
            extraction_results[item["name"]] = {
                "dir": str(out_dir),
                "retained_width": target_width,
                "reduction_pct": item["reduction_pct"],
                "max_params": max_params
            }
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)

        budget = ParameterBudget(max_parameters=max_params)
        res = ZUCE.extract(
            base_model_id,
            capability=spec,
            budget=budget,
            output_dir=out_dir,
            min_retention=0.01,
            max_samples=16
        )
        print(f"    Extracted successfully: {res.extracted_parameters:,} params | Retained Width: {res.retained_width}")
        extraction_results[item["name"]] = {
            "dir": str(out_dir),
            "retained_width": res.retained_width,
            "reduction_pct": item["reduction_pct"],
            "teacher_params": res.teacher_parameters,
            "extracted_params": res.extracted_parameters
        }

    return extraction_results

# ============================================================
# PHASE 2: MULTI-MODEL BENCHMARK
# ============================================================

def benchmark_single_model(model_path: str, model_label: str, device="cpu", dtype=torch.float32):
    print("\n" + "=" * 90)
    print(f"BENCHMARKING: {model_label}")
    print(f"Path: {model_path}")
    print("=" * 90)

    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", 24)
    intermediate_size = getattr(cfg, "intermediate_size", 3584)
    if hasattr(cfg, "text_config"):
        num_layers = getattr(cfg.text_config, "num_hidden_layers", num_layers)
        intermediate_size = getattr(cfg.text_config, "intermediate_size", intermediate_size)

    # Safe generate helper
    def safe_generate(prompt_text, max_new_tokens):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        vocab_limit = getattr(model.config, "vocab_size", len(tokenizer))
        if hasattr(model.config, "text_config"):
            vocab_limit = getattr(model.config.text_config, "vocab_size", vocab_limit)
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], min=0, max=vocab_limit - 1)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_tokens = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return gen_text, out.shape[1] - prompt_len

    # 1. Hardware & Latency
    print("  [1/4] Measuring TTFT, Throughput, and TPOT...")
    bench_prompt = "def binary_search(arr, target):\n    '''Binary search algorithm'''\n"
    _ = safe_generate(bench_prompt, max_new_tokens=4) # warmup
    
    t0 = time.time()
    _, _ = safe_generate(bench_prompt, max_new_tokens=1)
    ttft_ms = (time.time() - t0) * 1000

    t0 = time.time()
    _, gen_tokens = safe_generate(bench_prompt, max_new_tokens=32)
    gen_time = time.time() - t0
    tok_per_sec = gen_tokens / max(gen_time, 1e-5)
    tpot_ms = (gen_time / max(gen_tokens, 1)) * 1000

    # 2. Coding & Algorithms
    print("  [2/4] Testing Coding & Algorithm Syntax...")
    code_results = []
    syntax_passed = 0
    for q in CODING_PROMPTS:
        t_start = time.time()
        gen_text, _ = safe_generate(q["prompt"], max_new_tokens=60)
        q_time = time.time() - t_start
        full_code = q["prompt"] + gen_text
        is_syn_valid = check_python_syntax(full_code) or check_python_syntax(gen_text)
        if is_syn_valid:
            syntax_passed += 1
        code_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "syntax_valid": is_syn_valid,
            "snippet": gen_text[:160]
        })
    code_pass_rate = (syntax_passed / len(CODING_PROMPTS)) * 100

    # 3. Instruction Following
    print("  [3/4] Testing Instruction Following...")
    inst_results = []
    inst_passed = 0
    for q in INSTRUCTION_PROMPTS:
        t_start = time.time()
        gen_text, _ = safe_generate(q["prompt"], max_new_tokens=50)
        q_time = time.time() - t_start
        valid = False
        if q["type"] == "json":
            valid = check_json_validity(gen_text)
        elif q["type"] == "bullets":
            bullets = [line for line in gen_text.split("\n") if line.strip().startswith("-")]
            valid = len(bullets) >= 2
        if valid:
            inst_passed += 1
        inst_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "format_valid": valid,
            "snippet": gen_text[-120:]
        })
    inst_pass_rate = (inst_passed / len(INSTRUCTION_PROMPTS)) * 100

    # 4. Multilingual / Thai
    print("  [4/4] Testing Multilingual Thai Language...")
    thai_results = []
    thai_ok_count = 0
    for q in THAI_PROMPTS:
        t_start = time.time()
        gen_text, _ = safe_generate(q["prompt"], max_new_tokens=50)
        q_time = time.time() - t_start
        has_thai = bool(re.search(r"[\u0E00-\u0E7F]", gen_text))
        if has_thai:
            thai_ok_count += 1
        thai_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "has_thai": has_thai,
            "snippet": gen_text.strip()[:140]
        })
    thai_pass_rate = (thai_ok_count / len(THAI_PROMPTS)) * 100

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_label": model_label,
        "model_path": str(model_path),
        "parameters": total_params,
        "parameters_m": round(total_params / 1e6, 2),
        "intermediate_size": intermediate_size,
        "layers": num_layers,
        "speed": {
            "ttft_ms": round(ttft_ms, 1),
            "tpot_ms": round(tpot_ms, 1),
            "throughput_tok_s": round(tok_per_sec, 2),
        },
        "scores": {
            "coding_syntax_pass_pct": round(code_pass_rate, 1),
            "instruction_format_pass_pct": round(inst_pass_rate, 1),
            "thai_language_pass_pct": round(thai_pass_rate, 1)
        },
        "details": {
            "coding": code_results,
            "instruction": inst_results,
            "thai": thai_results
        }
    }

def main():
    base_model_id = "Qwen/Qwen3.5-0.8B"
    
    # 1. Run extractions
    extractions = run_zuce_extractions(base_model_id)

    # 2. Run benchmark sweep
    models_to_benchmark = [
        (base_model_id, "Qwen3.5-0.8B (Original 100% Width)"),
        (extractions["zuce_10pct"]["dir"], "Qwen3.5-0.8B (ZUCE 10% Pruned / 90% Width)"),
        (extractions["zuce_30pct"]["dir"], "Qwen3.5-0.8B (ZUCE 30% Pruned / 70% Width)"),
        (extractions["zuce_50pct"]["dir"], "Qwen3.5-0.8B (ZUCE 50% Pruned / 50% Width)"),
    ]

    all_benchmarks = []
    for path, label in models_to_benchmark:
        res = benchmark_single_model(path, label, device="cpu", dtype=torch.float32)
        all_benchmarks.append(res)

    report_path = Path("outputs/zuce_qwen35_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("ALL ZUCE EXTRACTIONS & BENCHMARKS COMPLETED!")
    print(f"Report saved to: {report_path.resolve()}")
    print("=" * 90)

if __name__ == "__main__":
    main()
