"""
Comprehensive Multi-Dimensional Benchmark Suite
Compares:
1. Qwen/Qwen3.5-0.8B (Latest Hybrid Linear Attention Sub-1B Model)
2. Qwen/Qwen2.5-1.5B (Original 1.54B Dense Transformer Base Model)
3. Specialist Sub-0.86B (Local Extracted Coding Specialist)
4. Specialist Optimal-1.03B (Local Extracted Optimal Specialist)
"""

import os
import sys
import re
import time
import json
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# EVALUATION PROMPTS ACROSS DIVERSE DOMAINS
# ============================================================

CODING_PROMPTS = [
    {
        "id": "code_1",
        "title": "Fibonacci DP",
        "prompt": "Write a Python function `fibonacci(n)` using dynamic programming with O(n) time and O(1) space.\n```python\ndef fibonacci(n):",
        "type": "code"
    },
    {
        "id": "code_2",
        "title": "Two Sum Hash Map",
        "prompt": "Write a Python function `two_sum(nums, target)` that returns the 0-based indices of two numbers that add up to target in O(n) time.\n```python\ndef two_sum(nums, target):",
        "type": "code"
    },
    {
        "id": "code_3",
        "title": "Binary Search",
        "prompt": "Write a Python function `binary_search(arr, target)` that returns index of target in sorted arr, or -1 if not found.\n```python\ndef binary_search(arr, target):",
        "type": "code"
    },
    {
        "id": "code_4",
        "title": "Valid Parentheses Stack",
        "prompt": "Write a Python function `is_valid(s)` using a stack to verify if brackets '()[]{}' in string s are closed in correct order.\n```python\ndef is_valid(s):",
        "type": "code"
    }
]

MATH_LOGIC_PROMPTS = [
    {
        "id": "math_1",
        "title": "Multi-step Arithmetic",
        "prompt": "Solve this step-by-step: A store sells apples for $3 each and oranges for $2 each. Sarah buys 4 apples and 6 oranges. She pays with a $50 bill. How much change does she receive?\nStep-by-step calculation:",
        "expected": "26"
    },
    {
        "id": "math_2",
        "title": "Rate & Time Problem",
        "prompt": "Solve this step-by-step: A train travels 120 miles in 2 hours, then 180 miles in 3 hours. What is the average speed of the entire trip in miles per hour?\nStep-by-step calculation:",
        "expected": "60"
    }
]

INSTRUCTION_FOLLOWING_PROMPTS = [
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

MULTILINGUAL_THAI_PROMPTS = [
    {
        "id": "thai_1",
        "title": "Thai Translation",
        "prompt": "Translate this sentence to natural Thai:\n\"Artificial Intelligence and compact language models enable efficient edge computing.\"\nคำแปลภาษาไทย:",
        "type": "thai"
    },
    {
        "id": "thai_2",
        "title": "Thai Question Answering",
        "prompt": "คำถาม: จงอธิบายความแตกต่างระหว่าง RAM และ ROM ในคอมพิวเตอร์อย่างกระชับ\nคำตอบ:",
        "type": "thai"
    }
]

def check_python_syntax(code_str: str) -> bool:
    """Check if Python code fragment is syntactically valid."""
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
    """Check if generated text contains valid JSON."""
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
# EVALUATION RUNNER
# ============================================================

def evaluate_model_pipeline(model_path: str, model_label: str, device="cpu", dtype=torch.float32):
    print("\n" + "=" * 90)
    print(f"EVALUATING MODEL: {model_label}")
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
    load_time = time.time() - t0_load

    # Count parameters & examine architecture
    total_params = sum(p.numel() for p in model.parameters())
    cfg = model.config
    
    # Check text config for hybrid architecture (Qwen3.5)
    is_hybrid = False
    attn_type = "Full Self-Attention"
    num_layers = getattr(cfg, "num_hidden_layers", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    intermediate_size = getattr(cfg, "intermediate_size", None)
    vocab_size = getattr(cfg, "vocab_size", len(tokenizer))
    max_pos = getattr(cfg, "max_position_embeddings", 32768)

    if hasattr(cfg, "text_config"):
        tcfg = cfg.text_config
        is_hybrid = True
        attn_type = "Hybrid (Gated DeltaNet Linear + Periodic Full Attention 3:1)"
        num_layers = getattr(tcfg, "num_hidden_layers", num_layers)
        hidden_size = getattr(tcfg, "hidden_size", hidden_size)
        intermediate_size = getattr(tcfg, "intermediate_size", intermediate_size)
        vocab_size = getattr(tcfg, "vocab_size", vocab_size)
        max_pos = getattr(tcfg, "max_position_embeddings", max_pos)

    print(f"Architecture: {attn_type}")
    print(f"Params: {total_params:,} ({total_params/1e6:.2f}M) | Layers: {num_layers} | Hidden: {hidden_size} | MLP: {intermediate_size} | Vocab: {vocab_size} | Max Context: {max_pos:,}")

    # Helper for safe generation with vocab clamping
    def safe_generate(prompt_text, max_new_tokens, stop_strings=None):
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

    # 1. Hardware & Latency Benchmark
    print("\n--- [1/5] Speed & Latency Benchmark ---")
    bench_prompt = "def quick_sort(arr):\n    '''Quick sort algorithm implementation'''\n"
    
    # Warmup
    _ = safe_generate(bench_prompt, max_new_tokens=4)
    
    # Measure TTFT
    t0 = time.time()
    _, _ = safe_generate(bench_prompt, max_new_tokens=1)
    ttft_ms = (time.time() - t0) * 1000

    # Measure Throughput (32 tokens)
    t0 = time.time()
    gen_str, gen_tokens = safe_generate(bench_prompt, max_new_tokens=32)
    gen_time = time.time() - t0
    tok_per_sec = gen_tokens / max(gen_time, 1e-5)
    tpot_ms = (gen_time / max(gen_tokens, 1)) * 1000

    print(f"TTFT: {ttft_ms:.1f}ms | TPOT: {tpot_ms:.1f}ms/tok | Throughput: {tok_per_sec:.2f} tok/s | Total Time: {gen_time:.2f}s")

    # 2. Coding Benchmark
    print("\n--- [2/5] Coding & Algorithms Benchmark ---")
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
        print(f"  * {q['title']:<25} | Time: {q_time:.2f}s | Syntax: {'[OK]' if is_syn_valid else '[FAIL]'}")
        code_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "syntax_valid": is_syn_valid,
            "generated_snippet": gen_text[:200]
        })
    code_pass_rate = (syntax_passed / len(CODING_PROMPTS)) * 100

    # 3. Math & Logic Reasoning
    print("\n--- [3/5] Math & Multi-Step Logic Benchmark ---")
    math_results = []
    math_correct_count = 0
    for q in MATH_LOGIC_PROMPTS:
        t_start = time.time()
        gen_text, _ = safe_generate(q["prompt"], max_new_tokens=60)
        q_time = time.time() - t_start
        has_expected = q["expected"] in gen_text
        if has_expected:
            math_correct_count += 1
        print(f"  * {q['title']:<25} | Time: {q_time:.2f}s | Expected Num ({q['expected']}): {'[FOUND]' if has_expected else '[NOT FOUND]'}")
        math_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "expected": q["expected"],
            "contains_expected": has_expected,
            "output_snippet": gen_text.strip()[:150]
        })
    math_pass_rate = (math_correct_count / len(MATH_LOGIC_PROMPTS)) * 100

    # 4. Instruction Following & Constraints
    print("\n--- [4/5] Instruction Following & Formatting Benchmark ---")
    inst_results = []
    inst_passed = 0
    for q in INSTRUCTION_FOLLOWING_PROMPTS:
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
        print(f"  * {q['title']:<25} | Time: {q_time:.2f}s | Format Adherence: {'[OK]' if valid else '[FAIL]'}")
        inst_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "format_valid": valid,
            "snippet": gen_text[-120:]
        })
    inst_pass_rate = (inst_passed / len(INSTRUCTION_FOLLOWING_PROMPTS)) * 100

    # 5. Multilingual & Thai Comprehension
    print("\n--- [5/5] Multilingual / Thai Language Benchmark ---")
    thai_results = []
    for q in MULTILINGUAL_THAI_PROMPTS:
        t_start = time.time()
        gen_text, _ = safe_generate(q["prompt"], max_new_tokens=50)
        q_time = time.time() - t_start
        has_thai = bool(re.search(r"[\u0E00-\u0E7F]", gen_text))
        print(f"  * {q['title']:<32} | Time: {q_time:.2f}s | Thai Generated: {'[YES]' if has_thai else '[NO]'}")
        thai_results.append({
            "title": q["title"],
            "time_sec": round(q_time, 2),
            "has_thai": has_thai,
            "snippet": gen_text.strip()[:140]
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_label": model_label,
        "model_path": str(model_path),
        "architecture_type": attn_type,
        "parameters": total_params,
        "parameters_m": round(total_params / 1e6, 2),
        "layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "max_context": max_pos,
        "speed": {
            "ttft_ms": round(ttft_ms, 1),
            "tpot_ms": round(tpot_ms, 1),
            "throughput_tok_s": round(tok_per_sec, 2),
            "gen_time_32tok_s": round(gen_time, 2)
        },
        "scores": {
            "coding_syntax_pass_pct": round(code_pass_rate, 1),
            "math_expected_found_pct": round(math_pass_rate, 1),
            "instruction_format_pass_pct": round(inst_pass_rate, 1)
        },
        "details": {
            "coding": code_results,
            "math": math_results,
            "instruction": inst_results,
            "thai": thai_results
        }
    }

def main():
    models_to_test = [
        ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B (Hybrid Linear Attn)"),
        ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B (Dense Base Transformer)"),
        (r"d:\llm_code\outputs\specialist_sub_0.86b_safetensors", "Specialist-0.86B (Extracted Coding)"),
        (r"d:\llm_code\outputs\specialist_optimal_1.03b_safetensors", "Specialist-1.03B (Optimal Extracted)")
    ]

    device = "cpu"
    dtype = torch.float32

    all_results = []
    for path, label in models_to_test:
        if not Path(path).exists() and not path.startswith("Qwen/"):
            print(f"Skipping {label} (Path not found: {path})")
            continue
        res = evaluate_model_pipeline(path, label, device=device, dtype=dtype)
        all_results.append(res)

    # Save output report
    report_file = Path("outputs/comprehensive_multi_model_benchmark_report.json")
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("ALL MODELS BENCHMARKED SUCCESSFULLY!")
    print(f"Report saved to: {report_file.resolve()}")
    print("=" * 100)

if __name__ == "__main__":
    main()
