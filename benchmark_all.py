"""
MASTER BENCHMARK SUITE FOR ALL EXTRACTED SPECIALISTS & BASE MODELS
Evaluates 7 Models across:
1. Coding: HumanEval+ and MBPP+ (via EvalPlus)
2. General Survival: HellaSwag, ARC-Challenge, GSM8K (via LM-Eval)
3. Hardware Efficiency: Peak VRAM, TTFT, TPOT, Throughput (tok/s), Disk Size
4. Computes Pareto Frontier (Coding Accuracy vs. Parameters vs. Latency)
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Safe SSL patch for Windows cert store
import certifi
import ssl

orig_load = ssl.SSLContext.load_default_certs
def safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        orig_load(self, purpose)
    except ssl.SSLError:
        self.load_verify_locations(cafile=certifi.where())
ssl.SSLContext.load_default_certs = safe_load_default_certs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# CONFIG
# ============================================================
MODELS = {
    "base_1.54b": r"Qwen/Qwen2.5-1.5B",
    "specialist_1.03b": r"d:\llm_code\outputs\specialist_optimal_1.03b_safetensors",
    "specialist_0.97b": r"d:\llm_code\outputs\specialist_sub_billion_968m_safetensors",
    "specialist_0.95b_v100k": r"d:\llm_code\outputs\specialist_sub_0.95b_safetensors",
    "specialist_0.89b_v100k": r"d:\llm_code\outputs\specialist_sub_0.89b_safetensors",
    "specialist_0.86b_v85k": r"d:\llm_code\outputs\specialist_sub_0.86b_safetensors",
    "nonuniform_0.97b": r"d:\llm_code\outputs\exp_smart_nonuniform_4500",
}

RESULT_ROOT = Path("benchmark_results")
RESULT_ROOT.mkdir(exist_ok=True)

# ============================================================
# COMMAND RUNNER
# ============================================================
def run(cmd, log_file):
    print()
    print("=" * 100)
    print("RUN:")
    print(" ".join(cmd))
    print("=" * 100)
    start = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
    code = proc.wait()
    elapsed = time.time() - start
    return {
        "return_code": code,
        "seconds": elapsed,
    }

# ============================================================
# HARDWARE EFFICIENCY BENCHMARK
# ============================================================
def measure_hardware_efficiency(model_path, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[Hardware Profiler] Measuring VRAM, TTFT, TPOT for {model_path}...")
    if not torch.cuda.is_available():
        return {"vram_mb": 0, "ttft_ms": 0, "tpot_ms": 0, "throughput_tok_s": 0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(model_path if Path(model_path).exists() else "Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    prompt = "def binary_search(arr, target):\n    '''Binary search algorithm in Python'''\n"
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    vocab_limit = getattr(model.config, "vocab_size", 151936)
    inputs["input_ids"] = torch.clamp(inputs["input_ids"], min=0, max=vocab_limit - 1)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    torch.cuda.synchronize()
    # Measure TTFT (Time To First Token)
    t0 = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    ttft = (time.time() - t0) * 1000 # ms

    # Measure TPOT & Throughput (128 tokens)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    torch.cuda.synchronize()
    total_time = time.time() - t0
    num_gen = out.shape[1] - inputs["input_ids"].shape[1]
    tpot = (total_time / max(num_gen, 1)) * 1000 # ms / token
    throughput = num_gen / max(total_time, 1e-6)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB

    del model
    torch.cuda.empty_cache()

    return {
        "peak_vram_mb": round(peak_vram, 1),
        "ttft_ms": round(ttft, 2),
        "tpot_ms": round(tpot, 2),
        "throughput_tok_s": round(throughput, 1)
    }

# ============================================================
# EVALPLUS (HumanEval+ / MBPP+)
# ============================================================
def run_evalplus(name, model_path, dataset):
    outdir = RESULT_ROOT / name
    outdir.mkdir(exist_ok=True)
    log = outdir / f"evalplus_{dataset}.log"
    cmd = [
        sys.executable,
        "run_eval_single.py",
        "evalplus_codegen",
        "--model",
        model_path,
        "--dataset",
        dataset,
        "--backend",
        "hf",
        "--greedy",
        "--root",
        str(outdir),
    ]
    res_codegen = run(cmd, log)
    
    # Check generated samples file
    samples_candidates = list(outdir.glob(f"**/*{dataset}*evalplus*.jsonl")) + list(outdir.glob(f"**/*.jsonl"))
    if samples_candidates:
        samples_file = str(samples_candidates[0])
        eval_log = outdir / f"evalplus_{dataset}_eval.log"
        eval_cmd = [
            sys.executable,
            "run_eval_single.py",
            "evalplus_evaluate",
            "--dataset",
            dataset,
            "--samples",
            samples_file,
        ]
        res_eval = run(eval_cmd, eval_log)
        return {"codegen": res_codegen, "eval": res_eval, "samples": samples_file}
    return {"codegen": res_codegen}

# ============================================================
# LM-EVAL (HellaSwag, ARC-Challenge, GSM8K)
# ============================================================
def run_lm_eval(name, model_path):
    outdir = RESULT_ROOT / name
    outdir.mkdir(exist_ok=True)
    output_json = outdir / "lm_eval_output"
    cmd = [
        sys.executable,
        "run_eval_single.py",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},dtype=bfloat16",
        "--tasks",
        "hellaswag,arc_challenge,gsm8k",
        "--device",
        "cuda:0",
        "--batch_size",
        "auto",
        "--output_path",
        str(output_json),
    ]
    return run(cmd, outdir / "lm_eval.log")

# ============================================================
# PARAMETER COUNT
# ============================================================
def count_parameters(model_path):
    import torch
    from transformers import AutoModelForCausalLM
    print(f"Loading for parameter count: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    total = sum(p.numel() for p in model.parameters())
    del model
    return total

# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================
def main():
    print("=" * 100)
    print("MASTER BENCHMARK SUITE FOR ALL MODELS (EvalPlus + LM-Eval + Hardware Efficiency)")
    print("=" * 100)

    summary = {}

    for name, model_path in MODELS.items():
        print()
        print("#" * 100)
        print("MODEL:", name)
        print("PATH :", model_path)
        print("#" * 100)

        summary[name] = {}

        # 1. Parameter Count
        try:
            params = count_parameters(model_path)
            summary[name]["parameters"] = params
            summary[name]["parameters_m"] = round(params / 1e6, 2)
            print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
        except Exception as e:
            print("Parameter count failed:", e)
            summary[name]["parameters_error"] = str(e)

        # 2. Hardware Efficiency (VRAM, TTFT, TPOT, tok/s)
        try:
            hw = measure_hardware_efficiency(model_path)
            summary[name]["hardware"] = hw
            print(f"Hardware: Peak VRAM={hw['peak_vram_mb']}MB | TTFT={hw['ttft_ms']}ms | TPOT={hw['tpot_ms']}ms | {hw['throughput_tok_s']} tok/s")
        except Exception as e:
            print("Hardware benchmark failed:", e)
            summary[name]["hardware_error"] = str(e)

        # 3. HumanEval+
        try:
            summary[name]["humaneval"] = run_evalplus(name, model_path, "humaneval")
        except Exception as e:
            print("HumanEval failed:", e)
            summary[name]["humaneval_error"] = str(e)

        # 4. MBPP+
        try:
            summary[name]["mbpp"] = run_evalplus(name, model_path, "mbpp")
        except Exception as e:
            print("MBPP failed:", e)
            summary[name]["mbpp_error"] = str(e)

        # 5. LM-Eval (General controls)
        try:
            summary[name]["lm_eval"] = run_lm_eval(name, model_path)
        except Exception as e:
            print("LM-Eval failed:", e)
            summary[name]["lm_eval_error"] = str(e)

        # Save intermediate summary
        with open(RESULT_ROOT / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 100)
    print("MASTER BENCHMARK FINISHED")
    print("=" * 100)
    print("Results saved in:", RESULT_ROOT.resolve())

if __name__ == "__main__":
    main()
