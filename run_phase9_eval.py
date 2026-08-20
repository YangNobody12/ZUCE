"""
Runner for Phase 9: Comprehensive Benchmark & Efficiency Evaluation
Evaluates HumanEval / MBPP syntax pass rate, math reasoning, latency, and VRAM memory.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.evaluation.benchmarks import CapabilityEvaluator
from cap_extract.evaluation.profiler import ModelProfiler

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Evaluation and Profiling")
    parser.add_argument("--model_dir", type=str, default="./outputs/mini_model_0.5b", help="Model path to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Model from {args.model_dir} for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)

    # 1. Capability Benchmarking
    evaluator = CapabilityEvaluator(model, tokenizer)
    eval_report = evaluator.run_full_suite()

    # 2. Performance Profiling
    profiler = ModelProfiler(model, tokenizer)
    prof_report = profiler.profile_inference()

    print("\n[OK] Phase 9 Evaluation & Profiling complete.")

if __name__ == "__main__":
    main()
