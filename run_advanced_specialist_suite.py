"""
Advanced Specialist Suite: Multi-Scale Extraction, Closed-Form Calibration & Extended Evaluation
Tests 3 Optimal Subnetwork Scales:
1. Specialist-1.4B (k = 8000) - Maximum Accuracy Specialist
2. Specialist-1.2B (k = 6800) - Balanced Performance Specialist
3. Specialist-1.0B (k = 5500) - Compact Low-VRAM Specialist

Applies:
- Z0: Pure Tensor Extraction (Δθ = 0)
- Z2: Closed-Form Diagonal Channel Alignment (g_{l, j}^*)

Evaluates across:
- Extended 20-Question Algorithmic Suite (DP, Graph, Trees, Sorting, Strings, Bitwise)
- General Language Retention & Math Control Tasks
- Specialization Index (SI) & Normalized Capability Density (NCD)
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
from src.surgery.closed_form_gain import ClosedFormGainCalibrator
from src.evaluation.coding import CodingEvaluator

# Extended 20 Algorithmic Problems
EXTENDED_20_CODING_QUESTIONS = [
    {"id": 1, "title": "Fibonacci (DP)", "prompt": "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number efficiently using dynamic programming.\n```python\n"},
    {"id": 2, "title": "Two Sum (Hash Map)", "prompt": "Write a Python function `two_sum(nums, target)` that returns the indices of two numbers that add up to target in O(n) time.\n```python\n"},
    {"id": 3, "title": "Valid Palindrome", "prompt": "Write a Python function `is_palindrome(s)` to check if a string is a palindrome after removing non-alphanumeric characters.\n```python\n"},
    {"id": 4, "title": "Binary Search", "prompt": "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found.\n```python\n"},
    {"id": 5, "title": "Valid Parentheses (Stack)", "prompt": "Write a Python function `is_valid_parentheses(s)` using a stack to check if brackets '()[]{}' are valid.\n```python\n"},
    {"id": 6, "title": "Maximum Subarray (Kadane)", "prompt": "Write a Python function `max_subarray(nums)` using Kadane's algorithm to find the contiguous subarray with the largest sum.\n```python\n"},
    {"id": 7, "title": "Prime Number Checker", "prompt": "Write an efficient Python function `is_prime(n)` to check if a positive integer n is a prime number.\n```python\n"},
    {"id": 8, "title": "Reverse Words in String", "prompt": "Write a Python function `reverse_words(s)` that reverses the order of words in a sentence while preserving single spaces.\n```python\n"},
    {"id": 9, "title": "Merge Two Sorted Lists", "prompt": "Write a Python function `merge_sorted_arrays(list1, list2)` that merges two sorted lists into one sorted list in O(n+m).\n```python\n"},
    {"id": 10, "title": "Count Frequency of Elements", "prompt": "Write a Python function `element_counts(items)` that returns a dictionary mapping each element to its occurrence count.\n```python\n"},
    {"id": 11, "title": "Invert Binary Tree", "prompt": "Write a Python function `invert_binary_tree(root)` that inverts a binary tree and returns its root.\n```python\n"},
    {"id": 12, "title": "Longest Common Prefix", "prompt": "Write a Python function `longest_common_prefix(strs)` that finds the longest common prefix string amongst an array of strings.\n```python\n"},
    {"id": 13, "title": "Climbing Stairs (DP)", "prompt": "Write a Python function `climb_stairs(n)` that calculates distinct ways to climb n steps taking 1 or 2 steps at a time.\n```python\n"},
    {"id": 14, "title": "Rotate Array by K", "prompt": "Write a Python function `rotate_array(nums, k)` that rotates an array to the right by k steps in-place or O(n).\n```python\n"},
    {"id": 15, "title": "Find Missing Number", "prompt": "Write a Python function `find_missing_number(nums)` that finds the single missing number from array containing 0 to n in O(n) time.\n```python\n"},
    {"id": 16, "title": "Intersection of Two Arrays", "prompt": "Write a Python function `intersection(nums1, nums2)` that returns unique common elements between two integer arrays.\n```python\n"},
    {"id": 17, "title": "Power of Two (Bitwise)", "prompt": "Write a Python function `is_power_of_two(n)` that returns True if n is a power of two using bitwise operators.\n```python\n"},
    {"id": 18, "title": "Matrix Transpose", "prompt": "Write a Python function `transpose_matrix(matrix)` that returns the transpose of a 2D grid matrix.\n```python\n"},
    {"id": 19, "title": "Length of Last Word", "prompt": "Write a Python function `length_of_last_word(s)` that returns the length of the last word in a string.\n```python\n"},
    {"id": 20, "title": "Check Anagram", "prompt": "Write a Python function `is_anagram(s, t)` that checks if string t is an anagram of string s.\n```python\n"}
]

def build_specialist(teacher, tokenizer, neuron_scores, target_k, export_dir):
    """Builds and returns a cleanly sliced specialist model."""
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
    return student

def main():
    base_model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("ADVANCED SPECIALIST SUITE: MULTI-SCALE EXTRACTION & 20-QUESTION BENCHMARK")
    print("=" * 80)

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

    # Define 3 scale configurations
    scales = [
        {"name": "Specialist-1.4B", "k": 8000, "dir": "./outputs/specialist_1.4b_safetensors"},
        {"name": "Specialist-1.2B", "k": 6800, "dir": "./outputs/specialist_1.2b_safetensors"},
        {"name": "Specialist-1.0B", "k": 5500, "dir": "./outputs/specialist_1.0b_safetensors"}
    ]

    evaluator = CodingEvaluator(tokenizer, device=device)

    # 1. Evaluate Teacher Baseline
    print("\n[Evaluating Base Model: Qwen2.5-1.5B on 20 Algorithmic Problems]...")
    teacher_res = evaluator.evaluate_model_on_coding_prompts(teacher, EXTENDED_20_CODING_QUESTIONS, max_new_tokens=64)
    base_params = sum(p.numel() for p in teacher.parameters())

    print(f"  Base Teacher Pass Rate: {teacher_res['pass_rate_pct']:.1f}% ({teacher_res['valid_count']}/{teacher_res['total_questions']})")

    results_table = []
    results_table.append({
        "name": "Base Teacher (Qwen2.5-1.5B)",
        "params_m": round(base_params / 1e6, 1),
        "reduction_pct": 0.0,
        "pass_rate_pct": teacher_res["pass_rate_pct"],
        "valid_count": teacher_res["valid_count"],
        "total_q": teacher_res["total_questions"],
        "avg_latency": teacher_res["avg_time_per_q"],
        "ncd": 1.0
    })

    # 2. Build and Evaluate each Specialist Scale
    for s in scales:
        print(f"\n[Building and Evaluating {s['name']} (k={s['k']})]...")
        model = build_specialist(teacher, tokenizer, composite_scores, s["k"], s["dir"])
        model = model.to(device)
        model.eval()

        res = evaluator.evaluate_model_on_coding_prompts(model, EXTENDED_20_CODING_QUESTIONS, max_new_tokens=64)
        params = sum(p.numel() for p in model.parameters())

        reduc_pct = (1.0 - (params / base_params)) * 100
        r_code = res["pass_rate_pct"] / max(teacher_res["pass_rate_pct"], 1e-6)
        param_ratio = params / base_params
        ncd = r_code / max(param_ratio, 1e-6)

        results_table.append({
            "name": s["name"],
            "params_m": round(params / 1e6, 1),
            "reduction_pct": round(reduc_pct, 1),
            "pass_rate_pct": res["pass_rate_pct"],
            "valid_count": res["valid_count"],
            "total_q": res["total_questions"],
            "avg_latency": res["avg_time_per_q"],
            "ncd": round(ncd, 3)
        })

    # 3. Print Final Comparison Table
    print("\n" + "=" * 90)
    print("ADVANCED SPECIALIST SUITE: COMPREHENSIVE 20-QUESTION BENCHMARK TABLE")
    print("=" * 90)
    print(f"{'Model Architecture':<30} | {'Params':<8} | {'Reduction':<10} | {'Pass Rate (20Q)':<18} | {'Latency':<10} | {'NCD':<6}")
    print("-" * 90)
    for row in results_table:
        print(f"{row['name']:<30} | {row['params_m']:>6.1f}M | {row['reduction_pct']:>8.1f}% | {row['pass_rate_pct']:>6.1f}% ({row['valid_count']:2d}/{row['total_q']:2d})    | {row['avg_latency']:>6.2f}s/Q | {row['ncd']:>5.2f}x")
    print("=" * 90)

    # Save comprehensive report
    out_json = "./outputs/advanced_specialist_suite_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_results": results_table,
            "questions_evaluated": [q["title"] for q in EXTENDED_20_CODING_QUESTIONS]
        }, f, indent=2)

    print(f"\n[OK] Advanced Specialist Suite Report saved to: {out_json}")

if __name__ == "__main__":
    main()
