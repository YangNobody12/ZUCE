"""
Real Algorithmic Exam & Functional Unit-Test Sandbox (HumanEval Standard Format)
Tests 10 real algorithmic problems with canonical docstring prompts and sandboxed unit tests.
Evaluates both Base Model and Specialist Model.
"""

import os
import sys
import re
import time
import json
from typing import Tuple, Dict, List, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANONICAL_REAL_EXAM = [
    {
        "id": 1,
        "title": "Two Sum (Hash Map)",
        "prompt": """def two_sum(nums: list, target: int) -> list:
    \"\"\"
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    >>> two_sum([2, 7, 11, 15], 9)
    [0, 1]
    \"\"\"
""",
        "entry_point": "two_sum",
        "unit_tests": """
assert two_sum([2, 7, 11, 15], 9) in [[0, 1], [1, 0]]
assert two_sum([3, 2, 4], 6) in [[1, 2], [2, 1]]
assert two_sum([3, 3], 6) in [[0, 1], [1, 0]]
"""
    },
    {
        "id": 2,
        "title": "Fibonacci (DP)",
        "prompt": """def fibonacci(n: int) -> int:
    \"\"\"
    Returns the n-th Fibonacci number efficiently.
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    \"\"\"
""",
        "entry_point": "fibonacci",
        "unit_tests": """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(10) == 55
assert fibonacci(20) == 6765
"""
    },
    {
        "id": 3,
        "title": "Valid Palindrome",
        "prompt": """def is_palindrome(s: str) -> bool:
    \"\"\"
    Returns True if s is a palindrome considering only alphanumeric characters and ignoring cases.
    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    >>> is_palindrome("race a car")
    False
    \"\"\"
""",
        "entry_point": "is_palindrome",
        "unit_tests": """
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome(" ") == True
assert is_palindrome("ab_a") == True
"""
    },
    {
        "id": 4,
        "title": "Maximum Subarray (Kadane)",
        "prompt": """def max_subarray(nums: list) -> int:
    \"\"\"
    Find the contiguous subarray with the largest sum and return its sum.
    >>> max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    6
    \"\"\"
""",
        "entry_point": "max_subarray",
        "unit_tests": """
assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([5, 4, -1, 7, 8]) == 23
"""
    },
    {
        "id": 5,
        "title": "Valid Parentheses (Stack)",
        "prompt": """def is_valid_parentheses(s: str) -> bool:
    \"\"\"
    Given a string s containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    >>> is_valid_parentheses("()")
    True
    >>> is_valid_parentheses("()[]{}")
    True
    >>> is_valid_parentheses("(]")
    False
    \"\"\"
""",
        "entry_point": "is_valid_parentheses",
        "unit_tests": """
assert is_valid_parentheses("()") == True
assert is_valid_parentheses("()[]{}") == True
assert is_valid_parentheses("(]") == False
assert is_valid_parentheses("([)]") == False
assert is_valid_parentheses("{[]}") == True
"""
    },
    {
        "id": 6,
        "title": "Binary Search",
        "prompt": """def binary_search(arr: list, target: int) -> int:
    \"\"\"
    Given a sorted array of integers arr and a target, return index of target or -1 if not found.
    >>> binary_search([-1, 0, 3, 5, 9, 12], 9)
    4
    >>> binary_search([-1, 0, 3, 5, 9, 12], 2)
    -1
    \"\"\"
""",
        "entry_point": "binary_search",
        "unit_tests": """
assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4
assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1
assert binary_search([5], 5) == 0
"""
    },
    {
        "id": 7,
        "title": "Prime Number Checker",
        "prompt": """def is_prime(n: int) -> bool:
    \"\"\"
    Returns True if positive integer n is a prime number, False otherwise.
    >>> is_prime(2)
    True
    >>> is_prime(17)
    True
    >>> is_prime(4)
    False
    \"\"\"
""",
        "entry_point": "is_prime",
        "unit_tests": """
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(17) == True
assert is_prime(1) == False
assert is_prime(4) == False
assert is_prime(15) == False
"""
    },
    {
        "id": 8,
        "title": "Reverse Words in String",
        "prompt": """def reverse_words(s: str) -> str:
    \"\"\"
    Given an input string s, reverse the order of the words separated by a single space.
    >>> reverse_words("the sky is blue")
    'blue is sky the'
    >>> reverse_words("  hello world  ")
    'world hello'
    \"\"\"
""",
        "entry_point": "reverse_words",
        "unit_tests": """
assert reverse_words("the sky is blue") == "blue is sky the"
assert reverse_words("  hello world  ") == "world hello"
assert reverse_words("a good   example") == "example good a"
"""
    },
    {
        "id": 9,
        "title": "Merge Two Sorted Lists",
        "prompt": """def merge_sorted_arrays(list1: list, list2: list) -> list:
    \"\"\"
    Merge two sorted lists list1 and list2 into one sorted list.
    >>> merge_sorted_arrays([1, 2, 4], [1, 3, 4])
    [1, 1, 2, 3, 4, 4]
    \"\"\"
""",
        "entry_point": "merge_sorted_arrays",
        "unit_tests": """
assert merge_sorted_arrays([1, 2, 4], [1, 3, 4]) == [1, 1, 2, 3, 4, 4]
assert merge_sorted_arrays([], []) == []
assert merge_sorted_arrays([], [0]) == [0]
"""
    },
    {
        "id": 10,
        "title": "Climbing Stairs (DP)",
        "prompt": """def climb_stairs(n: int) -> int:
    \"\"\"
    You are climbing a staircase. It takes n steps to reach the top.
    Each time you can either climb 1 or 2 steps. How many distinct ways can you climb to the top?
    >>> climb_stairs(2)
    2
    >>> climb_stairs(3)
    3
    \"\"\"
""",
        "entry_point": "climb_stairs",
        "unit_tests": """
assert climb_stairs(2) == 2
assert climb_stairs(3) == 3
assert climb_stairs(5) == 8
"""
    }
]

def clean_completion(prompt: str, raw_generated: str) -> str:
    """Extracts completion body stopping at next function/class."""
    comp = raw_generated[len(prompt):]
    # Truncate at next function def or class or main
    stop_patterns = [r"\ndef ", r"\nclass ", r"\nif __name__", r"\n# Example", r"\nprint\("]
    for sp in stop_patterns:
        m = re.search(sp, comp)
        if m:
            comp = comp[:m.start()]
    return prompt + comp

def execute_in_sandbox(full_code: str, unit_tests_str: str) -> Tuple[bool, str]:
    """Executes full Python code against unit test assertions in sandbox."""
    full_exec_code = f"{full_code}\n\n# Unit Tests\n{unit_tests_str}\n"
    local_env = {}
    try:
        exec(full_exec_code, {}, local_env)
        return True, "PASSED all unit tests!"
    except AssertionError:
        return False, "AssertionError: Output did not match expected unit tests."
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"

def evaluate_exam_on_model(model_path: str, model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"LOADING MODEL: {model_name} ({model_path})")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f} M")

    results = []
    passed_count = 0
    total_time = 0.0

    for item in CANONICAL_REAL_EXAM:
        q_id = item["id"]
        title = item["title"]
        prompt = item["prompt"]
        entry_point = item["entry_point"]
        unit_tests = item["unit_tests"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        total_time += elapsed

        raw_gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_code = clean_completion(prompt, raw_gen)

        passed, verdict = execute_in_sandbox(full_code, unit_tests)
        if passed:
            passed_count += 1

        badge = "✅ PASS (100%)" if passed else "❌ FAIL"
        print(f"  [Q{q_id:02d}/10] {title:<30} | {badge:<14} ({elapsed:.2f}s) | {verdict}")

        results.append({
            "id": q_id,
            "title": title,
            "passed": passed,
            "verdict": verdict,
            "time_sec": round(elapsed, 3),
            "generated_code": full_code
        })

    pass_rate = (passed_count / len(CANONICAL_REAL_EXAM)) * 100.0
    print("-" * 90)
    print(f"SCORE: {model_name} -> Passed: {passed_count}/{len(CANONICAL_REAL_EXAM)} ({pass_rate:.1f}%) | Avg Latency: {total_time/len(CANONICAL_REAL_EXAM):.2f}s")
    print("-" * 90)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "model_path": model_path,
        "parameters_m": round(total_params / 1e6, 2),
        "passed_count": passed_count,
        "pass_rate_pct": pass_rate,
        "avg_latency_sec": round(total_time / len(CANONICAL_REAL_EXAM), 3),
        "results": results
    }

def main():
    print("=" * 90)
    print("🎯 CANONICAL REAL ALGORITHMIC EXAM & SANDBOX BENCHMARK")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    models_to_test = [
        ("Qwen/Qwen2.5-1.5B", "1. Base Teacher Model (1.54B Full)"),
        ("./outputs/specialist_optimal_1.03b_safetensors", "2. 🌟 ZUCE Specialist-1.03B (-33.1% Sliced)")
    ]

    all_reports = []
    for path, name in models_to_test:
        if os.path.exists(path) or path.startswith("Qwen/"):
            rep = evaluate_exam_on_model(path, name, device=device, dtype=dtype)
            all_reports.append(rep)

    out_file = "./outputs/canonical_real_exam_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Canonical Real Exam Report saved to: {out_file}")

if __name__ == "__main__":
    main()
