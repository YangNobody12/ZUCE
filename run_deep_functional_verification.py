"""
Deep Functional Correctness & Edge-Case Verification Suite
Tests 10 real algorithmic problems with exhaustive functional unit test cases:
- Verifies exact returned values (not just syntax)
- Checks regular cases, edge cases, negative numbers, boundaries, empty inputs
- Prints input -> returned output -> expected output -> verdict for every single test case
"""

import os
import sys
import re
import time
import json
from typing import Tuple, Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEEP_EXAM_PROBLEMS = [
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
        "test_cases": [
            {"args": ([2, 7, 11, 15], 9), "checker": lambda res: sorted(res) == [0, 1], "desc": "Standard [2,7,11,15], target=9 -> [0, 1]"},
            {"args": ([3, 2, 4], 6), "checker": lambda res: sorted(res) == [1, 2], "desc": "Standard [3,2,4], target=6 -> [1, 2]"},
            {"args": ([3, 3], 6), "checker": lambda res: sorted(res) == [0, 1], "desc": "Duplicate values [3,3], target=6 -> [0, 1]"},
            {"args": ([-1, -2, -3, -4, -5], -8), "checker": lambda res: sorted(res) == [2, 4], "desc": "Negative numbers [-1,-2,-3,-4,-5], target=-8 -> [2, 4]"}
        ]
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
        "test_cases": [
            {"args": (0,), "expected": 0, "desc": "Base case n=0 -> 0"},
            {"args": (1,), "expected": 1, "desc": "Base case n=1 -> 1"},
            {"args": (2,), "expected": 1, "desc": "n=2 -> 1"},
            {"args": (10,), "expected": 55, "desc": "n=10 -> 55"},
            {"args": (15,), "expected": 610, "desc": "n=15 -> 610"},
            {"args": (20,), "expected": 6765, "desc": "n=20 -> 6765"}
        ]
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
        "test_cases": [
            {"args": ("A man, a plan, a canal: Panama",), "expected": True, "desc": "Classic phrase with punctuation -> True"},
            {"args": ("race a car",), "expected": False, "desc": "Non-palindrome -> False"},
            {"args": ("   ",), "expected": True, "desc": "Edge: whitespace only -> True"},
            {"args": (".,",), "expected": True, "desc": "Edge: punctuation only -> True"},
            {"args": ("a",), "expected": True, "desc": "Edge: single char -> True"},
            {"args": ("0P",), "expected": False, "desc": "Alphanumeric non-palindrome '0P' -> False"}
        ]
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
        "test_cases": [
            {"args": ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), "expected": 6, "desc": "Standard mixed array -> 6"},
            {"args": ([1],), "expected": 1, "desc": "Single positive element -> 1"},
            {"args": ([5, 4, -1, 7, 8],), "expected": 23, "desc": "Mostly positive array -> 23"},
            {"args": ([-5, -2, -8, -1],), "expected": -1, "desc": "Edge: all negative numbers -> -1"},
            {"args": ([-100],), "expected": -100, "desc": "Edge: single negative -> -100"}
        ]
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
        "test_cases": [
            {"args": ("()",), "expected": True, "desc": "Single pair '()' -> True"},
            {"args": ("()[]{}",), "expected": True, "desc": "Multiple pairs '()[]{}' -> True"},
            {"args": ("(]",), "expected": False, "desc": "Mismatched '(]' -> False"},
            {"args": ("([)]",), "expected": False, "desc": "Interleaved '([)]' -> False"},
            {"args": ("{[]}",), "expected": True, "desc": "Nested '{[]}' -> True"},
            {"args": ("(",), "expected": False, "desc": "Edge: unclosed '(' -> False"},
            {"args": ("]",), "expected": False, "desc": "Edge: unopened ']' -> False"},
            {"args": ("",), "expected": True, "desc": "Edge: empty string -> True"}
        ]
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
        "test_cases": [
            {"args": ([-1, 0, 3, 5, 9, 12], 9), "expected": 4, "desc": "Found at index 4 -> 4"},
            {"args": ([-1, 0, 3, 5, 9, 12], 2), "expected": -1, "desc": "Not found -> -1"},
            {"args": ([5], 5), "expected": 0, "desc": "Single element found -> 0"},
            {"args": ([5], 2), "expected": -1, "desc": "Single element not found -> -1"},
            {"args": ([1, 3, 5, 7, 9], 1), "expected": 0, "desc": "Boundary left -> 0"},
            {"args": ([1, 3, 5, 7, 9], 9), "expected": 4, "desc": "Boundary right -> 4"}
        ]
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
        "test_cases": [
            {"args": (2,), "expected": True, "desc": "Smallest prime 2 -> True"},
            {"args": (3,), "expected": True, "desc": "Prime 3 -> True"},
            {"args": (17,), "expected": True, "desc": "Prime 17 -> True"},
            {"args": (97,), "expected": True, "desc": "Prime 97 -> True"},
            {"args": (1,), "expected": False, "desc": "Edge: 1 is not prime -> False"},
            {"args": (0,), "expected": False, "desc": "Edge: 0 is not prime -> False"},
            {"args": (-5,), "expected": False, "desc": "Edge: negative is not prime -> False"},
            {"args": (4,), "expected": False, "desc": "Composite 4 -> False"},
            {"args": (15,), "expected": False, "desc": "Composite 15 -> False"},
            {"args": (100,), "expected": False, "desc": "Composite 100 -> False"}
        ]
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
        "test_cases": [
            {"args": ("the sky is blue",), "expected": "blue is sky the", "desc": "Standard sentence -> 'blue is sky the'"},
            {"args": ("  hello world  ",), "expected": "world hello", "desc": "Leading and trailing spaces -> 'world hello'"},
            {"args": ("a good   example",), "expected": "example good a", "desc": "Multiple spaces between words -> 'example good a'"},
            {"args": ("word",), "expected": "word", "desc": "Single word -> 'word'"}
        ]
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
        "test_cases": [
            {"args": ([1, 2, 4], [1, 3, 4]), "expected": [1, 1, 2, 3, 4, 4], "desc": "Overlapping sorted lists -> [1, 1, 2, 3, 4, 4]"},
            {"args": ([], []), "expected": [], "desc": "Both empty -> []"},
            {"args": ([], [0]), "expected": [0], "desc": "One empty, one with [0] -> [0]"},
            {"args": ([1, 2, 3], [4, 5, 6]), "expected": [1, 2, 3, 4, 5, 6], "desc": "Non-overlapping disjoint lists -> [1, 2, 3, 4, 5, 6]"}
        ]
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
        "test_cases": [
            {"args": (1,), "expected": 1, "desc": "n=1 step -> 1"},
            {"args": (2,), "expected": 2, "desc": "n=2 steps -> 2"},
            {"args": (3,), "expected": 3, "desc": "n=3 steps -> 3"},
            {"args": (4,), "expected": 5, "desc": "n=4 steps -> 5"},
            {"args": (5,), "expected": 8, "desc": "n=5 steps -> 8"},
            {"args": (10,), "expected": 89, "desc": "n=10 steps -> 89"}
        ]
    }
]

def clean_and_repair_code(prompt: str, generated_text: str) -> str:
    """Cleans completion and normalizes indentation."""
    comp = generated_text[len(prompt):]
    stop_tokens = [r"\ndef ", r"\nclass ", r"\nif __name__", r"\n# Example", r"\nprint\("]
    for st in stop_tokens:
        m = re.search(st, comp)
        if m:
            comp = comp[:m.start()]
    
    full_code = prompt + comp
    
    # Indentation normalization
    lines = full_code.split("\n")
    repaired = []
    in_func = False
    for line in lines:
        if line.startswith("def "):
            in_func = True
            repaired.append(line)
        elif in_func:
            if line.strip() == "":
                repaired.append(line)
            elif not line.startswith("    ") and not line.startswith("\t"):
                repaired.append("    " + line.lstrip())
            else:
                repaired.append(line)
        else:
            repaired.append(line)
    return "\n".join(repaired)

def evaluate_functional_problem(func_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> Tuple[bool, int, int, List[Dict[str, Any]]]:
    """
    Executes compiled code and runs every single test case dynamically.
    Returns: (all_passed, passed_cases, total_cases, case_logs)
    """
    sandbox_env = {}
    try:
        exec(func_code, sandbox_env, sandbox_env)
    except Exception as e:
        return False, 0, len(test_cases), [{"error": f"Compilation Error: {str(e)}"}]

    if entry_point not in sandbox_env:
        return False, 0, len(test_cases), [{"error": f"Function {entry_point} not found in sandbox."}]

    target_func = sandbox_env[entry_point]
    case_logs = []
    passed_cases = 0

    for idx, tc in enumerate(test_cases, 1):
        args = tc["args"]
        desc = tc["desc"]
        try:
            actual_res = target_func(*args)
            if "checker" in tc:
                is_ok = tc["checker"](actual_res)
                expected_val = "(Checker function)"
            else:
                expected_val = tc["expected"]
                is_ok = (actual_res == expected_val)

            if is_ok:
                passed_cases += 1
                verdict = "PASS"
            else:
                verdict = "FAIL"

            case_logs.append({
                "case_id": idx,
                "desc": desc,
                "input_args": str(args),
                "actual_output": str(actual_res),
                "expected_output": str(expected_val),
                "verdict": verdict
            })
        except Exception as e:
            case_logs.append({
                "case_id": idx,
                "desc": desc,
                "input_args": str(args),
                "actual_output": f"Exception: {type(e).__name__}: {str(e)}",
                "expected_output": str(tc.get("expected", "")),
                "verdict": "FAIL"
            })

    all_passed = (passed_cases == len(test_cases))
    return all_passed, passed_cases, len(test_cases), case_logs

def main():
    print("=" * 95)
    print("🔬 DEEP FUNCTIONAL CORRECTNESS & EXHAUSTIVE TEST CASE VERIFICATION")
    print("=" * 95)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"\n[1/2] Loading Model: {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model Parameters: {total_params/1e6:.1f}M\n")

    print("=" * 95)
    print("[2/2] EXECUTING REAL PROBLEMS & EVALUATING ALL EDGE CASES")
    print("=" * 95)

    grand_total_cases = 0
    grand_passed_cases = 0
    problem_results = []
    total_time = 0.0

    for prob in DEEP_EXAM_PROBLEMS:
        p_id = prob["id"]
        title = prob["title"]
        prompt = prob["prompt"]
        entry_point = prob["entry_point"]
        test_cases = prob["test_cases"]

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
        cleaned_code = clean_and_repair_code(prompt, raw_gen)

        all_ok, p_pass, p_tot, logs = evaluate_functional_problem(cleaned_code, entry_point, test_cases)
        grand_passed_cases += p_pass
        grand_total_cases += p_tot

        status_icon = "✅ 100% CORRECT" if all_ok else "❌ PARTIAL/FAIL"
        print(f"\n[Problem {p_id:02d}/10] {title:<32} | {status_icon} ({p_pass}/{p_tot} cases passed) in {elapsed:.2f}s")
        
        # Print each test case detail
        for c in logs:
            c_icon = "  ✓" if c["verdict"] == "PASS" else "  ✗"
            print(f"  {c_icon} {c['desc']}")
            print(f"      Input: {c['input_args']} -> Actual: {c['actual_output']} | Expected: {c['expected_output']}")

        problem_results.append({
            "id": p_id,
            "title": title,
            "all_passed": all_ok,
            "passed_cases": p_pass,
            "total_cases": p_tot,
            "latency_sec": round(elapsed, 3),
            "generated_code": cleaned_code,
            "test_case_details": logs
        })

    # Final Summary
    overall_case_pass_rate = (grand_passed_cases / grand_total_cases) * 100.0
    problems_passed = sum(1 for p in problem_results if p["all_passed"])

    print("\n" + "=" * 95)
    print("🏆 FINAL DEEP FUNCTIONAL VERIFICATION SCORECARD")
    print("=" * 95)
    print(f"  Total Algorithmic Problems Solved Perfectly : {problems_passed} / {len(DEEP_EXAM_PROBLEMS)} ({(problems_passed/len(DEEP_EXAM_PROBLEMS))*100:.1f}%) 🌟")
    print(f"  Exhaustive Test Cases Passed (Inc. Edge Cases): {grand_passed_cases} / {grand_total_cases} ({overall_case_pass_rate:.1f}%) 🎯")
    print(f"  Total Time Taken                            : {total_time:.2f}s (Avg {total_time/len(DEEP_EXAM_PROBLEMS):.2f}s / problem)")
    print("=" * 95)

    out_file = "./outputs/deep_functional_verification_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_tested": model_name,
            "problems_passed_count": problems_passed,
            "problems_total_count": len(DEEP_EXAM_PROBLEMS),
            "functional_problem_pass_rate_pct": (problems_passed / len(DEEP_EXAM_PROBLEMS)) * 100.0,
            "test_cases_passed_count": grand_passed_cases,
            "test_cases_total_count": grand_total_cases,
            "test_cases_pass_rate_pct": overall_case_pass_rate,
            "total_latency_sec": round(total_time, 2),
            "problems": problem_results
        }, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Exhaustive Functional Report saved to: {out_file}")

if __name__ == "__main__":
    main()
