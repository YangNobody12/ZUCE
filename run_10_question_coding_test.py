"""
10-Question Coding Evaluation Benchmark Suite
Tests 10 fundamental algorithmic and programming problems on both Base (1.5B) and Extracted (0.5B) Models.
"""

import os
import sys
import re
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cap_extract.utils import prepare_inputs

TEN_CODING_QUESTIONS = [
    {
        "id": 1,
        "title": "Fibonacci (Dynamic Programming)",
        "prompt": "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number efficiently using dynamic programming.\n```python\ndef fibonacci(n):"
    },
    {
        "id": 2,
        "title": "Two Sum (Hash Map)",
        "prompt": "Write a Python function `two_sum(nums, target)` that returns the indices of two numbers that add up to target in O(n) time.\n```python\ndef two_sum(nums, target):"
    },
    {
        "id": 3,
        "title": "Valid Palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` to check if a string is a palindrome after removing non-alphanumeric characters.\n```python\ndef is_palindrome(s):"
    },
    {
        "id": 4,
        "title": "Binary Search",
        "prompt": "Write a Python function `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found.\n```python\ndef binary_search(arr, target):"
    },
    {
        "id": 5,
        "title": "Valid Parentheses",
        "prompt": "Write a Python function `is_valid_parentheses(s)` using a stack to check if brackets '()[]{}' are valid.\n```python\ndef is_valid_parentheses(s):"
    },
    {
        "id": 6,
        "title": "Maximum Subarray (Kadane's)",
        "prompt": "Write a Python function `max_subarray(nums)` using Kadane's algorithm to find the contiguous subarray with the largest sum.\n```python\ndef max_subarray(nums):"
    },
    {
        "id": 7,
        "title": "Prime Number Checker",
        "prompt": "Write an efficient Python function `is_prime(n)` to check if a positive integer n is a prime number.\n```python\ndef is_prime(n):"
    },
    {
        "id": 8,
        "title": "Reverse Words in String",
        "prompt": "Write a Python function `reverse_words(s)` that reverses the order of words in a sentence while preserving single spaces.\n```python\ndef reverse_words(s):"
    },
    {
        "id": 9,
        "title": "Merge Two Sorted Lists",
        "prompt": "Write a Python function `merge_sorted_arrays(list1, list2)` that merges two sorted lists into one sorted list in O(n+m).\n```python\ndef merge_sorted_arrays(list1, list2):"
    },
    {
        "id": 10,
        "title": "Count Frequency of Elements",
        "prompt": "Write a Python function `element_counts(items)` that returns a dictionary mapping each element to its occurrence count.\n```python\ndef element_counts(items):"
    }
]

def check_syntax(code_text: str) -> bool:
    """Extract code and check if valid Python syntax."""
    # Find code inside ```python ... ``` or check raw text
    code_match = re.search(r"```(?:python)?\s*(.*?)(?:```|$)", code_text, re.DOTALL)
    candidate = code_match.group(1) if code_match else code_text
    
    # Try parsing code block
    try:
        compile(candidate, "<string>", "exec")
        return True
    except Exception:
        # Try finding standalone def function
        func_match = re.search(r"(def\s+\w+\(.*?\):(?:\n\s+.*)+)", candidate)
        if func_match:
            try:
                compile(func_match.group(1), "<string>", "exec")
                return True
            except Exception:
                pass
        return False

def evaluate_model(model_name_or_dir: str, model_label: str, device: str, dtype: torch.dtype):
    print(f"\n" + "="*80)
    print(f"LOADING AND EVALUATING: {model_label} ({model_name_or_dir})")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)

    model.eval()
    results = []
    total_time = 0.0
    valid_syntax_count = 0

    for item in TEN_CODING_QUESTIONS:
        q_id = item["id"]
        title = item["title"]
        prompt = item["prompt"]

        raw_inputs = tokenizer(prompt, return_tensors="pt")
        inputs = prepare_inputs(raw_inputs, device)

        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        total_time += elapsed

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the new tokens
        prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_part = gen_text[prompt_len:].strip()

        is_valid = check_syntax(gen_text)
        if is_valid:
            valid_syntax_count += 1

        print(f"  [Q{q_id:02d}/10] {title:<35} | Time: {elapsed:.2f}s | Valid Syntax: {'YES' if is_valid else 'NO'}")

        results.append({
            "id": q_id,
            "title": title,
            "prompt": prompt,
            "full_output": gen_text,
            "generated_part": generated_part,
            "time_sec": round(elapsed, 3),
            "valid_syntax": is_valid
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_label": model_label,
        "valid_syntax_count": valid_syntax_count,
        "pass_rate_pct": round((valid_syntax_count / len(TEN_CODING_QUESTIONS)) * 100, 1),
        "total_time_sec": round(total_time, 2),
        "avg_time_per_q": round(total_time / len(TEN_CODING_QUESTIONS), 3),
        "questions": results
    }

def main():
    base_model_path = "Qwen/Qwen2.5-1.5B"
    mini_model_path = "./outputs/mini_model_0.5b"
    output_report = "./outputs/ten_question_coding_test_report.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("="*80)
    print("10-QUESTION CODING BENCHMARK TEST")
    print(f"Device: {device} | Dtype: {dtype}")
    print("="*80)

    # 1. Run 10 questions on Extracted 0.5B Model
    mini_report = evaluate_model(mini_model_path, "Extracted Mini Model (~0.5B)", device, dtype)

    # 2. Run 10 questions on Base 1.5B Model
    base_report = evaluate_model(base_model_path, "Dense Base Model (1.5B)", device, dtype)

    # 3. Print Comprehensive Comparison Table
    print("\n" + "="*96)
    print(f"{'Q#':<4} | {'Question Title':<32} | {'Base 1.5B (Time/Syntax)':<24} | {'Mini 0.5B (Time/Syntax)':<24}")
    print("="*96)

    for i in range(len(TEN_CODING_QUESTIONS)):
        b_q = base_report["questions"][i]
        m_q = mini_report["questions"][i]

        b_status = f"{b_q['time_sec']:.2f}s | {'[VALID]' if b_q['valid_syntax'] else '[INVALID]'}"
        m_status = f"{m_q['time_sec']:.2f}s | {'[VALID]' if m_q['valid_syntax'] else '[INVALID]'}"

        print(f"Q{b_q['id']:<3} | {b_q['title']:<32} | {b_status:<24} | {m_status:<24}")

    print("-" * 96)
    print(f"{'SUMMARY':<38} | Pass: {base_report['pass_rate_pct']}% ({base_report['total_time_sec']}s)     | Pass: {mini_report['pass_rate_pct']}% ({mini_report['total_time_sec']}s)")
    print("="*96)

    # Save detailed JSON report
    combined = {
        "device": device,
        "base_model_report": base_report,
        "mini_model_report": mini_report
    }
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n[OK] Complete 10-Question Test Report saved to: {output_report}")

if __name__ == "__main__":
    main()
