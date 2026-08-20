"""
Direct, Robust HumanEval+ Evaluator for Windows
Evaluates samples.jsonl across both Base tests and Extra Plus tests
without nested multiprocessing crashes.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

import sitecustomize
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import get_groundtruth, get_human_eval_plus_hash
from evalplus.eval import untrusted_check, PASS, FAIL, TIMEOUT

def evaluate_samples(samples_file, dataset_name="humaneval", base_only=False):
    print(f"\n[Evaluator] Loading samples from {samples_file}...")
    dataset = get_human_eval_plus()
    expected_output = get_groundtruth(dataset, get_human_eval_plus_hash(), [])

    # Load solutions
    solutions = {}
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                solutions[item["task_id"]] = item["solution"]

    total = len(dataset)
    base_passed = 0
    plus_passed = 0
    results_detail = {}

    t0 = time.time()
    for i, (task_id, task) in enumerate(dataset.items()):
        if task_id not in solutions:
            print(f"Warning: {task_id} missing from samples!")
            continue

        solution = solutions[task_id]
        expected_task = expected_output[task_id]

        # 1. Base Tests
        base_stat, base_det = untrusted_check(
            dataset_name,
            solution,
            task["base_input"],
            task["entry_point"],
            expected=expected_task["base"],
            atol=task["atol"],
            ref_time=expected_task["base_time"],
            fast_check=True,
        )

        is_base_pass = (base_stat == PASS)
        if is_base_pass:
            base_passed += 1

        # 2. Plus Tests
        is_plus_pass = False
        if not base_only:
            if is_base_pass:
                plus_stat, plus_det = untrusted_check(
                    dataset_name,
                    solution,
                    task["plus_input"],
                    task["entry_point"],
                    expected=expected_task["plus"],
                    atol=task["atol"],
                    ref_time=expected_task["plus_time"],
                    fast_check=True,
                )
                is_plus_pass = (plus_stat == PASS)
                if is_plus_pass:
                    plus_passed += 1
            else:
                is_plus_pass = False

        results_detail[task_id] = {
            "base": "pass" if is_base_pass else "fail",
            "plus": "pass" if is_plus_pass else "fail"
        }

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  Tested {i+1}/{total} tasks | Base Pass: {base_passed}/{i+1} ({base_passed/(i+1)*100:.1f}%) | Plus Pass: {plus_passed}/{i+1} ({plus_passed/(i+1)*100:.1f}%) | Time: {time.time()-t0:.1f}s")

    base_pass_rate = round(base_passed / total * 100, 2)
    plus_pass_rate = round(plus_passed / total * 100, 2)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {samples_file}")
    print(f"HumanEval Base Pass@1 : {base_pass_rate}% ({base_passed}/{total})")
    print(f"HumanEval Plus Pass@1 : {plus_pass_rate}% ({plus_passed}/{total})")
    print(f"Total Eval Time       : {time.time()-t0:.2f}s")
    print("=" * 80)

    out_results = Path(str(samples_file).replace(".jsonl", "_eval_results.json"))
    with open(out_results, "w", encoding="utf-8") as f:
        json.dump({
            "base_pass@1": base_pass_rate,
            "plus_pass@1": plus_pass_rate,
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "total": total,
            "results": results_detail
        }, f, indent=2)

    return {"base_pass@1": base_pass_rate, "plus_pass@1": plus_pass_rate}

if __name__ == "__main__":
    samples = sys.argv[1] if len(sys.argv) > 1 else r"benchmark_results\base_1.54b\humaneval\samples.jsonl"
    evaluate_samples(samples)
