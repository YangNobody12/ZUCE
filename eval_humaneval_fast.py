"""
Ultra-Fast, Thread-Interrupt-Protected HumanEval+ Evaluator for Windows
- Uses _thread.interrupt_main() for 100% reliable timeout on Windows
- Evaluates all 164 tasks in 2-5 seconds
- Exact match and float tolerance matching EvalPlus groundtruth
"""

import os
import sys
import json
import time
import math
import _thread
import threading
import contextlib
import numpy as np
from pathlib import Path

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

# Safe SSL
import sitecustomize
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import get_groundtruth, get_human_eval_plus_hash

@contextlib.contextmanager
def time_limit(seconds=1.5):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutError("Execution timed out!")
    finally:
        timer.cancel()

def is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(is_floats(i) for i in x)
    return False

def run_test_suite(fn, inputs, expected_outputs, entry_point, atol, timeout_per_input=1.5):
    for i, inp in enumerate(inputs):
        exp = expected_outputs[i]
        try:
            with time_limit(timeout_per_input):
                out = fn(*inp)
        except Exception:
            return False

        exact_match = (out == exp)

        if entry_point == "find_zero":
            try:
                poly_res = sum(coeff * math.pow(out, p) for p, coeff in enumerate(inp[0]))
                if abs(poly_res) <= atol:
                    continue
            except Exception:
                return False

        if atol == 0 and is_floats(exp):
            current_atol = 1e-6
        else:
            current_atol = atol

        if not exact_match and current_atol != 0:
            try:
                if type(out) != type(exp):
                    return False
                if isinstance(exp, (list, tuple)) and len(out) != len(exp):
                    return False
                if not np.allclose(out, exp, rtol=1e-07, atol=current_atol):
                    return False
            except Exception:
                return False
        elif not exact_match:
            return False

    return True

def eval_single_task(task, solution, expected_task):
    if not solution or not solution.strip():
        return {"base": False, "plus": False, "error": "Empty solution"}

    entry_point = task["entry_point"]
    atol = task["atol"]

    exec_globals = {}
    try:
        with time_limit(1.5):
            exec(solution, exec_globals)
        if entry_point not in exec_globals:
            return {"base": False, "plus": False, "error": f"Entry point '{entry_point}' not found"}
        fn = exec_globals[entry_point]
    except Exception as e:
        return {"base": False, "plus": False, "error": str(e)}

    base_pass = run_test_suite(fn, task["base_input"], expected_task["base"], entry_point, atol)
    plus_pass = False
    if base_pass:
        plus_pass = run_test_suite(fn, task["plus_input"], expected_task["plus"], entry_point, atol)

    return {"base": base_pass, "plus": plus_pass}

def evaluate_samples_fast(samples_file):
    print(f"\n[Fast Evaluator] Evaluating {samples_file}...")
    dataset = get_human_eval_plus()
    expected = get_groundtruth(dataset, get_human_eval_plus_hash(), [])

    solutions = {}
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                solutions[item["task_id"]] = item["solution"]

    total = len(dataset)
    base_passed = 0
    plus_passed = 0
    details = {}

    t0 = time.time()
    for i, (task_id, task) in enumerate(dataset.items()):
        sol = solutions.get(task_id, "")
        exp = expected[task_id]
        res = eval_single_task(task, sol, exp)
        if res.get("base", False):
            base_passed += 1
        if res.get("plus", False):
            plus_passed += 1
        details[task_id] = res

        if (i + 1) % 40 == 0 or (i + 1) == total:
            print(f"  Evaluated {i+1:3d}/{total} | Base Pass: {base_passed:2d}/{i+1:3d} ({base_passed/(i+1)*100:.1f}%) | Plus Pass: {plus_passed:2d}/{i+1:3d} ({plus_passed/(i+1)*100:.1f}%) | Time: {time.time()-t0:.1f}s", flush=True)

    base_pass_rate = round(base_passed / total * 100, 2)
    plus_pass_rate = round(plus_passed / total * 100, 2)
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {samples_file}")
    print(f"HumanEval Base Pass@1 : {base_pass_rate}% ({base_passed}/{total})")
    print(f"HumanEval Plus Pass@1 : {plus_pass_rate}% ({plus_passed}/{total})")
    print(f"Total Eval Time       : {elapsed:.2f}s ({total/max(elapsed, 1e-4):.1f} tasks/s)")
    print("=" * 80)

    out_file = Path(str(samples_file).replace(".jsonl", "_eval_results.json"))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "base_pass@1": base_pass_rate,
            "plus_pass@1": plus_pass_rate,
            "base_passed": base_passed,
            "plus_passed": plus_passed,
            "total": total,
            "details": details
        }, f, indent=2)

    return {"base_pass@1": base_pass_rate, "plus_pass@1": plus_pass_rate}

if __name__ == "__main__":
    samples = sys.argv[1] if len(sys.argv) > 1 else r"benchmark_results\base_1.54b\humaneval\samples.jsonl"
    evaluate_samples_fast(samples)
