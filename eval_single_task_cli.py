"""
Single task evaluator subprocess for Windows.
Reads task_id and samples_file, evaluates the task against Base and Plus tests.
"""

import sys
import json
import math
import numpy as np

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

import sitecustomize
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import get_groundtruth, get_human_eval_plus_hash

def is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(is_floats(i) for i in x)
    return False

def main():
    task_id = sys.argv[1]
    samples_file = sys.argv[2]

    # Find solution
    solution = None
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["task_id"] == task_id:
                    solution = item["solution"]
                    break

    if solution is None:
        print(json.dumps({"base": False, "plus": False, "error": "Task not found in samples"}))
        return

    dataset = get_human_eval_plus()
    expected = get_groundtruth(dataset, get_human_eval_plus_hash(), [])

    task = dataset[task_id]
    expected_task = expected[task_id]

    entry_point = task["entry_point"]
    atol = task["atol"]

    exec_globals = {}
    try:
        exec(solution, exec_globals)
        if entry_point not in exec_globals:
            print(json.dumps({"base": False, "plus": False, "error": "Entry point not found"}))
            return
        fn = exec_globals[entry_point]
    except Exception as e:
        print(json.dumps({"base": False, "plus": False, "error": str(e)}))
        return

    def run_tests(inputs, expected_outputs):
        for i, inp in enumerate(inputs):
            exp = expected_outputs[i]
            try:
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

    base_pass = run_tests(task["base_input"], expected_task["base"])
    plus_pass = False
    if base_pass:
        plus_pass = run_tests(task["plus_input"], expected_task["plus"])

    print(json.dumps({"base": base_pass, "plus": plus_pass}))

if __name__ == "__main__":
    main()
