import json
import time
import sitecustomize
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import get_groundtruth, get_human_eval_plus_hash
from eval_humaneval_fast import eval_single_task

dataset = get_human_eval_plus()
expected = get_groundtruth(dataset, get_human_eval_plus_hash(), [])

solutions = {}
with open(r"benchmark_results\base_1.54b\humaneval\samples.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        solutions[item["task_id"]] = item["solution"]

print("Testing tasks 40-80 sequentially...")
for i, (task_id, task) in enumerate(dataset.items()):
    if i < 40:
        continue
    t0 = time.time()
    sol = solutions.get(task_id, "")
    res = eval_single_task(task, sol, expected[task_id])
    elapsed = time.time() - t0
    print(f"Task {i:3d} ({task_id:15s}): Base={res['base']}, Plus={res['plus']} ({elapsed:.3f}s)")
    if elapsed > 1.0:
        print(f"  SLOW TASK! Solution:\n{sol}")
    if i >= 80:
        break
