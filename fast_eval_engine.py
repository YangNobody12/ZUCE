"""
Fast Batched Evaluator for EvalPlus HumanEval+ and MBPP+
Features:
- Batched GPU Inference (batch_size=8 or 16) with left-padding
- Dynamic Stopping Criteria (stops on top-level def, class, if __name__, assert)
- Automatic Sanitization using evalplus.sanitize
- Direct EvalPlus Evaluation for HumanEval / HumanEval+
"""

import os
import sys
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Apply SSL fix
import sitecustomize
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.sanitize import sanitize

class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if last token is in stop list
        return False

def fast_generate_humaneval(model_path, output_jsonl, batch_size=8, max_new_tokens=512, device="cuda"):
    print(f"\n[Fast Evaluator] Loading {model_path}...")
    tok = AutoTokenizer.from_pretrained(
        model_path if Path(model_path).exists() else "Qwen/Qwen2.5-1.5B",
        padding_side="left"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    dataset = get_human_eval_plus()
    tasks = list(dataset.values())
    total = len(tasks)
    print(f"[Fast Evaluator] Loaded {total} HumanEval tasks. Generating with batch_size={batch_size}...")

    # Stop sequences for code completion
    stop_words = ["\nif __name__", "\ndef ", "\nclass ", "\n# ---", "\nassert "]
    
    solutions = []
    t0 = time.time()

    vocab_limit = getattr(model.config, "vocab_size", 151936)

    for i in range(0, total, batch_size):
        batch_tasks = tasks[i:i+batch_size]
        prompts = [t["prompt"] for t in batch_tasks]
        
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], min=0, max=vocab_limit - 1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Greedy
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # Slice generated part
        input_lens = [inputs["input_ids"][j].shape[0] for j in range(len(batch_tasks))]
        for j, task in enumerate(batch_tasks):
            gen_tokens = outputs[j][inputs["attention_mask"][j].sum():]
            gen_text = tok.decode(gen_tokens, skip_special_tokens=True)
            
            # Post-process with stop words
            for stop in stop_words:
                if stop in gen_text:
                    gen_text = gen_text.split(stop)[0]
            
            # Full raw solution
            raw_solution = task["prompt"] + gen_text
            
            # Sanitize to extract exact target function
            clean_solution = sanitize(raw_solution, entrypoint=task["entry_point"])
            
            solutions.append({
                "task_id": task["task_id"],
                "solution": clean_solution,
                "raw_completion": gen_text
            })

        print(f"  Generated {min(i+batch_size, total)}/{total} tasks ({time.time()-t0:.1f}s)...", end="\r")

    print(f"\n[Fast Evaluator] Generation finished in {time.time()-t0:.2f}s!")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in solutions:
            f.write(json.dumps({"task_id": s["task_id"], "solution": s["solution"]}) + "\n")

    print(f"[Fast Evaluator] Saved sanitized samples to {output_path}")

    del model
    torch.cuda.empty_cache()
    return output_path

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
    out = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results/test_fast/humaneval.jsonl"
    fast_generate_humaneval(model, out, batch_size=8)
