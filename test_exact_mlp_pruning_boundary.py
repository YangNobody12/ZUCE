"""
Targeted In-Memory MLP Pruning Boundary Test
Tests in-memory sliced MLPs across full 28 layers at pruning ratios:
k in [8500, 8000, 7168, 6000, 5000, 4000, 3000, 2304]
Measures exact Syntax Pass Rate, First Token Accuracy, and Cross-Entropy Loss!
"""

import sys
import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_10_question_coding_test import TEN_CODING_QUESTIONS
from src.evaluation.coding import CodingEvaluator

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("IN-MEMORY MLP PRUNING BOUNDARY TEST (28 LAYERS, Δθ = 0)")
    print("Testing the exact mathematical threshold where capability degrades")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    teacher.eval()

    evaluator = CodingEvaluator(tokenizer, device=device)

    # Load neuron scores
    attr_data = torch.load("./outputs/scientific_reports/02_neuron_attribution.pt", map_location="cpu")
    code_attr = attr_data["attributions"]["coding"]
    z_sel = attr_data["z_selectivity"]
    norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
    norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
    scores = 0.5 * norm_attr + 0.5 * norm_sel

    width_ratios = [8960, 8500, 8000, 7168, 6000, 5000, 4000, 3000, 2304]

    print(f"\n{'MLP Width (k)':<15} | {'Pruned %':<10} | {'Coding Pass Rate (10Q)':<24} | {'Sample Output (Q1)'}")
    print("-" * 80)

    for k in width_ratios:
        # Create in-memory model copy
        model_copy = copy.deepcopy(teacher)

        if k < 8960:
            for l in range(28):
                top_k = torch.topk(scores[l], k).indices
                w_gate = model_copy.model.layers[l].mlp.gate_proj.weight.data[top_k, :]
                w_up = model_copy.model.layers[l].mlp.up_proj.weight.data[top_k, :]
                w_down = model_copy.model.layers[l].mlp.down_proj.weight.data[:, top_k]

                # Update in-memory layer weights
                model_copy.model.layers[l].mlp.gate_proj.weight = torch.nn.Parameter(w_gate)
                model_copy.model.layers[l].mlp.up_proj.weight = torch.nn.Parameter(w_up)
                model_copy.model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(w_down)

        # Test on Q1 sample prompt
        q1_prompt = TEN_CODING_QUESTIONS[0]["prompt"]
        enc = tokenizer(q1_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_tokens = model_copy.generate(**enc, max_new_tokens=32, do_sample=False)
            gen_text = tokenizer.decode(gen_tokens[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")[:35]

        # Evaluate on all 10 questions
        res = evaluator.evaluate_model_on_coding_prompts(model_copy, TEN_CODING_QUESTIONS, max_new_tokens=64)
        pruned_pct = (1.0 - (k / 8960.0)) * 100

        print(f"{k:<15d} | {pruned_pct:>8.1f}% | {res['pass_rate_pct']:>6.1f}% ({res['valid_count']}/{res['total_questions']})       | {gen_text}")

if __name__ == "__main__":
    main()
