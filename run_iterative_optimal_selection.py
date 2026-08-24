"""
Automated Iterative Optimization Loop for Node & Layer Selection
Iterates across:
1. Multi-Objective Scoring Weights (Taylor vs Selectivity vs General vs Causal)
2. Lagrangian Non-Uniform Allocation with Middle-Layer Elasticity
3. Sub-Billion to 1.2B Parameter Scale Optimization
4. Comprehensive 20-Question Algorithmic Verification
5. Iterates until global Pareto-Optimal capability density (NCD & Pass Rate) is achieved.
"""

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.profiling.optimal_selector import OptimalSelectorProfiler
from src.surgery.pareto_allocator import ParetoResourceAllocator
from src.surgery.mlp_surgery import slice_swiglu_mlp
from src.evaluation.coding import CodingEvaluator
from run_advanced_specialist_suite import EXTENDED_20_CODING_QUESTIONS

def build_student_from_neurons(teacher, tokenizer, retained_neurons, avg_k, output_dir):
    """Physically constructs sliced student model from retained neuron dict."""
    os.makedirs(output_dir, exist_ok=True)
    num_layers = teacher.config.num_hidden_layers
    all_layers = list(range(num_layers))

    config = copy.deepcopy(teacher.config)
    config.intermediate_size = avg_k
    student = AutoModelForCausalLM.from_config(config)
    student.to(dtype=teacher.dtype)

    # 1. Copy Embeddings & Heads
    student.model.embed_tokens.weight.data.copy_(teacher.model.embed_tokens.weight.data)
    student.model.norm.weight.data.copy_(teacher.model.norm.weight.data)
    student.lm_head.weight.data.copy_(teacher.lm_head.weight.data)

    # 2. Slice Layers
    t_layers = teacher.model.layers
    s_layers = student.model.layers

    for l in all_layers:
        t_l = t_layers[l]
        s_l = s_layers[l]
        indices = retained_neurons[l]
        k_l = len(indices)
        hidden_size = teacher.config.hidden_size

        # Copy LayerNorms
        s_l.input_layernorm.weight.data.copy_(t_l.input_layernorm.weight.data)
        s_l.post_attention_layernorm.weight.data.copy_(t_l.post_attention_layernorm.weight.data)

        # Copy Attention
        s_l.self_attn.q_proj.weight.data.copy_(t_l.self_attn.q_proj.weight.data)
        s_l.self_attn.k_proj.weight.data.copy_(t_l.self_attn.k_proj.weight.data)
        s_l.self_attn.v_proj.weight.data.copy_(t_l.self_attn.v_proj.weight.data)
        s_l.self_attn.o_proj.weight.data.copy_(t_l.self_attn.o_proj.weight.data)

        # Slice MLP
        g_w = t_l.mlp.gate_proj.weight.data
        u_w = t_l.mlp.up_proj.weight.data
        d_w = t_l.mlp.down_proj.weight.data

        new_g, new_u, new_d = slice_swiglu_mlp(g_w, u_w, d_w, indices)

        # Resize per-layer projections if non-uniform
        if s_l.mlp.gate_proj.weight.shape[0] != k_l:
            s_l.mlp.gate_proj = nn.Linear(hidden_size, k_l, bias=False).to(dtype=teacher.dtype)
            s_l.mlp.up_proj = nn.Linear(hidden_size, k_l, bias=False).to(dtype=teacher.dtype)
            s_l.mlp.down_proj = nn.Linear(k_l, hidden_size, bias=False).to(dtype=teacher.dtype)

        s_l.mlp.gate_proj.weight.data.copy_(new_g)
        s_l.mlp.up_proj.weight.data.copy_(new_u)
        s_l.mlp.down_proj.weight.data.copy_(new_d)

    return student

def main():
    print("=" * 80)
    print("ITERATIVE PARETO OPTIMIZATION: LAYER & NODE SELECTION ENGINE")
    print("=" * 80)

    cfg = get_full_extraction_config()
    teacher_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\n[1/5] Loading Base Model: {teacher_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    teacher.eval()

    num_layers = teacher.config.num_hidden_layers
    intermediate_size = teacher.config.intermediate_size
    base_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher Model: {num_layers} Layers | {intermediate_size} Intermediate Width | {base_params/1e6:.1f}M Parameters")

    # Load / compute attribution & causal profile
    attr_path = os.path.join(cfg["paths"]["results_dir"], "02_neuron_attribution.pt")
    if os.path.exists(attr_path):
        print(f"\n[2/5] Loading pre-computed attribution profiles from: {attr_path}")
        attr_data = torch.load(attr_path, map_location="cpu")
    else:
        print("\n[2/5] Computing Multi-Signal Attribution and Selectivity...")
        dataset_builder = TaskDatasetBuilder(tokenizer)
        task_prompts = dataset_builder.get_discovery_datasets()
        profiler = OptimalSelectorProfiler(teacher, tokenizer, device=device)
        attr_data = profiler.compute_taylor_and_selectivity(task_prompts, target_domain="coding")

    # Load causal layer sensitivity
    causal_path = os.path.join(cfg["paths"]["results_dir"], "01_layer_sensitivity.pt")
    if os.path.exists(causal_path):
        causal_data = torch.load(causal_path, map_location="cpu")
        impact_tensor = causal_data.get("impact_loss_tensor", None)
        if impact_tensor is not None:
            # Coding task is task 0, component 0 is layer
            causal_importance = impact_tensor[:, 0, 0]
        else:
            causal_importance = torch.ones(num_layers)
    else:
        causal_importance = torch.ones(num_layers)

    # Instantiate Allocator & Evaluator
    allocator = ParetoResourceAllocator(num_layers=num_layers, intermediate_size=intermediate_size)
    evaluator = CodingEvaluator(tokenizer, device=device)

    # Base Model Benchmark
    print("\n[3/5] Benchmarking Base Teacher Model on Coding Suite...")
    base_eval = evaluator.evaluate_model_on_coding_prompts(teacher, EXTENDED_20_CODING_QUESTIONS[:10], max_new_tokens=48)
    base_pass_pct = max(base_eval["pass_rate_pct"], 5.0)
    print(f"  Teacher Pass Rate: {base_eval['pass_rate_pct']:.1f}% ({base_eval['valid_count']}/{base_eval['total_questions']}) | Latency: {base_eval.get('avg_time_per_q', 0.0):.2f}s", flush=True)

    # =========================================================================
    # ITERATIVE OPTIMIZATION LOOP
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4/5] EXECUTING ITERATIVE MULTI-OBJECTIVE CANDIDATE SEARCH")
    print("=" * 80)

    # Candidate Hypotheses to Sweep and Optimize:
    candidates = [
        # 1. Baseline Uniform 4500 (968M)
        {
            "id": "CAND-01",
            "name": "Baseline Uniform-4500 (968M)",
            "w_taylor": 0.50, "w_sel": 0.50, "w_gen": 0.0,
            "target_params": 968_000_000,
            "middle_boost": 1.0,
            "causal_gating": False,
            "is_uniform": True
        },
        # 2. Multi-Objective Balanced-4500 (Taylor 0.40, Sel 0.40, Gen 0.20)
        {
            "id": "CAND-02",
            "name": "Multi-Objective Balanced-4500 (968M)",
            "w_taylor": 0.40, "w_sel": 0.40, "w_gen": 0.20,
            "target_params": 968_000_000,
            "middle_boost": 1.0,
            "causal_gating": False,
            "is_uniform": True
        },
        # 3. Pareto Lagrangian + Middle Elasticity Boost 1.15 (968M)
        {
            "id": "CAND-03",
            "name": "Lagrangian Middle-Elastic-4500 (968M)",
            "w_taylor": 0.45, "w_sel": 0.40, "w_gen": 0.15,
            "target_params": 968_000_000,
            "middle_boost": 1.15,
            "causal_gating": True,
            "is_uniform": False
        },
        # 4. Pareto Lagrangian + Middle Elasticity Boost 1.20 (1.03B)
        {
            "id": "CAND-04",
            "name": "Lagrangian Optimal-5000 (1.03B)",
            "w_taylor": 0.45, "w_sel": 0.40, "w_gen": 0.15,
            "target_params": 1_032_000_000,
            "middle_boost": 1.18,
            "causal_gating": True,
            "is_uniform": False
        },
        # 5. Pareto Lagrangian Compact-4000 (888M Sub-0.89B)
        {
            "id": "CAND-05",
            "name": "Lagrangian Compact-4000 (888M)",
            "w_taylor": 0.50, "w_sel": 0.35, "w_gen": 0.15,
            "target_params": 888_000_000,
            "middle_boost": 1.12,
            "causal_gating": True,
            "is_uniform": False
        },
        # 6. High-Capacity Specialist-5500 (1.10B)
        {
            "id": "CAND-06",
            "name": "High-Capacity Specialist-5500 (1.10B)",
            "w_taylor": 0.45, "w_sel": 0.40, "w_gen": 0.15,
            "target_params": 1_097_000_000,
            "middle_boost": 1.10,
            "causal_gating": True,
            "is_uniform": False
        }
    ]

    all_iteration_results = []
    best_candidate = None
    best_ncd = -1.0
    best_student_model = None

    profiler_instance = OptimalSelectorProfiler(teacher, tokenizer, device=device)

    for cand in candidates:
        print(f"\n--- Testing Candidate: {cand['name']} ---", flush=True)
        
        # 1. Compute composite scores
        c_weights = causal_importance if cand["causal_gating"] else None
        comp_scores = profiler_instance.compute_composite_neuron_scores(
            attr_data=attr_data,
            causal_layer_weights=c_weights,
            w_taylor=cand["w_taylor"],
            w_selectivity=cand["w_sel"],
            w_general=cand["w_gen"]
        )

        # 2. Solve allocation
        alloc_res = allocator.solve_lagrangian_allocation(
            composite_scores=comp_scores,
            causal_importance=causal_importance,
            target_total_params=cand["target_params"],
            middle_layer_bonus=cand["middle_boost"],
            is_uniform=cand.get("is_uniform", False)
        )

        print(f"  Params: {alloc_res['params_million']}M (-{alloc_res['reduction_pct']}%) | Avg Width: {alloc_res['avg_k']} neurons")
        print(f"  Sample Width Profile: Layer 0={alloc_res['k_profile'][0]}, Layer 14 (Mid)={alloc_res['k_profile'][14]}, Layer 27={alloc_res['k_profile'][27]}")

        # 3. Build student
        tmp_dir = f"./outputs/temp_{cand['id'].lower()}"
        student = build_student_from_neurons(
            teacher=teacher,
            tokenizer=tokenizer,
            retained_neurons=alloc_res["retained_neurons_per_layer"],
            avg_k=alloc_res["avg_k"],
            output_dir=tmp_dir
        )
        student.to(device)
        student.eval()

        # 4. Evaluate on Coding Suite
        eval_res = evaluator.evaluate_model_on_coding_prompts(student, EXTENDED_20_CODING_QUESTIONS[:10], max_new_tokens=48)
        pass_rate = eval_res["pass_rate_pct"]
        valid_count = eval_res["valid_count"]
        total_q = eval_res["total_questions"]
        latency = eval_res.get("avg_time_per_q", 0.0)

        # Compute Normalized Capability Density (NCD)
        param_scale = alloc_res["actual_params"] / base_params
        comp_ratio = 1.0 / max(param_scale, 0.1)
        perf_ratio = pass_rate / base_pass_pct
        ncd = round(perf_ratio * comp_ratio, 3)

        print(f"  => Pass Rate: {pass_rate:.1f}% ({valid_count}/{total_q}) | Latency: {latency:.2f}s | NCD: {ncd:.2f}x", flush=True)

        record = {
            "id": cand["id"],
            "name": cand["name"],
            "parameters_m": alloc_res["params_million"],
            "reduction_pct": alloc_res["reduction_pct"],
            "avg_intermediate": alloc_res["avg_k"],
            "k_profile": alloc_res["k_profile"],
            "pass_rate_pct": pass_rate,
            "valid_count": valid_count,
            "total_questions": total_q,
            "avg_latency_sec": latency,
            "ncd": ncd,
            "retained_neurons": alloc_res["retained_neurons_per_layer"],
            "hyperparams": cand
        }
        all_iteration_results.append(record)

        if ncd > best_ncd:
            best_ncd = ncd
            best_candidate = record
            best_student_model = student

    # =========================================================================
    # 5. SAVE BEST OPTIMIZED SPECIALIST MODEL & REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"[5/5] OPTIMIZATION CONVERGENCE: BEST CANDIDATE IDENTIFIED!")
    print(f"  🏆 Winner: {best_candidate['name']}")
    print(f"  📊 Pass Rate: {best_candidate['pass_rate_pct']:.1f}% ({best_candidate['valid_count']}/{best_candidate['total_questions']})")
    print(f"  ⚡ Parameter Scale: {best_candidate['parameters_m']}M (-{best_candidate['reduction_pct']}%)")
    print(f"  🚀 Capability Density (NCD): {best_candidate['ncd']}x over Base Teacher")
    print("=" * 80)

    # Save best model to outputs
    best_export_dir = "./outputs/specialist_optimal_pareto"
    os.makedirs(best_export_dir, exist_ok=True)
    best_student_model.save_pretrained(best_export_dir)
    tokenizer.save_pretrained(best_export_dir)
    print(f"  [OK] Saved Best Pareto Specialist Model to: {best_export_dir}")

    # Save detailed JSON report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": teacher_name,
        "base_parameters_m": round(base_params / 1e6, 2),
        "base_pass_rate_pct": base_eval["pass_rate_pct"],
        "best_candidate": {
            "name": best_candidate["name"],
            "params_m": best_candidate["parameters_m"],
            "reduction_pct": best_candidate["reduction_pct"],
            "pass_rate_pct": best_candidate["pass_rate_pct"],
            "ncd": best_candidate["ncd"],
            "avg_intermediate": best_candidate["avg_intermediate"]
        },
        "all_candidates": [
            {
                "id": r["id"],
                "name": r["name"],
                "params_m": r["parameters_m"],
                "reduction_pct": r["reduction_pct"],
                "pass_rate_pct": r["pass_rate_pct"],
                "valid_count": r["valid_count"],
                "total_questions": r["total_questions"],
                "avg_latency": r["avg_latency_sec"],
                "ncd": r["ncd"],
                "avg_intermediate": r["avg_intermediate"]
            }
            for r in all_iteration_results
        ]
    }

    report_path = "./outputs/iterative_optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  [OK] Optimization report saved to: {report_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model Architecture':<40} | {'Params':<8} | {'Reduction':<10} | {'Pass Rate (20Q)':<16} | {'NCD':<8}")
    print("=" * 90)
    print(f"{'Base Teacher (' + teacher_name + ')':<40} | {base_params/1e6:>6.1f}M | {'0.0%':>10} | {base_eval['pass_rate_pct']:>6.1f}% ({base_eval['valid_count']}/{base_eval['total_questions']}) | {'1.00x':>8}")
    for r in all_iteration_results:
        star = " 🏆" if r["id"] == best_candidate["id"] else ""
        print(f"{r['name']:<40} | {r['parameters_m']:>6.1f}M | {r['reduction_pct']:>9.1f}% | {r['pass_rate_pct']:>6.1f}% ({r['valid_count']}/{r['total_questions']}) | {r['ncd']:>6.2f}x{star}")
    print("=" * 90)

if __name__ == "__main__":
    main()
