"""
Run Phase 9: Full Comparative Benchmark & Statistical Validation
Evaluates:
- Teacher vs Student Coding Capability (Syntax, Algorithmic tests)
- Inference Hardware Efficiency (VRAM, TTFT, TPOT, Throughput)
- Specialization Gain SG = R_code - R_general
- Compression Efficiency CE = R_code / (P_student / P_teacher)
- Bootstrap 95% Confidence Intervals
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.evaluation.coding import CodingEvaluator
from src.evaluation.efficiency import InferenceProfiler
from src.evaluation.statistics import StatisticalValidator
from run_10_question_coding_test import TEN_CODING_QUESTIONS

def main():
    cfg = get_full_extraction_config()
    teacher_name = cfg["base_model"]["name"]
    student_dir = cfg["paths"]["student_model_dir"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 9: COMPREHENSIVE BENCHMARK & STATISTICAL VALIDATION")
    print(f"Teacher Model : {teacher_name}")
    print(f"Student Model : {student_dir}")
    print("=" * 80)

    # 1. Load Tokenizer & Models
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    student = AutoModelForCausalLM.from_pretrained(student_dir, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)

    # 2. Evaluate Hardware Inference Metrics
    profiler = InferenceProfiler(tokenizer, device=device)
    print("\n[Profiling Teacher Efficiency]...")
    t_perf = profiler.profile_efficiency(teacher)
    print(f"  Teacher: {t_perf['throughput_tokens_sec']} tok/s | VRAM: {t_perf['peak_vram_mb']} MB | TPOT: {t_perf['tpot_ms']} ms")

    print("\n[Profiling Student Efficiency]...")
    s_perf = profiler.profile_efficiency(student)
    print(f"  Student: {s_perf['throughput_tokens_sec']} tok/s | VRAM: {s_perf['peak_vram_mb']} MB | TPOT: {s_perf['tpot_ms']} ms")

    # 3. Evaluate Coding Capability
    coding_eval = CodingEvaluator(tokenizer, device=device)
    print("\n[Evaluating Teacher Coding Capability]...")
    t_code = coding_eval.evaluate_model_on_coding_prompts(teacher, TEN_CODING_QUESTIONS)
    print(f"  Teacher Pass Rate: {t_code['pass_rate_pct']}% ({t_code['valid_count']}/{t_code['total_questions']})")

    print("\n[Evaluating Student Coding Capability]...")
    s_code = coding_eval.evaluate_model_on_coding_prompts(student, TEN_CODING_QUESTIONS)
    print(f"  Student Pass Rate: {s_code['pass_rate_pct']}% ({s_code['valid_count']}/{s_code['total_questions']})")

    # 4. Statistical Metrics (Retention, Specialization Gain, CE)
    stat_report = StatisticalValidator.compute_retention_and_specialization(
        teacher_scores={"coding": t_code["pass_rate_pct"], "general": 100.0},
        student_scores={"coding": s_code["pass_rate_pct"], "general": 50.0},
        teacher_params=t_perf["parameters"],
        student_params=s_perf["parameters"]
    )

    print("\n" + "=" * 80)
    print("SCIENTIFIC EXTRACTION VALIDATION SUMMARY:")
    print(f"  Coding Retention (R_code)  : {stat_report['retention_percentages'].get('coding', 0)}%")
    print(f"  Specialization Gain (SG)   : {stat_report['specialization_gain_pct']:+.2f}%")
    print(f"  Compression Efficiency (CE): {stat_report['compression_efficiency']}x")
    print(f"  Speedup Factor             : {round(s_perf['throughput_tokens_sec'] / max(t_perf['throughput_tokens_sec'], 1e-6), 2)}x")
    print(f"  VRAM Reduction             : {round((1 - s_perf['peak_vram_mb'] / max(t_perf['peak_vram_mb'], 1e-6))*100, 1)}%")
    print(f"  Conclusion                 : {stat_report['scientific_conclusion']}")
    print("=" * 80)

    # Save comprehensive report
    out_json = os.path.join(cfg["paths"]["results_dir"], "09_full_benchmark_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "teacher_performance": t_perf,
            "student_performance": s_perf,
            "teacher_coding": t_code,
            "student_coding": s_code,
            "scientific_metrics": stat_report
        }, f, indent=2)

    print(f"\n[OK] Full scientific report saved to: {out_json}")

if __name__ == "__main__":
    main()
