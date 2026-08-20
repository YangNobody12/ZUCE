"""
Master Experiment: Closed-Form Residual Calibration & Distortion Analysis
Executes the 4 key experiments:
1. Layer-by-layer Residual Distortion (Energy, Cosine Direction, Drift)
2. Fixed Gain Grid (1.0, 1.5, 1.97, 2.5, 3.0, 3.89)
3. Layer-wise Closed-Form Scalar Gain (g_l^*)
4. Channel-wise Diagonal Closed-Form Gain (g_{l, j}^*)
5. Full Comparative Benchmark Table
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
from src.profiling.residual_distortion import ResidualDistortionProfiler
from src.surgery.closed_form_gain import ClosedFormGainCalibrator
from src.evaluation.coding import CodingEvaluator
from run_10_question_coding_test import TEN_CODING_QUESTIONS

def main():
    cfg = get_full_extraction_config()
    teacher_name = cfg["base_model"]["name"]
    student_dir = cfg["paths"]["student_model_dir"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("CLOSED-FORM RESIDUAL CALIBRATION & DISTORTION ANALYSIS")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    student = AutoModelForCausalLM.from_pretrained(student_dir, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    coding_prompts = dataset_builder.get_discovery_datasets()["coding"]

    # =========================================================================
    # EXPERIMENT 1: Layer-by-layer Residual Distortion Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: RESIDUAL DISTORTION METRIC TABLE")
    print("=" * 80)

    distortion_profiler = ResidualDistortionProfiler(teacher, student, tokenizer, device=device)
    distortion_table = distortion_profiler.profile_residual_distortion(coding_prompts)

    print(f"{'Layer':>6} | {'Energy (E_l)':>14} | {'Cosine (C_l)':>14} | {'Residual Drift (D_l)':>20}")
    print("-" * 65)
    for row in distortion_table:
        print(f"{row['layer']:>6d} | {row['energy_retention']:>14.3f} | {row['cosine_direction']:>14.3f} | {row['residual_drift']:>20.3f}")

    # =========================================================================
    # EXPERIMENT 2, 3, 4: Gain Calibration Calculations
    # =========================================================================
    gain_calibrator = ClosedFormGainCalibrator(teacher, student, tokenizer, device=device)

    # 3. Layerwise closed-form scalar gains g_l^*
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: COMPUTING LAYERWISE CLOSED-FORM GAINS g_l^*")
    print("=" * 80)
    layerwise_gains = gain_calibrator.compute_layerwise_scalar_gains(coding_prompts)
    for l_idx, g in enumerate(layerwise_gains):
        print(f"  Layer {l_idx:2d}: g_{l_idx:2d}^* = {g:.4f}")

    # 4. Channelwise diagonal closed-form gains g_{l, j}^*
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: COMPUTING CHANNELWISE CLOSED-FORM GAINS g_{l, j}^*")
    print("=" * 80)
    channelwise_gains = gain_calibrator.compute_channelwise_diagonal_gains(coding_prompts)
    print(f"  Computed diagonal vectors across {len(channelwise_gains)} layers (dimension: {channelwise_gains[0].shape[0]}).")

    # =========================================================================
    # BENCHMARK EVALUATION ACROSS ALL METHODS
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING 10-QUESTION CODING BENCHMARK ACROSS ALL CALIBRATION METHODS")
    print("=" * 80)

    evaluator = CodingEvaluator(tokenizer, device=device)

    # Build model variants
    model_variants = {}

    # Z0: No gain (Raw Slicing)
    model_variants["Z0: No Gain (Pure Extraction)"] = student

    # Z1: Fixed Gains
    fixed_gain_values = [1.5, 1.97, 2.5, 3.89]
    for fg in fixed_gain_values:
        m_fg = ClosedFormGainCalibrator.apply_scalar_gains_to_model(student, [fg] * len(student.model.layers))
        model_variants[f"Z1: Fixed Gain {fg:.2f}"] = m_fg

    # Z2: Layerwise Closed-Form Gain
    m_layerwise = ClosedFormGainCalibrator.apply_scalar_gains_to_model(student, layerwise_gains)
    model_variants["Z2: Layerwise Closed-Form (g_l^*)"] = m_layerwise

    # Z2: Channelwise Closed-Form Gain
    m_channelwise = ClosedFormGainCalibrator.apply_channel_gains_to_model(student, channelwise_gains)
    model_variants["Z2: Channelwise Closed-Form (g_{l,j}^*)"] = m_channelwise

    # Add Teacher for baseline reference
    model_variants["Teacher (Base 1.54B)"] = teacher

    benchmark_summary = []
    print(f"{'Method / Calibration':<42} | {'Pass Rate (10Q)':<16} | {'Avg Time/Q':<12}")
    print("-" * 75)

    for method_name, model_inst in model_variants.items():
        model_inst = model_inst.to(device)
        eval_res = evaluator.evaluate_model_on_coding_prompts(model_inst, TEN_CODING_QUESTIONS, max_new_tokens=128)
        
        pass_rate_str = f"{eval_res['pass_rate_pct']:.1f}% ({eval_res['valid_count']}/{eval_res['total_questions']})"
        time_str = f"{eval_res['avg_time_per_q']:.2f}s"
        
        print(f"{method_name:<42} | {pass_rate_str:<16} | {time_str:<12}")
        
        benchmark_summary.append({
            "method": method_name,
            "pass_rate_pct": eval_res["pass_rate_pct"],
            "valid_count": eval_res["valid_count"],
            "avg_time_sec": eval_res["avg_time_per_q"]
        })

    # Save complete report
    out_json = os.path.join(cfg["paths"]["results_dir"], "closed_form_residual_calibration_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "distortion_table": distortion_table,
            "layerwise_scalar_gains": layerwise_gains,
            "benchmark_summary": benchmark_summary
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[OK] Full experiment report successfully saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
