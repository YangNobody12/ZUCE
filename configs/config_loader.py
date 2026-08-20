"""
Configuration Loader for Capability-Aware Model Extraction.
Loads YAML configurations into typed dictionaries with full default safety.
"""

import os
import yaml
from typing import Dict, Any, Optional

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Load a YAML file safely."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def get_full_extraction_config(
    base_config_path: str = "configs/base_model.yaml",
    capability_config_path: str = "configs/coding.yaml",
    budget_config_path: str = "configs/extraction_500m.yaml"
) -> Dict[str, Any]:
    """Combines base, capability, and budget configuration files."""
    base_cfg = load_yaml_config(base_config_path) if os.path.exists(base_config_path) else {}
    cap_cfg = load_yaml_config(capability_config_path) if os.path.exists(capability_config_path) else {}
    budget_cfg = load_yaml_config(budget_config_path) if os.path.exists(budget_config_path) else {}

    merged = {
        "base_model": base_cfg.get("base_model", {
            "name": "Qwen/Qwen2.5-1.5B",
            "torch_dtype": "bfloat16",
            "device": "cuda",
            "max_seq_len": 512,
            "vocab_size": 151936
        }),
        "paths": base_cfg.get("paths", {
            "output_dir": "./outputs",
            "results_dir": "./outputs/scientific_reports",
            "student_model_dir": "./outputs/specialist_student_model",
            "checkpoint_dir": "./outputs/checkpoints"
        }),
        "capability": cap_cfg.get("capability", {
            "target_domain": "coding",
            "contrast_domains": ["math", "general"],
            "layer_alphas": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            "num_profiling_samples": 16,
            "lambda_correlation": 0.4,
            "lambda_gradient": 0.3,
            "lambda_causal_interaction": 0.3,
            "min_retention_ratio": 0.60
        }),
        "student_target": budget_cfg.get("student_target", {
            "target_parameter_budget": 550000000,
            "target_size_label": "0.5B",
            "min_layers": 14,
            "max_layers": 18,
            "min_intermediate_size": 2048,
            "max_intermediate_size": 2816,
            "calib_epochs": 5,
            "calib_lr": 3e-5,
            "distill_epochs": 15,
            "distill_lr": 5e-5,
            "distill_temperature": 2.0,
            "lambda_ce": 0.35,
            "lambda_kd": 0.35,
            "lambda_hidden": 0.15,
            "lambda_circuit": 0.15
        })
    }

    # Ensure directories exist
    for p in merged["paths"].values():
        os.makedirs(p, exist_ok=True)

    return merged
