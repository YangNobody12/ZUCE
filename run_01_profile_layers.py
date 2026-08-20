"""
Run Phase 1: Component & Layer Sensitivity Profiling
Measures task-conditioned KL-Divergence and Delta Loss across 28 layers and 10 components.
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
from src.profiling.layer_sensitivity import ComponentSensitivityProfiler

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 1: COMPONENT & LAYER SENSITIVITY PROFILING")
    print(f"Base Model : {model_name}")
    print(f"Device     : {device} ({dtype})")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    discovery_data = dataset_builder.get_discovery_datasets()

    profiler = ComponentSensitivityProfiler(model, tokenizer, device=device)
    results = profiler.profile_component_sensitivity(discovery_data)

    out_pt = os.path.join(cfg["paths"]["results_dir"], "01_layer_sensitivity.pt")
    out_json = os.path.join(cfg["paths"]["results_dir"], "01_layer_sensitivity.json")

    torch.save(results, out_pt)
    
    # Save JSON summary of layer impact
    summary = {
        "num_layers": results["num_layers"],
        "components": results["components"],
        "tasks": results["tasks"],
        "layer_impact_summary": results["impact_loss_tensor"].sum(dim=1).tolist()
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Phase 1 Complete] Layer sensitivity saved to: {out_pt}")

if __name__ == "__main__":
    main()
