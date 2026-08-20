"""
Run Phase 6: Student Architecture Search & Physical Model Surgery
Searches optimal monotonic layer ordering under parameter budget and extracts standalone student model.
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from src.surgery.layer_mapping import LayerMappingOptimizer
from src.surgery.weight_mapper import PhysicalWeightMapper

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    student_dir = cfg["paths"]["student_model_dir"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    target_layers = cfg["student_target"].get("min_layers", 16)
    target_intermediate = cfg["student_target"].get("min_intermediate_size", 2304)

    print("=" * 80)
    print("PHASE 6: BUDGET ARCHITECTURE SEARCH & PHYSICAL MODEL SURGERY")
    print(f"Target Budget : {cfg['student_target']['target_size_label']} (Max: {cfg['student_target']['target_parameter_budget']:,} params)")
    print(f"Target Config : {target_layers} Layers | {target_intermediate} Intermediate Size")
    print("=" * 80)

    # Load layer sensitivity summary to rank layers
    sens_path = os.path.join(cfg["paths"]["results_dir"], "01_layer_sensitivity.json")
    if os.path.exists(sens_path):
        with open(sens_path, "r", encoding="utf-8") as f:
            sens_data = json.load(f)
        layer_scores = [sum(row) for row in sens_data["layer_impact_summary"]]
    else:
        layer_scores = [1.0] * 28

    # Monotonic layer selection
    mapper = LayerMappingOptimizer(num_total_layers=28)
    retained_layers = mapper.select_monotonic_layers(layer_scores, target_num_layers=target_layers)
    print(f"  Selected Monotonic Layers ({len(retained_layers)}/28): {retained_layers}")

    # Load neuron mask or attribution to rank neurons per layer
    mask_path = os.path.join(cfg["paths"]["results_dir"], "04_capability_mask.pt")
    if os.path.exists(mask_path):
        mask_data = torch.load(mask_path, map_location="cpu")
        c_mask = mask_data["continuous_mask"]
    else:
        c_mask = torch.ones((28, 8960))

    retained_neurons = {}
    for l_idx in retained_layers:
        top_idx = torch.topk(c_mask[l_idx], target_intermediate).indices.tolist()
        retained_neurons[l_idx] = sorted(top_idx)

    # Load base teacher model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="cpu")

    # Perform physical surgery
    weight_mapper = PhysicalWeightMapper(teacher, tokenizer)
    student = weight_mapper.construct_and_slice_student(
        retained_layers=retained_layers,
        retained_neurons_per_layer=retained_neurons,
        target_intermediate_size=target_intermediate,
        output_dir=student_dir
    )

    out_json = os.path.join(cfg["paths"]["results_dir"], "06_surgery_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "target_size_label": cfg["student_target"]["target_size_label"],
            "retained_layers": retained_layers,
            "target_intermediate_size": target_intermediate,
            "student_parameters": sum(p.numel() for p in student.parameters()),
            "output_directory": student_dir
        }, f, indent=2)

    print(f"\n[Phase 6 Complete] Physical student model successfully created in: {student_dir}")

if __name__ == "__main__":
    main()
