"""
Run Phase 5: Scientific Validation Gate
Executes 5 Causal Hypothesis Tests:
1. Necessity Test
2. Specificity Test
3. Sufficiency Test
4. vs Random Mask Baseline
5. Stability on unseen Validation Set

CRITICAL: If the gate fails, halts execution before physical surgery.
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
from src.masks.causal_validation import ScientificValidationGate

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 5: SCIENTIFIC VALIDATION GATE")
    print("=" * 80)

    mask_path = os.path.join(cfg["paths"]["results_dir"], "04_capability_mask.pt")
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}. Run run_04_optimize_mask.py first.")
        sys.exit(1)

    mask_data = torch.load(mask_path, map_location="cpu")
    binary_mask = mask_data["binary_mask"].to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    val_data = dataset_builder.get_validation_datasets()

    gate = ScientificValidationGate(model, tokenizer, device=device)
    gate_results = gate.run_scientific_validation_suite(
        capability_mask=binary_mask,
        val_dataset_dict=val_data,
        min_retention_ratio=cfg["capability"].get("min_retention_ratio", 0.60)
    )

    out_json = os.path.join(cfg["paths"]["results_dir"], "05_validation_gate_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(gate_results, f, indent=2)

    if not gate_results["gate_passed"]:
        print("\n[SCIENTIFIC HALT] Validation Gate FAILED. Do not proceed to physical surgery until hypothesis is refined.")
        sys.exit(1)
    else:
        print("\n[GO] Validation Gate PASSED. Proceeding to Phase 6 (Student Architecture Search & Surgery).")

if __name__ == "__main__":
    main()
