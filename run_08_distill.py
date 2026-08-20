"""
Run Phase 8: Stage B Multi-Objective Knowledge Distillation
Transfers domain capability and logits distribution from Teacher to Student.
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
from src.distillation.trainer import DistillationTrainer

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    student_dir = cfg["paths"]["student_model_dir"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 8: STAGE B MULTI-OBJECTIVE KNOWLEDGE DISTILLATION")
    print(f"Teacher Model : {model_name}")
    print(f"Student Model : {student_dir}")
    print(f"Device        : {device} ({dtype})")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(student_dir)
    student = AutoModelForCausalLM.from_pretrained(student_dir, torch_dtype=dtype)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    calib_corpus = dataset_builder.get_calibration_dataset()

    trainer = DistillationTrainer(student, teacher, tokenizer, device=device)
    distill_results = trainer.run_stage_b_distillation(
        corpus=calib_corpus,
        epochs=cfg["student_target"].get("distill_epochs", 15),
        lr=cfg["student_target"].get("distill_lr", 5e-5),
        batch_size=cfg["student_target"].get("distill_batch_size", 2),
        temperature=cfg["student_target"].get("distill_temperature", 2.0)
    )

    # Save distilled model
    student.save_pretrained(student_dir)
    tokenizer.save_pretrained(student_dir)

    out_json = os.path.join(cfg["paths"]["results_dir"], "08_distillation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(distill_results, f, indent=2)

    print(f"\n[Phase 8 Complete] Distilled student model saved to: {student_dir}")

if __name__ == "__main__":
    main()
