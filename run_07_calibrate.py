"""
Run Phase 7: Stage A Residual Stream Calibration
Performs fast domain calibration on student model to stabilize residual representations.
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
    print("PHASE 7: STAGE A RESIDUAL STREAM CALIBRATION")
    print(f"Student Model : {student_dir}")
    print(f"Device        : {device} ({dtype})")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(student_dir)
    student = AutoModelForCausalLM.from_pretrained(student_dir, torch_dtype=dtype)
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    calib_corpus = dataset_builder.get_calibration_dataset()

    trainer = DistillationTrainer(student, teacher, tokenizer, device=device)
    final_loss = trainer.run_stage_a_calibration(
        corpus=calib_corpus,
        epochs=cfg["student_target"].get("calib_epochs", 5),
        lr=cfg["student_target"].get("calib_lr", 3e-5),
        batch_size=cfg["student_target"].get("calib_batch_size", 2)
    )

    # Save calibrated checkpoint
    student.save_pretrained(student_dir)
    tokenizer.save_pretrained(student_dir)

    out_json = os.path.join(cfg["paths"]["results_dir"], "07_calibration_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"final_calibration_loss": round(final_loss, 4)}, f, indent=2)

    print(f"\n[Phase 7 Complete] Calibrated model saved to: {student_dir} (Final Loss: {final_loss:.4f})")

if __name__ == "__main__":
    main()
