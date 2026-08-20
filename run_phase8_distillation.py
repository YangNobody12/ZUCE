"""
Runner for Phase 8: Circuit-Aware Knowledge Distillation
Distills capability knowledge from Dense Teacher (1.5B) to Mini Student (0.5B).
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.training.distillation import CircuitDistiller

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Circuit Distillation")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-1.5B", help="Teacher model name")
    parser.add_argument("--student_model_dir", type=str, default="./outputs/mini_model_0.5b", help="Student mini model dir")
    parser.add_argument("--domain", type=str, default="coding", choices=["coding", "math", "translation"], help="Target domain")
    parser.add_argument("--epochs", type=int, default=2, help="Distillation epochs")
    args = parser.parse_args()

    config = ExtractionConfig(
        base_model_name=args.teacher_model,
        target_capability=args.domain,
        ft_epochs=args.epochs
    )

    if not os.path.exists(args.student_model_dir):
        print(f"Error: Student model not found at {args.student_model_dir}. Run Phase 6 first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Teacher Model ({config.base_model_name}) and Student Model ({args.student_model_dir})...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_dir)
    
    teacher = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        teacher = teacher.to(device)

    student = AutoModelForCausalLM.from_pretrained(
        args.student_model_dir,
        torch_dtype=dtype
    ).to(device)

    distiller = CircuitDistiller(teacher, student, tokenizer, config)
    distiller.train_distillation(epochs=args.epochs)

if __name__ == "__main__":
    main()
