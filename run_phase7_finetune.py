"""
Runner for Phase 7: Domain-Specific Recovery Fine-Tuning
Fine-tunes the extracted mini model to repair representation boundaries.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.training.finetune import DomainFineTuner

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Fine-Tuning")
    parser.add_argument("--mini_model_dir", type=str, default="./outputs/mini_model_0.5b", help="Path to extracted mini model")
    parser.add_argument("--domain", type=str, default="coding", choices=["coding", "math", "translation"], help="Domain")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    config = ExtractionConfig(
        target_capability=args.domain,
        ft_epochs=args.epochs,
        ft_learning_rate=args.lr
    )

    if not os.path.exists(args.mini_model_dir):
        print(f"Error: Mini model directory {args.mini_model_dir} not found. Run Phase 6 first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Extracted Mini Model from {args.mini_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.mini_model_dir)
    mini_model = AutoModelForCausalLM.from_pretrained(args.mini_model_dir, torch_dtype=dtype).to(device)

    fine_tuner = DomainFineTuner(mini_model, tokenizer, config)
    fine_tuner.train(epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
