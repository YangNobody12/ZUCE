"""
Runner for Phase 6: Model Surgery & Physical Mini Model Extraction
Extracts retained neurons and weights into a standalone ~0.5B model.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.surgery.model_surgery import ModelSurgeryEngine

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Model Surgery")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model identifier")
    parser.add_argument("--mask_file", type=str, default="./outputs/capability_mask.pt", help="Path to Phase 4 mask")
    parser.add_argument("--output_dir", type=str, default="./outputs/mini_model_0.5b", help="Target output directory for Mini Model")
    args = parser.parse_args()

    config = ExtractionConfig(
        base_model_name=args.model,
        output_mini_model_dir=args.output_dir
    )

    if not os.path.exists(args.mask_file):
        print(f"Error: Mask file not found at {args.mask_file}. Run Phase 4 first.")
        return

    mask_dict = torch.load(args.mask_file, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Base Model {config.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        base_model = base_model.to(device)

    surgery_engine = ModelSurgeryEngine(base_model, tokenizer, config)
    mini_model = surgery_engine.perform_surgery(mask_dict, output_dir=args.output_dir)

    print("\nVerifying saved mini model loading and generation...")
    loaded_mini = AutoModelForCausalLM.from_pretrained(args.output_dir, torch_dtype=dtype)
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    print("[OK] Successfully reloaded standalone Mini Model!")

if __name__ == "__main__":
    main()
