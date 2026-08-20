"""
Runner for Phase 5: Runtime Mask Verification
Validates capability preservation dynamically by hooking masks during generation.
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.masking.runtime_mask import RuntimeMaskEngine

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Runtime Mask Testing")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model identifier")
    parser.add_argument("--mask_file", type=str, default="./outputs/capability_mask.pt", help="Path to Phase 4 mask")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to test")
    args = parser.parse_args()

    config = ExtractionConfig(base_model_name=args.model)
    
    if not os.path.exists(args.mask_file):
        print(f"Error: Mask file not found at {args.mask_file}. Run Phase 4 first.")
        return

    mask_dict = torch.load(args.mask_file, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Base Model {config.base_model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)

    engine = RuntimeMaskEngine(model, tokenizer, config)
    results = engine.evaluate_runtime_mask(mask_dict, max_samples=args.samples)

    print(f"\nPhase 5 Complete. Mean Generation Similarity: {results['mean_similarity']:.4f}")

if __name__ == "__main__":
    main()
