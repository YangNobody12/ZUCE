"""
Runner for Phase 1: Layer Analysis
Computes Layer Importance Matrix across Coding, Math, and Translation domains.
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.analyzer.layer_analyzer import LayerAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Layer Importance Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model identifier")
    parser.add_argument("--samples", type=int, default=10, help="Number of calibration samples per domain")
    args = parser.parse_args()

    config = ExtractionConfig(
        base_model_name=args.model,
        num_calibration_samples=args.samples
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Base Model {config.base_model_name} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)

    analyzer = LayerAnalyzer(model, tokenizer, config)
    matrix = analyzer.generate_layer_importance_matrix()

    print("\nLayer Analysis Sample Output (Layer 0, Layer 14, Layer 27):")
    for d in ["coding", "math", "translation"]:
        print(f"\n[{d.upper()}] Composite Importance Scores:")
        for l in [0, min(14, model.config.num_hidden_layers-1), model.config.num_hidden_layers-1]:
            score = matrix["domains"][d][l]["composite_importance"]
            print(f"  Layer {l:2d}: {score:.6f}")

if __name__ == "__main__":
    main()
