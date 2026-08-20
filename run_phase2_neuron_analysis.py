"""
Runner for Phase 2: Fine-Grained Neuron & Head Analysis
Computes Neuron and Attention Head Importance Tensors & Selectivity Indices.
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.analyzer.neuron_analyzer import NeuronAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Neuron & Head Analysis")
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

    analyzer = NeuronAnalyzer(model, tokenizer, config)
    results = analyzer.run_full_neuron_analysis()

    print("\nNeuron Analysis Summary:")
    for d in ["coding", "math", "translation"]:
        n_imp = results["neuron_importance"][d]
        print(f"  Domain: {d:<12} | Mean Neuron Attribution: {n_imp.mean().item():.6f} | Max Neuron: {n_imp.max().item():.6f}")

if __name__ == "__main__":
    main()
