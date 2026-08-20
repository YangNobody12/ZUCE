"""
Runner for Phase 4: Expert Mask Generation
Generates binary capability masks for layers, neurons, and attention heads.
"""

import os
import torch
import argparse

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.masking.mask_generator import MaskGenerator

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Expert Mask Generation")
    parser.add_argument("--domain", type=str, default="coding", choices=["coding", "math", "translation"], help="Target capability domain")
    parser.add_argument("--neuron_data", type=str, default="./outputs/neuron_importance_matrix.pt", help="Path to Phase 2 output")
    parser.add_argument("--neuron_ratio", type=float, default=0.40, help="Ratio of neurons to retain (e.g. 0.40)")
    parser.add_argument("--head_ratio", type=float, default=0.50, help="Ratio of attention heads to retain")
    parser.add_argument("--layer_ratio", type=float, default=1.00, help="Ratio of layers to retain (e.g. 0.50)")
    args = parser.parse_args()

    config = ExtractionConfig(
        target_capability=args.domain,
        target_neuron_retention_ratio=args.neuron_ratio,
        target_head_retention_ratio=args.head_ratio,
        target_layer_retention_ratio=args.layer_ratio
    )

    if not os.path.exists(args.neuron_data):
        print(f"Error: Neuron data not found at {args.neuron_data}. Run Phase 2 first.")
        return

    print(f"Loading Neuron Importance data from {args.neuron_data}...")
    neuron_data = torch.load(args.neuron_data, map_location="cpu")

    generator = MaskGenerator(config)
    mask_dict = generator.generate_capability_mask(
        neuron_data,
        target_domain=args.domain,
        neuron_retention_ratio=args.neuron_ratio,
        head_retention_ratio=args.head_ratio,
        layer_retention_ratio=args.layer_ratio
    )

    print("\nMask Generation Complete.")
    print(f"Domain: {mask_dict['domain']}")
    print(f"Neuron Mask Shape: {mask_dict['neuron_mask'].shape}")
    print(f"Head Mask Shape  : {mask_dict['head_mask'].shape}")

if __name__ == "__main__":
    main()
