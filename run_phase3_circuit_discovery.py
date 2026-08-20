"""
Runner for Phase 3: Circuit Discovery
Constructs inter-layer graph networks and applies community detection to isolate domain circuits.
"""

import os
import torch
import argparse

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.analyzer.circuit_discovery import CircuitDiscovery

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Circuit Discovery")
    parser.add_argument("--neuron_data", type=str, default="./outputs/neuron_importance_matrix.pt", help="Path to Phase 2 output")
    parser.add_argument("--top_k_neurons", type=int, default=128, help="Top neurons per layer to include in circuit")
    parser.add_argument("--top_k_heads", type=int, default=4, help="Top heads per layer to include in circuit")
    args = parser.parse_args()

    config = ExtractionConfig()
    
    if not os.path.exists(args.neuron_data):
        print(f"Error: Neuron importance data not found at {args.neuron_data}. Please run Phase 2 first.")
        return

    print(f"Loading Neuron Importance data from {args.neuron_data}...")
    neuron_data = torch.load(args.neuron_data, map_location="cpu")

    discoverer = CircuitDiscovery(config)
    circuits = discoverer.discover_circuits(
        neuron_data,
        top_k_neurons_per_layer=args.top_k_neurons,
        top_k_heads_per_layer=args.top_k_heads
    )

    print("\nCircuits Discovered Summary:")
    for d, c in circuits.items():
        print(f"  Domain: {d:<12} | Nodes: {c['total_nodes']} | Edges: {c['total_edges']} | Sub-communities: {c['num_sub_communities']}")

if __name__ == "__main__":
    main()
