"""
Run Phase 3: Causal Interaction & Circuit Discovery
Computes causal synergy J_ij, builds graph G = (V, E), and extracts core circuit pathways.
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
from src.circuits.interaction import CausalInteractionProfiler
from src.circuits.graph_builder import CircuitGraphBuilder
from src.circuits.community import CircuitCommunityExtractor

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 3: CAUSAL INTERACTION & CIRCUIT DISCOVERY")
    print(f"Base Model : {model_name}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    discovery_data = dataset_builder.get_discovery_datasets()
    code_prompts = discovery_data["coding"]

    # Sample representative layer pairs across depth (strictly bounded < num_layers)
    num_layers = model.config.num_hidden_layers
    sample_pairs = []
    for i in range(0, num_layers - 4, 4):
        if i + 2 < num_layers:
            sample_pairs.append((i, i + 2))
        if i + 4 < num_layers:
            sample_pairs.append((i, i + 4))

    interaction_profiler = CausalInteractionProfiler(model, tokenizer, device=device)
    synergies = interaction_profiler.compute_layer_interaction_matrix(code_prompts, sample_pairs)

    graph_builder = CircuitGraphBuilder(
        lambda_corr=cfg["capability"]["lambda_correlation"],
        lambda_grad=cfg["capability"]["lambda_gradient"],
        lambda_causal=cfg["capability"]["lambda_causal_interaction"]
    )
    circuit_graph = graph_builder.build_circuit_graph(num_layers, synergies, {})

    extractor = CircuitCommunityExtractor()
    clusters = extractor.extract_core_circuit_clusters(circuit_graph)

    out_json = os.path.join(cfg["paths"]["results_dir"], "03_circuit_graph.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "graph": circuit_graph,
            "core_clusters": clusters,
            "synergies": synergies
        }, f, indent=2)

    print(f"\n[Phase 3 Complete] Circuit graph with {circuit_graph['meta']['num_nodes']} nodes and {circuit_graph['meta']['num_edges']} edges saved to: {out_json}")

if __name__ == "__main__":
    main()
