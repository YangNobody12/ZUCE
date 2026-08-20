"""
Phase 3: Computational Circuit Discovery Engine
Constructs inter-layer neuron-attention bipartite graphs, computes informational flow,
and applies community detection to isolate domain-specific computational circuits.
"""

import os
import json
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple

from ..configs.base_config import ExtractionConfig

class CircuitDiscovery:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    def discover_circuits(
        self,
        neuron_importance_data: Dict[str, Any],
        top_k_neurons_per_layer: int = 128,
        top_k_heads_per_layer: int = 4
    ) -> Dict[str, Any]:
        """
        Builds a cross-layer directed graph connecting top active heads and neurons
        and partitions them into domain-specific functional circuits.
        """
        print("\n" + "="*70)
        print("PHASE 3: CIRCUIT DISCOVERY & GRAPH COMMUNITY DETECTION")
        print("="*70)

        meta = neuron_importance_data["meta"]
        num_layers = meta["num_layers"]
        intermediate_size = meta["intermediate_size"]
        num_heads = meta["num_heads"]

        circuits = {}
        target_domains = ["coding", "math", "translation"]

        for domain in target_domains:
            print(f"\n--- Discovering Circuit for Domain: {domain.upper()} ---")
            n_imp = neuron_importance_data["neuron_importance"][domain] # [num_layers, intermediate_size]
            h_imp = neuron_importance_data["head_importance"][domain]   # [num_layers, num_heads]

            G = nx.DiGraph()

            # 1. Select top-k nodes per layer for this domain
            selected_nodes_by_layer = {}
            for l in range(num_layers):
                # Top heads in layer l
                top_heads = torch.topk(h_imp[l], min(top_k_heads_per_layer, num_heads)).indices.tolist()
                # Top neurons in layer l
                top_neurons = torch.topk(n_imp[l], min(top_k_neurons_per_layer, intermediate_size)).indices.tolist()

                for h in top_heads:
                    node_id = f"L{l}_H{h}"
                    weight = float(h_imp[l, h].item())
                    G.add_node(node_id, type="head", layer=l, index=h, score=weight, domain=domain)

                for n in top_neurons:
                    node_id = f"L{l}_N{n}"
                    weight = float(n_imp[l, n].item())
                    G.add_node(node_id, type="neuron", layer=l, index=n, score=weight, domain=domain)

                selected_nodes_by_layer[l] = {"heads": top_heads, "neurons": top_neurons}

            # 2. Add edges: Layer L Attention -> Layer L Neurons -> Layer L+1 Attention
            for l in range(num_layers):
                heads = selected_nodes_by_layer[l]["heads"]
                neurons = selected_nodes_by_layer[l]["neurons"]

                # Connect Heads(L) -> Neurons(L)
                for h in heads:
                    h_node = f"L{l}_H{h}"
                    h_weight = G.nodes[h_node]["score"]
                    for n in neurons:
                        n_node = f"L{l}_N{n}"
                        n_weight = G.nodes[n_node]["score"]
                        # Edge weight proportional to geometric mean of attribution scores
                        edge_weight = float(np.sqrt(max(1e-8, h_weight * n_weight)))
                        G.add_edge(h_node, n_node, weight=edge_weight)

                # Connect Neurons(L) -> Heads(L+1)
                if l + 1 < num_layers:
                    next_heads = selected_nodes_by_layer[l + 1]["heads"]
                    for n in neurons:
                        n_node = f"L{l}_N{n}"
                        n_weight = G.nodes[n_node]["score"]
                        for nh in next_heads:
                            nh_node = f"L{l+1}_H{nh}"
                            nh_weight = G.nodes[nh_node]["score"]
                            edge_weight = float(np.sqrt(max(1e-8, n_weight * nh_weight)))
                            G.add_edge(n_node, nh_node, weight=edge_weight)

            # 3. Community Detection using Greedy Modularity on undirected projection
            undirected_G = G.to_undirected()
            try:
                communities = list(nx.community.greedy_modularity_communities(undirected_G, weight="weight"))
            except Exception as e:
                # Fallback to connected components
                communities = list(nx.connected_components(undirected_G))

            print(f"Graph built for {domain}: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
            print(f"Identified {len(communities)} modular sub-circuits")

            # Format extracted sub-circuit structure
            circuit_data = {
                "domain": domain,
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "num_sub_communities": len(communities),
                "nodes": {n: G.nodes[n] for n in G.nodes()},
                "layer_summary": {
                    l: {
                        "active_heads": selected_nodes_by_layer[l]["heads"],
                        "active_neurons_count": len(selected_nodes_by_layer[l]["neurons"]),
                        "top_neuron_indices": selected_nodes_by_layer[l]["neurons"][:10]
                    } for l in range(num_layers)
                }
            }
            circuits[domain] = circuit_data

        # Save to JSON
        out_file = self.config.circuit_output_path
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(circuits, f, indent=2)

        print(f"\n[Phase 3 Complete] Computational Circuits saved to: {out_file}")
        return circuits
