"""
Phase 4B: Circuit Graph Builder
Constructs functional circuit graph G = (V, E) connecting attention heads,
layer components, and MLP neuron clusters with multi-signal edge weights:
w_{ij} = lambda_1 C_{ij} + lambda_2 G_{ij} + lambda_3 J_{ij}
"""

import json
from typing import Dict, List, Any, Tuple

class CircuitGraphBuilder:
    def __init__(
        self,
        lambda_corr: float = 0.4,
        lambda_grad: float = 0.3,
        lambda_causal: float = 0.3
    ):
        self.lambda_corr = lambda_corr
        self.lambda_grad = lambda_grad
        self.lambda_causal = lambda_causal

    def build_circuit_graph(
        self,
        num_layers: int,
        layer_synergies: Dict[str, Any],
        layer_sensitivities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds graph dictionary with nodes and weighted edges representing the functional circuit.
        """
        nodes = []
        for l in range(num_layers):
            nodes.append({
                "id": f"layer_{l}",
                "type": "layer",
                "layer_idx": l
            })
            nodes.append({
                "id": f"mlp_{l}",
                "type": "mlp",
                "layer_idx": l
            })
            nodes.append({
                "id": f"attn_{l}",
                "type": "attention",
                "layer_idx": l
            })

        edges = []
        # Add internal layer-to-subcomponent edges
        for l in range(num_layers):
            edges.append({"source": f"layer_{l}", "target": f"attn_{l}", "weight": 1.0})
            edges.append({"source": f"attn_{l}", "target": f"mlp_{l}", "weight": 1.0})

        # Add cross-layer causal synergy edges
        for k, v in layer_synergies.items():
            l_i = v["layer_i"]
            l_j = v["layer_j"]
            synergy = max(0.0, float(v["synergy_J"]))
            weight = self.lambda_causal * synergy
            edges.append({
                "source": f"layer_{l_i}",
                "target": f"layer_{l_j}",
                "weight": round(weight, 4),
                "type": "causal_synergy"
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "lambda_weights": [self.lambda_corr, self.lambda_grad, self.lambda_causal]
            }
        }
