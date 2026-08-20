"""
Base configuration classes for Capability-aware Model Extraction.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os

@dataclass
class ExtractionConfig:
    # Model Configuration
    base_model_name: str = "Qwen/Qwen2.5-1.5B"
    device: str = "cuda"  # auto fallback to cpu if cuda unavailable
    torch_dtype: str = "bfloat16" # "float16", "bfloat16", "float32"
    
    # Target Capability
    # Options: "coding", "math", "translation", "custom"
    target_capability: str = "coding"
    
    # Phase 1 & 2: Analysis Parameters
    num_calibration_samples: int = 32
    max_seq_len: int = 512
    layer_alphas: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    
    # Phase 3: Circuit Discovery
    correlation_threshold: float = 0.65
    top_k_edges_per_node: int = 10
    
    # Phase 4 & 5: Masking
    # Ratio of neurons/heads to retain in the extracted capability model
    target_neuron_retention_ratio: float = 0.40  # Keep top 40% most important neurons
    target_head_retention_ratio: float = 0.50    # Keep top 50% most important attention heads
    target_layer_retention_ratio: float = 1.00   # Or reduce layers (e.g. 0.75)
    
    # Phase 6: Model Surgery
    output_mini_model_dir: str = "./outputs/mini_model_0.5b"
    extracted_model_name: str = "Qwen2.5-0.5B-Extracted-Coding"
    
    # Phase 7 & 8: Fine-Tuning & Distillation
    ft_learning_rate: float = 2e-5
    ft_epochs: int = 3
    ft_batch_size: int = 4
    kd_alpha: float = 0.4       # Weight for Logit KD Loss
    ce_beta: float = 0.3        # Weight for Task CE Loss
    circuit_gamma: float = 0.3  # Weight for Circuit Activation Matching Loss
    distill_temperature: float = 2.0
    
    # Storage & Artifact Paths
    output_dir: str = "./outputs"
    matrix_output_path: str = "./outputs/layer_importance_matrix.json"
    neuron_output_path: str = "./outputs/neuron_importance_matrix.pt"
    circuit_output_path: str = "./outputs/circuit_graph.json"
    mask_output_path: str = "./outputs/capability_mask.pt"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_mini_model_dir, exist_ok=True)
