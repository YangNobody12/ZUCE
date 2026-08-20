"""
Phase 7C: Student Architecture Config Builder
Builds valid HuggingFace PretrainedConfig adhering to GQA constraints and layer_types.
"""

import copy
from transformers import AutoConfig, PretrainedConfig
from typing import List, Optional

class StudentConfigBuilder:
    @staticmethod
    def build_student_config(
        base_config: PretrainedConfig,
        num_layers: int,
        intermediate_size: int,
        retained_layer_indices: List[int]
    ) -> PretrainedConfig:
        """Constructs new PretrainedConfig reflecting the extracted student architecture."""
        student_config = copy.deepcopy(base_config)
        student_config.num_hidden_layers = num_layers
        student_config.intermediate_size = intermediate_size

        # Fix Qwen2.5 layer_types validation constraint
        if hasattr(student_config, "layer_types") and student_config.layer_types is not None:
            student_config.layer_types = [student_config.layer_types[i] for i in retained_layer_indices]

        return student_config
