"""
Phase 6: Model Surgery Engine
Performs physical model surgery by rewriting HuggingFace configuration, slicing
MLP and Attention weight matrices based on Capability Masks, and saving a standalone Mini Model (~0.5B).
"""

import os
import copy
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional

from .weight_indexer import slice_mlp_weights, slice_attention_weights
from ..configs.base_config import ExtractionConfig

class ModelSurgeryEngine:
    def __init__(self, base_model: nn.Module, tokenizer: Any, config: Optional[ExtractionConfig] = None):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()

    def perform_surgery(
        self,
        mask_dict: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> nn.Module:
        """
        Executes physical model extraction:
        1. Derives new architectural dimensions from capability mask.
        2. Clones and modifies HuggingFace configuration.
        3. Instantiates empty mini model.
        4. Transfers and slices weights from base model into mini model.
        5. Saves standalone model artifacts.
        """
        print("\n" + "="*70)
        print("PHASE 6: MODEL SURGERY & STRUCTURAL EXTRACTION (~0.5B)")
        print("="*70)

        out_path = output_dir or self.config.output_mini_model_dir
        os.makedirs(out_path, exist_ok=True)

        base_config = self.base_model.config
        meta = mask_dict["meta"]
        retained_neurons = mask_dict["retained_neuron_indices"]
        retained_layers = mask_dict.get("retained_layer_indices", list(range(meta["num_layers"])))
        num_layers = len(retained_layers)

        # Calculate new uniform intermediate size (equal to length of retained neurons per layer)
        sample_layer_idx = retained_layers[0]
        new_intermediate_size = len(retained_neurons[sample_layer_idx])

        print(f"Original Configuration : {base_config.num_hidden_layers} Layers | Intermediate Size: {base_config.intermediate_size}")
        print(f"Extracted Architecture : {num_layers} Layers | Intermediate Size: {new_intermediate_size}")

        # 1. Create Mini Model Configuration
        mini_config = copy.deepcopy(base_config)
        mini_config.intermediate_size = new_intermediate_size
        mini_config.num_hidden_layers = num_layers
        if hasattr(mini_config, "layer_types") and mini_config.layer_types is not None:
            mini_config.layer_types = mini_config.layer_types[:num_layers]

        # 2. Instantiate Mini Model on CPU/Target Device with new architecture
        dtype = getattr(self.base_model, "dtype", torch.float16)
        print(f"Instantiating Mini Model architecture with dtype: {dtype}...")

        # Initialize mini model structure
        mini_model = AutoModelForCausalLM.from_config(mini_config).to(dtype=dtype)
        
        # 3. Copy Shared Weights (Embeddings, Final Norm, LM Head)
        print("Copying Embeddings and Output Heads...")
        base_layers = self._get_layers(self.base_model)
        mini_layers = self._get_layers(mini_model)

        # Copy Embeddings
        if hasattr(self.base_model.model, "embed_tokens"):
            mini_model.model.embed_tokens.weight.data.copy_(self.base_model.model.embed_tokens.weight.data)

        # Copy Final LayerNorm
        if hasattr(self.base_model.model, "norm"):
            mini_model.model.norm.weight.data.copy_(self.base_model.model.norm.weight.data)

        # Copy LM Head
        if hasattr(self.base_model, "lm_head"):
            mini_model.lm_head.weight.data.copy_(self.base_model.lm_head.weight.data)

        # 4. Copy and Slice Decoder Layer Weights
        print("Slicing and copying layer weights...")
        for new_l_idx, orig_l_idx in enumerate(retained_layers):
            b_layer = base_layers[orig_l_idx]
            m_layer = mini_layers[new_l_idx]
            retained_idx = retained_neurons[orig_l_idx]

            # Copy LayerNorms
            if hasattr(b_layer, "input_layernorm"):
                m_layer.input_layernorm.weight.data.copy_(b_layer.input_layernorm.weight.data)
            if hasattr(b_layer, "post_attention_layernorm"):
                m_layer.post_attention_layernorm.weight.data.copy_(b_layer.post_attention_layernorm.weight.data)

            # Copy Self-Attention weights directly (keeping all attention heads intact for stability)
            if hasattr(b_layer, "self_attn"):
                m_layer.self_attn.q_proj.weight.data.copy_(b_layer.self_attn.q_proj.weight.data)
                m_layer.self_attn.k_proj.weight.data.copy_(b_layer.self_attn.k_proj.weight.data)
                m_layer.self_attn.v_proj.weight.data.copy_(b_layer.self_attn.v_proj.weight.data)
                m_layer.self_attn.o_proj.weight.data.copy_(b_layer.self_attn.o_proj.weight.data)

            # Slice MLP Projection Weights
            if hasattr(b_layer, "mlp"):
                gate_w = b_layer.mlp.gate_proj.weight.data
                up_w = b_layer.mlp.up_proj.weight.data
                down_w = b_layer.mlp.down_proj.weight.data

                new_gate, new_up, new_down = slice_mlp_weights(gate_w, up_w, down_w, retained_idx)

                m_layer.mlp.gate_proj.weight.data.copy_(new_gate)
                m_layer.mlp.up_proj.weight.data.copy_(new_up)
                m_layer.mlp.down_proj.weight.data.copy_(new_down)

        # 5. Parameter Count Comparison
        base_params = sum(p.numel() for p in self.base_model.parameters())
        mini_params = sum(p.numel() for p in mini_model.parameters())

        print("\n" + "="*50)
        print(f"Base Model Parameters : {base_params / 1e9:.3f} B ({base_params:,} params)")
        print(f"Mini Model Parameters : {mini_params / 1e9:.3f} B ({mini_params:,} params)")
        print(f"Size Reduction        : {(1 - mini_params / base_params) * 100:.2f}%")
        print("="*50)

        # 6. Save Model & Tokenizer
        print(f"Saving extracted mini model to: {out_path}...")
        mini_model.save_pretrained(out_path)
        self.tokenizer.save_pretrained(out_path)

        print("\n[Phase 6 Complete] Model Surgery successfully extracted and saved Mini Model.")
        return mini_model

    def _get_layers(self, model: nn.Module) -> nn.ModuleList:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        elif hasattr(model, "layers"):
            return model.layers
        else:
            raise AttributeError("Unable to locate decoder layers.")
