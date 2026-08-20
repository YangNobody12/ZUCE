"""
Phase 7D: Physical Weight Mapper
Transfers embeddings, Attention heads, LayerNorms, and sliced MLPs into the student model.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional

from .mlp_surgery import slice_swiglu_mlp
from .config_builder import StudentConfigBuilder

class PhysicalWeightMapper:
    def __init__(self, teacher_model: nn.Module, tokenizer: Any):
        self.teacher = teacher_model
        self.tokenizer = tokenizer

    def construct_and_slice_student(
        self,
        retained_layers: List[int],
        retained_neurons_per_layer: Dict[int, List[int]],
        target_intermediate_size: int,
        output_dir: str
    ) -> nn.Module:
        """
        Instantiates empty student model and physically transfers sliced weights.
        """
        os.makedirs(output_dir, exist_ok=True)
        dtype = getattr(self.teacher, "dtype", torch.bfloat16)

        # 1. Build Config
        student_config = StudentConfigBuilder.build_student_config(
            self.teacher.config,
            num_layers=len(retained_layers),
            intermediate_size=target_intermediate_size,
            retained_layer_indices=retained_layers
        )

        # 2. Instantiate Student
        print(f"\n[Surgery] Instantiating Student Model: {len(retained_layers)} Layers | {target_intermediate_size} Intermediate Size...")
        student = AutoModelForCausalLM.from_config(student_config).to(dtype=dtype)

        # 3. Copy Shared Layers (Embeddings, Final Norm, LM Head)
        print("  Copying token embeddings and output heads...")
        if hasattr(self.teacher.model, "embed_tokens"):
            student.model.embed_tokens.weight.data.copy_(self.teacher.model.embed_tokens.weight.data)
        if hasattr(self.teacher.model, "norm"):
            student.model.norm.weight.data.copy_(self.teacher.model.norm.weight.data)
        if hasattr(self.teacher, "lm_head"):
            student.lm_head.weight.data.copy_(self.teacher.lm_head.weight.data)

        # 4. Slicing Decoder Layers
        print("  Transferring and slicing layer weights...")
        t_layers = self.teacher.model.layers
        s_layers = student.model.layers

        for new_idx, orig_idx in enumerate(retained_layers):
            t_l = t_layers[orig_idx]
            s_l = s_layers[new_idx]
            neurons = retained_neurons_per_layer[orig_idx]

            # Copy LayerNorms
            s_l.input_layernorm.weight.data.copy_(t_l.input_layernorm.weight.data)
            s_l.post_attention_layernorm.weight.data.copy_(t_l.post_attention_layernorm.weight.data)

            # Copy Self-Attention (preserving attention projections & biases)
            s_l.self_attn.q_proj.weight.data.copy_(t_l.self_attn.q_proj.weight.data)
            s_l.self_attn.k_proj.weight.data.copy_(t_l.self_attn.k_proj.weight.data)
            s_l.self_attn.v_proj.weight.data.copy_(t_l.self_attn.v_proj.weight.data)
            s_l.self_attn.o_proj.weight.data.copy_(t_l.self_attn.o_proj.weight.data)

            if t_l.self_attn.q_proj.bias is not None and s_l.self_attn.q_proj.bias is not None:
                s_l.self_attn.q_proj.bias.data.copy_(t_l.self_attn.q_proj.bias.data)
            if t_l.self_attn.k_proj.bias is not None and s_l.self_attn.k_proj.bias is not None:
                s_l.self_attn.k_proj.bias.data.copy_(t_l.self_attn.k_proj.bias.data)
            if t_l.self_attn.v_proj.bias is not None and s_l.self_attn.v_proj.bias is not None:
                s_l.self_attn.v_proj.bias.data.copy_(t_l.self_attn.v_proj.bias.data)
            if getattr(t_l.self_attn.o_proj, "bias", None) is not None and getattr(s_l.self_attn.o_proj, "bias", None) is not None:
                s_l.self_attn.o_proj.bias.data.copy_(t_l.self_attn.o_proj.bias.data)

            # Slice MLP
            g_w = t_l.mlp.gate_proj.weight.data
            u_w = t_l.mlp.up_proj.weight.data
            d_w = t_l.mlp.down_proj.weight.data

            new_g, new_u, new_d = slice_swiglu_mlp(g_w, u_w, d_w, neurons)
            s_l.mlp.gate_proj.weight.data.copy_(new_g)
            s_l.mlp.up_proj.weight.data.copy_(new_u)
            s_l.mlp.down_proj.weight.data.copy_(new_d)

        # Parameter accounting
        t_params = sum(p.numel() for p in self.teacher.parameters())
        s_params = sum(p.numel() for p in student.parameters())

        print(f"\n  Teacher Parameters: {t_params/1e9:.3f} B ({t_params:,})")
        print(f"  Student Parameters: {s_params/1e9:.3f} B ({s_params:,})")
        print(f"  Compression Ratio : {(1 - s_params/t_params)*100:.2f}% reduction")

        # Save to disk
        student.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"  [OK] Saved initial physical student model to: {output_dir}")

        return student
