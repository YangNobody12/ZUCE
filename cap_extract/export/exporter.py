"""
Phase 10: Multi-Format Model Exporter
Exports extracted and distilled models to HuggingFace Safetensors, ONNX, GGUF, and TensorRT-LLM configurations.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class ModelExporter:
    def __init__(self, model: nn.Module, tokenizer: Any, output_base_dir: str = "./outputs/exports"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

    def export_huggingface(self, save_name: str = "hf_mini_model") -> str:
        """
        Exports the model in standard HuggingFace Safetensors format.
        """
        target_path = os.path.join(self.output_base_dir, save_name)
        os.makedirs(target_path, exist_ok=True)

        print(f"\n[Export HF] Saving HuggingFace Safetensors to: {target_path}")
        self.model.save_pretrained(target_path, safe_serialization=True)
        self.tokenizer.save_pretrained(target_path)
        print("[OK] HuggingFace export completed.")
        return target_path

    def export_onnx(self, save_name: str = "mini_model.onnx", opset_version: int = 17) -> str:
        """
        Exports the model to ONNX with dynamic batch and sequence length axes.
        """
        target_path = os.path.join(self.output_base_dir, save_name)
        print(f"\n[Export ONNX] Exporting ONNX model to: {target_path}...")

        dummy_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.model.device)
        dummy_attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long, device=self.model.device)

        self.model.eval()
        try:
            torch.onnx.export(
                self.model,
                (dummy_input_ids, dummy_attention_mask),
                target_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=opset_version,
                do_constant_folding=True
            )
            print("[OK] ONNX export successfully generated.")
        except Exception as e:
            print(f"! Note on ONNX export: {e}")

        return target_path

    def generate_gguf_conversion_script(self, hf_dir: str, save_name: str = "convert_gguf.bat") -> str:
        """
        Generates a helper batch script for converting the HuggingFace model into GGUF format
        using llama.cpp.
        """
        script_path = os.path.join(self.output_base_dir, save_name)
        content = f"""@echo off
REM GGUF Conversion Helper Script for Capability-Extracted Mini Model
REM Requires llama.cpp repo cloned or python convert_hf_to_gguf.py installed

echo [1/3] Converting HuggingFace to GGUF F16...
python convert_hf_to_gguf.py "{hf_dir}" --outfile "{self.output_base_dir}\\model_f16.gguf" --outtype f16

echo [2/3] Quantizing to Q8_0...
llama-quantize "{self.output_base_dir}\\model_f16.gguf" "{self.output_base_dir}\\model_q8_0.gguf" q8_0

echo [3/3] Quantizing to Q4_K_M...
llama-quantize "{self.output_base_dir}\\model_f16.gguf" "{self.output_base_dir}\\model_q4_k_m.gguf" q4_k_m

echo GGUF export and quantization finished!
pause
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[OK] GGUF conversion instructions script generated at: {script_path}")
        return script_path

    def generate_tensorrt_config(self, save_name: str = "tensorrt_llm_build.json") -> str:
        """
        Generates TensorRT-LLM build configuration for accelerated serving.
        """
        config_path = os.path.join(self.output_base_dir, save_name)
        cfg = getattr(self.model, "config", None)
        trt_cfg = {
            "builder_config": {
                "max_batch_size": 32,
                "max_input_len": 2048,
                "max_output_len": 1024,
                "max_beam_width": 1,
                "precision": "float16",
                "strongly_typed": True
            },
            "model_config": {
                "architecture": "Qwen2ForCausalLM",
                "hidden_size": getattr(cfg, "hidden_size", 2048),
                "intermediate_size": getattr(cfg, "intermediate_size", 3584),
                "num_hidden_layers": getattr(cfg, "num_hidden_layers", 28),
                "num_attention_heads": getattr(cfg, "num_attention_heads", 16),
                "num_key_value_heads": getattr(cfg, "num_key_value_heads", 16),
                "vocab_size": getattr(cfg, "vocab_size", 151936)
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(trt_cfg, f, indent=2)

        print(f"[OK] TensorRT-LLM configuration generated at: {config_path}")
        return config_path
