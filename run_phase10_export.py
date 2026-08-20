"""
Runner for Phase 10: Model Export (HuggingFace, GGUF, ONNX, TensorRT-LLM)
"""

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from cap_extract.export.exporter import ModelExporter

def main():
    parser = argparse.ArgumentParser(description="Phase 10: Multi-Format Model Export")
    parser.add_argument("--model_dir", type=str, default="./outputs/mini_model_0.5b", help="Model path to export")
    parser.add_argument("--export_dir", type=str, default="./outputs/exports", help="Export destination directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading Model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)

    exporter = ModelExporter(model, tokenizer, output_base_dir=args.export_dir)

    # 1. HuggingFace Safetensors
    hf_path = exporter.export_huggingface("qwen2.5_0.5b_extracted")

    # 2. ONNX Model
    exporter.export_onnx("qwen2.5_0.5b_extracted.onnx")

    # 3. GGUF Conversion Script
    exporter.generate_gguf_conversion_script(hf_path, "convert_to_gguf.bat")

    # 4. TensorRT Config
    exporter.generate_tensorrt_config("tensorrt_llm_config.json")

    print(f"\n[OK] All Phase 10 Exports completed in: {args.export_dir}")

if __name__ == "__main__":
    main()
