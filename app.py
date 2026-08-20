"""
Interactive Web Application for Capability-Aware Specialist Model Exploration
Features:
1. Live Side-by-Side Code Generation & Latency Profiler (Base vs Specialist)
2. Interactive Circuit & Layer Redundancy Visualizer
3. Multi-Scale Benchmark Explorer & Performance Metrics
"""

import os
import sys
import time
import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Model Paths
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SPECIALIST_1_0B_PATH = "./outputs/specialist_1.0b_safetensors"
SPECIALIST_1_4B_PATH = "./outputs/specialist_1.4b_safetensors"

print("[Init] Loading tokenizers...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Cached models
loaded_models = {}

def get_model(model_key):
    if model_key in loaded_models:
        return loaded_models[model_key]
    
    print(f"[Loading] {model_key} onto {device}...")
    if model_key == "Base Teacher (1.54B)":
        m = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    elif model_key == "Specialist-1.0B (1.10B, -28.9%)":
        m = AutoModelForCausalLM.from_pretrained(SPECIALIST_1_0B_PATH, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    elif model_key == "Specialist-1.4B (1.42B, -8.0%)":
        m = AutoModelForCausalLM.from_pretrained(SPECIALIST_1_4B_PATH, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    else:
        raise ValueError(f"Unknown model: {model_key}")
    
    m.eval()
    loaded_models[model_key] = m
    return m

def generate_code_comparison(prompt, model_choice, max_tokens, temperature, top_p):
    """Generates code and returns output text, latency, and tokens/sec."""
    model = get_model(model_choice)
    
    formatted_prompt = prompt if "```python" in prompt else f"{prompt}\n```python\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        if temperature > 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    elapsed = time.time() - start_time
    
    gen_tokens_count = outputs.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = gen_tokens_count / max(elapsed, 1e-4)
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    metrics = f"⏱️ Time: {elapsed:.2f}s | 🚀 Speed: {tok_per_sec:.1f} tokens/s | 📝 Tokens Generated: {gen_tokens_count}"
    return decoded, metrics

def load_benchmark_report():
    report_file = "./outputs/advanced_specialist_suite_report.json"
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("benchmark_results", [])
    return []

# Build Gradio UI
with gr.Blocks(title="Capability-Aware Model Extraction Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Capability-Aware Specialist Model Extraction Dashboard")
    gr.Markdown("> **Zero-Update Subnetwork Extraction ($\Delta \theta = 0$) from Dense LLMs (Qwen2.5-1.5B)**")
    
    with gr.Tab("💻 Live Code Playground & Comparison"):
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=[
                        "Specialist-1.0B (1.10B, -28.9%)",
                        "Specialist-1.4B (1.42B, -8.0%)",
                        "Base Teacher (1.54B)"
                    ],
                    value="Specialist-1.0B (1.10B, -28.9%)",
                    label="Select Model"
                )
                prompt_input = gr.Textbox(
                    lines=4,
                    label="Coding Prompt",
                    placeholder="Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number.",
                    value="Write a Python function `two_sum(nums, target)` that returns the indices of two numbers that add up to target in O(n) time."
                )
                with gr.Row():
                    max_tokens_slider = gr.Slider(minimum=16, maximum=256, value=64, step=16, label="Max New Tokens")
                    temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature")
                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top P")
                
                run_btn = gr.Button("⚡ Generate Code", variant="primary")
            
            with gr.Column(scale=1):
                output_code = gr.Code(label="Generated Python Code", language="python", lines=12)
                metrics_text = gr.Markdown(label="Performance Metrics")
        
        run_btn.click(
            fn=generate_code_comparison,
            inputs=[prompt_input, model_dropdown, max_tokens_slider, temperature_slider, top_p_slider],
            outputs=[output_code, metrics_text]
        )
    
    with gr.Tab("📊 Multi-Scale Benchmark Results"):
        gr.Markdown("### Comprehensive 20-Question Algorithmic Evaluation")
        gr.Markdown("""
| Model Architecture | Parameters | Reduction (%) | Pass Rate (20Q) | Latency | NCD (Capability Density) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Teacher (Qwen2.5-1.5B)** | **1,543.7 M** | **0.0%** | **5.0% ( 1/20)** | 2.79s/Q | 1.00x |
| **Specialist-1.4B ($k=8000$)** | **1,419.9 M** | **8.0%** | **15.0% ( 3/20)** | 2.82s/Q | 3.26x |
| **Specialist-1.2B ($k=6800$)** | **1,265.0 M** | **18.1%** | **35.0% ( 7/20)** | 2.62s/Q | 8.54x |
| **Specialist-1.0B ($k=5500$)** | **1,097.3 M** | **28.9%** | **35.0% ( 7/20)** 🌟 | **2.11s/Q** ⚡ | **9.85x** 🏆 |
        """)
        gr.Markdown("🌟 **Specialist-1.0B outperforms Base Teacher by 7x** while running **24% faster** under strict $\Delta \theta = 0$!")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
