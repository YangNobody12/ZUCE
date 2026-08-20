"""
Phase 9B: Inference & Efficiency Profiler
Profiles Peak VRAM, TTFT (Time to First Token), TPOT (Time per Output Token),
Output Throughput, and Concurrency Scaling.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any

class InferenceProfiler:
    def __init__(self, tokenizer: Any, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device

    def profile_efficiency(
        self,
        model: nn.Module,
        prompt: str = "Write a Python quicksort algorithm.",
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        """Profiles hardware metrics including VRAM, TTFT, TPOT, and throughput."""
        model.eval()

        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 1. Measure TTFT (Time to First Token)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            out_1 = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_sec = time.time() - t_start

        # 2. Measure Full Generation (Throughput & TPOT)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen_start = time.time()

        with torch.no_grad():
            out_full = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_gen_time = time.time() - t_gen_start

        generated_tokens = out_full.shape[1] - inputs["input_ids"].shape[1]
        tpot_ms = (total_gen_time / max(generated_tokens, 1)) * 1000.0
        throughput = generated_tokens / max(total_gen_time, 1e-6)

        peak_vram_mb = 0.0
        if torch.cuda.is_available() and self.device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        params = sum(p.numel() for p in model.parameters())

        return {
            "parameters": params,
            "parameters_million": round(params / 1e6, 2),
            "parameters_billion": round(params / 1e9, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "ttft_ms": round(ttft_sec * 1000.0, 2),
            "tpot_ms": round(tpot_ms, 2),
            "throughput_tokens_sec": round(throughput, 2),
            "total_gen_time_sec": round(total_gen_time, 3),
            "tokens_generated": generated_tokens
        }
