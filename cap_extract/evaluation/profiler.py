"""
Phase 9: Efficiency & Resource Profiler
Measures latency (TTFT, TPOT), peak VRAM memory, and throughput (tokens/sec).
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ..utils import prepare_inputs

class ModelProfiler:
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def profile_inference(
        self,
        prompt: str = "def quicksort(arr):",
        max_new_tokens: int = 128,
        warmup_runs: int = 2,
        test_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Profiles inference speed and memory footprint.
        """
        device = self.model.device
        raw_inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = prepare_inputs(raw_inputs, device)

        # Warmup
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup_runs):
                self.model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)

        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        latencies = []
        tokens_generated = []

        for _ in range(test_runs):
            if torch.cuda.is_available() and "cuda" in str(device):
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            if torch.cuda.is_available() and "cuda" in str(device):
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            gen_len = out.shape[1] - inputs["input_ids"].shape[1]
            latencies.append(t1 - t0)
            tokens_generated.append(gen_len)

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        throughput = avg_tokens / max(1e-6, avg_latency)
        tpot_ms = (avg_latency / max(1, avg_tokens)) * 1000.0

        peak_vram_mb = 0.0
        if torch.cuda.is_available() and "cuda" in str(device):
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        param_count = sum(p.numel() for p in self.model.parameters())

        report = {
            "parameters_million": round(param_count / 1e6, 2),
            "avg_latency_sec": round(avg_latency, 4),
            "tokens_per_second": round(throughput, 2),
            "time_per_token_ms": round(tpot_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2)
        }

        print("\n" + "="*50)
        print("INFERENCE EFFICIENCY PROFILE")
        print("="*50)
        print(f"Parameters    : {report['parameters_million']} M")
        print(f"Throughput    : {report['tokens_per_second']} tokens/sec")
        print(f"Time/Token    : {report['time_per_token_ms']} ms")
        print(f"Peak VRAM     : {report['peak_vram_mb']} MB")
        print("="*50)

        return report
