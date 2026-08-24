"""
Simple 1-Click Quickstart
Run with: python examples/quickstart.py
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zuce import ZUCE

def main():
    print("=" * 70)
    print("🚀 ZUCE-AI: QUICKSTART RUNNER")
    print("=" * 70)
    
    model = "Qwen/Qwen2.5-1.5B"
    
    # 1. Inspect
    report = ZUCE.inspect(model)
    print(f"1. Model : {report.model_type} ({report.num_parameters/1e6:.1f}M params)")
    
    # 2. Compress (AMPQ)
    ampq = ZUCE.quantize_ampq(model, group_size=128)
    print(f"2. Quant : {ampq.average_bits_per_weight}b/weight (-{ampq.vram_reduction_pct}% VRAM)")
    
    # 3. Fuse Multi-Experts
    fusion = ZUCE.fuse_teachers(model, top_k=2)
    print(f"3. Fusion: Dynamic Router Active (Saved {fusion.vram_savings_pct}% VRAM)")
    
    # 4. Test Real Code
    exam = ZUCE.evaluate_exam(model, max_tokens=128)
    print(f"4. Exam  : Passed {exam.passed_problems}/{exam.total_problems} ({exam.functional_pass_rate_pct:.1f}%) 🌟")
    print("=" * 70)

if __name__ == "__main__":
    main()
