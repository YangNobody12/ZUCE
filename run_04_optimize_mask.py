"""
Run Phase 4: Soft Capability Mask Optimization
Learns continuous mask M with L1 regularized loss and produces binary capability mask.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.masks.mask_optimizer import MaskOptimizer

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 4: SOFT CAPABILITY MASK OPTIMIZATION")
    print(f"Target Capability: {cfg['capability']['target_domain'].upper()}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    discovery_data = dataset_builder.get_discovery_datasets()
    code_prompts = discovery_data["coding"]

    optimizer = MaskOptimizer(model, tokenizer, device=device)
    mask_results = optimizer.optimize_capability_mask(
        prompts=code_prompts,
        epochs=cfg["capability"].get("soft_mask_epochs", 5),
        lr=cfg["capability"].get("soft_mask_lr", 0.02),
        l1_lambda=cfg["capability"].get("l1_lambda", 0.005),
        threshold=cfg["capability"].get("binary_threshold", 0.5)
    )

    # Load Phase 2 neuron attribution and Z-selectivity to refine mask ranking
    attr_path = os.path.join(cfg["paths"]["results_dir"], "02_neuron_attribution.pt")
    if os.path.exists(attr_path):
        attr_data = torch.load(attr_path, map_location="cpu")
        code_attr = attr_data["attributions"]["coding"]
        z_sel = attr_data["z_selectivity"]
        
        # Normalize and compute Composite Capability Score
        norm_attr = (code_attr - code_attr.min()) / (code_attr.max() - code_attr.min() + 1e-8)
        norm_sel = (z_sel - z_sel.min()) / (z_sel.max() - z_sel.min() + 1e-8)
        composite_score = 0.5 * norm_attr + 0.5 * norm_sel
        
        target_k = cfg["student_target"].get("target_intermediate_size", 2304)
        refined_binary = torch.zeros_like(composite_score)
        for l_idx in range(composite_score.shape[0]):
            top_k_idx = torch.topk(composite_score[l_idx], target_k).indices
            refined_binary[l_idx, top_k_idx] = 1.0
            
        mask_results["binary_mask"] = refined_binary
        mask_results["continuous_mask"] = composite_score
        mask_results["active_neuron_ratio"] = float(refined_binary.mean().item())
        mask_results["sparsity_pct"] = round((1.0 - refined_binary.mean().item()) * 100, 2)

    out_pt = os.path.join(cfg["paths"]["results_dir"], "04_capability_mask.pt")
    torch.save(mask_results, out_pt)

    print(f"\n[Phase 4 Complete] Optimized Mask saved to: {out_pt}")
    print(f"  Neuron Sparsity: {mask_results['sparsity_pct']}% (Retained: {mask_results['active_neuron_ratio']*100:.1f}%)")

if __name__ == "__main__":
    main()
