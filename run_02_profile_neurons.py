"""
Run Phase 2: Neuron Attribution & Domain Selectivity
Calculates First-Order Taylor series attribution and Z-score Domain Selectivity.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.profiling.neuron_attribution import NeuronAttributionProfiler

def main():
    cfg = get_full_extraction_config()
    model_name = cfg["base_model"]["name"]
    device = "cuda" if torch.cuda.is_available() and cfg["base_model"]["device"] == "cuda" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("PHASE 2: NEURON ATTRIBUTION & Z-SCORE DOMAIN SELECTIVITY")
    print(f"Base Model : {model_name}")
    print(f"Target     : {cfg['capability']['target_domain'].upper()}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model = model.to(device)

    dataset_builder = TaskDatasetBuilder(tokenizer)
    discovery_data = dataset_builder.get_discovery_datasets()

    profiler = NeuronAttributionProfiler(model, tokenizer, device=device)
    results = profiler.profile_all_domains_with_selectivity(
        discovery_data,
        target_domain=cfg["capability"]["target_domain"]
    )

    out_pt = os.path.join(cfg["paths"]["results_dir"], "02_neuron_attribution.pt")
    torch.save(results, out_pt)

    print(f"\n[Phase 2 Complete] Max Z-Selectivity: {results['z_selectivity'].max():.4f}")
    print(f"Saved neuron attribution data to: {out_pt}")

if __name__ == "__main__":
    main()
