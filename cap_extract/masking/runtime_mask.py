"""
Phase 5: Runtime Mask Execution & Capability Preservation Engine
Dynamically applies capability masks in forward passes to validate capability preservation
without modifying underlying model weights (Non-destructive testing).
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
from tqdm import tqdm

from ..analyzer.hooks import HookManager
from ..configs.base_config import ExtractionConfig
from ..datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS, TRANSLATION_PROMPTS, GENERAL_PROMPTS
from ..utils import prepare_inputs

class RuntimeMaskEngine:
    def __init__(self, model: nn.Module, tokenizer: Any, config: Optional[ExtractionConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()
        self.hook_mgr = HookManager(self.model)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7
    ) -> str:
        """Generate response given a text prompt."""
        raw_inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        inputs = prepare_inputs(raw_inputs, self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def apply_runtime_mask(self, mask_dict: Dict[str, Any]):
        """Inject forward pre-hooks to dynamically zero out pruned neurons."""
        self.hook_mgr.clear()
        neuron_mask = mask_dict["neuron_mask"].to(self.model.device)
        num_layers = neuron_mask.shape[0]

        for l in range(num_layers):
            layer_mask = neuron_mask[l]
            self.hook_mgr.register_neuron_mask_hook(l, layer_mask)

    def remove_runtime_mask(self):
        """Remove all active mask hooks."""
        self.hook_mgr.clear()

    def evaluate_runtime_mask(
        self,
        mask_dict: Dict[str, Any],
        test_prompts: Optional[List[str]] = None,
        max_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Runs side-by-side generation comparing Baseline vs Masked output.
        """
        print("\n" + "="*70)
        print("PHASE 5: RUNTIME MASK VALIDATION & NON-DESTRUCTIVE TESTING")
        print("="*70)

        domain = mask_dict.get("domain", "coding")
        if test_prompts is None:
            if domain == "coding":
                test_prompts = CODING_PROMPTS
            elif domain == "math":
                test_prompts = MATH_PROMPTS
            elif domain == "translation":
                test_prompts = TRANSLATION_PROMPTS
            else:
                test_prompts = CODING_PROMPTS

        prompts_to_test = test_prompts[:max_samples]
        results = []
        similarities = []

        for idx, p in enumerate(prompts_to_test):
            print(f"\n--- Sample [{idx+1}/{len(prompts_to_test)}] ---")
            print(f"Prompt: {p[:100]}...")

            # 1. Baseline generation (no mask)
            self.remove_runtime_mask()
            baseline_output = self.generate_text(p, max_new_tokens=100)

            # 2. Masked generation
            self.apply_runtime_mask(mask_dict)
            masked_output = self.generate_text(p, max_new_tokens=100)
            self.remove_runtime_mask()

            # 3. Text Similarity
            sim = SequenceMatcher(None, baseline_output, masked_output).ratio()
            similarities.append(sim)

            print(f"Baseline Output (Snippet): {baseline_output[len(p):len(p)+120].strip()}...")
            print(f"Masked Output   (Snippet): {masked_output[len(p):len(p)+120].strip()}...")
            print(f"Similarity Score         : {sim:.4f}")

            results.append({
                "prompt": p,
                "baseline": baseline_output,
                "masked": masked_output,
                "similarity": sim
            })

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        print("\n" + "-"*50)
        print(f"Domain: {domain.upper()} | Mean Baseline-to-Mask Similarity: {avg_sim:.4f}")
        print("-"*50)

        return {
            "domain": domain,
            "mean_similarity": avg_sim,
            "samples": results
        }
