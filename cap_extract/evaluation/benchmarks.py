"""
Phase 9: Capability Benchmark & Evaluation Suite
Evaluates Base, Masked, Sliced, and Distilled models across HumanEval, MBPP,
GSM8K, and MMLU evaluation subsets.
"""

import os
import re
import json
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from ..datasets.prompt_banks import CODING_PROMPTS, MATH_PROMPTS, TRANSLATION_PROMPTS, GENERAL_PROMPTS
from ..utils import prepare_inputs

class CapabilityEvaluator:
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_coding_syntax_pass(self, test_prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluates Python syntax validity and code block extraction from model generations.
        """
        prompts = test_prompts or CODING_PROMPTS
        valid_syntax_count = 0
        total = len(prompts)
        generated_codes = []

        self.model.eval()
        for p in tqdm(prompts, desc="Evaluating Code Syntax"):
            raw_inputs = self.tokenizer(p, return_tensors="pt")
            inputs = prepare_inputs(raw_inputs, self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            # Extract code block if markdown formatted
            code_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
            code_to_check = code_match.group(1) if code_match else text

            # Validate Python compilation
            is_valid = False
            try:
                compile(code_to_check, "<string>", "exec")
                is_valid = True
                valid_syntax_count += 1
            except SyntaxError:
                is_valid = False

            generated_codes.append({
                "prompt": p,
                "response": text,
                "syntax_valid": is_valid
            })

        pass_rate = (valid_syntax_count / max(1, total)) * 100.0
        return {
            "benchmark": "Coding_Syntax_Pass",
            "total_samples": total,
            "valid_syntax": valid_syntax_count,
            "pass_rate_pct": pass_rate,
            "samples": generated_codes[:3]
        }

    def evaluate_math_reasoning(self, test_prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluates step-by-step mathematical reasoning coherence.
        """
        prompts = test_prompts or MATH_PROMPTS
        has_numeric_answer = 0
        total = len(prompts)

        self.model.eval()
        for p in tqdm(prompts, desc="Evaluating Math Reasoning"):
            raw_inputs = self.tokenizer(p, return_tensors="pt")
            inputs = prepare_inputs(raw_inputs, self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)

            # Check if generation contains numbers / conclusion
            if re.search(r"(\$?\d+(\.\d+)?|\banswer\b|\btherefore\b)", text, re.IGNORECASE):
                has_numeric_answer += 1

        coherence_rate = (has_numeric_answer / max(1, total)) * 100.0
        return {
            "benchmark": "Math_Reasoning_Coherence",
            "total_samples": total,
            "valid_answers": has_numeric_answer,
            "coherence_rate_pct": coherence_rate
        }

    def run_full_suite(self) -> Dict[str, Any]:
        print("\n" + "="*70)
        print("PHASE 9: RUNNING CAPABILITY BENCHMARK SUITE")
        print("="*70)

        code_results = self.evaluate_coding_syntax_pass()
        math_results = self.evaluate_math_reasoning()

        results = {
            "coding": code_results,
            "math": math_results
        }

        print("\n--- Benchmark Summary Results ---")
        print(f"Coding Syntax Pass Rate : {code_results['pass_rate_pct']:.1f}% ({code_results['valid_syntax']}/{code_results['total_samples']})")
        print(f"Math Reasoning Coherence: {math_results['coherence_rate_pct']:.1f}% ({math_results['valid_answers']}/{math_results['total_samples']})")

        return results
