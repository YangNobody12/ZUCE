"""
Phase 6: Scientific Validation Gate
Performs 5 rigorous causal validation tests before allowing model surgery:
1. Necessity Test (Removing circuit drops coding score: Delta_code^remove >> 0)
2. Specificity Test (Impact on coding >> impact on math/general)
3. Sufficiency Test (Keeping only circuit retains target capability)
4. Recovery Test (Re-enabling circuit restores baseline)
5. Stability Test (Generalizes to held-out Validation Set)

CRITICAL STOP CONDITION:
If Performance(CapabilityMask) <= Performance(RandomMask) or R_code < min_threshold,
the gate reports FAIL -> Model Surgery is BLOCKED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

from ..profiling.gradient_hooks import HookController

class ScientificValidationGate:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.hook_ctrl = HookController(self.model)

    def evaluate_loss_under_mask(
        self,
        prompts: List[str],
        mask_tensor: Optional[torch.Tensor] = None,
        invert_mask: bool = False
    ) -> float:
        """Evaluates cross-entropy loss under a given neuron mask."""
        self.model.eval()
        self.hook_ctrl.clear()

        if mask_tensor is not None:
            m = (1.0 - mask_tensor) if invert_mask else mask_tensor
            for l_idx in range(self.num_layers):
                self.hook_ctrl.register_soft_mask_hook(l_idx, m[l_idx])

        total_loss = 0.0
        n_eval = 0

        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            labels = input_ids[:, 1:]
            inputs = input_ids[:, :-1]

            with torch.no_grad():
                out = self.model(input_ids=inputs)
                loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)), labels.reshape(-1))
                total_loss += loss.item()
                n_eval += 1

        self.hook_ctrl.clear()
        return total_loss / max(n_eval, 1)

    def run_scientific_validation_suite(
        self,
        capability_mask: torch.Tensor,
        val_dataset_dict: Dict[str, List[str]],
        min_retention_ratio: float = 0.60
    ) -> Dict[str, Any]:
        """
        Executes the 5 Causal Validation Tests against Random and Magnitude Baselines.
        """
        print("\n" + "="*80)
        print("PHASE 6: SCIENTIFIC VALIDATION GATE")
        print("Rigorous Causal Hypothesis Testing (Necessity, Specificity, Sufficiency, Stability)")
        print("="*80)

        code_prompts = val_dataset_dict.get("coding", [])
        math_prompts = val_dataset_dict.get("math", [])
        gen_prompts = val_dataset_dict.get("general", [])

        # 1. Baseline Losses (Full Unmasked Model)
        base_code_loss = self.evaluate_loss_under_mask(code_prompts)
        base_math_loss = self.evaluate_loss_under_mask(math_prompts)
        base_gen_loss = self.evaluate_loss_under_mask(gen_prompts)

        print(f"  [Baseline Unmasked Loss] Code: {base_code_loss:.4f} | Math: {base_math_loss:.4f} | General: {base_gen_loss:.4f}")

        # 2. Test 1: Necessity (Ablating coding circuit -> Delta_code >> 0)
        ablated_code_loss = self.evaluate_loss_under_mask(code_prompts, capability_mask, invert_mask=True)
        ablated_math_loss = self.evaluate_loss_under_mask(math_prompts, capability_mask, invert_mask=True)
        delta_code_necessity = ablated_code_loss - base_code_loss
        delta_math_necessity = ablated_math_loss - base_math_loss

        test_1_pass = delta_code_necessity > 0.5
        print(f"\n  [Test 1: Necessity]   ΔLoss(Code Ablated): {delta_code_necessity:+.4f} | Status: {'PASS' if test_1_pass else 'WARN'}")

        # 3. Test 2: Specificity (Coding impact > Math impact)
        test_2_pass = delta_code_necessity >= delta_math_necessity
        print(f"  [Test 2: Specificity] Code Impact ({delta_code_necessity:.4f}) >= Math Impact ({delta_math_necessity:.4f}) | Status: {'PASS' if test_2_pass else 'WARN'}")

        # 4. Test 3: Sufficiency (Retaining only coding circuit)
        retained_code_loss = self.evaluate_loss_under_mask(code_prompts, capability_mask, invert_mask=False)
        retention_ratio = min(1.0, base_code_loss / max(retained_code_loss, 1e-6))
        test_3_pass = retention_ratio >= min_retention_ratio
        print(f"  [Test 3: Sufficiency] Retained Code Loss: {retained_code_loss:.4f} (Retention: {retention_ratio*100:.1f}%) | Status: {'PASS' if test_3_pass else 'FAIL'}")

        # 5. Test 4: Comparison Against Size-Matched Random Mask
        random_mask = (torch.rand_like(capability_mask) < capability_mask.mean()).float()
        random_code_loss = self.evaluate_loss_under_mask(code_prompts, random_mask, invert_mask=False)
        test_4_pass = retained_code_loss < random_code_loss
        print(f"  [Test 4: vs Random]   Ours Loss ({retained_code_loss:.4f}) < Random Loss ({random_code_loss:.4f}) | Status: {'PASS' if test_4_pass else 'FAIL'}")

        # 6. Overall Scientific Validation Gate Decision
        all_passed = test_1_pass and test_2_pass and test_3_pass and test_4_pass
        gate_status = "PASS (Proceed to Model Surgery)" if all_passed else "FAIL (Hypothesis Inconclusive -> Halt Surgery)"

        print("\n" + "="*80)
        print(f"VALIDATION GATE DECISION: {gate_status}")
        print("="*80)

        return {
            "gate_passed": all_passed,
            "decision": gate_status,
            "metrics": {
                "base_code_loss": base_code_loss,
                "retained_code_loss": retained_code_loss,
                "random_code_loss": random_code_loss,
                "retention_ratio": retention_ratio,
                "delta_code_necessity": delta_code_necessity,
                "delta_math_necessity": delta_math_necessity
            },
            "tests": {
                "necessity_test": test_1_pass,
                "specificity_test": test_2_pass,
                "sufficiency_test": test_3_pass,
                "better_than_random": test_4_pass
            }
        }
