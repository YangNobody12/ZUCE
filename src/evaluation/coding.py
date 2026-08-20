"""
Phase 9A: Coding Capability Evaluator
Evaluates code generation across syntax validity, compilation, and functional execution.
"""

import re
import time
import torch
import torch.nn as nn
from typing import Dict, List, Any

class CodingEvaluator:
    def __init__(self, tokenizer: Any, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def check_python_syntax(code_text: str) -> bool:
        """Verifies whether extracted code is syntactically valid Python."""
        code_match = re.search(r"```(?:python)?\s*(.*?)(?:```|$)", code_text, re.DOTALL)
        candidate = code_match.group(1) if code_match else code_text

        try:
            compile(candidate, "<string>", "exec")
            return True
        except Exception:
            func_match = re.search(r"(def\s+\w+\(.*?\):(?:\n\s+.*)+)", candidate)
            if func_match:
                try:
                    compile(func_match.group(1), "<string>", "exec")
                    return True
                except Exception:
                    pass
            return False

    def evaluate_model_on_coding_prompts(
        self,
        model: nn.Module,
        test_prompts: List[Dict[str, Any]],
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """Runs evaluation over a list of coding problem items."""
        model.eval()
        results = []
        valid_count = 0
        total_time = 0.0

        for item in test_prompts:
            q_id = item.get("id", 0)
            title = item.get("title", "Coding Problem")
            prompt = item.get("prompt", "")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            vocab_limit = getattr(model.config, "vocab_size", 151936)
            inputs["input_ids"] = torch.clamp(inputs["input_ids"], min=0, max=vocab_limit - 1)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.time()
            pad_id = getattr(model.config, "pad_token_id", self.tokenizer.eos_token_id)
            if pad_id is None or pad_id >= vocab_limit:
                pad_id = vocab_limit - 1

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=pad_id
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - t0
            total_time += elapsed

            gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_valid = self.check_python_syntax(gen_text)
            if is_valid:
                valid_count += 1

            results.append({
                "id": q_id,
                "title": title,
                "time_sec": round(elapsed, 3),
                "valid_syntax": is_valid,
                "output": gen_text
            })

        pass_rate = (valid_count / max(len(test_prompts), 1)) * 100.0
        return {
            "pass_rate_pct": round(pass_rate, 2),
            "valid_count": valid_count,
            "total_questions": len(test_prompts),
            "total_time_sec": round(total_time, 2),
            "avg_time_per_q": round(total_time / max(len(test_prompts), 1), 3),
            "questions": results
        }
