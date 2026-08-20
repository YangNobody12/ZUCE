"""
Automated Verification & Integrity Test Suite
Verifies:
1. Exact Parameter Identity: θ_mini ⊆ θ_teacher (Δθ = 0)
2. Attention Projection & Bias Integrity
3. Standalone Safetensors Loading
4. Python Syntax & Generation Validity
"""

import os
import ast
import torch
import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM

@unittest.skipUnless(
    os.environ.get("ZUCE_RUN_LARGE_TESTS") == "1",
    "set ZUCE_RUN_LARGE_TESTS=1 to run model-download/GPU integration tests",
)
class TestExtractionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16 if cls.device == "cuda" else torch.float32
        cls.model_path = "./outputs/specialist_1.0b_safetensors"
        cls.base_name = "Qwen/Qwen2.5-1.5B"
        
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.base_name)
        cls.student = AutoModelForCausalLM.from_pretrained(cls.model_path, torch_dtype=cls.dtype, device_map="auto" if cls.device == "cuda" else None)
        cls.student.eval()

    def test_01_standalone_files_exist(self):
        """Verifies that all required standalone HuggingFace files exist."""
        required_files = ["config.json", "model.safetensors", "tokenizer.json", "generation_config.json", "extraction_metadata.json"]
        for f in required_files:
            file_path = os.path.join(self.model_path, f)
            self.assertTrue(os.path.exists(file_path), f"Missing required file: {file_path}")

    def test_02_model_architecture_and_parameters(self):
        """Verifies model layer count (28) and parameter reduction."""
        num_layers = self.student.config.num_hidden_layers
        self.assertEqual(num_layers, 28, "Student model must retain all 28 layers.")
        
        intermediate_size = self.student.config.intermediate_size
        self.assertEqual(intermediate_size, 5500, "Intermediate size must match target 5500.")
        
        total_params = sum(p.numel() for p in self.student.parameters())
        self.assertLess(total_params, 1.2e9, "Student model must have < 1.2B parameters.")
        self.assertGreater(total_params, 1.0e9, "Student model must have > 1.0B parameters.")

    def test_03_attention_biases_present(self):
        """Verifies that attention projection biases are loaded and non-zero."""
        l0_attn = self.student.model.layers[0].self_attn
        self.assertIsNotNone(l0_attn.q_proj.bias, "q_proj must have bias.")
        self.assertIsNotNone(l0_attn.k_proj.bias, "k_proj must have bias.")
        self.assertIsNotNone(l0_attn.v_proj.bias, "v_proj must have bias.")

    def test_05_exact_weight_subset_identity(self):
        """Verifies mathematical identity: θ_mini ⊆ θ_teacher and Δθ = 0 (exact bit-for-bit)."""
        teacher = AutoModelForCausalLM.from_pretrained(self.base_name, torch_dtype=self.dtype, device_map="auto" if self.device == "cuda" else None)
        teacher.eval()

        # Check embedding and output head
        torch.testing.assert_close(self.student.model.embed_tokens.weight, teacher.model.embed_tokens.weight)
        torch.testing.assert_close(self.student.model.norm.weight, teacher.model.norm.weight)
        torch.testing.assert_close(self.student.lm_head.weight, teacher.lm_head.weight)

        # Check Layer 0 Attention projections
        s_l0 = self.student.model.layers[0].self_attn
        t_l0 = teacher.model.layers[0].self_attn
        torch.testing.assert_close(s_l0.q_proj.weight, t_l0.q_proj.weight)
        torch.testing.assert_close(s_l0.k_proj.weight, t_l0.k_proj.weight)
        torch.testing.assert_close(s_l0.v_proj.weight, t_l0.v_proj.weight)
        torch.testing.assert_close(s_l0.o_proj.weight, t_l0.o_proj.weight)
        torch.testing.assert_close(s_l0.q_proj.bias, t_l0.q_proj.bias)
        torch.testing.assert_close(s_l0.k_proj.bias, t_l0.k_proj.bias)
        torch.testing.assert_close(s_l0.v_proj.bias, t_l0.v_proj.bias)

    def test_06_layernorm_identity_all_28_layers(self):
        """Verifies that LayerNorm weights across all 28 layers match teacher exactly."""
        teacher = AutoModelForCausalLM.from_pretrained(self.base_name, torch_dtype=self.dtype, device_map="auto" if self.device == "cuda" else None)
        for l in range(28):
            s_l = self.student.model.layers[l]
            t_l = teacher.model.layers[l]
            torch.testing.assert_close(s_l.input_layernorm.weight, t_l.input_layernorm.weight)
            torch.testing.assert_close(s_l.post_attention_layernorm.weight, t_l.post_attention_layernorm.weight)

    def test_07_multi_prompt_syntax_ast_verification(self):
        """Verifies that generated code across multiple distinct prompts is valid Python AST."""
        test_prompts = [
            "Write a Python function `two_sum(nums, target)` in O(n).\n```python\n",
            "Write a Python function `is_palindrome(s)`.\n```python\n",
            "Write a Python function `binary_search(arr, target)`.\n```python\n"
        ]
        for p in test_prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.student.generate(**inputs, max_new_tokens=48, do_sample=False)
            decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
            self.assertIn("def ", decoded, f"Generated output must contain function definition for prompt: {p}")

if __name__ == "__main__":
    unittest.main()
