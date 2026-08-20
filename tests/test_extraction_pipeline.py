"""
Automated Unit and Integration Tests for Capability-aware Model Extraction Pipeline.
Uses a lightweight transformer architecture to verify Phases 1-10 end-to-end.
"""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

from cap_extract.configs.base_config import ExtractionConfig
from cap_extract.analyzer.hooks import HookManager
from cap_extract.analyzer.layer_analyzer import LayerAnalyzer
from cap_extract.analyzer.neuron_analyzer import NeuronAnalyzer
from cap_extract.analyzer.circuit_discovery import CircuitDiscovery
from cap_extract.masking.mask_generator import MaskGenerator
from cap_extract.masking.runtime_mask import RuntimeMaskEngine
from cap_extract.surgery.model_surgery import ModelSurgeryEngine
from cap_extract.training.losses import CapabilityDistillationLoss
from cap_extract.evaluation.benchmarks import CapabilityEvaluator
from cap_extract.evaluation.profiler import ModelProfiler
from cap_extract.export.exporter import ModelExporter

class BatchDict(dict):
    def to(self, device):
        return BatchDict({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})

class MockTokenizer:
    """Lightweight dummy tokenizer for testing."""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", max_length=128, truncation=True, padding=False):
        # Generate deterministic token IDs from text
        tokens = [abs(hash(w)) % (self.vocab_size - 2) + 2 for w in text.split()]
        if not tokens:
            tokens = [2]
        tokens = tokens[:max_length]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return BatchDict({"input_ids": input_ids, "attention_mask": attention_mask})

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "def test_func(): return True"

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w") as f:
            f.write('{"model_type": "mock"}')

@pytest.fixture
def test_setup():
    test_dir = "./test_outputs"
    os.makedirs(test_dir, exist_ok=True)

    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256
    )
    model = Qwen2ForCausalLM(config).eval()
    tokenizer = MockTokenizer(vocab_size=1000)

    extract_cfg = ExtractionConfig(
        output_dir=test_dir,
        output_mini_model_dir=os.path.join(test_dir, "mini_model"),
        matrix_output_path=os.path.join(test_dir, "layer_matrix.json"),
        neuron_output_path=os.path.join(test_dir, "neuron_data.pt"),
        circuit_output_path=os.path.join(test_dir, "circuit.json"),
        mask_output_path=os.path.join(test_dir, "mask.pt"),
        num_calibration_samples=2,
        target_neuron_retention_ratio=0.5
    )

    yield model, tokenizer, extract_cfg

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_hook_manager(test_setup):
    model, tokenizer, cfg = test_setup
    hm = HookManager(model)
    assert hm.num_layers() == 2

    # Test activation capture
    hm.register_activation_capture("all")
    inputs = tokenizer("test prompt for hook capture")
    out = model(inputs["input_ids"])
    assert "layer_0" in hm.activations
    assert "mlp_neuron_0" in hm.activations
    assert "attn_0" in hm.activations
    hm.clear()
    assert len(hm.activations) == 0

def test_phases_1_to_6_end_to_end(test_setup):
    model, tokenizer, cfg = test_setup

    # Phase 1: Layer Analyzer
    l_analyzer = LayerAnalyzer(model, tokenizer, cfg)
    layer_matrix = l_analyzer.generate_layer_importance_matrix()
    assert "coding" in layer_matrix["domains"]
    assert 0 in layer_matrix["domains"]["coding"]

    # Phase 2: Neuron Analyzer
    n_analyzer = NeuronAnalyzer(model, tokenizer, cfg)
    n_data = n_analyzer.run_full_neuron_analysis()
    assert n_data["neuron_importance"]["coding"].shape == (2, 128)
    assert n_data["head_importance"]["coding"].shape == (2, 4)

    # Phase 3: Circuit Discovery
    circ_engine = CircuitDiscovery(cfg)
    circuits = circ_engine.discover_circuits(n_data, top_k_neurons_per_layer=32, top_k_heads_per_layer=2)
    assert "coding" in circuits
    assert circuits["coding"]["total_nodes"] > 0

    # Phase 4: Expert Mask Generation
    mask_gen = MaskGenerator(cfg)
    mask_dict = mask_gen.generate_capability_mask(n_data, target_domain="coding", neuron_retention_ratio=0.5)
    assert mask_dict["neuron_mask"].shape == (2, 128)
    assert mask_dict["neuron_mask"].sum().item() == 2 * 64 # 50% of 128 = 64 per layer

    # Phase 5: Runtime Mask
    rt_engine = RuntimeMaskEngine(model, tokenizer, cfg)
    rt_results = rt_engine.evaluate_runtime_mask(mask_dict, max_samples=2)
    assert "mean_similarity" in rt_results

    # Phase 6: Model Surgery (Extract Mini Model)
    surgery = ModelSurgeryEngine(model, tokenizer, cfg)
    mini_model = surgery.perform_surgery(mask_dict, output_dir=cfg.output_mini_model_dir)
    assert mini_model.config.intermediate_size == 64

    # Verify mini model forward pass
    inputs = tokenizer("test prompt for mini model")
    mini_out = mini_model(inputs["input_ids"])
    assert mini_out.logits.shape[-1] == 1000

def test_distillation_loss():
    loss_fn = CapabilityDistillationLoss(alpha_kd=0.4, beta_ce=0.3, gamma_circuit=0.3)
    s_logits = torch.randn(2, 8, 1000, requires_grad=True)
    t_logits = torch.randn(2, 8, 1000)
    labels = torch.randint(0, 1000, (2, 8))
    s_hidden = [torch.randn(2, 8, 64, requires_grad=True)]
    t_hidden = [torch.randn(2, 8, 64)]

    loss, metrics = loss_fn(s_logits, t_logits, labels, s_hidden, t_hidden)
    loss.backward()
    assert s_logits.grad is not None
    assert metrics["loss_total"] > 0.0

def test_evaluation_and_profiler(test_setup):
    model, tokenizer, cfg = test_setup
    evaluator = CapabilityEvaluator(model, tokenizer)
    results = evaluator.evaluate_coding_syntax_pass(["def add(a, b): return a + b"])
    assert "pass_rate_pct" in results

    profiler = ModelProfiler(model, tokenizer)
    report = profiler.profile_inference("test", max_new_tokens=10, warmup_runs=1, test_runs=2)
    assert "tokens_per_second" in report

def test_export(test_setup):
    model, tokenizer, cfg = test_setup
    exporter = ModelExporter(model, tokenizer, output_base_dir=os.path.join(cfg.output_dir, "exports"))
    hf_dir = exporter.export_huggingface("test_hf_export")
    assert os.path.exists(os.path.join(hf_dir, "config.json"))

    bat_script = exporter.generate_gguf_conversion_script(hf_dir)
    assert os.path.exists(bat_script)

    trt_cfg = exporter.generate_tensorrt_config()
    assert os.path.exists(trt_cfg)
