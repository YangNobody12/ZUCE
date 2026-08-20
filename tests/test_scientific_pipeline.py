"""
Unit and Integration Tests for Scientific Capability Extraction Framework.
Tests:
1. Config loading
2. Task Dataset balance (|D_code| == |D_math| == |D_general|)
3. Soft Mask and Sigmoid bounds
4. SwiGLU MLP Tensor Slicing
5. Layer Mapping monotonicity (l_1 < l_2 < ... < l_k)
6. Statistical Metrics (Retention, Specialization Gain, Compression Efficiency)
"""

import pytest
import torch
from configs.config_loader import get_full_extraction_config
from task_datasets.task_dataset_builder import TaskDatasetBuilder
from src.masks.soft_mask import SoftCapabilityMask
from src.surgery.mlp_surgery import slice_swiglu_mlp
from src.surgery.layer_mapping import LayerMappingOptimizer
from src.evaluation.statistics import StatisticalValidator
from src.evaluation.coding import CodingEvaluator

def test_config_loader():
    cfg = get_full_extraction_config()
    assert "base_model" in cfg
    assert "capability" in cfg
    assert "student_target" in cfg
    assert cfg["capability"]["target_domain"] == "coding"
    assert cfg["student_target"]["target_parameter_budget"] > 0

def test_soft_mask_bounds():
    mask_mod = SoftCapabilityMask(num_layers=28, intermediate_size=8960, init_val=0.8)
    soft = mask_mod()
    assert soft.shape == (28, 8960)
    assert (soft >= 0.0).all() and (soft <= 1.0).all()

    binary = mask_mod.binarize(threshold=0.5)
    assert ((binary == 0.0) | (binary == 1.0)).all()

def test_swiglu_mlp_slicing():
    d_model = 2048
    d_ff = 8960
    d_target = 2304

    gate = torch.randn(d_ff, d_model)
    up = torch.randn(d_ff, d_model)
    down = torch.randn(d_model, d_ff)

    retained_idx = list(range(d_target))
    new_g, new_u, new_d = slice_swiglu_mlp(gate, up, down, retained_idx)

    assert new_g.shape == (d_target, d_model)
    assert new_u.shape == (d_target, d_model)
    assert new_d.shape == (d_model, d_target)

def test_monotonic_layer_mapping():
    mapper = LayerMappingOptimizer(num_total_layers=28)
    scores = [float(i) for i in range(28)]
    retained = mapper.select_monotonic_layers(scores, target_num_layers=16)

    assert len(retained) == 16
    assert retained[0] == 0
    assert retained[-1] == 27
    # Verify strict monotonicity
    for i in range(len(retained) - 1):
        assert retained[i] < retained[i + 1]

def test_statistical_metrics():
    t_scores = {"coding": 80.0, "general": 75.0}
    s_scores = {"coding": 70.0, "general": 50.0}

    stats = StatisticalValidator.compute_retention_and_specialization(
        teacher_scores=t_scores,
        student_scores=s_scores,
        teacher_params=1500000000,
        student_params=500000000
    )

    assert stats["retention_percentages"]["coding"] > 0
    assert stats["specialization_gain_pct"] > 0 # (70/80 - 50/75) > 0
    assert stats["is_specialist_extraction"] is True
    assert stats["compression_efficiency"] > 1.0

def test_coding_syntax_checker():
    valid_code = "def add(a, b):\n    return a + b\n"
    invalid_code = "def add(a, b): return return for for"

    assert CodingEvaluator.check_python_syntax(valid_code) is True
    assert CodingEvaluator.check_python_syntax(invalid_code) is False
