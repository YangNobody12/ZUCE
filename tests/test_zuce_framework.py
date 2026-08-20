from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import (
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from zuce import (
    BudgetInfeasibleError,
    CapabilitySpec,
    ParameterBudget,
    QualityGateError,
    UnsupportedArchitectureError,
    ZUCE,
)
from zuce.adapters import GatedMLPAdapter, adapter_for, inspect_compatibility
from zuce.budget import allocate_parameter_budget
from zuce.datasets import load_texts
from zuce.profiling import runtime_masks, select_neurons
from zuce.surgery import build_extracted_model
from zuce.verification import state_dict_fingerprint, verify_exact_subset


class TinyTokenizer:
    vocab_size = 64
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        ids = [2 + (ord(char) % (self.vocab_size - 2)) for char in text][:max_length]
        if len(ids) < 2:
            ids = [2, 3]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)

    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "TinyTokenizer"}), encoding="utf-8"
        )


def tiny_models():
    common = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=1,
    )
    return [
        Qwen2ForCausalLM(Qwen2Config(**common)),
        Qwen3ForCausalLM(Qwen3Config(**common)),
        LlamaForCausalLM(LlamaConfig(**common)),
        MistralForCausalLM(MistralConfig(**common)),
        GemmaForCausalLM(GemmaConfig(**common, head_dim=4)),
    ]


@pytest.mark.parametrize("model", tiny_models(), ids=["qwen2", "qwen3", "llama", "mistral", "gemma"])
def test_gated_adapter_slices_exact_subsets(model):
    model.eval()
    adapter = adapter_for(model)
    assert isinstance(adapter, GatedMLPAdapter)
    compatibility = inspect_compatibility(model)
    assert compatibility.can_profile and compatibility.can_surgery
    before = state_dict_fingerprint(model)
    scores = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    selected = select_neurons(scores, 16)
    student = build_extracted_model(model, adapter, selected, 16)
    proof = verify_exact_subset(model, student, adapter, selected)
    assert proof["verified"]
    assert state_dict_fingerprint(model) == before
    assert student.config.num_hidden_layers == model.config.num_hidden_layers
    assert student.config.vocab_size == model.config.vocab_size
    assert student.config.intermediate_size == 16
    input_ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad(), runtime_masks(model, adapter, selected):
        masked_logits = model(input_ids=input_ids).logits
    output = student(input_ids=input_ids)
    assert output.logits.shape == (1, 3, 64)
    torch.testing.assert_close(output.logits, masked_logits, rtol=1e-4, atol=1e-5)


def test_unknown_model_inspects_but_cannot_surgery():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_inner=32, n_layer=2, n_head=4))
    report = ZUCE.inspect(model, tokenizer=TinyTokenizer())
    assert report.can_inspect
    assert report.can_profile
    assert not report.can_surgery


def test_unknown_model_profiles_then_fails_safely(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_inner=32, n_layer=2, n_head=4))
    output_dir = tmp_path / "unsupported"
    with pytest.raises(UnsupportedArchitectureError) as raised:
        ZUCE.extract(
            model=model,
            tokenizer=TinyTokenizer(),
            capability=CapabilitySpec(target=["profile this unknown architecture"]),
            budget=ParameterBudget(10_000_000),
            output_dir=output_dir,
            max_samples=1,
            max_length=24,
        )
    assert "layer_sensitivity" in raised.value.details
    assert not output_dir.exists()


def test_budget_below_full_depth_floor_is_structured_error():
    model = tiny_models()[0]
    adapter = adapter_for(model)
    full = allocate_parameter_budget(model, adapter, sum(p.numel() for p in model.parameters()))
    with pytest.raises(BudgetInfeasibleError) as raised:
        allocate_parameter_budget(model, adapter, full.minimum_parameters - 1)
    assert raised.value.details["minimum_parameters"] == full.minimum_parameters


def test_dataset_jsonl_chat_iterable_and_validation(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '\n'.join([
            json.dumps({"text": "plain text"}),
            json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
        ]),
        encoding="utf-8",
    )
    assert load_texts(path, TinyTokenizer()) == ["plain text", "user: hello"]
    assert load_texts(["one", {"text": "two"}]) == ["one", "two"]
    with pytest.raises(Exception):
        load_texts([{"wrong": "field"}])


def test_end_to_end_api_budget_manifest_and_verification(tmp_path):
    torch.manual_seed(7)
    model = tiny_models()[0].eval()
    tokenizer = TinyTokenizer()
    adapter = adapter_for(model)
    full = allocate_parameter_budget(model, adapter, sum(p.numel() for p in model.parameters()))
    target_width = 24
    budget = full.constant_parameters + full.parameters_per_width * target_width
    output_dir = tmp_path / "artifact"
    before = state_dict_fingerprint(model)
    result = ZUCE.extract(
        model=model,
        tokenizer=tokenizer,
        capability=CapabilitySpec(
            name="coding",
            target=["def add(a, b): return a + b"],
            contrasts={"math": ["Solve two plus two step by step"]},
        ),
        budget=ParameterBudget(budget),
        output_dir=output_dir,
        max_samples=1,
        max_length=32,
        min_retention=0.0,
        seed=7,
    )
    assert result.extracted_parameters <= budget
    assert result.retained_width == target_width
    assert state_dict_fingerprint(model) == before
    assert (output_dir / "model.safetensors").is_file()
    manifest = json.loads((output_dir / "zuce_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preserved"] == ["decoder_depth", "attention", "embeddings", "vocabulary", "tokenizer"]
    assert ZUCE.verify(output_dir)["zero_update_verified"]


def test_quality_failure_does_not_publish_artifact(tmp_path, monkeypatch):
    import zuce.pipeline as pipeline

    model = tiny_models()[0].eval()
    tokenizer = TinyTokenizer()
    adapter = adapter_for(model)
    allocation = allocate_parameter_budget(model, adapter, sum(p.numel() for p in model.parameters()))
    budget = allocation.minimum_parameters
    monkeypatch.setattr(pipeline, "retention_from_nll", lambda *_args: 0.0)
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(QualityGateError) as raised:
        ZUCE.extract(
            model=model,
            tokenizer=tokenizer,
            capability=CapabilitySpec(target=["target capability sample"]),
            budget=ParameterBudget(budget),
            output_dir=output_dir,
            max_samples=1,
            max_length=24,
            min_retention=0.6,
        )
    assert not output_dir.exists()
    assert Path(raised.value.details["diagnostic_path"]).is_dir()
