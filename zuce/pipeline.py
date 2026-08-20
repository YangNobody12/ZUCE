"""End-to-end transactional ZUCE extraction pipeline."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .adapters import adapter_for, inspect_compatibility
from .budget import BudgetAllocation, allocate_parameter_budget
from .datasets import load_texts
from .errors import QualityGateError, UnsupportedArchitectureError, VerificationError
from .profiling import (
    evaluate_nll,
    profile_capability,
    retention_from_nll,
    runtime_masks,
    select_neurons,
)
from .surgery import build_extracted_model
from .types import ExtractionResult, ZUCEConfig
from .verification import state_dict_fingerprint, verify_exact_subset


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_model_and_tokenizer(
    model_source: Any,
    *,
    tokenizer: Any | None,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(model_source, nn.Module):
        model = model_source
        if tokenizer is None:
            source_name = getattr(model.config, "_name_or_path", None)
            if not source_name:
                raise ValueError("A tokenizer is required when passing an in-memory model")
            tokenizer = AutoTokenizer.from_pretrained(source_name, trust_remote_code=trust_remote_code)
    else:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_source, trust_remote_code=trust_remote_code)
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        resolved_dtype = _dtype_from_name(dtype)
        if resolved_dtype != "auto":
            # torch_dtype remains supported across the full Transformers >=4.45 range.
            load_kwargs["torch_dtype"] = resolved_dtype
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device
    current_devices = {parameter.device.type for parameter in model.parameters()}
    if current_devices != {target_device}:
        model.to(target_device)
    model.eval()
    return model, tokenizer


def _estimate_minimum_passing_width(
    model: nn.Module,
    tokenizer: Any,
    adapter: Any,
    scores: torch.Tensor,
    texts: list[str],
    max_length: int,
    teacher_nll: float,
    current_width: int,
    original_width: int,
    minimum_retention: float,
    steps: int,
    allocation: BudgetAllocation,
) -> dict[str, Any]:
    low = current_width + 1
    high = original_width
    tested: list[dict[str, float | int]] = []
    best = original_width
    remaining = max(1, steps)
    while low <= high and remaining > 0:
        width = (low + high) // 2
        selected = select_neurons(scores, width)
        with runtime_masks(model, adapter, selected):
            metrics = evaluate_nll(model, tokenizer, texts, max_length)
        retention = retention_from_nll(teacher_nll, float(metrics["nll"]))
        tested.append({"width": width, "retention": retention})
        if retention >= minimum_retention:
            best = width
            high = width - 1
        else:
            low = width + 1
        remaining -= 1
    return {
        "estimated_minimum_passing_width": best,
        "estimated_minimum_passing_parameters": allocation.constant_parameters
        + best * allocation.parameters_per_width,
        "tested": tested,
    }


def _diagnostic_directory(output_dir: Path) -> Path:
    base = output_dir.with_name(output_dir.name + ".diagnostics")
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for index in range(1, 10_000):
        candidate = output_dir.with_name(output_dir.name + f".diagnostics-{index}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("Could not allocate a diagnostic directory")


def run_extraction(config: ZUCEConfig, tokenizer: Any | None = None) -> ExtractionResult:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        config.model,
        tokenizer=tokenizer,
        device=config.device,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
    )
    compatibility = inspect_compatibility(model)
    adapter = adapter_for(model)
    if not compatibility.can_profile:
        raise UnsupportedArchitectureError(
            "The model can be inspected but exposes no safe profiling endpoint",
            compatibility=compatibility.to_dict(),
        )

    target_texts = load_texts(config.capability.target, tokenizer, config.max_samples)
    contrast_texts = {
        name: load_texts(source, tokenizer, config.max_samples)
        for name, source in config.capability.contrasts.items()
    }
    teacher_before = state_dict_fingerprint(model)
    profile = profile_capability(
        model,
        tokenizer,
        adapter,
        target_texts,
        contrast_texts,
        config.max_length,
    )
    teacher_after_profile = state_dict_fingerprint(model)
    if teacher_after_profile != teacher_before:
        raise VerificationError("Teacher parameters changed during capability profiling")
    if not compatibility.can_surgery:
        raise UnsupportedArchitectureError(
            "Profiling completed, but physical extraction is not registered for this architecture",
            compatibility=compatibility.to_dict(),
            layer_sensitivity=profile.layer_sensitivity,
        )

    allocation = allocate_parameter_budget(model, adapter, config.budget.max_parameters)
    selected = select_neurons(profile.scores, allocation.retained_width)
    teacher_metrics = evaluate_nll(model, tokenizer, target_texts, config.max_length)
    with runtime_masks(model, adapter, selected):
        masked_metrics = evaluate_nll(model, tokenizer, target_texts, config.max_length)
    masked_retention = retention_from_nll(float(teacher_metrics["nll"]), float(masked_metrics["nll"]))

    if masked_retention < config.min_retention:
        estimate = _estimate_minimum_passing_width(
            model,
            tokenizer,
            adapter,
            profile.scores,
            target_texts,
            config.max_length,
            float(teacher_metrics["nll"]),
            allocation.retained_width,
            allocation.original_width,
            config.min_retention,
            config.diagnostic_search_steps,
            allocation,
        )
        diagnostic_dir = _diagnostic_directory(output_dir)
        diagnostic = {
            "status": "quality_gate_failed",
            "threshold": config.min_retention,
            "candidate_retention": masked_retention,
            "candidate_width": allocation.retained_width,
            "candidate_parameters": allocation.extracted_parameters,
            **estimate,
        }
        _json_dump(diagnostic_dir / "evaluation_report.json", diagnostic)
        if state_dict_fingerprint(model) != teacher_before:
            raise VerificationError("Teacher parameters changed during diagnostic masking")
        raise QualityGateError(
            "The budget-constrained candidate did not pass the capability retention gate",
            diagnostic_path=str(diagnostic_dir),
            **diagnostic,
        )

    student = build_extracted_model(model, adapter, selected, allocation.retained_width)
    subset_proof = verify_exact_subset(model, student, adapter, selected)
    student_metrics = evaluate_nll(student, tokenizer, target_texts, config.max_length)
    retention = retention_from_nll(float(teacher_metrics["nll"]), float(student_metrics["nll"]))
    if retention < config.min_retention:
        raise QualityGateError(
            "Physical model did not match the validated runtime mask",
            candidate_retention=retention,
            threshold=config.min_retention,
        )

    teacher_after = state_dict_fingerprint(model)
    if teacher_after != teacher_before:
        raise VerificationError("Teacher parameters changed during extraction")
    artifact_fingerprint = state_dict_fingerprint(student)
    actual_parameters = sum(parameter.numel() for parameter in student.parameters())
    if actual_parameters > config.budget.max_parameters or actual_parameters != allocation.extracted_parameters:
        raise VerificationError(
            "Exported parameter count violates budget accounting",
            expected=allocation.extracted_parameters,
            actual=actual_parameters,
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        student.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        manifest = {
            "schema_version": "zuce-0.1",
            "framework": "ZUCE — Zero-Update Capability Extraction",
            "status": "verified",
            "config": config.to_dict(),
            "compatibility": compatibility.to_dict(),
            "adapter": adapter.name,
            "teacher_parameters": allocation.teacher_parameters,
            "extracted_parameters": actual_parameters,
            "parameter_reduction": 1.0 - actual_parameters / allocation.teacher_parameters,
            "original_intermediate_size": allocation.original_width,
            "retained_intermediate_size": allocation.retained_width,
            "retained_neuron_indices": {str(key): value for key, value in selected.items()},
            "preserved": ["decoder_depth", "attention", "embeddings", "vocabulary", "tokenizer"],
            "profile": {
                "samples": profile.samples,
                "layer_sensitivity": profile.layer_sensitivity,
            },
        }
        proof = {
            "teacher_fingerprint_before": teacher_before,
            "teacher_fingerprint_after": teacher_after,
            "teacher_unchanged": teacher_before == teacher_after,
            "artifact_fingerprint": artifact_fingerprint,
            "subset_verified": subset_proof["verified"],
            **subset_proof,
        }
        evaluation = {
            "quality_gate": "passed",
            "minimum_retention": config.min_retention,
            "capability_retention": retention,
            "runtime_mask_retention": masked_retention,
            "teacher": teacher_metrics,
            "runtime_mask": masked_metrics,
            "extracted": student_metrics,
            "estimated_weight_memory_bytes": actual_parameters
            * next(student.parameters()).element_size(),
        }
        _json_dump(staging / "zuce_manifest.json", manifest)
        _json_dump(staging / "zero_update_proof.json", proof)
        _json_dump(staging / "evaluation_report.json", evaluation)
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return ExtractionResult(
        output_dir=str(output_dir),
        manifest_path=str(output_dir / "zuce_manifest.json"),
        proof_path=str(output_dir / "zero_update_proof.json"),
        evaluation_path=str(output_dir / "evaluation_report.json"),
        teacher_parameters=allocation.teacher_parameters,
        extracted_parameters=actual_parameters,
        retained_width=allocation.retained_width,
        capability_retention=retention,
        compatibility=compatibility,
    )
