"""Stable Python API for ZUCE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from .adapters import inspect_compatibility
from .pipeline import load_model_and_tokenizer, run_extraction
from .types import CapabilitySpec, CompatibilityReport, ExtractionResult, ParameterBudget, ZUCEConfig
from .verification import verify_artifact_directory


class ZUCE:
    """Namespace for inspection, zero-update extraction, and verification."""

    @staticmethod
    def inspect(
        model: str | Path | nn.Module,
        *,
        tokenizer: Any | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> CompatibilityReport:
        loaded, _ = load_model_and_tokenizer(
            model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        return inspect_compatibility(loaded)

    @staticmethod
    def extract(
        model: str | Path | nn.Module,
        capability: CapabilitySpec,
        budget: ParameterBudget,
        output_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        max_samples: int = 32,
        max_length: int = 512,
        min_retention: float = 0.60,
        seed: int = 42,
    ) -> ExtractionResult:
        config = ZUCEConfig(
            model=model,
            capability=capability,
            budget=budget,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_samples=max_samples,
            max_length=max_length,
            min_retention=min_retention,
            seed=seed,
        )
        return run_extraction(config, tokenizer=tokenizer)

    @staticmethod
    def extract_config(config: ZUCEConfig, *, tokenizer: Any | None = None) -> ExtractionResult:
        return run_extraction(config, tokenizer=tokenizer)

    @staticmethod
    def verify(output_dir: str | Path) -> dict[str, Any]:
        return verify_artifact_directory(output_dir)

