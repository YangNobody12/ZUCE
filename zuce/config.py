"""YAML configuration loading for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import DatasetValidationError
from .types import CapabilitySpec, ParameterBudget, ZUCEConfig


def config_from_mapping(data: Mapping[str, Any]) -> ZUCEConfig:
    try:
        capability_data = data["capability"]
        budget_data = data["budget"]
        return ZUCEConfig(
            model=data["model"],
            capability=CapabilitySpec(
                target=capability_data["target"],
                contrasts=capability_data.get("contrasts", {}),
                name=capability_data.get("name", "custom"),
            ),
            budget=ParameterBudget(max_parameters=int(budget_data["max_parameters"])),
            output_dir=data["output_dir"],
            device=data.get("device", "auto"),
            dtype=data.get("dtype", "auto"),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            max_samples=int(data.get("max_samples", 32)),
            max_length=int(data.get("max_length", 512)),
            min_retention=float(data.get("min_retention", 0.60)),
            seed=int(data.get("seed", 42)),
            diagnostic_search_steps=int(data.get("diagnostic_search_steps", 4)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError("Invalid ZUCE configuration", error=str(exc)) from exc


def load_config(path: str | Path) -> ZUCEConfig:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetValidationError("Could not load ZUCE YAML configuration", path=str(config_path)) from exc
    if not isinstance(data, Mapping):
        raise DatasetValidationError("ZUCE YAML root must be a mapping")
    return config_from_mapping(data)

