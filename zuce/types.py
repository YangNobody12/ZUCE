"""Public dataclasses used by the ZUCE API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


DatasetSource = Any


@dataclass(slots=True)
class CapabilitySpec:
    """Target examples and optional named contrast datasets."""

    target: DatasetSource
    contrasts: Mapping[str, DatasetSource] = field(default_factory=dict)
    name: str = "custom"


@dataclass(slots=True)
class ParameterBudget:
    """Hard upper bound for the number of exported parameters."""

    max_parameters: int

    def __post_init__(self) -> None:
        if self.max_parameters <= 0:
            raise ValueError("max_parameters must be a positive integer")


@dataclass(slots=True)
class ZUCEConfig:
    """Complete, serializable extraction configuration."""

    model: Any
    capability: CapabilitySpec
    budget: ParameterBudget
    output_dir: str | Path
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    max_samples: int = 32
    max_length: int = 512
    min_retention: float = 0.60
    seed: int = 42
    diagnostic_search_steps: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_retention <= 1.0:
            raise ValueError("min_retention must be between 0 and 1")
        if self.max_samples <= 0 or self.max_length < 2:
            raise ValueError("max_samples must be positive and max_length must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        def source_description(source: Any) -> Any:
            if isinstance(source, Path):
                return str(source)
            if isinstance(source, (str, int, float, bool, dict, type(None))):
                return source
            if isinstance(source, list):
                return {"type": "list", "records": len(source)}
            return {"type": type(source).__name__}

        model_name = (
            str(self.model)
            if isinstance(self.model, (str, Path))
            else getattr(self.model.config, "_name_or_path", None) or self.model.__class__.__name__
        )
        return {
            "model": model_name,
            "capability": {
                "name": self.capability.name,
                "target": source_description(self.capability.target),
                "contrasts": {
                    key: source_description(value) for key, value in self.capability.contrasts.items()
                },
            },
            "budget": {"max_parameters": self.budget.max_parameters},
            "output_dir": str(self.output_dir),
            "device": self.device,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "max_samples": self.max_samples,
            "max_length": self.max_length,
            "min_retention": self.min_retention,
            "seed": self.seed,
            "diagnostic_search_steps": self.diagnostic_search_steps,
        }


@dataclass(slots=True)
class CompatibilityReport:
    model_type: str
    architecture: str
    adapter: str | None
    can_inspect: bool
    can_profile: bool
    can_surgery: bool
    reasons: list[str] = field(default_factory=list)
    num_parameters: int | None = None
    num_layers: int | None = None
    intermediate_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractionResult:
    output_dir: str
    manifest_path: str
    proof_path: str
    evaluation_path: str
    teacher_parameters: int
    extracted_parameters: int
    retained_width: int
    capability_retention: float
    compatibility: CompatibilityReport

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compatibility"] = self.compatibility.to_dict()
        return data
