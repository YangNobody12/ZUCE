"""ZUCE — Zero-Update Capability Extraction."""

from .adapters import ModelAdapter
from .api import ZUCE
from .errors import (
    BudgetInfeasibleError,
    DatasetValidationError,
    QualityGateError,
    UnsupportedArchitectureError,
    VerificationError,
    ZUCEError,
)
from .types import CapabilitySpec, CompatibilityReport, ExtractionResult, ParameterBudget, ZUCEConfig

__all__ = [
    "ZUCE",
    "ZUCEConfig",
    "CapabilitySpec",
    "ParameterBudget",
    "CompatibilityReport",
    "ExtractionResult",
    "ModelAdapter",
    "ZUCEError",
    "UnsupportedArchitectureError",
    "BudgetInfeasibleError",
    "DatasetValidationError",
    "QualityGateError",
    "VerificationError",
]

__version__ = "0.1.0"
