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
from .types import (
    AMPQConfig,
    AMPQResult,
    CapabilitySpec,
    CompatibilityReport,
    ExamResult,
    ExtractionResult,
    FusionConfig,
    FusionResult,
    ParameterBudget,
    ZUCEConfig,
)

__all__ = [
    "ZUCE",
    "ZUCEConfig",
    "CapabilitySpec",
    "ParameterBudget",
    "CompatibilityReport",
    "ExtractionResult",
    "AMPQConfig",
    "AMPQResult",
    "FusionConfig",
    "FusionResult",
    "ExamResult",
    "ModelAdapter",
    "ZUCEError",
    "UnsupportedArchitectureError",
    "BudgetInfeasibleError",
    "DatasetValidationError",
    "QualityGateError",
    "VerificationError",
]

__version__ = "4.0.0"
