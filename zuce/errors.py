"""Structured errors raised by ZUCE."""

from __future__ import annotations


class ZUCEError(RuntimeError):
    """Base error for all framework failures."""

    code = "zuce_error"

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.details = details


class UnsupportedArchitectureError(ZUCEError):
    code = "unsupported_architecture"


class BudgetInfeasibleError(ZUCEError):
    code = "budget_infeasible"


class DatasetValidationError(ZUCEError):
    code = "dataset_validation"


class QualityGateError(ZUCEError):
    code = "quality_gate_failed"


class VerificationError(ZUCEError):
    code = "verification_failed"

