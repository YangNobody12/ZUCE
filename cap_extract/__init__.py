"""Compatibility namespace for the pre-ZUCE research package.

New applications should import the strict zero-update API from :mod:`zuce`.
Legacy submodules remain available during the v0.x migration window.
"""

import warnings

warnings.warn(
    "'cap_extract' is deprecated; use 'zuce' for the supported zero-update API",
    DeprecationWarning,
    stacklevel=2,
)

from zuce import (  # noqa: E402,F401
    CapabilitySpec,
    CompatibilityReport,
    ExtractionResult,
    ModelAdapter,
    ParameterBudget,
    ZUCE,
    ZUCEConfig,
)

__version__ = "0.1.0"

__all__ = [
    "ZUCE",
    "ZUCEConfig",
    "CapabilitySpec",
    "ParameterBudget",
    "CompatibilityReport",
    "ExtractionResult",
    "ModelAdapter",
]
