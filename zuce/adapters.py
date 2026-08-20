"""Architecture adapter contracts, built-ins, and compatibility inspection."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import torch.nn as nn

from .types import CompatibilityReport


@dataclass(slots=True)
class MLPParts:
    gate: nn.Module
    up: nn.Module
    down: nn.Module


class ModelAdapter(ABC):
    """Public extension point for architecture-aware profiling and surgery."""

    name = "base"
    model_types: tuple[str, ...] = ()
    supports_surgery = False

    def matches(self, model: nn.Module) -> bool:
        return getattr(model.config, "model_type", "") in self.model_types

    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        """Return decoder layers in execution order."""

    @abstractmethod
    def get_profile_module(self, layer: nn.Module) -> nn.Module:
        """Return the module whose input is the MLP intermediate activation."""

    def intermediate_size(self, model: nn.Module) -> int | None:
        value = getattr(model.config, "intermediate_size", None)
        return int(value) if value is not None else None

    def patch_config(self, config: Any, retained_width: int) -> Any:
        raise NotImplementedError

    def mlp_parts(self, layer: nn.Module) -> MLPParts:
        raise NotImplementedError


class GenericInspectionAdapter(ModelAdapter):
    """Read-only structural inspection for unknown CausalLM architectures."""

    name = "generic-inspection"

    def matches(self, model: nn.Module) -> bool:
        return True

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        candidates = (
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        )
        for parent_name, layers_name in candidates:
            parent = getattr(model, parent_name, None)
            layers = getattr(parent, layers_name, None) if parent is not None else None
            if isinstance(layers, nn.ModuleList):
                return layers
        layers = getattr(model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        raise AttributeError("Could not locate a decoder ModuleList")

    def get_profile_module(self, layer: nn.Module) -> nn.Module:
        mlp = getattr(layer, "mlp", None)
        for name in ("down_proj", "c_proj", "dense_4h_to_h"):
            module = getattr(mlp, name, None) if mlp is not None else None
            if isinstance(module, nn.Module):
                return module
        raise AttributeError("Could not locate a safe MLP reduction module")

    def intermediate_size(self, model: nn.Module) -> int | None:
        configured = super().intermediate_size(model)
        if configured is not None:
            return configured
        try:
            module = self.get_profile_module(self.get_layers(model)[0])
        except (AttributeError, IndexError):
            return None
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            return None
        return int(max(weight.shape))


class GatedMLPAdapter(ModelAdapter):
    """Physical slicing for Qwen2, Llama, Mistral, and Gemma model types."""

    name = "gated-mlp-v1"
    model_types = ("qwen2", "llama", "mistral", "gemma")
    supports_surgery = True

    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None) if backbone is not None else None
        if not isinstance(layers, nn.ModuleList):
            raise AttributeError("Expected model.model.layers for gated-MLP adapter")
        return layers

    def mlp_parts(self, layer: nn.Module) -> MLPParts:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise AttributeError("Decoder layer has no mlp module")
        parts = MLPParts(
            gate=getattr(mlp, "gate_proj", None),
            up=getattr(mlp, "up_proj", None),
            down=getattr(mlp, "down_proj", None),
        )
        if not all(isinstance(part, nn.Linear) for part in (parts.gate, parts.up, parts.down)):
            raise AttributeError("Expected Linear gate_proj/up_proj/down_proj modules")
        return parts

    def get_profile_module(self, layer: nn.Module) -> nn.Module:
        return self.mlp_parts(layer).down

    def patch_config(self, config: Any, retained_width: int) -> Any:
        result = copy.deepcopy(config)
        result.intermediate_size = int(retained_width)
        return result


_ADAPTERS: list[ModelAdapter] = [GatedMLPAdapter()]
_GENERIC = GenericInspectionAdapter()


def register_adapter(adapter: ModelAdapter, *, prepend: bool = True) -> None:
    if prepend:
        _ADAPTERS.insert(0, adapter)
    else:
        _ADAPTERS.append(adapter)


def registered_adapters() -> Iterable[ModelAdapter]:
    return tuple(_ADAPTERS)


def adapter_for(model: nn.Module) -> ModelAdapter:
    for adapter in _ADAPTERS:
        if adapter.matches(model):
            return adapter
    return _GENERIC


def inspect_compatibility(model: nn.Module) -> CompatibilityReport:
    adapter = adapter_for(model)
    reasons: list[str] = []
    try:
        layers = adapter.get_layers(model)
    except AttributeError as exc:
        layers = []
        reasons.append(str(exc))

    can_profile = False
    if layers:
        try:
            adapter.get_profile_module(layers[0])
            can_profile = True
        except AttributeError as exc:
            reasons.append(str(exc))

    can_surgery = bool(adapter.supports_surgery and can_profile)
    if not can_surgery:
        reasons.append(
            f"model_type '{getattr(model.config, 'model_type', 'unknown')}' has no registered physical-surgery adapter"
        )

    return CompatibilityReport(
        model_type=str(getattr(model.config, "model_type", "unknown")),
        architecture=model.__class__.__name__,
        adapter=adapter.name,
        can_inspect=True,
        can_profile=can_profile,
        can_surgery=can_surgery,
        reasons=reasons,
        num_parameters=sum(parameter.numel() for parameter in model.parameters()),
        num_layers=len(layers) if layers else None,
        intermediate_size=adapter.intermediate_size(model),
    )


__all__ = [
    "ModelAdapter",
    "GenericInspectionAdapter",
    "GatedMLPAdapter",
    "MLPParts",
    "adapter_for",
    "inspect_compatibility",
    "register_adapter",
    "registered_adapters",
]
