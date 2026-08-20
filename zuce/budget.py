"""Exact parameter accounting for uniform MLP-width extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .adapters import ModelAdapter
from .errors import BudgetInfeasibleError


@dataclass(slots=True)
class BudgetAllocation:
    retained_width: int
    original_width: int
    teacher_parameters: int
    extracted_parameters: int
    constant_parameters: int
    parameters_per_width: int
    minimum_parameters: int


def allocate_parameter_budget(
    model: nn.Module,
    adapter: ModelAdapter,
    max_parameters: int,
) -> BudgetAllocation:
    original_width = adapter.intermediate_size(model)
    if original_width is None or original_width <= 0:
        raise BudgetInfeasibleError("Could not determine the model intermediate size")
    variable_parameters = 0
    for layer in adapter.get_layers(model):
        parts = adapter.mlp_parts(layer)
        variable_parameters += parts.gate.weight.numel()
        variable_parameters += parts.up.weight.numel()
        variable_parameters += parts.down.weight.numel()
        if getattr(parts.gate, "bias", None) is not None:
            variable_parameters += parts.gate.bias.numel()
        if getattr(parts.up, "bias", None) is not None:
            variable_parameters += parts.up.bias.numel()
    teacher_parameters = sum(parameter.numel() for parameter in model.parameters())
    if variable_parameters % original_width:
        raise BudgetInfeasibleError("MLP parameter layout is not uniformly sliceable")
    parameters_per_width = variable_parameters // original_width
    constant_parameters = teacher_parameters - variable_parameters
    minimum_parameters = constant_parameters + parameters_per_width
    if max_parameters < minimum_parameters:
        raise BudgetInfeasibleError(
            "Parameter budget is below the minimum full-depth architecture",
            max_parameters=max_parameters,
            minimum_parameters=minimum_parameters,
        )
    retained_width = min(original_width, (max_parameters - constant_parameters) // parameters_per_width)
    extracted_parameters = constant_parameters + retained_width * parameters_per_width
    return BudgetAllocation(
        retained_width=int(retained_width),
        original_width=int(original_width),
        teacher_parameters=teacher_parameters,
        extracted_parameters=int(extracted_parameters),
        constant_parameters=constant_parameters,
        parameters_per_width=parameters_per_width,
        minimum_parameters=minimum_parameters,
    )

