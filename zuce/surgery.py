"""Physical MLP slicing for registered zero-update adapters."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from .adapters import ModelAdapter
from .errors import VerificationError


_SLICE_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.gate_proj.bias",
    ".mlp.up_proj.weight",
    ".mlp.up_proj.bias",
    ".mlp.down_proj.weight",
)


def _first_float_dtype(model: nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def build_extracted_model(
    teacher: nn.Module,
    adapter: ModelAdapter,
    selected: Mapping[int, list[int]],
    retained_width: int,
    target_device: torch.device | str | None = None,
) -> nn.Module:
    """Instantiate the same HF architecture and copy exact source tensor subsets."""

    new_config = adapter.patch_config(teacher.config, retained_width)
    student = teacher.__class__(new_config)
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    mismatched: list[str] = []

    with torch.no_grad():
        for name, target in student_state.items():
            source = teacher_state.get(name)
            if source is None:
                raise VerificationError("Extracted architecture introduced an unknown tensor", tensor=name)
            if source.shape == target.shape:
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            else:
                mismatched.append(name)
        unexpected = [name for name in mismatched if not name.endswith(_SLICE_SUFFIXES)]
        if unexpected:
            raise VerificationError(
                "Architecture config changed tensors outside the supported MLP dimensions",
                tensors=unexpected,
            )

        teacher_layers = adapter.get_layers(teacher)
        student_layers = adapter.get_layers(student)
        if len(teacher_layers) != len(student_layers):
            raise VerificationError("ZUCE v0.1 must preserve every decoder layer")
        for layer_index, indices in selected.items():
            index = torch.tensor(indices, device=teacher_layers[layer_index].mlp.down_proj.weight.device)
            source = adapter.mlp_parts(teacher_layers[layer_index])
            target = adapter.mlp_parts(student_layers[layer_index])
            target.gate.weight.copy_(
                source.gate.weight.index_select(0, index).to(target.gate.weight.device, target.gate.weight.dtype)
            )
            target.up.weight.copy_(
                source.up.weight.index_select(0, index).to(target.up.weight.device, target.up.weight.dtype)
            )
            target.down.weight.copy_(
                source.down.weight.index_select(1, index).to(target.down.weight.device, target.down.weight.dtype)
            )
            if source.gate.bias is not None:
                target.gate.bias.copy_(
                    source.gate.bias.index_select(0, index).to(target.gate.bias.device, target.gate.bias.dtype)
                )
            if source.up.bias is not None:
                target.up.bias.copy_(
                    source.up.bias.index_select(0, index).to(target.up.bias.device, target.up.bias.dtype)
                )

    if hasattr(student, "tie_weights"):
        student.tie_weights()
    dtype = _first_float_dtype(teacher)
    if target_device is None:
        try:
            target_device = next(teacher.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    student.to(device=target_device, dtype=dtype)
    student.eval()
    return student

