"""Teacher fingerprints, subset proofs, and standalone artifact verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .adapters import ModelAdapter
from .errors import VerificationError


def _update_tensor_hash(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def state_dict_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        _update_tensor_hash(digest, tensor)
    return digest.hexdigest()


def verify_exact_subset(
    teacher: nn.Module,
    student: nn.Module,
    adapter: ModelAdapter,
    selected: Mapping[int, list[int]],
) -> dict[str, int | bool]:
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    checked = 0
    for name, student_tensor in student_state.items():
        teacher_tensor = teacher_state.get(name)
        if teacher_tensor is None:
            raise VerificationError("Student contains a tensor absent from teacher", tensor=name)
        if student_tensor.shape == teacher_tensor.shape:
            if not torch.equal(student_tensor.detach().cpu(), teacher_tensor.detach().cpu()):
                raise VerificationError("Unchanged tensor differs from teacher", tensor=name)
            checked += 1

    teacher_layers = adapter.get_layers(teacher)
    student_layers = adapter.get_layers(student)
    sliced = 0
    for layer_index, indices in selected.items():
        index = torch.tensor(indices, dtype=torch.long)
        source = adapter.mlp_parts(teacher_layers[layer_index])
        target = adapter.mlp_parts(student_layers[layer_index])
        comparisons = (
            (target.gate.weight, source.gate.weight.index_select(0, index.to(source.gate.weight.device))),
            (target.up.weight, source.up.weight.index_select(0, index.to(source.up.weight.device))),
            (target.down.weight, source.down.weight.index_select(1, index.to(source.down.weight.device))),
        )
        for actual, expected in comparisons:
            if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
                raise VerificationError("Sliced MLP tensor is not an exact teacher subset", layer=layer_index)
            sliced += 1
        for target_module, source_module in ((target.gate, source.gate), (target.up, source.up)):
            if getattr(source_module, "bias", None) is not None:
                expected_bias = source_module.bias.index_select(0, index.to(source_module.bias.device))
                if not torch.equal(target_module.bias.detach().cpu(), expected_bias.detach().cpu()):
                    raise VerificationError("Sliced MLP bias is not an exact teacher subset", layer=layer_index)
                sliced += 1
    return {"verified": True, "unchanged_tensors_checked": checked, "sliced_tensors_checked": sliced}


def verify_artifact_directory(output_dir: str | Path) -> dict[str, Any]:
    """Check required reports and saved-model integrity without reloading the teacher."""

    from transformers import AutoModelForCausalLM

    path = Path(output_dir)
    required = ["zuce_manifest.json", "zero_update_proof.json", "evaluation_report.json", "config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise VerificationError("Artifact directory is incomplete", missing=missing)
    manifest = json.loads((path / "zuce_manifest.json").read_text(encoding="utf-8"))
    proof = json.loads((path / "zero_update_proof.json").read_text(encoding="utf-8"))
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=False)
    actual = state_dict_fingerprint(model)
    expected = proof.get("artifact_fingerprint")
    if actual != expected:
        raise VerificationError("Artifact fingerprint mismatch", expected=expected, actual=actual)
    return {
        "verified": True,
        "schema_version": manifest.get("schema_version"),
        "artifact_fingerprint": actual,
        "zero_update_verified": bool(proof.get("teacher_unchanged") and proof.get("subset_verified")),
    }

