"""Zero-update attribution, selectivity, masking, and evaluation utilities."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from .adapters import ModelAdapter


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def tokenize_text(tokenizer: Any, text: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    allowed = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    if "input_ids" not in allowed:
        raise ValueError("Tokenizer did not return input_ids")
    return allowed


def causal_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    if input_ids.shape[1] < 2:
        raise ValueError("A calibration sample must contain at least two tokens")
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="sum",
    )
    return loss, int(shifted_labels.numel())


@dataclass(slots=True)
class ProfileResult:
    scores: torch.Tensor
    target_attribution: torch.Tensor
    contrast_attribution: dict[str, torch.Tensor]
    layer_sensitivity: list[float]
    samples: dict[str, int]


def _attribution_for_texts(
    model: nn.Module,
    tokenizer: Any,
    adapter: ModelAdapter,
    texts: list[str],
    max_length: int,
) -> torch.Tensor:
    layers = adapter.get_layers(model)
    width = adapter.intermediate_size(model)
    if width is None:
        raise ValueError("Adapter could not determine intermediate size")
    totals = torch.zeros((len(layers), width), dtype=torch.float64)
    counts = torch.zeros(len(layers), dtype=torch.int64)
    device = model_device(model)
    model.eval()

    for text in texts:
        inputs = tokenize_text(tokenizer, text, max_length, device)
        captured: list[torch.Tensor | None] = [None] * len(layers)
        handles = []
        for layer_index, layer in enumerate(layers):
            module = adapter.get_profile_module(layer)

            def capture(_module: nn.Module, args: tuple[Any, ...], _index: int = layer_index) -> None:
                if args and isinstance(args[0], torch.Tensor):
                    captured[_index] = args[0]

            handles.append(module.register_forward_pre_hook(capture))
        try:
            outputs = model(**inputs, use_cache=False)
            loss_sum, token_count = causal_nll(outputs.logits, inputs["input_ids"])
            loss = loss_sum / token_count
            active = [value for value in captured if value is not None and value.requires_grad]
            if not active:
                raise RuntimeError("No differentiable MLP activations were captured")
            gradients = torch.autograd.grad(loss, active, allow_unused=True)
            gradient_iter = iter(gradients)
            for layer_index, activation in enumerate(captured):
                if activation is None or not activation.requires_grad:
                    continue
                gradient = next(gradient_iter)
                if gradient is None:
                    continue
                value = (activation.detach() * gradient.detach()).abs().mean(dim=tuple(range(activation.ndim - 1)))
                if value.numel() != width:
                    raise RuntimeError(
                        f"Layer {layer_index} exposed {value.numel()} neurons, expected {width}"
                    )
                totals[layer_index] += value.double().cpu()
                counts[layer_index] += 1
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)

    for layer_index in range(len(layers)):
        if counts[layer_index] > 0:
            totals[layer_index] /= int(counts[layer_index])
    return totals.float()


def _row_minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = values.amin(dim=1, keepdim=True)
    maximum = values.amax(dim=1, keepdim=True)
    return (values - minimum) / (maximum - minimum + 1e-12)


def profile_capability(
    model: nn.Module,
    tokenizer: Any,
    adapter: ModelAdapter,
    target_texts: list[str],
    contrast_texts: Mapping[str, list[str]],
    max_length: int,
) -> ProfileResult:
    target = _attribution_for_texts(model, tokenizer, adapter, target_texts, max_length)
    contrasts = {
        name: _attribution_for_texts(model, tokenizer, adapter, texts, max_length)
        for name, texts in contrast_texts.items()
    }
    normalized_target = _row_minmax(target)
    if contrasts:
        stack = torch.stack(list(contrasts.values()))
        other_mean = stack.mean(dim=0)
        other_std = stack.std(dim=0, unbiased=False)
        selectivity = (target - other_mean) / (other_std + 1e-8)
        normalized_selectivity = _row_minmax(selectivity)
        scores = 0.7 * normalized_target + 0.3 * normalized_selectivity
    else:
        scores = normalized_target
    return ProfileResult(
        scores=scores,
        target_attribution=target,
        contrast_attribution=contrasts,
        layer_sensitivity=target.sum(dim=1).tolist(),
        samples={"target": len(target_texts), **{name: len(texts) for name, texts in contrast_texts.items()}},
    )


def select_neurons(scores: torch.Tensor, retained_width: int) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    for layer_index, row in enumerate(scores):
        indices = torch.argsort(row, descending=True, stable=True)[:retained_width].tolist()
        selected[layer_index] = sorted(int(index) for index in indices)
    return selected


@contextmanager
def runtime_masks(
    model: nn.Module,
    adapter: ModelAdapter,
    selected: Mapping[int, list[int]],
) -> Iterator[None]:
    handles = []
    layers = adapter.get_layers(model)
    for layer_index, layer in enumerate(layers):
        module = adapter.get_profile_module(layer)
        indices = selected[layer_index]

        def mask_input(_module: nn.Module, args: tuple[Any, ...], _indices: list[int] = indices):
            activation = args[0]
            mask = torch.zeros(activation.shape[-1], device=activation.device, dtype=activation.dtype)
            mask[_indices] = 1
            return (activation * mask, *args[1:])

        handles.append(module.register_forward_pre_hook(mask_input))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate_nll(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
) -> dict[str, float | int]:
    device = model_device(model)
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenize_text(tokenizer, text, max_length, device)
            outputs = model(**inputs, use_cache=False)
            loss, count = causal_nll(outputs.logits, inputs["input_ids"])
            total_loss += float(loss.item())
            total_tokens += count
    elapsed = time.perf_counter() - started
    nll = total_loss / max(total_tokens, 1)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "tokens": total_tokens,
        "seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
    }


def retention_from_nll(teacher_nll: float, extracted_nll: float) -> float:
    return min(1.0, math.exp(teacher_nll - extracted_nll))

