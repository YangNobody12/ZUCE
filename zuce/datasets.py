"""Dataset normalization for JSONL, HF Dataset objects, and Python iterables."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import DatasetValidationError
from .presets import PRESETS


def _render_messages(messages: Any, tokenizer: Any | None) -> str:
    if not isinstance(messages, list) or not messages:
        raise DatasetValidationError("'messages' must be a non-empty list")
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or "content" not in message:
            raise DatasetValidationError("Each chat message must contain a 'content' field")
        rendered.append(f"{message.get('role', 'user')}: {message['content']}")
    return "\n".join(rendered)


def _record_to_text(record: Any, tokenizer: Any | None, text_field: str = "text") -> str:
    if isinstance(record, str):
        text = record
    elif isinstance(record, Mapping):
        if text_field in record:
            text = record[text_field]
        elif "messages" in record:
            text = _render_messages(record["messages"], tokenizer)
        else:
            raise DatasetValidationError(
                "Dataset record must be a string or contain 'text' or 'messages'",
                record_keys=list(record.keys()),
            )
    else:
        raise DatasetValidationError("Unsupported dataset record type", record_type=type(record).__name__)
    if not isinstance(text, str) or not text.strip():
        raise DatasetValidationError("Dataset text must be a non-empty string")
    return text.strip()


def _load_hf_descriptor(descriptor: Mapping[str, Any]) -> Iterable[Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetValidationError(
            "Hugging Face dataset descriptors require the 'datasets' extra: pip install zuce[datasets]"
        ) from exc
    name = descriptor.get("hf_dataset")
    if not name:
        raise DatasetValidationError("HF dataset descriptor requires 'hf_dataset'")
    return load_dataset(
        name,
        descriptor.get("config"),
        split=descriptor.get("split", "train"),
        streaming=bool(descriptor.get("streaming", False)),
    )


def load_texts(source: Any, tokenizer: Any | None = None, limit: int | None = None) -> list[str]:
    """Normalize a supported dataset source into a bounded list of text samples."""

    text_field = "text"
    if isinstance(source, (str, Path)):
        source_text = str(source)
        preset_name = source_text.removeprefix("preset:")
        if preset_name in PRESETS:
            records: Iterable[Any] = PRESETS[preset_name]
        else:
            path = Path(source)
            if not path.is_file():
                raise DatasetValidationError(
                    "Dataset path does not exist; use preset:coding, preset:math, or preset:translation for a preset",
                    source=source_text,
                )
            if path.suffix.lower() != ".jsonl":
                raise DatasetValidationError("Only JSONL files are supported by path", path=str(path))
            try:
                records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetValidationError("Could not read JSONL dataset", path=str(path)) from exc
    elif isinstance(source, Mapping) and "hf_dataset" in source:
        text_field = str(source.get("text_field", "text"))
        records = _load_hf_descriptor(source)
    elif isinstance(source, Mapping):
        records = [source]
    elif isinstance(source, Iterable):
        records = source
    else:
        raise DatasetValidationError("Unsupported dataset source", source_type=type(source).__name__)

    texts: list[str] = []
    for record in records:
        texts.append(_record_to_text(record, tokenizer, text_field=text_field))
        if limit is not None and len(texts) >= limit:
            break
    if not texts:
        raise DatasetValidationError("Dataset contains no usable examples")
    return texts

