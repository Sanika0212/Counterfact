"""Canonical JSON conversion and content fingerprints."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert common Python values into plain JSON types (deterministically)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((to_jsonable(v) for v in value), key=canonical)
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump(mode="json"))
    return {"__repr__": repr(value)}


def canonical(value: Any) -> str:
    """Stable JSON text: sorted keys, no whitespace, non-ASCII kept."""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    """Short SHA-256 of the canonical JSON form."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:20]
