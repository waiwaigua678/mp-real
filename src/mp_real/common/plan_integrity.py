"""Canonical hashing and immutability helpers for reviewed motion plans."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.runtime.models import ActionSpec, RobotState

PLAN_HASH_SCHEMA_VERSION = 1


class PlanIntegrityError(RuntimeError):
    """A reviewed motion plan no longer matches its canonical payload."""


class FrozenMapping(Mapping[str, Any]):
    """Small immutable mapping that remains friendly to ``dataclasses.asdict``."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        items = tuple(sorted(((str(key), freeze_jsonish(value)) for key, value in dict(values or {}).items())))
        self._items = items
        self._dict = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return {key: value for key, value in self._items}


def readonly_array(value: Any, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Copy an array-like value and make the returned ndarray read-only."""

    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def readonly_optional_array(value: Any, *, dtype: np.dtype | type = np.float32) -> np.ndarray | None:
    if value is None:
        return None
    return readonly_array(value, dtype=dtype)


def freeze_jsonish(value: Any) -> Any:
    """Recursively freeze JSON-like metadata without changing scalar values."""

    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_jsonish(item) for item in value)
    if isinstance(value, np.ndarray):
        return readonly_array(value, dtype=value.dtype)
    return value


def freeze_robot_state(state: RobotState) -> RobotState:
    """Return a plan-owned RobotState whose values and metadata cannot mutate."""

    return RobotState(
        readonly_array(state.values),
        float(state.timestamp_monotonic),
        int(state.timestamp_monotonic_ns),
        state.source_timestamp_ns,
        freeze_jsonish(state.health) if state.health is not None else None,
    )


def freeze_action_spec(spec: ActionSpec) -> ActionSpec:
    """Return a plan-owned ActionSpec copy.

    ActionSpec is already frozen, but its capabilities mapping is a mutable
    dict.  Plan objects keep a copy so mutating the original ActionSpec cannot
    alter a reviewed plan.
    """

    copied = ActionSpec.from_dict(spec.to_dict())
    object.__setattr__(copied, "capabilities", FrozenMapping(copied.capabilities))
    return copied


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(canonicalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonicalize(value: Any) -> Any:
    """Convert plan payloads to a stable, platform-independent JSON tree."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical float must be finite")
        return {"__float64__": float(value).hex()}
    if isinstance(value, np.ndarray):
        return _canonical_array(value)
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: canonicalize(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(key): canonicalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    return value


def _canonical_array(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
        raise ValueError("canonical ndarray must be finite")
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    elif array.dtype.byteorder not in {"<", "|", "="}:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    return {
        "__ndarray__": {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data_hex": array.tobytes(order="C").hex(),
        }
    }
