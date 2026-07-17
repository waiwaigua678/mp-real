"""Semantic Baseline differences grouped by experimental variable."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from mp_real.evaluation.baseline.models import Baseline, json_copy


@dataclasses.dataclass(frozen=True)
class BaselineDiffItem:
    path: str
    category: str
    before: Any
    after: Any
    requires_derived_baseline: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BaselineDiff:
    baseline_a: str
    baseline_b: str
    items: tuple[BaselineDiffItem, ...]

    @property
    def compatible_for_run(self) -> bool:
        return not any(item.requires_derived_baseline for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_a": self.baseline_a,
            "baseline_b": self.baseline_b,
            "compatible_for_run": self.compatible_for_run,
            "items": [item.to_dict() for item in self.items],
        }


_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("action_spec", "robot"),
    ("state_schema", "robot"),
    ("camera_roles", "camera"),
    ("camera_config", "camera"),
    ("robot_config", "robot"),
    ("safety_config", "safety"),
    ("runtime_config", "runtime"),
    ("rtc_config", "RTC"),
    ("warmup_config", "runtime"),
    ("evaluation_protocol", "evaluation protocol"),
    ("initial_position_protocol", "initial state"),
    ("source", "initial state"),
    ("dataset_format", "recording"),
    ("policy_server_url", "model"),
    ("policy_label", "model"),
    ("policy_metadata", "model"),
    ("checkpoint_hash", "model"),
    ("prompt", "evaluation protocol"),
    ("task_name", "evaluation protocol"),
    ("robot_name", "robot"),
)
_DISPLAY_ONLY = frozenset({"name", "operator", "tags", "notes", "created_at", "parent_baseline_id", "derived_reason"})


def diff_baselines(before: Baseline, after: Baseline) -> BaselineDiff:
    left = before.without_references()
    right = after.without_references()
    values: list[BaselineDiffItem] = []
    for key in sorted(set(left) | set(right)):
        if key in {"baseline_id", "schema_version"}:
            continue
        _diff_value(key, left.get(key), right.get(key), values)
    return BaselineDiff(before.baseline_id, after.baseline_id, tuple(values))


def _diff_value(path: str, before: Any, after: Any, values: list[BaselineDiffItem]) -> None:
    if before == after:
        return
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after), key=str):
            _diff_value(f"{path}.{key}", before.get(key), after.get(key), values)
        return
    category = _category_for(path)
    values.append(
        BaselineDiffItem(
            path=path,
            category=category,
            before=json_copy(before),
            after=json_copy(after),
            requires_derived_baseline=path.split(".", 1)[0] not in _DISPLAY_ONLY,
        )
    )


def _category_for(path: str) -> str:
    root = path.split(".", 1)[0]
    for field, category in _CATEGORIES:
        if root == field:
            return category
    return "annotations" if root in _DISPLAY_ONLY else "recording"
