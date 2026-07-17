"""Versioned, JSON-safe contracts for reproducible evaluation baselines."""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from collections.abc import Mapping
from typing import Any

from mp_real.runtime.models import ActionSpec

BASELINE_SCHEMA_VERSION = 1


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def json_copy(value: Any) -> Any:
    """Copy only JSON-shaped values and reject accidental secret/object leakage."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_copy(item) for item in value]
    raise TypeError(f"baseline values must be JSON-compatible, got {type(value).__name__}")


def clean_runtime_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Remove credentials recursively before any durable Baseline write."""
    secret_fragments = (
        "api_key",
        "apikey",
        "access_key",
        "authorization",
        "token",
        "password",
        "secret",
    )

    def clean(value: Any, *, key: str | None = None) -> Any:
        normalized_key = key.lower().replace("-", "_") if key is not None else ""
        if any(fragment in normalized_key for fragment in secret_fragments):
            return None
        if isinstance(value, Mapping):
            return {str(item_key): clean(item, key=str(item_key)) for item_key, item in value.items()}
        if isinstance(value, tuple | list):
            return [clean(item) for item in value]
        return json_copy(value)

    return clean(snapshot)


@dataclasses.dataclass(frozen=True)
class BaselineRunReference:
    """A compact link to a real-robot evaluation; never contains episode data."""

    evaluation_id: str
    state: str
    summary: Mapping[str, Any]
    dataset_path: str | None = None
    attached_at: str = dataclasses.field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.evaluation_id:
            raise ValueError("evaluation_id cannot be empty")
        object.__setattr__(self, "summary", json_copy(self.summary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "state": self.state,
            "summary": json_copy(self.summary),
            "dataset_path": self.dataset_path,
            "attached_at": self.attached_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BaselineRunReference:
        return cls(
            evaluation_id=str(value["evaluation_id"]),
            state=str(value.get("state", "UNKNOWN")),
            summary=dict(value.get("summary", {})),
            dataset_path=_optional_text(value.get("dataset_path")),
            attached_at=str(value.get("attached_at", utc_now())),
        )


@dataclasses.dataclass(frozen=True)
class BaselineOpenLoopReference:
    """A compact link to one immutable open-loop output directory."""

    evaluation_id: str
    output_dir: str
    config_fingerprint: str | None
    status: str
    summary: Mapping[str, Any]
    attached_at: str = dataclasses.field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.evaluation_id or not self.output_dir:
            raise ValueError("open-loop evaluation_id and output_dir cannot be empty")
        object.__setattr__(self, "summary", json_copy(self.summary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "output_dir": self.output_dir,
            "config_fingerprint": self.config_fingerprint,
            "status": self.status,
            "summary": json_copy(self.summary),
            "attached_at": self.attached_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BaselineOpenLoopReference:
        return cls(
            evaluation_id=str(value["evaluation_id"]),
            output_dir=str(value["output_dir"]),
            config_fingerprint=_optional_text(value.get("config_fingerprint")),
            status=str(value.get("status", "unknown")),
            summary=dict(value.get("summary", {})),
            attached_at=str(value.get("attached_at", utc_now())),
        )


@dataclasses.dataclass(frozen=True)
class Baseline:
    """Immutable experiment inputs plus append-only compact result links."""

    baseline_id: str
    name: str
    robot_name: str
    task_name: str
    prompt: str
    policy_server_url: str
    policy_label: str
    policy_metadata: Mapping[str, Any]
    checkpoint_hash: str | None
    git_commit: str
    dataset_format: str
    action_spec: Mapping[str, Any]
    state_schema: tuple[str, ...]
    camera_roles: tuple[str, ...]
    camera_config: Mapping[str, Any]
    robot_config: Mapping[str, Any]
    safety_config: Mapping[str, Any]
    runtime_config: Mapping[str, Any]
    rtc_config: Mapping[str, Any]
    warmup_config: Mapping[str, Any]
    evaluation_protocol: Mapping[str, Any]
    initial_position_protocol: Mapping[str, Any]
    source: Mapping[str, Any]
    operator: str | None
    tags: tuple[str, ...]
    notes: str
    created_at: str
    schema_version: int = BASELINE_SCHEMA_VERSION
    parent_baseline_id: str | None = None
    derived_reason: str | None = None
    evaluation_runs: tuple[BaselineRunReference, ...] = ()
    open_loop_runs: tuple[BaselineOpenLoopReference, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != BASELINE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported baseline schema_version {self.schema_version}")
        for name, value in (
            ("baseline_id", self.baseline_id),
            ("name", self.name),
            ("robot_name", self.robot_name),
            ("task_name", self.task_name),
            ("policy_server_url", self.policy_server_url),
            ("policy_label", self.policy_label),
            ("git_commit", self.git_commit),
            ("dataset_format", self.dataset_format),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        spec = ActionSpec.from_dict(self.action_spec)
        if self.state_schema and self.state_schema != spec.state_names:
            raise ValueError("state_schema must match ActionSpec state_names")
        if self.camera_roles and self.camera_roles != spec.camera_roles:
            raise ValueError("camera_roles must match ActionSpec camera_roles")
        object.__setattr__(self, "action_spec", spec.to_dict())
        object.__setattr__(self, "state_schema", tuple(self.state_schema or spec.state_names))
        object.__setattr__(self, "camera_roles", tuple(self.camera_roles or spec.camera_roles))
        for name in (
            "policy_metadata",
            "camera_config",
            "robot_config",
            "safety_config",
            "runtime_config",
            "rtc_config",
            "warmup_config",
            "evaluation_protocol",
            "initial_position_protocol",
            "source",
        ):
            object.__setattr__(self, name, json_copy(getattr(self, name)))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))

    @classmethod
    def from_runtime_snapshot(
        cls,
        payload: Mapping[str, Any],
        *,
        runtime_config: Mapping[str, Any],
        git_commit: str,
        baseline_id: str | None = None,
        parent_baseline_id: str | None = None,
        derived_reason: str | None = None,
    ) -> Baseline:
        runtime = clean_runtime_snapshot(runtime_config)
        clean_payload = clean_runtime_snapshot(payload)
        spec_value = clean_payload.get("action_spec") or runtime.get("action_spec")
        if not isinstance(spec_value, Mapping):
            raise ValueError("current runtime must publish an ActionSpec")
        spec = ActionSpec.from_dict(spec_value)
        runtime_values = _runtime_values(runtime)
        source = {
            "dataset": _optional_text(clean_payload.get("source_dataset")),
            "episode": _optional_int(clean_payload.get("source_episode")),
            "sample": _optional_int(clean_payload.get("source_sample")),
        }
        return cls(
            baseline_id=baseline_id or str(clean_payload.get("baseline_id") or uuid.uuid4().hex),
            name=str(clean_payload.get("name", "baseline")),
            robot_name=str(clean_payload.get("robot_name", runtime.get("robot", ""))),
            task_name=str(clean_payload.get("task_name", "unnamed-task")),
            prompt=str(clean_payload.get("prompt", runtime.get("prompt", ""))),
            policy_server_url=str(clean_payload.get("policy_server_url", runtime.get("server_url", ""))),
            policy_label=str(clean_payload.get("policy_label", "unlabeled-policy")),
            policy_metadata=dict(runtime.get("policy_metadata", clean_payload.get("policy_metadata", {}))),
            checkpoint_hash=_optional_text(clean_payload.get("checkpoint_hash")),
            git_commit=git_commit,
            dataset_format=str(clean_payload.get("dataset_format", "lerobot_v2.1")),
            action_spec=spec.to_dict(),
            state_schema=spec.state_names,
            camera_roles=spec.camera_roles,
            camera_config=_config_section(runtime, "camera_config"),
            robot_config=_config_section(runtime, "robot_config"),
            safety_config=_config_section(runtime, "safety_config"),
            runtime_config=runtime_values,
            rtc_config=_select(runtime, _RTC_KEYS),
            warmup_config=_select(runtime, _WARMUP_KEYS),
            evaluation_protocol={
                "planned_episodes": _positive_int(clean_payload.get("planned_episodes", 1), "planned_episodes"),
                "max_episode_duration_s": _positive_float(
                    clean_payload.get("max_episode_duration_s", clean_payload.get("max_episode_seconds", 60.0)),
                    "max_episode_duration_s",
                ),
            },
            initial_position_protocol=json_copy(
                clean_payload.get("initial_position_protocol")
                or runtime.get("recorded_start")
                or {"mode": "manual_reset"}
            ),
            source=source,
            operator=_optional_text(clean_payload.get("operator")),
            tags=_tags(clean_payload.get("tags", ())),
            notes=str(clean_payload.get("notes", "")),
            created_at=str(clean_payload.get("created_at", utc_now())),
            parent_baseline_id=parent_baseline_id,
            derived_reason=derived_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_id": self.baseline_id,
            "name": self.name,
            "robot_name": self.robot_name,
            "task_name": self.task_name,
            "prompt": self.prompt,
            "policy_server_url": self.policy_server_url,
            "policy_label": self.policy_label,
            "policy_metadata": json_copy(self.policy_metadata),
            "checkpoint_hash": self.checkpoint_hash,
            "git_commit": self.git_commit,
            "dataset_format": self.dataset_format,
            "action_spec": json_copy(self.action_spec),
            "state_schema": list(self.state_schema),
            "camera_roles": list(self.camera_roles),
            "camera_config": json_copy(self.camera_config),
            "robot_config": json_copy(self.robot_config),
            "safety_config": json_copy(self.safety_config),
            "runtime_config": json_copy(self.runtime_config),
            "rtc_config": json_copy(self.rtc_config),
            "warmup_config": json_copy(self.warmup_config),
            "evaluation_protocol": json_copy(self.evaluation_protocol),
            "initial_position_protocol": json_copy(self.initial_position_protocol),
            "source": json_copy(self.source),
            "operator": self.operator,
            "tags": list(self.tags),
            "notes": self.notes,
            "created_at": self.created_at,
            "parent_baseline_id": self.parent_baseline_id,
            "derived_reason": self.derived_reason,
            "evaluation_runs": [item.to_dict() for item in self.evaluation_runs],
            "open_loop_runs": [item.to_dict() for item in self.open_loop_runs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Baseline:
        return cls(
            schema_version=int(value.get("schema_version", BASELINE_SCHEMA_VERSION)),
            baseline_id=str(value["baseline_id"]),
            name=str(value["name"]),
            robot_name=str(value["robot_name"]),
            task_name=str(value["task_name"]),
            prompt=str(value.get("prompt", "")),
            policy_server_url=str(value["policy_server_url"]),
            policy_label=str(value["policy_label"]),
            policy_metadata=dict(value.get("policy_metadata", {})),
            checkpoint_hash=_optional_text(value.get("checkpoint_hash")),
            git_commit=str(value["git_commit"]),
            dataset_format=str(value["dataset_format"]),
            action_spec=dict(value["action_spec"]),
            state_schema=tuple(str(item) for item in value.get("state_schema", ())),
            camera_roles=tuple(str(item) for item in value.get("camera_roles", ())),
            camera_config=dict(value.get("camera_config", {})),
            robot_config=dict(value.get("robot_config", {})),
            safety_config=dict(value.get("safety_config", {})),
            runtime_config=dict(value.get("runtime_config", {})),
            rtc_config=dict(value.get("rtc_config", {})),
            warmup_config=dict(value.get("warmup_config", {})),
            evaluation_protocol=dict(value.get("evaluation_protocol", {})),
            initial_position_protocol=dict(value.get("initial_position_protocol", {})),
            source=dict(value.get("source", {})),
            operator=_optional_text(value.get("operator")),
            tags=_tags(value.get("tags", ())),
            notes=str(value.get("notes", "")),
            created_at=str(value.get("created_at", utc_now())),
            parent_baseline_id=_optional_text(value.get("parent_baseline_id")),
            derived_reason=_optional_text(value.get("derived_reason")),
            evaluation_runs=tuple(BaselineRunReference.from_dict(item) for item in value.get("evaluation_runs", ())),
            open_loop_runs=tuple(BaselineOpenLoopReference.from_dict(item) for item in value.get("open_loop_runs", ())),
        )

    def without_references(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("evaluation_runs")
        value.pop("open_loop_runs")
        return value


_RTC_KEYS = frozenset({"use_rtc", "rtc_replan_stride", "rtc_prefetch_steps", "rtc_exp_weight"})
_WARMUP_KEYS = frozenset(
    {
        "policy_warmup_enabled",
        "policy_warmup_timeout_s",
        "policy_warmup_requests",
        "policy_inference_timeout_s",
        "policy_connect_timeout_s",
        "policy_metadata_timeout_s",
        "policy_prefetch_first_chunk",
    }
)
_RUNTIME_KEYS = frozenset({"fps", "dataset_fps", "replan_steps", "max_steps", "resize_size", "log_timing"})


def _runtime_values(runtime: Mapping[str, Any]) -> dict[str, Any]:
    values = _select(runtime, _RUNTIME_KEYS)
    values.setdefault("control_fps", runtime.get("fps"))
    values.setdefault("dataset_fps", runtime.get("dataset_fps", runtime.get("fps")))
    return values


def _config_section(runtime: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = runtime.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"runtime {name} must be a JSON object")
    return json_copy(value)


def _select(value: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {key: json_copy(value[key]) for key in sorted(keys) if key in value}


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, tuple | list):
        return tuple(str(item) for item in value)
    raise ValueError("tags must be a list, tuple, or comma-separated string")
