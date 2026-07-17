"""Immutable data contracts for safe robot trajectory replay."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from mp_real.runtime.models import ActionSpec


class ReplayError(RuntimeError):
    """Base error for replay planning or execution."""


class ReplayValidationError(ReplayError):
    """The source episode cannot be safely replayed."""


class ReplayPlanStaleError(ReplayError):
    """A reviewed plan no longer matches the active replay identity."""


class ReplayState(enum.StrEnum):
    IDLE = "idle"
    PLANNING = "planning"
    VALIDATED = "validated"
    CONNECTING = "connecting"
    MOVING_TO_START = "moving_to_start"
    ARMED = "armed"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


class ReplayTimingMode(enum.StrEnum):
    RECORDED_TIMESTAMPS = "recorded"
    FIXED_FPS = "fixed"


class ReplayMode(enum.StrEnum):
    COMMAND_REPLAY = "command"
    STATE_TRAJECTORY_FOLLOWING = "state"


@dataclasses.dataclass(frozen=True)
class ReplaySafetyIssue:
    code: str
    message: str
    sample_index: int | None = None
    dimension: int | None = None


@dataclasses.dataclass(frozen=True)
class ReplaySafetyReport:
    """Complete offline preflight output displayed before robot connection."""

    errors: tuple[ReplaySafetyIssue, ...] = ()
    warnings: tuple[ReplaySafetyIssue, ...] = ()
    converted_fields: tuple[str, ...] = ()
    skipped_fields: tuple[str, ...] = ()
    maximum_observed_delta: float | None = None
    maximum_observed_velocity: float | None = None
    maximum_observed_acceleration: float | None = None
    expected_duration_s: float | None = None
    start_state: np.ndarray | None = None
    end_state: np.ndarray | None = None
    plan_hash: str | None = None
    source_dataset_id: str | None = None
    source_dataset_hash: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise ReplayValidationError("; ".join(issue.message for issue in self.errors))


@dataclasses.dataclass(frozen=True)
class ReplayConstraints:
    """Conservative generic limits; vendor limits are checked again live."""

    min_interval_s: float = 0.002
    max_interval_s: float = 1.0
    max_step: float = 0.05
    max_velocity: float = 0.5
    max_acceleration: float = 2.0
    tracking_tolerance: float = 0.05
    max_tracking_error: float = 0.15
    max_control_overrun_s: float = 0.10
    lower_limits: tuple[float, ...] | None = None
    upper_limits: tuple[float, ...] | None = None
    move_to_start_constraints: Any | None = None

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name in {"lower_limits", "upper_limits", "move_to_start_constraints"}:
                continue
            if getattr(self, field.name) <= 0:
                raise ValueError(f"{field.name} must be positive")
        if self.min_interval_s > self.max_interval_s:
            raise ValueError("min_interval_s must not exceed max_interval_s")
        if self.tracking_tolerance > self.max_tracking_error:
            raise ValueError("tracking_tolerance must not exceed max_tracking_error")
        if (self.lower_limits is None) != (self.upper_limits is None):
            raise ValueError("lower_limits and upper_limits must be configured together")
        if self.lower_limits is not None and self.upper_limits is not None:
            if len(self.lower_limits) != len(self.upper_limits):
                raise ValueError("joint limit vectors must have matching dimensions")
            if not np.isfinite(self.lower_limits).all() or not np.isfinite(self.upper_limits).all():
                raise ValueError("joint limits must be finite")
            if any(lower >= upper for lower, upper in zip(self.lower_limits, self.upper_limits, strict=True)):
                raise ValueError("each joint lower limit must be less than upper limit")


@dataclasses.dataclass(frozen=True)
class ReplayActionSource:
    """A semantically declared standard action source for command replay."""

    action_source: str
    action_mode: str
    action_spec: ActionSpec
    arm_count: int
    gripper_indices: tuple[int, ...]
    gripper_semantics: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.action_source or not self.action_mode or self.arm_count <= 0:
            raise ValueError("action source and action mode cannot be empty")


@dataclasses.dataclass(frozen=True)
class StateTrajectorySource:
    """A state source explicitly labelled as trajectory following."""

    state_spec: ActionSpec
    arm_count: int
    gripper_indices: tuple[int, ...]
    gripper_semantics: tuple[str, ...]
    interpolation: str = "none"


@dataclasses.dataclass(frozen=True)
class ReplayStep:
    source_sample_index: int
    frame_index: int
    source_timestamp_s: float
    target_offset_ns: int
    target: np.ndarray
    expected_state: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float32).copy()
        state = np.asarray(self.expected_state, dtype=np.float32).copy()
        if target.ndim != 1 or state.ndim != 1 or not np.isfinite(target).all() or not np.isfinite(state).all():
            raise ValueError("replay step target/state must be finite vectors")
        if self.source_sample_index < 0 or self.frame_index < 0 or self.target_offset_ns < 0:
            raise ValueError("replay step indices and target time must be non-negative")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "expected_state", state)


@dataclasses.dataclass(frozen=True)
class ReplayPlan:
    """An immutable validated trajectory, safe to review and hash."""

    plan_id: str
    session_id: str
    generation_id: int
    dataset_id: str
    dataset_hash: str
    episode_index: int
    start_sample: int
    end_sample: int
    robot_name: str
    mode: ReplayMode
    timing_mode: ReplayTimingMode
    speed_scale: float
    action_spec: ActionSpec
    source: ReplayActionSource | StateTrajectorySource
    steps: tuple[ReplayStep, ...]
    constraints: ReplayConstraints
    created_at_monotonic_ns: int
    plan_hash: str

    def __post_init__(self) -> None:
        if (
            self.generation_id < 0
            or self.episode_index < 0
            or self.start_sample < 0
            or self.end_sample < self.start_sample
        ):
            raise ValueError("invalid replay plan indices")
        if not self.dataset_id or not self.dataset_hash or not self.robot_name:
            raise ValueError("replay plan identity cannot be empty")
        if not 0 < self.speed_scale <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        if not self.steps:
            raise ValueError("replay plan must contain at least one step")
        if self.steps[0].source_sample_index != self.start_sample:
            raise ValueError("first replay step must equal start_sample")
        if self.steps[-1].source_sample_index != self.end_sample:
            raise ValueError("last replay step must equal end_sample")

    @property
    def expected_duration_s(self) -> float:
        return self.steps[-1].target_offset_ns / 1e9

    @property
    def start_state(self) -> np.ndarray:
        return self.steps[0].expected_state.copy()

    @property
    def end_state(self) -> np.ndarray:
        return self.steps[-1].expected_state.copy()


@dataclasses.dataclass(frozen=True)
class ReplayPlanningResult:
    plan: ReplayPlan | None
    report: ReplaySafetyReport


@dataclasses.dataclass(frozen=True)
class RobotReplayCursor:
    """Read-only controller snapshot.  Only RobotReplayController creates it."""

    state: ReplayState
    source_sample_index: int | None = None
    sent_sample_index: int | None = None
    acknowledged_sample_index: int | None = None
    progress_ratio: float = 0.0
    elapsed_s: float = 0.0
    tracking_error: float | None = None
    timestamp_monotonic_ns: int = 0
    session_id: str | None = None
    generation_id: int | None = None
    plan_hash: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress_ratio <= 1.0:
            raise ValueError("progress_ratio must be in [0, 1]")
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", time.monotonic_ns())


def build_plan_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def new_plan_identity() -> tuple[str, str]:
    return uuid.uuid4().hex, uuid.uuid4().hex


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    return value


def json_safe(value: Any) -> Any:
    """Public JSON conversion for CLI/Web status payloads."""
    return _json_safe(value)
