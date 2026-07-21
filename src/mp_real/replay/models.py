"""Immutable data contracts for safe robot trajectory replay."""

from __future__ import annotations

import dataclasses
import enum
import secrets
import time
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from mp_real.common.plan_integrity import (
    PLAN_HASH_SCHEMA_VERSION,
    FrozenMapping,
    PlanIntegrityError,
    canonical_hash,
    freeze_action_spec,
    freeze_jsonish,
    readonly_array,
    readonly_optional_array,
)
from mp_real.runtime.models import ActionSpec


class ReplayError(RuntimeError):
    """Base error for replay planning or execution."""


class ReplayValidationError(ReplayError):
    """The source episode cannot be safely replayed."""


class ReplayPlanStaleError(ReplayError):
    """A reviewed plan no longer matches the active replay identity."""


class ReplayPlanIntegrityError(ReplayPlanStaleError, PlanIntegrityError):
    """A replay plan's stored hash no longer matches its canonical payload."""


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


class ReplayAcknowledgementStrategy(enum.StrEnum):
    IMMEDIATE_INTERFACE_ACK = "immediate_interface_ack"
    FEEDBACK_THRESHOLD = "feedback_threshold"
    FOLLOWER_WINDOW = "follower_window"
    STATE_TRAJECTORY_SETTLE = "state_trajectory_settle"


@dataclasses.dataclass(frozen=True)
class ReplaySafetyIssue:
    code: str
    message: str
    sample_index: int | None = None
    dimension: int | None = None
    severity: str = "error"
    source: str | None = None


@dataclasses.dataclass(frozen=True)
class ReplaySafetyReport:
    """Complete offline preflight output displayed before robot connection."""

    errors: tuple[ReplaySafetyIssue, ...] = ()
    warnings: tuple[ReplaySafetyIssue, ...] = ()
    unavailable_checks: tuple[ReplaySafetyIssue, ...] = ()
    passed_checks: tuple[ReplaySafetyIssue, ...] = ()
    converted_fields: tuple[str, ...] = ()
    skipped_fields: tuple[str, ...] = ()
    maximum_observed_delta: float | None = None
    maximum_observed_velocity: float | None = None
    maximum_observed_acceleration: float | None = None
    maximum_observed_joint_delta: float | None = None
    maximum_observed_joint_velocity: float | None = None
    maximum_observed_joint_acceleration: float | None = None
    maximum_observed_gripper_delta: float | None = None
    maximum_observed_gripper_velocity: float | None = None
    maximum_observed_gripper_acceleration: float | None = None
    expected_duration_s: float | None = None
    start_state: np.ndarray | None = None
    end_state: np.ndarray | None = None
    plan_hash: str | None = None
    source_dataset_id: str | None = None
    source_dataset_hash: str | None = None
    safety_policy: str | None = None
    safety_profile_hash: str | None = None
    safety_profile: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    development_override: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "unavailable_checks", tuple(self.unavailable_checks))
        object.__setattr__(self, "passed_checks", tuple(self.passed_checks))
        object.__setattr__(self, "start_state", readonly_optional_array(self.start_state))
        object.__setattr__(self, "end_state", readonly_optional_array(self.end_state))
        object.__setattr__(self, "safety_profile", freeze_jsonish(self.safety_profile))
        object.__setattr__(self, "development_override", freeze_jsonish(self.development_override))

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
    # Backward-compatible generic aliases. H5 applies these only to non-gripper
    # joint dimensions unless explicit joint_* values are supplied.
    max_step: float = 0.05
    max_velocity: float = 0.5
    max_acceleration: float = 2.0
    tracking_tolerance: float = 0.05
    max_tracking_error: float = 0.15
    max_control_overrun_s: float = 0.10
    lower_limits: tuple[float, ...] | None = None
    upper_limits: tuple[float, ...] | None = None
    joint_max_step: float | None = None
    joint_max_velocity: float | None = None
    joint_max_acceleration: float | None = None
    joint_tracking_error: float | None = None
    joint_lower_limits: tuple[float, ...] | None = None
    joint_upper_limits: tuple[float, ...] | None = None
    gripper_min: tuple[float, ...] | None = None
    gripper_max: tuple[float, ...] | None = None
    gripper_max_step: float | None = None
    gripper_command_mode: str = "position"
    gripper_settle_timeout_s: float = 1.0
    gripper_tracking_threshold: float | None = None
    gripper_transition_hysteresis: float = 0.0
    gripper_indices: tuple[int, ...] | None = None
    acknowledgement_strategy: ReplayAcknowledgementStrategy = ReplayAcknowledgementStrategy.FEEDBACK_THRESHOLD
    feedback_poll_interval_s: float = 0.01
    acknowledgement_timeout_s: float = 1.0
    feedback_freshness_timeout_s: float | None = None
    follower_window_samples: int = 1
    state_trajectory_settle_cycles: int = 2
    sustained_tracking_error_limit: int = 3
    extreme_tracking_error: float | None = None
    poll_feedback_while_paused: bool = False
    move_to_start_constraints: Any | None = None
    plan_expiration_s: float | None = 300.0

    def __post_init__(self) -> None:
        for field_name in (
            "min_interval_s",
            "max_interval_s",
            "max_step",
            "max_velocity",
            "max_acceleration",
            "tracking_tolerance",
            "max_tracking_error",
            "max_control_overrun_s",
            "gripper_settle_timeout_s",
            "feedback_poll_interval_s",
            "acknowledgement_timeout_s",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "joint_max_step",
            "joint_max_velocity",
            "joint_max_acceleration",
            "joint_tracking_error",
            "gripper_max_step",
            "gripper_tracking_threshold",
            "feedback_freshness_timeout_s",
            "extreme_tracking_error",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.plan_expiration_s is not None and self.plan_expiration_s <= 0:
            raise ValueError("plan_expiration_s must be positive when configured")
        if self.min_interval_s > self.max_interval_s:
            raise ValueError("min_interval_s must not exceed max_interval_s")
        if self.tracking_tolerance > self.max_tracking_error:
            raise ValueError("tracking_tolerance must not exceed max_tracking_error")
        if self.joint_tracking_error is not None and self.joint_tracking_error > self.effective_extreme_tracking_error:
            raise ValueError("joint_tracking_error must not exceed extreme tracking error")
        if (
            self.gripper_tracking_threshold is not None
            and self.gripper_tracking_threshold > self.effective_extreme_tracking_error
        ):
            raise ValueError("gripper_tracking_threshold must not exceed extreme tracking error")
        if self.gripper_transition_hysteresis < 0:
            raise ValueError("gripper_transition_hysteresis cannot be negative")
        if self.follower_window_samples < 0:
            raise ValueError("follower_window_samples cannot be negative")
        if self.state_trajectory_settle_cycles <= 0:
            raise ValueError("state_trajectory_settle_cycles must be positive")
        if self.sustained_tracking_error_limit <= 0:
            raise ValueError("sustained_tracking_error_limit must be positive")
        strategy = (
            self.acknowledgement_strategy
            if isinstance(self.acknowledgement_strategy, ReplayAcknowledgementStrategy)
            else ReplayAcknowledgementStrategy(str(self.acknowledgement_strategy))
        )
        object.__setattr__(self, "acknowledgement_strategy", strategy)
        mode = self.gripper_command_mode.lower()
        if mode not in {"position", "open", "closed", "open_closed"}:
            raise ValueError("gripper_command_mode must be one of position, open, closed, open_closed")
        object.__setattr__(self, "gripper_command_mode", mode)
        self._normalize_limit_pair("lower_limits", "upper_limits", "limit vectors")
        self._normalize_limit_pair("joint_lower_limits", "joint_upper_limits", "joint limit vectors")
        self._normalize_limit_pair("gripper_min", "gripper_max", "gripper range vectors")
        if self.gripper_indices is not None:
            object.__setattr__(self, "gripper_indices", tuple(int(index) for index in self.gripper_indices))

    @property
    def effective_joint_max_step(self) -> float:
        return self.max_step if self.joint_max_step is None else self.joint_max_step

    @property
    def effective_joint_max_velocity(self) -> float:
        return self.max_velocity if self.joint_max_velocity is None else self.joint_max_velocity

    @property
    def effective_joint_max_acceleration(self) -> float:
        return self.max_acceleration if self.joint_max_acceleration is None else self.joint_max_acceleration

    @property
    def effective_joint_tracking_error(self) -> float:
        return self.tracking_tolerance if self.joint_tracking_error is None else self.joint_tracking_error

    @property
    def effective_gripper_tracking_threshold(self) -> float:
        if self.gripper_tracking_threshold is not None:
            return self.gripper_tracking_threshold
        return self.effective_joint_tracking_error

    @property
    def effective_extreme_tracking_error(self) -> float:
        return self.max_tracking_error if self.extreme_tracking_error is None else self.extreme_tracking_error

    def _normalize_limit_pair(self, lower_name: str, upper_name: str, label: str) -> None:
        lower = getattr(self, lower_name)
        upper = getattr(self, upper_name)
        if (lower is None) != (upper is None):
            raise ValueError(f"{lower_name} and {upper_name} must be configured together")
        if lower is None or upper is None:
            return
        if len(lower) != len(upper):
            raise ValueError(f"{label} must have matching dimensions")
        lower_tuple = tuple(float(value) for value in lower)
        upper_tuple = tuple(float(value) for value in upper)
        if not np.isfinite(lower_tuple).all() or not np.isfinite(upper_tuple).all():
            raise ValueError(f"{label} must be finite")
        if any(lo >= hi for lo, hi in zip(lower_tuple, upper_tuple, strict=True)):
            raise ValueError(f"each {label} lower bound must be less than upper bound")
        object.__setattr__(self, lower_name, lower_tuple)
        object.__setattr__(self, upper_name, upper_tuple)


@dataclasses.dataclass(frozen=True)
class ReplayCommandRecord:
    command_id: str
    source_sample_index: int
    sent_timestamp_ns: int
    target: np.ndarray
    expected_state: np.ndarray
    acknowledgement_deadline_ns: int
    joint_tracking_threshold: float
    gripper_tracking_threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", readonly_array(self.target))
        object.__setattr__(self, "expected_state", readonly_array(self.expected_state))


@dataclasses.dataclass(frozen=True)
class ReplayFeedbackRecord:
    feedback_timestamp_ns: int
    robot_state: np.ndarray
    feedback_age_s: float | None
    matched_command_id: str | None
    instantaneous_tracking_error: float | None
    lag_adjusted_tracking_error: float | None
    acknowledged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_state", readonly_array(self.robot_state))


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
        if not self.action_source or not self.action_mode or self.arm_count < 0:
            raise ValueError("action source and action mode cannot be empty")
        object.__setattr__(self, "action_spec", freeze_action_spec(self.action_spec))
        object.__setattr__(self, "gripper_indices", tuple(int(index) for index in self.gripper_indices))
        object.__setattr__(self, "gripper_semantics", tuple(str(item) for item in self.gripper_semantics))


@dataclasses.dataclass(frozen=True)
class StateTrajectorySource:
    """A state source explicitly labelled as trajectory following."""

    state_spec: ActionSpec
    arm_count: int
    gripper_indices: tuple[int, ...]
    gripper_semantics: tuple[str, ...]
    interpolation: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_spec", freeze_action_spec(self.state_spec))
        object.__setattr__(self, "gripper_indices", tuple(int(index) for index in self.gripper_indices))
        object.__setattr__(self, "gripper_semantics", tuple(str(item) for item in self.gripper_semantics))


@dataclasses.dataclass(frozen=True)
class ReplayStep:
    source_sample_index: int
    frame_index: int
    source_timestamp_s: float
    target_offset_ns: int
    target: np.ndarray
    expected_state: np.ndarray

    def __post_init__(self) -> None:
        target = readonly_array(self.target)
        state = readonly_array(self.expected_state)
        if target.ndim != 1 or state.ndim != 1 or not np.isfinite(target).all() or not np.isfinite(state).all():
            raise ValueError("replay step target/state must be finite vectors")
        if self.source_sample_index < 0 or self.frame_index < 0 or self.target_offset_ns < 0:
            raise ValueError("replay step indices and target time must be non-negative")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "expected_state", state)

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "source_sample_index": self.source_sample_index,
            "frame_index": self.frame_index,
            "source_timestamp_s": self.source_timestamp_s,
            "target_offset_ns": self.target_offset_ns,
            "target": self.target,
            "expected_state": self.expected_state,
        }


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
    plan_hash: str = ""
    safety_flags: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    safety_profile_hash: str | None = None
    safety_policy: str | None = None
    resource_owner_id: str | None = None
    resource_lease_id: str | None = None
    source_data_identity: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    created_from_robot_state_hash: str | None = None
    expires_at_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_spec", freeze_action_spec(self.action_spec))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "safety_flags", freeze_jsonish(self.safety_flags))
        if not self.source_data_identity:
            object.__setattr__(
                self,
                "source_data_identity",
                FrozenMapping({"dataset_id": self.dataset_id, "dataset_hash": self.dataset_hash}),
            )
        else:
            object.__setattr__(self, "source_data_identity", freeze_jsonish(self.source_data_identity))
        if self.expires_at_monotonic_ns is None and self.constraints.plan_expiration_s is not None:
            object.__setattr__(
                self,
                "expires_at_monotonic_ns",
                self.created_at_monotonic_ns + int(self.constraints.plan_expiration_s * 1e9),
            )
        if (
            self.generation_id < 0
            or self.episode_index < 0
            or self.start_sample < 0
            or self.end_sample < self.start_sample
        ):
            raise ValueError("invalid replay plan indices")
        if not self.dataset_id or not self.dataset_hash or not self.robot_name:
            raise ValueError("replay plan identity cannot be empty")
        if self.source.arm_count <= 0:
            raise ValueError("replay plan source must declare at least one arm")
        if not 0 < self.speed_scale <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        if not self.steps:
            raise ValueError("replay plan must contain at least one step")
        if self.steps[0].source_sample_index != self.start_sample:
            raise ValueError("first replay step must equal start_sample")
        if self.steps[-1].source_sample_index != self.end_sample:
            raise ValueError("last replay step must equal end_sample")
        computed = self.recompute_plan_hash()
        if self.plan_hash:
            if not secrets.compare_digest(self.plan_hash, computed):
                raise ReplayPlanIntegrityError("replay plan hash does not match its canonical payload")
        else:
            object.__setattr__(self, "plan_hash", computed)

    @property
    def expected_duration_s(self) -> float:
        return self.steps[-1].target_offset_ns / 1e9

    @property
    def start_state(self) -> np.ndarray:
        return self.steps[0].expected_state.copy()

    @property
    def end_state(self) -> np.ndarray:
        return self.steps[-1].expected_state.copy()

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": PLAN_HASH_SCHEMA_VERSION,
            "plan_type": "replay",
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "generation_id": self.generation_id,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "source_data_identity": self.source_data_identity,
            "episode_index": self.episode_index,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "robot_name": self.robot_name,
            "mode": self.mode.value,
            "timing_mode": self.timing_mode.value,
            "speed_scale": self.speed_scale,
            "action_spec": self.action_spec,
            "state_schema": self.action_spec.state_fields,
            "action_schema": self.action_spec.action_fields,
            "source": self.source,
            "steps": [step.canonical_payload() for step in self.steps],
            "constraints": self.constraints,
            "safety_flags": self.safety_flags,
            "safety_profile_hash": self.safety_profile_hash,
            "safety_policy": self.safety_policy,
            "resource_owner_id": self.resource_owner_id,
            "resource_lease_id": self.resource_lease_id,
            "created_from_robot_state_hash": self.created_from_robot_state_hash,
            "created_at_monotonic_ns": self.created_at_monotonic_ns,
            "expires_at_monotonic_ns": self.expires_at_monotonic_ns,
        }

    def recompute_plan_hash(self) -> str:
        return build_plan_hash(self.canonical_payload())

    def require_integrity(self, *, now_monotonic_ns: int | None = None, check_expiration: bool = False) -> None:
        computed = self.recompute_plan_hash()
        if not secrets.compare_digest(self.plan_hash, computed):
            raise ReplayPlanIntegrityError("replay plan payload changed after review")
        if (
            check_expiration
            and self.expires_at_monotonic_ns is not None
            and (now_monotonic_ns or time.monotonic_ns()) > self.expires_at_monotonic_ns
        ):
            raise ReplayPlanStaleError("replay plan expired; generate a fresh plan")

    def with_integrity_context(
        self,
        *,
        generation_id: int | None = None,
        resource_owner_id: str | None = None,
        resource_lease_id: str | None = None,
        created_from_robot_state_hash: str | None = None,
        safety_profile_hash: str | None = None,
        safety_policy: str | None = None,
    ) -> ReplayPlan:
        return dataclasses.replace(
            self,
            generation_id=self.generation_id if generation_id is None else generation_id,
            resource_owner_id=self.resource_owner_id if resource_owner_id is None else resource_owner_id,
            resource_lease_id=self.resource_lease_id if resource_lease_id is None else resource_lease_id,
            created_from_robot_state_hash=(
                self.created_from_robot_state_hash
                if created_from_robot_state_hash is None
                else created_from_robot_state_hash
            ),
            safety_profile_hash=self.safety_profile_hash if safety_profile_hash is None else safety_profile_hash,
            safety_policy=self.safety_policy if safety_policy is None else safety_policy,
            plan_hash="",
        )


@dataclasses.dataclass(frozen=True)
class ReplayPlanningResult:
    plan: ReplayPlan | None
    report: ReplaySafetyReport


@dataclasses.dataclass(frozen=True)
class RobotReplayCursor:
    """Read-only controller snapshot.  Only RobotReplayController creates it."""

    state: ReplayState
    planned_sample_index: int | None = None
    # Backward-compatible alias for older UI/tests; H5 treats it as the last
    # planned source sample, not as proof of execution.
    source_sample_index: int | None = None
    sent_sample_index: int | None = None
    feedback_sample_index: int | None = None
    acknowledged_sample_index: int | None = None
    displayed_sample_index: int | None = None
    progress_ratio: float = 0.0
    sent_progress_ratio: float = 0.0
    feedback_progress_ratio: float = 0.0
    acknowledged_progress_ratio: float = 0.0
    lag_samples: int = 0
    elapsed_s: float = 0.0
    tracking_error: float | None = None
    instantaneous_tracking_error: float | None = None
    lag_adjusted_tracking_error: float | None = None
    max_tracking_error: float | None = None
    sustained_tracking_error_count: int = 0
    acknowledgement_strategy: str | None = None
    timestamp_monotonic_ns: int = 0
    session_id: str | None = None
    generation_id: int | None = None
    plan_hash: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        for name in ("progress_ratio", "sent_progress_ratio", "feedback_progress_ratio", "acknowledged_progress_ratio"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.lag_samples < 0:
            raise ValueError("lag_samples cannot be negative")
        if self.sustained_tracking_error_count < 0:
            raise ValueError("sustained_tracking_error_count cannot be negative")
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", time.monotonic_ns())


def build_plan_hash(payload: Mapping[str, Any]) -> str:
    return canonical_hash(payload)


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
