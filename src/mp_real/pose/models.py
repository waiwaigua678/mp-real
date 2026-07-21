from __future__ import annotations

import dataclasses
import secrets
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mp_real.common.plan_integrity import (
    PLAN_HASH_SCHEMA_VERSION,
    PlanIntegrityError,
    canonical_hash,
    freeze_action_spec,
    freeze_jsonish,
    freeze_robot_state,
    readonly_array,
)
from mp_real.runtime.models import ActionSpec, RobotState, VectorField


class PoseMoveError(RuntimeError):
    """Base error for a requested recorded-state move."""


class PoseValidationError(PoseMoveError):
    """The target cannot be safely mapped to the connected robot."""


class PosePlanStaleError(PoseMoveError):
    """A previously reviewed plan no longer matches live state or identity."""


class PosePlanIntegrityError(PosePlanStaleError, PlanIntegrityError):
    """A pose plan's stored hash no longer matches its canonical payload."""


class PoseMoveAborted(PoseMoveError):
    """The move stopped before verified completion."""


@dataclasses.dataclass(frozen=True)
class MappingEntry:
    """One explicit source-state to robot-state mapping.

    ``scale`` and ``offset`` are intentionally recorded rather than inferred;
    a rad/deg conversion is therefore visible in both the plan and audit log.
    """

    source_name: str
    target_name: str
    scale: float = 1.0
    offset: float = 0.0
    source_unit: str | None = None
    target_unit: str | None = None
    semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name or not self.target_name:
            raise ValueError("mapping names cannot be empty")
        if not np.isfinite(self.scale) or not np.isfinite(self.offset):
            raise ValueError("mapping scale and offset must be finite")


@dataclasses.dataclass(frozen=True)
class PoseMappingConfig:
    """Versioned, explicit mapping used only when schemas are not identical."""

    version: int
    entries: tuple[MappingEntry, ...]
    source_robot_name: str | None = None
    target_robot_name: str | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("mapping config version must be positive")
        if not self.entries:
            raise ValueError("mapping config must contain at least one entry")
        if len({entry.source_name for entry in self.entries}) != len(self.entries):
            raise ValueError("mapping config has duplicate source names")
        if len({entry.target_name for entry in self.entries}) != len(self.entries):
            raise ValueError("mapping config has duplicate target names")
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "metadata", freeze_jsonish(self.metadata))

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": self.version,
            "source_robot_name": self.source_robot_name,
            "target_robot_name": self.target_robot_name,
            "entries": [dataclasses.asdict(entry) for entry in self.entries],
            "metadata": _json_safe(self.metadata),
        }
        return _hash_payload(payload)


@dataclasses.dataclass(frozen=True)
class RecordedPoseTarget:
    """A target derived exclusively from ``RecordedSample.observation.state``."""

    dataset_id: str
    episode_index: int
    sample_index: int
    robot_name: str
    state_schema: tuple[str, ...]
    state_values: np.ndarray
    state_fields: tuple[VectorField, ...]
    joint_unit: str
    timestamp: float
    source_metadata: Mapping[str, Any]
    action_spec: ActionSpec
    target_id: str = ""

    def __post_init__(self) -> None:
        values = readonly_array(self.state_values)
        if not self.dataset_id or not self.robot_name:
            raise ValueError("dataset_id and robot_name cannot be empty")
        if self.episode_index < 0 or self.sample_index < 0:
            raise ValueError("episode_index and sample_index must be non-negative")
        if values.ndim != 1 or len(values) != self.action_spec.state_dim:
            raise ValueError("recorded state shape must match the ActionSpec state dimension")
        if len(self.state_schema) != len(values) or len(self.state_fields) != len(values):
            raise ValueError("state schema and fields must match recorded state dimension")
        if not np.isfinite(values).all():
            raise PoseValidationError("recorded state contains NaN or Inf")
        if tuple(field.name for field in self.state_fields) != self.state_schema:
            raise ValueError("state_schema must preserve the ActionSpec state field order")
        object.__setattr__(self, "state_values", values)
        object.__setattr__(self, "state_fields", tuple(self.state_fields))
        object.__setattr__(self, "source_metadata", freeze_jsonish(self.source_metadata))
        object.__setattr__(self, "action_spec", freeze_action_spec(self.action_spec))
        if not self.target_id:
            object.__setattr__(
                self,
                "target_id",
                _hash_payload(
                    {
                        "dataset_id": self.dataset_id,
                        "episode_index": self.episode_index,
                        "sample_index": self.sample_index,
                        "state_values": values.tolist(),
                        "source_metadata": _json_safe(self.source_metadata),
                    }
                ),
            )

    @property
    def gripper_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, field in enumerate(self.state_fields) if field.semantics == "gripper_open_fraction"
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "target_id": self.target_id,
            "dataset_id": self.dataset_id,
            "episode_index": self.episode_index,
            "sample_index": self.sample_index,
            "robot_name": self.robot_name,
            "state_schema": self.state_schema,
            "state_values": self.state_values,
            "state_fields": self.state_fields,
            "joint_unit": self.joint_unit,
            "timestamp": self.timestamp,
            "source_metadata": self.source_metadata,
            "action_spec": self.action_spec,
        }


@dataclasses.dataclass(frozen=True)
class PoseMotionConstraints:
    """Conservative, unit-preserving limits for one pose move."""

    control_period_s: float = 0.05
    max_joint_velocity: float = 0.10
    max_joint_acceleration: float = 0.30
    max_joint_step: float = 0.02
    max_gripper_step: float = 0.02
    tracking_tolerance: float = 0.05
    max_tracking_error: float = 0.15
    max_control_overrun_s: float = 0.10
    verify_timeout_s: float = 3.0
    keep_gripper: bool = False
    plan_expiration_s: float | None = 300.0

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name in {"keep_gripper", "plan_expiration_s"}:
                continue
            if getattr(self, field.name) <= 0:
                raise ValueError(f"{field.name} must be positive")
        if self.plan_expiration_s is not None and self.plan_expiration_s <= 0:
            raise ValueError("plan_expiration_s must be positive when configured")
        if self.tracking_tolerance > self.max_tracking_error:
            raise ValueError("tracking_tolerance must not exceed max_tracking_error")


@dataclasses.dataclass(frozen=True)
class PoseValidationIssue:
    code: str
    message: str
    dimension: int | None = None
    severity: str = "error"


@dataclasses.dataclass(frozen=True)
class PoseValidationReport:
    issues: tuple[PoseValidationIssue, ...] = ()
    warnings: tuple[PoseValidationIssue, ...] = ()
    mapping_fingerprint: str | None = None
    unavailable_checks: tuple[PoseValidationIssue, ...] = ()
    passed_checks: tuple[PoseValidationIssue, ...] = ()
    safety_policy: str | None = None
    safety_profile_hash: str | None = None
    safety_profile: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    development_override: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "unavailable_checks", tuple(self.unavailable_checks))
        object.__setattr__(self, "passed_checks", tuple(self.passed_checks))
        object.__setattr__(self, "safety_profile", freeze_jsonish(self.safety_profile))
        object.__setattr__(self, "development_override", freeze_jsonish(self.development_override))

    @property
    def errors(self) -> tuple[PoseValidationIssue, ...]:
        return self.issues

    @property
    def valid(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        if self.issues:
            raise PoseValidationError("; ".join(issue.message for issue in self.issues))


@dataclasses.dataclass(frozen=True)
class PoseSafetyLimits:
    """Known joint limits for the concrete robot's policy-facing state.

    Empty bounds are not silently treated as unlimited by an execution path;
    callers that require a physical move must obtain vendor validation too.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = readonly_array(self.lower)
        upper = readonly_array(self.upper)
        if lower.shape != upper.shape or lower.ndim != 1:
            raise ValueError("joint limit vectors must have matching one-dimensional shapes")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise ValueError("joint limits must be finite and lower than upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclasses.dataclass(frozen=True)
class ValidatedPoseTarget:
    """A recorded target expressed in the connected robot's state layout."""

    values: np.ndarray
    field_names: tuple[str, ...]
    gripper_indices: tuple[int, ...]
    mappings: tuple[MappingEntry, ...]
    report: PoseValidationReport

    def __post_init__(self) -> None:
        values = readonly_array(self.values)
        if values.ndim != 1 or len(values) != len(self.field_names):
            raise ValueError("validated target values and fields must have matching dimensions")
        object.__setattr__(self, "values", values)


@dataclasses.dataclass(frozen=True)
class PoseWaypoint:
    index: int
    target: np.ndarray
    scheduled_at_monotonic_ns: int

    def __post_init__(self) -> None:
        target = readonly_array(self.target)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError("pose waypoint must be a finite vector")
        object.__setattr__(self, "target", target)

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "target": self.target,
            "scheduled_at_monotonic_ns": self.scheduled_at_monotonic_ns,
        }


@dataclasses.dataclass(frozen=True)
class MoveToRecordedStatePlan:
    """A live-state-revalidated, immutable low-speed move plan."""

    plan_id: str
    target: RecordedPoseTarget
    current_state: RobotState
    target_state: np.ndarray
    per_dimension_delta: np.ndarray
    mapped_joint_names: tuple[str, ...]
    unit_conversions: tuple[MappingEntry, ...]
    gripper_indices: tuple[int, ...]
    waypoints: tuple[PoseWaypoint, ...]
    expected_duration_s: float
    constraints: PoseMotionConstraints
    safety_warnings: tuple[str, ...]
    required_confirmations: tuple[str, ...]
    mapping_fingerprint: str | None
    session_id: str
    generation_id: int
    created_at_monotonic_ns: int
    plan_hash: str = ""
    safety_flags: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    safety_profile_hash: str | None = None
    safety_policy: str | None = None
    resource_owner_id: str | None = None
    resource_lease_id: str | None = None
    created_from_robot_state_hash: str | None = None
    expires_at_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        current_state = freeze_robot_state(self.current_state)
        target_state = readonly_array(self.target_state)
        delta = readonly_array(self.per_dimension_delta)
        dimension = self.target.action_spec.state_dim
        if target_state.shape != (dimension,) or delta.shape != (dimension,):
            raise ValueError("plan target and delta must match state dimension")
        if len(self.waypoints) == 0:
            raise ValueError("plan must contain at least one waypoint")
        object.__setattr__(self, "current_state", current_state)
        object.__setattr__(self, "target_state", target_state)
        object.__setattr__(self, "per_dimension_delta", delta)
        object.__setattr__(self, "mapped_joint_names", tuple(str(item) for item in self.mapped_joint_names))
        object.__setattr__(self, "unit_conversions", tuple(self.unit_conversions))
        object.__setattr__(self, "gripper_indices", tuple(int(index) for index in self.gripper_indices))
        object.__setattr__(self, "waypoints", tuple(self.waypoints))
        object.__setattr__(self, "safety_warnings", tuple(str(item) for item in self.safety_warnings))
        object.__setattr__(self, "required_confirmations", tuple(str(item) for item in self.required_confirmations))
        object.__setattr__(self, "safety_flags", freeze_jsonish(self.safety_flags))
        if self.created_from_robot_state_hash is None:
            object.__setattr__(self, "created_from_robot_state_hash", _hash_payload({"current_state": current_state}))
        if self.expires_at_monotonic_ns is None and self.constraints.plan_expiration_s is not None:
            object.__setattr__(
                self,
                "expires_at_monotonic_ns",
                self.created_at_monotonic_ns + int(self.constraints.plan_expiration_s * 1e9),
            )
        computed = self.recompute_plan_hash()
        if self.plan_hash:
            if not secrets.compare_digest(self.plan_hash, computed):
                raise PosePlanIntegrityError("pose plan hash does not match its canonical payload")
        else:
            object.__setattr__(self, "plan_hash", computed)

    @classmethod
    def build(
        cls,
        *,
        target: RecordedPoseTarget,
        current_state: RobotState,
        target_state: np.ndarray,
        gripper_indices: Sequence[int] | None = None,
        mapped_joint_names: Sequence[str],
        conversions: Sequence[MappingEntry],
        constraints: PoseMotionConstraints,
        safety_warnings: Sequence[str] = (),
        required_confirmations: Sequence[str] = ("execute_low_speed_pose_move",),
        mapping_fingerprint: str | None = None,
        session_id: str | None = None,
        generation_id: int = 0,
        resource_owner_id: str | None = None,
        resource_lease_id: str | None = None,
        safety_profile_hash: str | None = None,
        safety_policy: str | None = None,
    ) -> MoveToRecordedStatePlan:
        now_ns = time.monotonic_ns()
        current = np.asarray(current_state.values, dtype=np.float32)
        desired = np.asarray(target_state, dtype=np.float32)
        if current.shape != desired.shape:
            raise PoseValidationError("current state and pose target dimension differ")
        if not np.isfinite(current).all() or not np.isfinite(desired).all():
            raise PoseValidationError("current state or pose target contains NaN or Inf")
        desired = desired.copy()
        plan_gripper_indices = tuple(target.gripper_indices if gripper_indices is None else gripper_indices)
        if constraints.keep_gripper:
            desired[list(plan_gripper_indices)] = current[list(plan_gripper_indices)]
        delta = desired - current
        joint_indices = tuple(index for index in range(len(desired)) if index not in plan_gripper_indices)
        max_ratio = 1.0
        maximum_joint_delta = 0.0
        if joint_indices:
            maximum_joint_delta = float(np.max(np.abs(delta[list(joint_indices)])))
            # Cubic ease-in/ease-out has a maximum normalized velocity of
            # 1.5.  Including it here keeps each emitted joint waypoint
            # within the configured position-step ceiling.
            max_ratio = max(max_ratio, 1.5 * maximum_joint_delta / constraints.max_joint_step)
        if plan_gripper_indices and not constraints.keep_gripper:
            max_ratio = max(
                max_ratio,
                float(np.max(np.abs(delta[list(plan_gripper_indices)]) / constraints.max_gripper_step)),
            )
        steps = max(1, int(np.ceil(max_ratio)))
        duration_s = max(
            steps * constraints.control_period_s,
            1.5 * maximum_joint_delta / constraints.max_joint_velocity,
            np.sqrt(6.0 * maximum_joint_delta / constraints.max_joint_acceleration),
        )
        steps = max(steps, int(np.ceil(duration_s / constraints.control_period_s)))
        waypoints: list[PoseWaypoint] = []
        for index in range(1, steps + 1):
            linear_fraction = index / steps
            joint_fraction = 3.0 * linear_fraction**2 - 2.0 * linear_fraction**3
            fractions = np.full(len(desired), joint_fraction, dtype=np.float32)
            if plan_gripper_indices:
                fractions[list(plan_gripper_indices)] = linear_fraction
            waypoints.append(
                PoseWaypoint(
                    index,
                    current + fractions * delta,
                    now_ns + int(index * constraints.control_period_s * 1e9),
                )
            )
        actual_duration_s = steps * constraints.control_period_s
        sid = session_id or uuid.uuid4().hex
        return cls(
            plan_id=uuid.uuid4().hex,
            target=target,
            current_state=current_state,
            target_state=desired,
            per_dimension_delta=delta,
            mapped_joint_names=tuple(mapped_joint_names),
            unit_conversions=tuple(conversions),
            gripper_indices=plan_gripper_indices,
            waypoints=tuple(waypoints),
            expected_duration_s=actual_duration_s,
            constraints=constraints,
            safety_warnings=tuple(safety_warnings),
            required_confirmations=tuple(required_confirmations),
            mapping_fingerprint=mapping_fingerprint,
            session_id=sid,
            generation_id=generation_id,
            created_at_monotonic_ns=now_ns,
            resource_owner_id=resource_owner_id,
            resource_lease_id=resource_lease_id,
            safety_profile_hash=safety_profile_hash,
            safety_policy=safety_policy,
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": PLAN_HASH_SCHEMA_VERSION,
            "plan_type": "move_to_recorded_state",
            "plan_id": self.plan_id,
            "target": self.target.canonical_payload(),
            "current_state": self.current_state,
            "target_state": self.target_state,
            "per_dimension_delta": self.per_dimension_delta,
            "mapped_joint_names": self.mapped_joint_names,
            "unit_conversions": self.unit_conversions,
            "gripper_indices": self.gripper_indices,
            "waypoints": [waypoint.canonical_payload() for waypoint in self.waypoints],
            "expected_duration_s": self.expected_duration_s,
            "constraints": self.constraints,
            "safety_warnings": self.safety_warnings,
            "required_confirmations": self.required_confirmations,
            "mapping_fingerprint": self.mapping_fingerprint,
            "session_id": self.session_id,
            "generation_id": self.generation_id,
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
        return _hash_payload(self.canonical_payload())

    def require_integrity(self, *, now_monotonic_ns: int | None = None, check_expiration: bool = False) -> None:
        computed = self.recompute_plan_hash()
        if not secrets.compare_digest(self.plan_hash, computed):
            raise PosePlanIntegrityError("pose plan payload changed after review")
        if (
            check_expiration
            and self.expires_at_monotonic_ns is not None
            and (now_monotonic_ns or time.monotonic_ns()) > self.expires_at_monotonic_ns
        ):
            raise PosePlanStaleError("pose plan expired; generate a fresh plan")


@dataclasses.dataclass(frozen=True)
class PoseMoveProgress:
    plan_id: str
    waypoint_index: int
    waypoint_count: int
    current_state: np.ndarray
    target_state: np.ndarray
    tracking_error: float
    monotonic_timestamp_ns: int


@dataclasses.dataclass(frozen=True)
class PoseMoveResult:
    plan_id: str
    status: str
    final_state: RobotState | None
    tracking_error: float | None
    message: str | None = None


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
    return value


def _hash_payload(value: Any) -> str:
    payload = value if isinstance(value, Mapping) else {"value": value}
    return canonical_hash(payload)
