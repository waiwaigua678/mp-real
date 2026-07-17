from __future__ import annotations

import dataclasses
import itertools
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

_observation_ids = itertools.count(1)
_observation_id_lock = threading.Lock()


def _next_observation_id() -> int:
    with _observation_id_lock:
        return next(_observation_ids)


@dataclasses.dataclass(frozen=True)
class VectorField:
    """One ordered element in a robot state or action vector.

    The policy/runtime continues to use dense numpy arrays.  This descriptor is
    deliberately metadata only: it makes the order and units explicit when an
    array crosses the recording boundary without imposing a shared robot
    layout.
    """

    name: str
    unit: str
    semantics: str


@dataclasses.dataclass(frozen=True)
class ActionProvenance:
    """Identity of the policy observation/chunk that produced an action."""

    observation_id: int | None = None
    chunk_cursor: int | None = None
    source_observation_ids: tuple[int, ...] = ()
    control_cycle_ns: int = 0


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    """Policy-facing action/state contract for a concrete robot."""

    action_dim: int
    state_dim: int
    joint_dof_per_arm: int
    joint_unit: str
    camera_roles: tuple[str, ...]
    supports_rtc: bool = True
    supports_interpolation: bool = True
    state_fields: tuple[VectorField, ...] = ()
    action_fields: tuple[VectorField, ...] = ()
    # These fields are deliberately appended so existing positional
    # ``ActionSpec(...)`` construction remains valid.
    schema_version: int = 1
    action_mode: str = "joint_position_target"
    arm_count: int | None = None
    gripper_indices: tuple[int, ...] = ()
    capabilities: Mapping[str, bool] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_dim <= 0 or self.state_dim <= 0:
            raise ValueError("ActionSpec dimensions must be positive")
        if self.joint_dof_per_arm < 0:
            raise ValueError("joint_dof_per_arm must be non-negative")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not self.action_mode.strip():
            raise ValueError("action_mode cannot be empty")
        if self.state_fields and len(self.state_fields) != self.state_dim:
            raise ValueError("state_fields length must match state_dim")
        if self.action_fields and len(self.action_fields) != self.action_dim:
            raise ValueError("action_fields length must match action_dim")
        derived_arm_count = (
            sum(field.semantics == "joint_position" for field in self.action_fields) // self.joint_dof_per_arm
            if self.action_fields and self.joint_dof_per_arm
            else 0
        )
        arm_count = derived_arm_count if self.arm_count is None else self.arm_count
        if arm_count < 0:
            raise ValueError("arm_count must be non-negative")
        derived_gripper_indices = tuple(
            index for index, field in enumerate(self.action_fields) if "gripper" in field.semantics.lower()
        )
        gripper_indices = self.gripper_indices or derived_gripper_indices
        if any(index < 0 or index >= self.action_dim for index in gripper_indices):
            raise ValueError("gripper_indices must refer to action dimensions")
        object.__setattr__(self, "arm_count", int(arm_count))
        object.__setattr__(self, "gripper_indices", tuple(int(index) for index in gripper_indices))
        object.__setattr__(self, "capabilities", {str(key): bool(value) for key, value in self.capabilities.items()})

    @property
    def state_field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.state_fields)

    @property
    def action_field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.action_fields)

    @property
    def state_names(self) -> tuple[str, ...]:
        """Backward-compatible explicit name alias used by persisted schemas."""
        return self.state_field_names

    @property
    def action_names(self) -> tuple[str, ...]:
        """Backward-compatible explicit name alias used by persisted schemas."""
        return self.action_field_names

    def to_dict(self) -> dict[str, Any]:
        """Serialize a complete contract without relying on dataclass internals.

        The redundant name lists make a persisted contract readable to tools
        that do not yet understand ``VectorField`` while the field descriptors
        remain the source of truth.
        """
        return {
            "schema_version": self.schema_version,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "joint_dof_per_arm": self.joint_dof_per_arm,
            "joint_unit": self.joint_unit,
            "camera_roles": list(self.camera_roles),
            "supports_rtc": self.supports_rtc,
            "supports_interpolation": self.supports_interpolation,
            "state_fields": [dataclasses.asdict(field) for field in self.state_fields],
            "action_fields": [dataclasses.asdict(field) for field in self.action_fields],
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "action_mode": self.action_mode,
            "arm_count": self.arm_count,
            "gripper_indices": list(self.gripper_indices),
            "capabilities": dict(self.capabilities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionSpec:
        """Load current and pre-schema-version ActionSpec snapshots."""
        joint_unit = str(payload.get("joint_unit", "unknown"))
        return cls(
            action_dim=int(payload["action_dim"]),
            state_dim=int(payload["state_dim"]),
            joint_dof_per_arm=int(payload.get("joint_dof_per_arm", 0)),
            joint_unit=joint_unit,
            camera_roles=tuple(str(item) for item in payload.get("camera_roles", ())),
            supports_rtc=bool(payload.get("supports_rtc", True)),
            supports_interpolation=bool(payload.get("supports_interpolation", True)),
            state_fields=_fields_from_payload(payload, "state", joint_unit),
            action_fields=_fields_from_payload(payload, "action", joint_unit),
            schema_version=int(payload.get("schema_version", 1)),
            action_mode=str(payload.get("action_mode", "joint_position_target")),
            arm_count=(int(payload["arm_count"]) if payload.get("arm_count") is not None else None),
            gripper_indices=tuple(int(item) for item in payload.get("gripper_indices", ())),
            capabilities={str(key): bool(value) for key, value in dict(payload.get("capabilities", {})).items()},
        )

    def validate_chunk(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise RuntimeError(f"Expected action chunk [T, >= {self.action_dim}], got {actions.shape}")
        return actions[:, : self.action_dim].copy()


def _fields_from_payload(payload: Mapping[str, Any], prefix: str, joint_unit: str) -> tuple[VectorField, ...]:
    fields = payload.get(f"{prefix}_fields", ())
    if fields:
        return tuple(VectorField(**dict(item)) for item in fields)
    # Legacy serialized schemas sometimes carried the readable names but not
    # full VectorField descriptors. Preserve those names instead of silently
    # erasing them during a Baseline round trip.
    return tuple(
        VectorField(str(name), joint_unit, "unspecified") for name in payload.get(f"{prefix}_names", ())
    )


@dataclasses.dataclass(frozen=True)
class RobotState:
    values: np.ndarray
    timestamp_monotonic: float
    timestamp_monotonic_ns: int = 0
    source_timestamp_ns: int | None = None
    health: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Keep the legacy float timestamp while making nanoseconds canonical."""
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", int(self.timestamp_monotonic * 1e9))


@dataclasses.dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    timestamp_monotonic: float
    camera_timestamp: float | None = None
    info: Mapping[str, Any] | None = None
    frame_id: int = 0
    timestamp_monotonic_ns: int = 0
    source_sequence: int | None = None
    capture_latency_ns: int | None = None

    def __post_init__(self) -> None:
        """Older camera implementations can still provide only the float timestamp."""
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", int(self.timestamp_monotonic * 1e9))


@dataclasses.dataclass(frozen=True)
class ObservationSnapshot:
    """An observation with timestamps retained outside the policy wire schema."""

    images: Mapping[str, CameraSample]
    image_masks: Mapping[str, np.bool_]
    state: RobotState
    prompt: str
    camera_params: Mapping[str, Mapping[str, Any] | None] | None = None
    captured_at_monotonic: float = dataclasses.field(default_factory=time.monotonic)
    observation_id: int = 0
    capture_started_ns: int = 0
    capture_finished_ns: int = 0
    state_timestamp_ns: int = 0
    camera_frame_ids: Mapping[str, int] = dataclasses.field(default_factory=dict)
    camera_timestamps_ns: Mapping[str, int] = dataclasses.field(default_factory=dict)
    max_camera_skew_ns: int = 0
    observation_age_ns: int = 0

    def __post_init__(self) -> None:
        """Fill derived timing metadata without changing the policy wire schema.

        ``max_camera_skew_ns`` is the range of camera timestamps in this
        snapshot. ``observation_age_ns`` is measured at capture completion
        against the oldest included state or camera source timestamp.
        """
        if self.observation_id <= 0:
            object.__setattr__(self, "observation_id", _next_observation_id())

        finished_ns = self.capture_finished_ns or int(self.captured_at_monotonic * 1e9)
        if finished_ns <= 0:
            finished_ns = time.monotonic_ns()
        started_ns = self.capture_started_ns or finished_ns
        state_timestamp_ns = self.state_timestamp_ns or self.state.timestamp_monotonic_ns
        camera_frame_ids = self.camera_frame_ids or {name: sample.frame_id for name, sample in self.images.items()}
        camera_timestamps_ns = self.camera_timestamps_ns or {
            name: sample.timestamp_monotonic_ns for name, sample in self.images.items()
        }
        timestamps = tuple(camera_timestamps_ns.values())
        max_camera_skew_ns = self.max_camera_skew_ns
        if max_camera_skew_ns == 0 and len(timestamps) > 1:
            max_camera_skew_ns = max(timestamps) - min(timestamps)
        source_timestamps = (state_timestamp_ns, *timestamps)
        observation_age_ns = self.observation_age_ns
        if observation_age_ns == 0 and source_timestamps:
            observation_age_ns = max(0, finished_ns - min(source_timestamps))

        object.__setattr__(self, "capture_started_ns", started_ns)
        object.__setattr__(self, "capture_finished_ns", finished_ns)
        object.__setattr__(self, "state_timestamp_ns", state_timestamp_ns)
        object.__setattr__(self, "camera_frame_ids", dict(camera_frame_ids))
        object.__setattr__(self, "camera_timestamps_ns", dict(camera_timestamps_ns))
        object.__setattr__(self, "max_camera_skew_ns", max_camera_skew_ns)
        object.__setattr__(self, "observation_age_ns", observation_age_ns)

    def to_policy_observation(self) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "images": {name: sample.image for name, sample in self.images.items()},
            "image_masks": dict(self.image_masks),
            "state": self.state.values,
            "prompt": self.prompt,
        }
        if self.camera_params is not None:
            observation["camera_params"] = dict(self.camera_params)
        return observation
