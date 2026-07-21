from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.common.plan_integrity import canonical_hash, freeze_jsonish, readonly_optional_array
from mp_real.runtime.models import ActionSpec


class SafetyPolicy(enum.StrEnum):
    STRICT = "strict"
    JOINT_SPACE_RECORDED_TRAJECTORY_ONLY = "joint_space_recorded_trajectory_only"
    DEVELOPMENT_OVERRIDE = "development_override"


@dataclasses.dataclass(frozen=True)
class SafetyCheckResult:
    code: str
    message: str
    dimension: int | None = None
    severity: str = "error"
    source: str | None = None
    sample_index: int | None = None


@dataclasses.dataclass(frozen=True)
class DevelopmentOverride:
    enabled: bool = False
    operator: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and (not self.operator or not self.reason):
            raise ValueError("development override requires operator and reason")

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "operator": self.operator, "reason": self.reason}


@dataclasses.dataclass(frozen=True)
class ArmHealthSnapshot:
    name: str
    connected: bool | None
    enabled: bool | None
    healthy: bool | None
    error_codes: tuple[str, ...] = ()
    stale_feedback: bool | None = None
    last_feedback_age_s: float | None = None
    communication_status: str = "unknown"
    stop_capability: bool | None = None
    raw_status: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("arm health name cannot be empty")
        if self.last_feedback_age_s is not None and self.last_feedback_age_s < 0:
            raise ValueError("last feedback age cannot be negative")
        object.__setattr__(self, "error_codes", tuple(str(code) for code in self.error_codes))
        object.__setattr__(self, "raw_status", freeze_jsonish(self.raw_status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "connected": self.connected,
            "enabled": self.enabled,
            "healthy": self.healthy,
            "error_codes": list(self.error_codes),
            "stale_feedback": self.stale_feedback,
            "last_feedback_age_s": self.last_feedback_age_s,
            "communication_status": self.communication_status,
            "stop_capability": self.stop_capability,
            "raw_status": dict(self.raw_status),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArmHealthSnapshot:
        return cls(
            name=str(value.get("name", "unknown")),
            connected=_optional_bool(value.get("connected")),
            enabled=_optional_bool(value.get("enabled")),
            healthy=_optional_bool(value.get("healthy")),
            error_codes=tuple(str(code) for code in value.get("error_codes", ())),
            stale_feedback=_optional_bool(value.get("stale_feedback")),
            last_feedback_age_s=(
                None if value.get("last_feedback_age_s") is None else float(value.get("last_feedback_age_s"))
            ),
            communication_status=str(value.get("communication_status", "unknown")),
            stop_capability=_optional_bool(value.get("stop_capability")),
            raw_status=dict(value.get("raw_status", {})),
        )


@dataclasses.dataclass(frozen=True)
class RobotHealthSnapshot:
    robot_name: str
    connected: bool | None
    enabled: bool | None
    healthy: bool | None
    error_codes: tuple[str, ...] = ()
    stale_feedback: bool | None = None
    last_feedback_age_s: float | None = None
    communication_status: str = "unknown"
    stop_capability: bool | None = None
    raw_status: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    arms: Mapping[str, ArmHealthSnapshot] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.robot_name:
            raise ValueError("robot health name cannot be empty")
        if self.last_feedback_age_s is not None and self.last_feedback_age_s < 0:
            raise ValueError("last feedback age cannot be negative")
        arms = {str(name): arm for name, arm in self.arms.items()}
        object.__setattr__(self, "error_codes", tuple(str(code) for code in self.error_codes))
        object.__setattr__(self, "raw_status", freeze_jsonish(self.raw_status))
        object.__setattr__(self, "arms", freeze_jsonish(arms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "connected": self.connected,
            "enabled": self.enabled,
            "healthy": self.healthy,
            "error_codes": list(self.error_codes),
            "stale_feedback": self.stale_feedback,
            "last_feedback_age_s": self.last_feedback_age_s,
            "communication_status": self.communication_status,
            "stop_capability": self.stop_capability,
            "raw_status": dict(self.raw_status),
            "arms": {
                name: (arm.to_dict() if isinstance(arm, ArmHealthSnapshot) else dict(arm))
                for name, arm in self.arms.items()
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RobotHealthSnapshot:
        arms = {
            str(name): arm if isinstance(arm, ArmHealthSnapshot) else ArmHealthSnapshot.from_mapping(dict(arm))
            for name, arm in dict(value.get("arms", {})).items()
        }
        return cls(
            robot_name=str(value.get("robot_name", "unknown")),
            connected=_optional_bool(value.get("connected")),
            enabled=_optional_bool(value.get("enabled")),
            healthy=_optional_bool(value.get("healthy")),
            error_codes=tuple(str(code) for code in value.get("error_codes", ())),
            stale_feedback=_optional_bool(value.get("stale_feedback")),
            last_feedback_age_s=(
                None if value.get("last_feedback_age_s") is None else float(value.get("last_feedback_age_s"))
            ),
            communication_status=str(value.get("communication_status", "unknown")),
            stop_capability=_optional_bool(value.get("stop_capability")),
            raw_status=dict(value.get("raw_status", {})),
            arms=arms,
        )


@dataclasses.dataclass(frozen=True)
class RobotSafetyProfile:
    robot_name: str
    robot_model: str
    joint_names: tuple[str, ...]
    joint_min: np.ndarray | None = None
    joint_max: np.ndarray | None = None
    soft_margin: float = 0.0
    max_joint_velocity: np.ndarray | None = None
    max_joint_acceleration: np.ndarray | None = None
    max_single_step_delta: np.ndarray | None = None
    gripper_min: np.ndarray | None = None
    gripper_max: np.ndarray | None = None
    gripper_indices: tuple[int, ...] = ()
    gripper_semantics: tuple[str, ...] = ()
    communication_timeout_s: float | None = None
    tracking_error_threshold: float | None = None
    health_error_mapping: Mapping[str, str] = dataclasses.field(default_factory=dict)
    stop_capability: bool | None = None
    workspace_validation_capability: bool = False
    profile_source: str = "repository_configuration"
    profile_version: int = 1
    parameter_sources: Mapping[str, str] = dataclasses.field(default_factory=dict)
    policy: SafetyPolicy = SafetyPolicy.STRICT
    development_override: DevelopmentOverride = dataclasses.field(default_factory=DevelopmentOverride)
    strict_unavailable_checks: tuple[str, ...] = (
        "joint_limit_validation_unavailable",
        "health_validation_unavailable",
        "feedback_freshness_unavailable",
        "stop_capability_unavailable",
        "workspace_validation_unavailable",
    )
    hardware_motion_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.robot_name or not self.robot_model:
            raise ValueError("safety profile robot name/model cannot be empty")
        if self.profile_version <= 0:
            raise ValueError("safety profile version must be positive")
        if self.soft_margin < 0:
            raise ValueError("soft margin cannot be negative")
        if self.communication_timeout_s is not None and self.communication_timeout_s <= 0:
            raise ValueError("communication timeout must be positive")
        if self.tracking_error_threshold is not None and self.tracking_error_threshold <= 0:
            raise ValueError("tracking error threshold must be positive")
        policy = self.policy if isinstance(self.policy, SafetyPolicy) else SafetyPolicy(str(self.policy))
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        object.__setattr__(self, "gripper_indices", tuple(int(index) for index in self.gripper_indices))
        object.__setattr__(self, "gripper_semantics", tuple(str(item) for item in self.gripper_semantics))
        object.__setattr__(self, "health_error_mapping", freeze_jsonish(self.health_error_mapping))
        object.__setattr__(self, "parameter_sources", freeze_jsonish(self.parameter_sources))
        object.__setattr__(
            self,
            "strict_unavailable_checks",
            tuple(str(item) for item in self.strict_unavailable_checks),
        )
        object.__setattr__(self, "joint_min", self._optional_vector(self.joint_min, len(self.joint_names), "joint_min"))
        object.__setattr__(self, "joint_max", self._optional_vector(self.joint_max, len(self.joint_names), "joint_max"))
        if (self.joint_min is None) != (self.joint_max is None):
            raise ValueError("joint_min and joint_max must be configured together")
        if self.joint_min is not None and self.joint_max is not None:
            if np.any(self.joint_min + self.soft_margin >= self.joint_max - self.soft_margin):
                raise ValueError("joint limits and soft margin leave no valid range")
        object.__setattr__(
            self,
            "max_joint_velocity",
            self._optional_vector(self.max_joint_velocity, len(self.joint_names), "max_joint_velocity"),
        )
        object.__setattr__(
            self,
            "max_joint_acceleration",
            self._optional_vector(self.max_joint_acceleration, len(self.joint_names), "max_joint_acceleration"),
        )
        object.__setattr__(
            self,
            "max_single_step_delta",
            self._optional_vector(self.max_single_step_delta, len(self.joint_names), "max_single_step_delta"),
        )
        object.__setattr__(
            self,
            "gripper_min",
            self._optional_vector(self.gripper_min, len(self.gripper_indices), "gripper_min"),
        )
        object.__setattr__(
            self,
            "gripper_max",
            self._optional_vector(self.gripper_max, len(self.gripper_indices), "gripper_max"),
        )
        if (self.gripper_min is None) != (self.gripper_max is None):
            raise ValueError("gripper_min and gripper_max must be configured together")
        if (
            self.gripper_min is not None
            and self.gripper_max is not None
            and np.any(self.gripper_min >= self.gripper_max)
        ):
            raise ValueError("gripper min must be less than gripper max")
        if policy is SafetyPolicy.DEVELOPMENT_OVERRIDE and not self.development_override.enabled:
            raise ValueError("development override policy requires an enabled override")

    @staticmethod
    def _optional_vector(value: Any, size: int, label: str) -> np.ndarray | None:
        vector = readonly_optional_array(value)
        if vector is None:
            return None
        if vector.shape != (size,):
            raise ValueError(f"{label} must have shape ({size},)")
        if not np.isfinite(vector).all():
            raise ValueError(f"{label} must be finite")
        return vector

    @property
    def profile_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    @property
    def joint_soft_min(self) -> np.ndarray | None:
        if self.joint_min is None:
            return None
        result = np.asarray(self.joint_min + self.soft_margin, dtype=np.float32)
        result.setflags(write=False)
        return result

    @property
    def joint_soft_max(self) -> np.ndarray | None:
        if self.joint_max is None:
            return None
        result = np.asarray(self.joint_max - self.soft_margin, dtype=np.float32)
        result.setflags(write=False)
        return result

    def blocks_unavailable(self, code: str) -> bool:
        if self.policy is SafetyPolicy.DEVELOPMENT_OVERRIDE and self.development_override.enabled:
            return False
        if (
            self.policy is SafetyPolicy.JOINT_SPACE_RECORDED_TRAJECTORY_ONLY
            and code == "workspace_validation_unavailable"
        ):
            return False
        aliases = {
            "joint_limits": "joint_limit_validation_unavailable",
            "health": "health_validation_unavailable",
            "feedback_freshness": "feedback_freshness_unavailable",
            "stop_capability": "stop_capability_unavailable",
            "workspace_validation": "workspace_validation_unavailable",
        }
        return code in self.strict_unavailable_checks or aliases.get(code) in self.strict_unavailable_checks

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "robot_name": self.robot_name,
            "robot_model": self.robot_model,
            "joint_names": list(self.joint_names),
            "joint_min": None if self.joint_min is None else self.joint_min.tolist(),
            "joint_max": None if self.joint_max is None else self.joint_max.tolist(),
            "soft_margin": self.soft_margin,
            "max_joint_velocity": None if self.max_joint_velocity is None else self.max_joint_velocity.tolist(),
            "max_joint_acceleration": (
                None if self.max_joint_acceleration is None else self.max_joint_acceleration.tolist()
            ),
            "max_single_step_delta": (
                None if self.max_single_step_delta is None else self.max_single_step_delta.tolist()
            ),
            "gripper_min": None if self.gripper_min is None else self.gripper_min.tolist(),
            "gripper_max": None if self.gripper_max is None else self.gripper_max.tolist(),
            "gripper_indices": list(self.gripper_indices),
            "gripper_semantics": list(self.gripper_semantics),
            "communication_timeout_s": self.communication_timeout_s,
            "tracking_error_threshold": self.tracking_error_threshold,
            "health_error_mapping": dict(self.health_error_mapping),
            "stop_capability": self.stop_capability,
            "workspace_validation_capability": self.workspace_validation_capability,
            "profile_source": self.profile_source,
            "profile_version": self.profile_version,
            "parameter_sources": dict(self.parameter_sources),
            "policy": self.policy.value,
            "development_override": self.development_override.to_dict(),
            "strict_unavailable_checks": list(self.strict_unavailable_checks),
            "hardware_motion_enabled": self.hardware_motion_enabled,
        }
        if include_hash:
            payload["profile_hash"] = self.profile_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobotSafetyProfile:
        override_value = payload.get("development_override", {})
        override = (
            override_value
            if isinstance(override_value, DevelopmentOverride)
            else DevelopmentOverride(**dict(override_value or {}))
        )
        return cls(
            robot_name=str(payload["robot_name"]),
            robot_model=str(payload["robot_model"]),
            joint_names=tuple(str(name) for name in payload.get("joint_names", ())),
            joint_min=_array_or_none(payload.get("joint_min")),
            joint_max=_array_or_none(payload.get("joint_max")),
            soft_margin=float(payload.get("soft_margin", 0.0)),
            max_joint_velocity=_array_or_none(payload.get("max_joint_velocity")),
            max_joint_acceleration=_array_or_none(payload.get("max_joint_acceleration")),
            max_single_step_delta=_array_or_none(payload.get("max_single_step_delta")),
            gripper_min=_array_or_none(payload.get("gripper_min")),
            gripper_max=_array_or_none(payload.get("gripper_max")),
            gripper_indices=tuple(int(index) for index in payload.get("gripper_indices", ())),
            gripper_semantics=tuple(str(item) for item in payload.get("gripper_semantics", ())),
            communication_timeout_s=None
            if payload.get("communication_timeout_s") is None
            else float(payload.get("communication_timeout_s")),
            tracking_error_threshold=(
                None
                if payload.get("tracking_error_threshold") is None
                else float(payload.get("tracking_error_threshold"))
            ),
            health_error_mapping=dict(payload.get("health_error_mapping", {})),
            stop_capability=_optional_bool(payload.get("stop_capability")),
            workspace_validation_capability=bool(payload.get("workspace_validation_capability", False)),
            profile_source=str(payload.get("profile_source", "user_supplied_configuration")),
            profile_version=int(payload.get("profile_version", 1)),
            parameter_sources=dict(payload.get("parameter_sources", {})),
            policy=SafetyPolicy(str(payload.get("policy", SafetyPolicy.STRICT.value))),
            development_override=override,
            strict_unavailable_checks=tuple(
                str(item)
                for item in payload.get(
                    "strict_unavailable_checks",
                    (
                        "joint_limit_validation_unavailable",
                        "health_validation_unavailable",
                        "feedback_freshness_unavailable",
                        "stop_capability_unavailable",
                        "workspace_validation_unavailable",
                    ),
                )
            ),
            hardware_motion_enabled=bool(payload.get("hardware_motion_enabled", False)),
        )

    @classmethod
    def from_action_spec(
        cls,
        *,
        robot_name: str,
        robot_model: str,
        action_spec: ActionSpec,
        profile_source: str = "repository_configuration",
        profile_version: int = 1,
        policy: SafetyPolicy = SafetyPolicy.STRICT,
        stop_capability: bool | None = None,
        hardware_motion_enabled: bool = False,
        development_override: DevelopmentOverride | None = None,
        parameter_sources: Mapping[str, str] | None = None,
    ) -> RobotSafetyProfile:
        joint_names = tuple(field.name for field in action_spec.state_fields if field.semantics == "joint_position")
        gripper_indices = tuple(
            index for index, field in enumerate(action_spec.state_fields) if "gripper" in field.semantics.lower()
        )
        gripper_semantics = tuple(action_spec.state_fields[index].semantics for index in gripper_indices)
        gripper_min = np.zeros(len(gripper_indices), dtype=np.float32) if gripper_indices else None
        gripper_max = np.ones(len(gripper_indices), dtype=np.float32) if gripper_indices else None
        return cls(
            robot_name=robot_name,
            robot_model=robot_model,
            joint_names=joint_names,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            gripper_indices=gripper_indices,
            gripper_semantics=gripper_semantics,
            stop_capability=stop_capability,
            profile_source=profile_source,
            profile_version=profile_version,
            parameter_sources={
                "joint_names": "repository_configuration",
                "gripper_range": "repository_configuration",
                **dict(parameter_sources or {}),
            },
            policy=policy,
            development_override=development_override or DevelopmentOverride(),
            hardware_motion_enabled=hardware_motion_enabled,
        )


def load_robot_safety_profile(path: Path | str) -> RobotSafetyProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RobotSafetyProfile.from_dict(payload)


def profile_for_robot(robot: Any) -> RobotSafetyProfile | None:
    getter = getattr(robot, "get_safety_profile", None)
    if callable(getter):
        return getter()
    profile = getattr(robot, "safety_profile", None)
    return profile if isinstance(profile, RobotSafetyProfile) else None


def health_from_state_mapping(value: Any) -> RobotHealthSnapshot | None:
    if isinstance(value, RobotHealthSnapshot):
        return value
    if isinstance(value, Mapping):
        try:
            return RobotHealthSnapshot.from_mapping(value)
        except (TypeError, ValueError):
            return None
    return None


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def aggregate_bool(values: Sequence[bool | None]) -> bool | None:
    if any(value is False for value in values):
        return False
    if values and all(value is True for value in values):
        return True
    return None
