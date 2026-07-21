from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from mp_real.runtime.models import ActionSpec, VectorField
from mp_real.safety.models import RobotHealthSnapshot, RobotSafetyProfile, SafetyCheckResult


@dataclasses.dataclass(frozen=True)
class SafetyValidationReport:
    errors: tuple[SafetyCheckResult, ...] = ()
    warnings: tuple[SafetyCheckResult, ...] = ()
    unavailable_checks: tuple[SafetyCheckResult, ...] = ()
    passed_checks: tuple[SafetyCheckResult, ...] = ()
    safety_profile: dict[str, Any] | None = None
    safety_profile_hash: str | None = None
    safety_policy: str | None = None
    development_override: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_motion_safety(
    *,
    profile: RobotSafetyProfile,
    action_spec: ActionSpec,
    values: np.ndarray,
    state_fields: Sequence[VectorField],
    robot_name: str,
    robot_model: str | None = None,
    health: RobotHealthSnapshot | None = None,
    require_hardware_motion_enabled: bool = True,
    recorded_trajectory_context: bool = False,
    sample_indices: Sequence[int] | None = None,
) -> SafetyValidationReport:
    values = np.asarray(values, dtype=np.float32)
    errors: list[SafetyCheckResult] = []
    warnings: list[SafetyCheckResult] = []
    unavailable: list[SafetyCheckResult] = []
    passed: list[SafetyCheckResult] = []

    def add_error(
        code: str,
        message: str,
        dimension: int | None = None,
        source: str | None = None,
        sample_index: int | None = None,
    ) -> None:
        errors.append(SafetyCheckResult(code, message, dimension, "error", source, sample_index))

    def add_warning(
        code: str,
        message: str,
        dimension: int | None = None,
        source: str | None = None,
        sample_index: int | None = None,
    ) -> None:
        warnings.append(SafetyCheckResult(code, message, dimension, "warning", source, sample_index))

    def add_passed(
        code: str,
        message: str,
        dimension: int | None = None,
        source: str | None = None,
        sample_index: int | None = None,
    ) -> None:
        passed.append(SafetyCheckResult(code, message, dimension, "passed", source, sample_index))

    def add_unavailable(
        code: str,
        message: str,
        dimension: int | None = None,
        source: str | None = None,
        sample_index: int | None = None,
    ) -> None:
        check = SafetyCheckResult(code, message, dimension, "unavailable", source, sample_index)
        unavailable.append(check)
        if profile.blocks_unavailable(code):
            add_error(code, message, dimension, source, sample_index)

    if profile.robot_name != robot_name:
        add_error("profile_robot_mismatch", "safety profile robot name does not match connected robot")
    else:
        add_passed("profile_robot_match", "safety profile robot name matches connected robot")

    if robot_model is not None and profile.robot_model != robot_model:
        add_error("profile_model_mismatch", "safety profile robot model does not match connected robot")
    elif robot_model is not None:
        add_passed("profile_model_match", "safety profile robot model matches connected robot")

    expected_dim = len(state_fields)
    expected_shape_valid = (
        values.shape == (expected_dim,) if values.ndim == 1 else values.ndim == 2 and values.shape[1] == expected_dim
    )
    if not expected_shape_valid:
        add_error("state_dimension_mismatch", "target vector dimension does not match its declared fields")
    elif expected_dim not in {action_spec.state_dim, action_spec.action_dim}:
        add_error("state_schema_dimension_mismatch", "declared fields do not match ActionSpec dimensions")
    else:
        add_passed("state_schema_dimension", "target vector dimension matches ActionSpec")

    if not np.isfinite(values).all():
        add_error("state_not_finite", "target vector contains NaN or Inf")
    else:
        add_passed("finite_state", "target vector values are finite")

    if tuple(state_fields) not in {action_spec.state_fields, action_spec.action_fields}:
        add_error("state_schema_mismatch", "target fields do not match connected robot ActionSpec")
    else:
        add_passed("state_schema_match", "target fields match connected robot ActionSpec")

    joint_indices = tuple(index for index, field in enumerate(state_fields) if field.semantics == "joint_position")
    joint_names = tuple(state_fields[index].name for index in joint_indices)
    if joint_names != profile.joint_names:
        add_error("profile_joint_names_mismatch", "safety profile joint names do not match target state schema")
    else:
        add_passed("profile_joint_names_match", "safety profile joint names match target state schema")

    if not expected_shape_valid:
        pass
    elif profile.joint_min is None or profile.joint_max is None:
        add_unavailable(
            "joint_limit_validation_unavailable",
            "joint soft limits are not configured in the safety profile",
        )
    elif len(joint_indices) == len(profile.joint_min):
        lower = profile.joint_soft_min
        upper = profile.joint_soft_max
        assert lower is not None and upper is not None
        matrix = values.reshape(1, expected_dim) if values.ndim == 1 else values
        joint_values = matrix[:, list(joint_indices)] if joint_indices else np.empty((len(matrix), 0), dtype=np.float32)
        violations = np.argwhere((joint_values < lower) | (joint_values > upper))
        for row, offset in violations:
            dimension = joint_indices[int(offset)]
            sample_index = None if sample_indices is None else int(sample_indices[int(row)])
            add_error(
                "joint_limit_exceeded",
                f"target {state_fields[dimension].name!r} is outside configured soft joint limits",
                dimension,
                profile.parameter_sources.get("joint_limits"),
                sample_index,
            )
        if not len(violations):
            add_passed("joint_limits", "all joint targets are within configured soft limits")

    if expected_shape_valid and profile.gripper_indices:
        if profile.gripper_min is None or profile.gripper_max is None:
            add_unavailable("gripper_range", "gripper range is not configured in the safety profile")
        else:
            matrix = values.reshape(1, expected_dim) if values.ndim == 1 else values
            for offset, dimension in enumerate(profile.gripper_indices):
                gripper_values = matrix[:, dimension]
                violations = np.flatnonzero(
                    (gripper_values < float(profile.gripper_min[offset]))
                    | (gripper_values > float(profile.gripper_max[offset]))
                )
                for row in violations:
                    sample_index = None if sample_indices is None else int(sample_indices[int(row)])
                    add_error(
                        "gripper_range",
                        "gripper target is outside configured range",
                        dimension,
                        None,
                        sample_index,
                    )
            if not any(issue.code == "gripper_range" for issue in errors):
                add_passed("gripper_range", "gripper targets are within configured range")

    _validate_health(profile, health, errors, warnings, unavailable, passed, add_error, add_unavailable, add_passed)

    if profile.workspace_validation_capability:
        add_passed("workspace_validation_capability", "workspace validation capability is declared")
    else:
        add_unavailable("workspace_validation_unavailable", "workspace/collision validation is unavailable")

    if recorded_trajectory_context and profile.policy.value == "joint_space_recorded_trajectory_only":
        add_warning(
            "recorded_trajectory_only",
            "workspace validation is unavailable; execution is limited to recorded joint-space trajectory criteria",
        )

    if require_hardware_motion_enabled and not profile.hardware_motion_enabled:
        add_error(
            "hardware_motion_blocked",
            "safety profile has not enabled real hardware motion for this robot",
            source=profile.profile_source,
        )

    if profile.policy.value == "development_override":
        add_warning(
            "development_override",
            "development override is active; operator and reason must be displayed before execution",
        )

    return SafetyValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        unavailable_checks=tuple(unavailable),
        passed_checks=tuple(passed),
        safety_profile=profile.to_dict(),
        safety_profile_hash=profile.profile_hash,
        safety_policy=profile.policy.value,
        development_override=profile.development_override.to_dict(),
    )


def _validate_health(
    profile: RobotSafetyProfile,
    health: RobotHealthSnapshot | None,
    errors: list[SafetyCheckResult],
    warnings: list[SafetyCheckResult],
    unavailable: list[SafetyCheckResult],
    passed: list[SafetyCheckResult],
    add_error: Any,
    add_unavailable: Any,
    add_passed: Any,
) -> None:
    del errors, warnings, unavailable, passed
    if health is None:
        add_unavailable("health_validation_unavailable", "robot health snapshot is unavailable")
    else:
        if health.connected is False:
            add_error("robot_disconnected", "robot health reports disconnected")
        elif health.connected is True:
            add_passed("robot_connected", "robot health reports connected")
        else:
            add_unavailable("health_validation_unavailable", "robot connection state is unavailable")

        if health.enabled is False:
            add_error("robot_disabled", "robot health reports disabled")
        elif health.enabled is True:
            add_passed("robot_enabled", "robot health reports enabled")
        else:
            add_unavailable("health_validation_unavailable", "robot enabled state is unavailable")

        if health.error_codes:
            add_error("robot_error_code", f"robot health reports error code(s): {', '.join(health.error_codes)}")
        elif not profile.health_error_mapping:
            add_unavailable("health_validation_unavailable", "robot health error-code mapping is unavailable")
        elif health.healthy is True:
            add_passed("robot_error_code", "robot health reports no active error codes")
        elif health.healthy is False:
            add_error("robot_unhealthy", "robot health reports unhealthy status")
        else:
            add_unavailable("health_validation_unavailable", "robot error-code mapping is unavailable")

        if profile.communication_timeout_s is None:
            add_unavailable(
                "feedback_freshness_unavailable",
                "communication timeout is not configured in the safety profile",
            )
        elif health.last_feedback_age_s is None:
            add_unavailable("feedback_freshness_unavailable", "SDK feedback timestamp is unavailable")
        elif health.stale_feedback is True or health.last_feedback_age_s > profile.communication_timeout_s:
            add_error("stale_feedback", "robot feedback is stale")
        else:
            add_passed("feedback_freshness", "robot feedback is fresh")

    if profile.stop_capability is True:
        add_passed("stop_capability", "robot stop capability is available")
    elif profile.stop_capability is False:
        add_error("stop_motion_unsupported", "robot stop capability is unavailable")
    else:
        add_unavailable("stop_capability_unavailable", "robot stop capability is unavailable")
