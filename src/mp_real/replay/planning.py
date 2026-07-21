"""Offline-only LeRobot v2.1 replay planning and validation."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from mp_real.data.models import EpisodeStatus, RecordedEpisodeSource
from mp_real.replay.models import (
    ReplayActionSource,
    ReplayConstraints,
    ReplayMode,
    ReplayPlan,
    ReplayPlanningResult,
    ReplaySafetyIssue,
    ReplaySafetyReport,
    ReplayStep,
    ReplayTimingMode,
    StateTrajectorySource,
    build_plan_hash,
    new_plan_identity,
)
from mp_real.runtime.models import ActionSpec
from mp_real.safety.models import RobotSafetyProfile
from mp_real.safety.validation import validate_motion_safety


class ReplayPlanner:
    """Build a plan without creating a robot, policy client, or camera."""

    def __init__(self, source: RecordedEpisodeSource) -> None:
        self._source = source

    def plan(
        self,
        *,
        robot_name: str,
        target_action_spec: ActionSpec,
        dataset_id: str | None = None,
        episode_index: int,
        start_sample: int | None = None,
        end_sample: int | None = None,
        mode: ReplayMode = ReplayMode.COMMAND_REPLAY,
        timing_mode: ReplayTimingMode = ReplayTimingMode.RECORDED_TIMESTAMPS,
        fps: float | None = None,
        speed_scale: float = 0.1,
        constraints: ReplayConstraints | None = None,
        generation_id: int = 1,
        resource_owner_id: str | None = None,
        safety_profile: RobotSafetyProfile | None = None,
    ) -> ReplayPlanningResult:
        constraints = constraints or ReplayConstraints()
        errors: list[ReplaySafetyIssue] = []
        warnings: list[ReplaySafetyIssue] = []
        converted: list[str] = []
        skipped: list[str] = []
        metadata = self._source.get_dataset_metadata()
        source_spec = self._source.get_action_spec()
        try:
            episode = self._source.get_episode_metadata(episode_index)
        except (IndexError, KeyError, ValueError) as exc:
            report = ReplaySafetyReport(errors=(ReplaySafetyIssue("episode_missing", str(exc)),))
            return ReplayPlanningResult(None, report)

        dataset_id = dataset_id or str(metadata.root)
        info = metadata.info
        source_robot = str(info.get("robot_type", ""))
        if metadata.status is not EpisodeStatus.COMPLETE or episode.status is not EpisodeStatus.COMPLETE:
            errors.append(ReplaySafetyIssue("episode_incomplete", "source dataset and episode must be complete"))
        if str(info.get("codebase_version", "")) != "v2.1":
            errors.append(ReplaySafetyIssue("lerobot_version", "replay requires LeRobot v2.1 data"))
        if source_robot != robot_name:
            errors.append(
                ReplaySafetyIssue(
                    "robot_name_mismatch",
                    f"recorded robot {source_robot!r} does not match target {robot_name!r}",
                )
            )
        if not 0 < speed_scale <= 1.0:
            errors.append(ReplaySafetyIssue("speed_scale", "speed scale must be in (0, 1]"))
        if timing_mode is ReplayTimingMode.FIXED_FPS and (fps is None or fps <= 0):
            errors.append(ReplaySafetyIssue("fixed_fps", "fixed timing mode requires a positive FPS"))

        start = 0 if start_sample is None else int(start_sample)
        end = episode.length - 1 if end_sample is None else int(end_sample)
        if start < 0 or end < start or end >= episode.length:
            errors.append(
                ReplaySafetyIssue(
                    "sample_range",
                    f"replay range [{start}, {end}] is outside episode length {episode.length}",
                )
            )
        source_contract = self._source_contract(mode, source_spec, info, target_action_spec, errors, warnings)
        if errors:
            return ReplayPlanningResult(
                None,
                ReplaySafetyReport(errors=tuple(errors), warnings=tuple(warnings), source_dataset_id=dataset_id),
            )

        samples = []
        previous_timestamp: float | None = None
        previous_frame: int | None = None
        for sample_index in range(start, end + 1):
            sample = self._source.get_sample(episode_index, sample_index)
            state = np.asarray(sample.state, dtype=np.float32)
            target = np.asarray(sample.action if mode is ReplayMode.COMMAND_REPLAY else sample.state, dtype=np.float32)
            if state.shape != (source_spec.state_dim,):
                errors.append(ReplaySafetyIssue("state_dimension", "source state dimension is invalid", sample_index))
            expected_dim = source_spec.action_dim if mode is ReplayMode.COMMAND_REPLAY else source_spec.state_dim
            if target.shape != (expected_dim,):
                errors.append(
                    ReplaySafetyIssue("target_dimension", "source replay target dimension is invalid", sample_index)
                )
            if not np.isfinite(state).all() or not np.isfinite(target).all():
                errors.append(ReplaySafetyIssue("not_finite", "source contains NaN or Inf", sample_index))
            timestamp = float(sample.timestamp)
            if not np.isfinite(timestamp):
                errors.append(ReplaySafetyIssue("timestamp_not_finite", "source timestamp is not finite", sample_index))
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                errors.append(
                    ReplaySafetyIssue(
                        "timestamp_not_monotonic", "source timestamps must strictly increase", sample_index
                    )
                )
            if previous_frame is not None and sample.frame_index != previous_frame + 1:
                errors.append(
                    ReplaySafetyIssue("frame_index_gap", "source frame_index is not contiguous", sample_index)
                )
            previous_timestamp = timestamp
            previous_frame = sample.frame_index
            samples.append((sample_index, sample.frame_index, timestamp, target.copy(), state.copy()))

        if errors:
            return ReplayPlanningResult(
                None,
                ReplaySafetyReport(errors=tuple(errors), warnings=tuple(warnings), source_dataset_id=dataset_id),
            )

        offsets_s, timing_warnings = self._timing_offsets(
            [sample[2] for sample in samples], timing_mode, fps, speed_scale, constraints
        )
        warnings.extend(timing_warnings)
        values = np.vstack([sample[3] for sample in samples])
        if values.shape[1] != target_action_spec.action_dim and mode is ReplayMode.COMMAND_REPLAY:
            errors.append(ReplaySafetyIssue("action_dimension_mismatch", "target action dimension differs from robot"))
        if values.shape[1] != target_action_spec.state_dim and mode is ReplayMode.STATE_TRAJECTORY_FOLLOWING:
            errors.append(ReplaySafetyIssue("state_dimension_mismatch", "target state dimension differs from robot"))
        if constraints.lower_limits is None:
            warnings.append(
                ReplaySafetyIssue(
                    "joint_limits_not_provided",
                    "generic trajectory limits are absent; connected vendor validation remains required",
                )
            )
        elif len(constraints.lower_limits) != values.shape[1]:
            errors.append(
                ReplaySafetyIssue("joint_limit_dimension", "configured joint limits do not match replay target")
            )
        else:
            lower = np.asarray(constraints.lower_limits, dtype=np.float32)
            upper = np.asarray(constraints.upper_limits, dtype=np.float32)
            violations = np.argwhere((values < lower) | (values > upper))
            for sample_offset, dimension in violations:
                errors.append(
                    ReplaySafetyIssue(
                        "joint_limit_exceeded",
                        "replay target is outside configured joint limits",
                        int(samples[int(sample_offset)][0]),
                        int(dimension),
                    )
                )
        max_delta, max_velocity, max_acceleration = self._kinematics(values, offsets_s)
        if max_delta > constraints.max_step:
            errors.append(
                ReplaySafetyIssue("max_step", f"maximum step {max_delta:.6f} exceeds {constraints.max_step:.6f}")
            )
        if max_velocity > constraints.max_velocity:
            errors.append(
                ReplaySafetyIssue(
                    "max_velocity", f"maximum velocity {max_velocity:.6f} exceeds {constraints.max_velocity:.6f}"
                )
            )
        if max_acceleration > constraints.max_acceleration:
            errors.append(
                ReplaySafetyIssue(
                    "max_acceleration",
                    f"maximum acceleration {max_acceleration:.6f} exceeds {constraints.max_acceleration:.6f}",
                )
            )
        safety_unavailable: list[ReplaySafetyIssue] = []
        safety_passed: list[ReplaySafetyIssue] = []
        safety_policy: str | None = None
        safety_profile_hash: str | None = None
        safety_profile_payload: Mapping[str, Any] = {}
        development_override: Mapping[str, Any] = {}
        if safety_profile is not None:
            target_fields = (
                target_action_spec.action_fields
                if mode is ReplayMode.COMMAND_REPLAY
                else target_action_spec.state_fields
            )
            safety_report = validate_motion_safety(
                profile=safety_profile,
                action_spec=target_action_spec,
                values=values,
                state_fields=target_fields,
                robot_name=robot_name,
                robot_model=safety_profile.robot_model,
                health=None,
                require_hardware_motion_enabled=True,
                recorded_trajectory_context=True,
                sample_indices=[sample[0] for sample in samples],
            )
            errors.extend(_safety_to_replay(issue) for issue in safety_report.errors)
            warnings.extend(_safety_to_replay(issue) for issue in safety_report.warnings)
            safety_unavailable.extend(_safety_to_replay(issue) for issue in safety_report.unavailable_checks)
            safety_passed.extend(_safety_to_replay(issue) for issue in safety_report.passed_checks)
            safety_policy = safety_report.safety_policy
            safety_profile_hash = safety_report.safety_profile_hash
            safety_profile_payload = safety_report.safety_profile or {}
            development_override = safety_report.development_override or {}
        dataset_hash = build_replay_source_hash(dataset_id, info, episode_index, start, end, samples, values)
        plan_id, session_id = new_plan_identity()
        plan = None
        if not errors:
            steps = tuple(
                ReplayStep(sample_index, frame_index, timestamp, int(offset * 1e9), target, state)
                for (sample_index, frame_index, timestamp, target, state), offset in zip(
                    samples, offsets_s, strict=True
                )
            )
            plan = ReplayPlan(
                plan_id=plan_id,
                session_id=session_id,
                generation_id=generation_id,
                dataset_id=dataset_id,
                dataset_hash=dataset_hash,
                episode_index=episode_index,
                start_sample=start,
                end_sample=end,
                robot_name=robot_name,
                mode=mode,
                timing_mode=timing_mode,
                speed_scale=speed_scale,
                action_spec=target_action_spec,
                source=source_contract,
                steps=steps,
                constraints=constraints,
                created_at_monotonic_ns=time.monotonic_ns(),
                resource_owner_id=resource_owner_id,
                safety_profile_hash=safety_profile_hash,
                safety_policy=safety_policy,
            )
        report = ReplaySafetyReport(
            errors=tuple(errors),
            warnings=tuple(warnings),
            unavailable_checks=tuple(safety_unavailable),
            passed_checks=tuple(safety_passed),
            converted_fields=tuple(converted),
            skipped_fields=tuple(skipped),
            maximum_observed_delta=max_delta,
            maximum_observed_velocity=max_velocity,
            maximum_observed_acceleration=max_acceleration,
            expected_duration_s=offsets_s[-1] if offsets_s else 0.0,
            start_state=samples[0][4].copy() if samples else None,
            end_state=samples[-1][4].copy() if samples else None,
            plan_hash=plan.plan_hash if plan is not None else None,
            source_dataset_id=dataset_id,
            source_dataset_hash=dataset_hash,
            safety_policy=safety_policy,
            safety_profile_hash=safety_profile_hash,
            safety_profile=safety_profile_payload,
            development_override=development_override,
        )
        return ReplayPlanningResult(plan, report)

    @staticmethod
    def _source_contract(
        mode: ReplayMode,
        source_spec: ActionSpec,
        info: Mapping[str, Any],
        target_spec: ActionSpec,
        errors: list[ReplaySafetyIssue],
        warnings: list[ReplaySafetyIssue],
    ) -> ReplayActionSource | StateTrajectorySource:
        if mode is ReplayMode.COMMAND_REPLAY:
            extension = info.get("mp_real", {})
            replay = extension.get("replay", {}) if isinstance(extension, Mapping) else {}
            action_source = str(replay.get("action_source", extension.get("action_source", "")))
            action_mode = str(replay.get("action_mode", extension.get("action_mode", "")))
            if action_source not in {"executed_action", "standard_action"}:
                errors.append(
                    ReplaySafetyIssue(
                        "action_source_unknown", "command replay requires an explicitly declared standard action source"
                    )
                )
            if action_mode != "joint_position_target":
                errors.append(
                    ReplaySafetyIssue("action_mode", "command replay requires joint_position_target action mode")
                )
            ReplayPlanner._require_action_contract(source_spec, target_spec, errors)
            arm_count, gripper_indices, gripper_semantics = ReplayPlanner._layout_metadata(source_spec, action=True)
            if arm_count <= 0:
                errors.append(ReplaySafetyIssue("arm_count", "recorded action layout has no declared arms"))
                arm_count = 1
            return ReplayActionSource(
                action_source or "unknown",
                action_mode or "unknown",
                source_spec,
                arm_count,
                gripper_indices,
                gripper_semantics,
            )
        if source_spec.state_dim != target_spec.state_dim or source_spec.action_dim != target_spec.action_dim:
            errors.append(
                ReplaySafetyIssue(
                    "state_action_layout", "state following requires matching robot state/action dimensions"
                )
            )
        ReplayPlanner._require_state_contract(source_spec, target_spec, errors)
        warnings.append(
            ReplaySafetyIssue("state_following", "state trajectory following is not command replay and may be smoothed")
        )
        arm_count, gripper_indices, gripper_semantics = ReplayPlanner._layout_metadata(source_spec, action=False)
        if arm_count <= 0:
            errors.append(ReplaySafetyIssue("arm_count", "recorded state layout has no declared arms"))
            arm_count = 1
        return StateTrajectorySource(source_spec, arm_count, gripper_indices, gripper_semantics)

    @staticmethod
    def _require_action_contract(source: ActionSpec, target: ActionSpec, errors: list[ReplaySafetyIssue]) -> None:
        if source.action_dim != target.action_dim:
            errors.append(ReplaySafetyIssue("action_dimension_mismatch", "recorded and robot action dimensions differ"))
            return
        if source.action_fields != target.action_fields:
            errors.append(ReplaySafetyIssue("action_spec_mismatch", "action names, units, or semantics differ"))
        if source.joint_unit != target.joint_unit:
            errors.append(ReplaySafetyIssue("joint_unit_mismatch", "recorded and robot joint units differ"))
        ReplayPlanner._require_layout_metadata(source, target, errors, action=True)

    @staticmethod
    def _require_state_contract(source: ActionSpec, target: ActionSpec, errors: list[ReplaySafetyIssue]) -> None:
        if source.state_dim != target.state_dim:
            errors.append(ReplaySafetyIssue("state_dimension_mismatch", "recorded and robot state dimensions differ"))
            return
        if source.state_fields != target.state_fields:
            errors.append(ReplaySafetyIssue("state_spec_mismatch", "state names, units, or semantics differ"))
        if source.joint_unit != target.joint_unit:
            errors.append(ReplaySafetyIssue("joint_unit_mismatch", "recorded and robot joint units differ"))
        ReplayPlanner._require_layout_metadata(source, target, errors, action=False)

    @staticmethod
    def _layout_metadata(spec: ActionSpec, *, action: bool) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
        fields = spec.action_fields if action else spec.state_fields
        joint_count = sum(field.semantics == "joint_position" for field in fields)
        arm_count = joint_count // spec.joint_dof_per_arm if spec.joint_dof_per_arm else 0
        gripper_indices = tuple(index for index, field in enumerate(fields) if "gripper" in field.semantics)
        return arm_count, gripper_indices, tuple(fields[index].semantics for index in gripper_indices)

    @staticmethod
    def _require_layout_metadata(
        source: ActionSpec, target: ActionSpec, errors: list[ReplaySafetyIssue], *, action: bool
    ) -> None:
        source_layout = ReplayPlanner._layout_metadata(source, action=action)
        target_layout = ReplayPlanner._layout_metadata(target, action=action)
        if source_layout[0] != target_layout[0]:
            errors.append(ReplaySafetyIssue("arm_count_mismatch", "recorded and robot arm counts differ"))
        if source_layout[1:] != target_layout[1:]:
            errors.append(ReplaySafetyIssue("gripper_layout_mismatch", "gripper indices or semantics differ"))

    @staticmethod
    def _timing_offsets(
        timestamps: list[float],
        mode: ReplayTimingMode,
        fps: float | None,
        speed_scale: float,
        constraints: ReplayConstraints,
    ) -> tuple[list[float], list[ReplaySafetyIssue]]:
        offsets = [0.0]
        warnings: list[ReplaySafetyIssue] = []
        if len(timestamps) < 2:
            return offsets, warnings
        for index in range(1, len(timestamps)):
            interval = (
                1.0 / float(fps) if mode is ReplayTimingMode.FIXED_FPS else timestamps[index] - timestamps[index - 1]
            )
            if interval < constraints.min_interval_s:
                warnings.append(ReplaySafetyIssue("timing_clamped_low", "source interval clamped to minimum", index))
                interval = constraints.min_interval_s
            if interval > constraints.max_interval_s:
                warnings.append(ReplaySafetyIssue("timing_clamped_high", "source interval clamped to maximum", index))
                interval = constraints.max_interval_s
            offsets.append(offsets[-1] + interval / speed_scale)
        return offsets, warnings

    @staticmethod
    def _kinematics(values: np.ndarray, offsets_s: list[float]) -> tuple[float, float, float]:
        if len(values) < 2:
            return 0.0, 0.0, 0.0
        deltas = np.abs(np.diff(values, axis=0))
        maximum_delta = float(np.max(deltas))
        intervals = np.diff(np.asarray(offsets_s, dtype=np.float64))
        velocity = deltas / intervals[:, None]
        maximum_velocity = float(np.max(velocity))
        if len(velocity) < 2:
            return maximum_delta, maximum_velocity, 0.0
        acceleration = np.abs(np.diff(velocity, axis=0)) / intervals[1:, None]
        return maximum_delta, maximum_velocity, float(np.max(acceleration))


def build_replay_source_hash(
    dataset_id: str,
    info: Mapping[str, Any],
    episode_index: int,
    start_sample: int,
    end_sample: int,
    samples: list[tuple[int, int, float, np.ndarray, np.ndarray]],
    targets: np.ndarray,
) -> str:
    """Hash the source data that affects replay motion and expected tracking."""

    return build_plan_hash(
        {
            "dataset": dataset_id,
            "info": info,
            "episode": episode_index,
            "range": (start_sample, end_sample),
            "timestamps": [sample[2] for sample in samples],
            "frame_indices": [sample[1] for sample in samples],
            "targets": targets,
            "expected_states": np.vstack([sample[4] for sample in samples]) if samples else np.empty((0, 0)),
        }
    )


def _safety_to_replay(issue: Any) -> ReplaySafetyIssue:
    return ReplaySafetyIssue(
        code=str(issue.code),
        message=str(issue.message),
        sample_index=issue.sample_index,
        dimension=issue.dimension,
        severity=str(issue.severity),
        source=issue.source,
    )
