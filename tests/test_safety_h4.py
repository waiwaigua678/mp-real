from __future__ import annotations

import dataclasses
import time
import unittest

import numpy as np

from mp_real.data.models import FakeRecordedEpisodeSource, RecordedSample
from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMoveResult,
    PoseValidationReport,
    RecordedPoseTarget,
)
from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import ReplayConstraints, ReplayPlanStaleError
from mp_real.replay.planning import ReplayPlanner
from mp_real.robots.piper.infer import Args as PiperArgs
from mp_real.robots.piper.infer import PiperArm, PiperRobot
from mp_real.robots.rm2.infer import Args as Rm2Args
from mp_real.robots.rm2.infer import MockArm, Rm2Robot
from mp_real.runtime.models import ActionSpec, RobotState, VectorField
from mp_real.safety.models import DevelopmentOverride, RobotSafetyProfile, SafetyPolicy


class _PiperFeedback:
    def __init__(self, values: np.ndarray) -> None:
        self.msg = np.asarray(values, dtype=np.float32)
        self.timestamp_monotonic_ns = time.monotonic_ns()


class _PiperStatus:
    enabled = True
    healthy = True
    error_code = 0


class _PiperGripperStatus:
    class _Msg:
        mode = "angle"
        value = 50.0

    msg = _Msg()


class _FakePiperSdkArm:
    def __init__(self, joints: np.ndarray | None = None) -> None:
        self.joints = np.zeros(6, dtype=np.float32) if joints is None else np.asarray(joints, dtype=np.float32)

    def get_joint_angles(self) -> _PiperFeedback:
        return _PiperFeedback(self.joints)

    def get_arm_status(self) -> _PiperStatus:
        return _PiperStatus()

    def stop(self) -> None:
        return None


class _FakePiperGripper:
    def get_gripper_status(self) -> _PiperGripperStatus:
        return _PiperGripperStatus()


def _target(spec: ActionSpec, *, robot_name: str, values: np.ndarray | None = None) -> RecordedPoseTarget:
    return RecordedPoseTarget(
        dataset_id="safe-dataset",
        episode_index=0,
        sample_index=0,
        robot_name=robot_name,
        state_schema=spec.state_field_names,
        state_values=np.asarray(values if values is not None else np.zeros(spec.state_dim), dtype=np.float32),
        state_fields=spec.state_fields,
        joint_unit=spec.joint_unit,
        timestamp=0.0,
        source_metadata={"dataset_status": "complete"},
        action_spec=spec,
    )


def _profile(
    robot_name: str,
    spec: ActionSpec,
    *,
    policy: SafetyPolicy = SafetyPolicy.STRICT,
    hardware_motion_enabled: bool = True,
    stop_capability: bool | None = True,
    workspace_validation_capability: bool = True,
) -> RobotSafetyProfile:
    joint_count = sum(field.semantics == "joint_position" for field in spec.state_fields)
    return dataclasses.replace(
        RobotSafetyProfile.from_action_spec(
            robot_name=robot_name,
            robot_model=robot_name,
            action_spec=spec,
            stop_capability=stop_capability,
            policy=policy,
            development_override=DevelopmentOverride(
                enabled=policy is SafetyPolicy.DEVELOPMENT_OVERRIDE,
                operator="test-operator" if policy is SafetyPolicy.DEVELOPMENT_OVERRIDE else None,
                reason="test override" if policy is SafetyPolicy.DEVELOPMENT_OVERRIDE else None,
            ),
            hardware_motion_enabled=hardware_motion_enabled,
        ),
        joint_min=np.full(joint_count, -1.0, dtype=np.float32),
        joint_max=np.full(joint_count, 1.0, dtype=np.float32),
        communication_timeout_s=1.0,
        health_error_mapping={"E42": "fake SDK error"},
        workspace_validation_capability=workspace_validation_capability,
        parameter_sources={
            "joint_limits": "user_supplied_configuration",
            "communication_timeout": "user_supplied_configuration",
            "stop_capability": "vendor_sdk",
        },
    )


def _piper_robot(*, profile: RobotSafetyProfile | None = None) -> PiperRobot:
    arm = _FakePiperSdkArm()
    return PiperRobot(
        PiperArm("left", arm, _FakePiperGripper(), 0.5),
        PiperArm("right", _FakePiperSdkArm(), _FakePiperGripper(), 0.5),
        PiperArgs(),
        safety_profile=profile,
    )


class SafetyProfileValidationTests(unittest.TestCase):
    def test_piper_pose_validation_distinguishes_unavailable_from_passed(self) -> None:
        robot = _piper_robot()
        report = robot.validate_pose_target(_target(robot.action_spec, robot_name="piper"))
        issue_codes = {issue.code for issue in report.issues}
        unavailable_codes = {issue.code for issue in report.unavailable_checks}
        passed_codes = {issue.code for issue in report.passed_checks}
        self.assertIn("joint_limit_validation_unavailable", issue_codes)
        self.assertIn("feedback_freshness_unavailable", issue_codes)
        self.assertIn("workspace_validation_unavailable", issue_codes)
        self.assertIn("joint_limit_validation_unavailable", unavailable_codes)
        self.assertNotIn("joint_limit_validation_unavailable", passed_codes)
        self.assertFalse(report.valid)

    def test_profiled_piper_pose_can_pass_without_unconditional_critical_issues(self) -> None:
        robot = _piper_robot()
        profile = _profile(
            "piper",
            robot.action_spec,
            policy=SafetyPolicy.JOINT_SPACE_RECORDED_TRAJECTORY_ONLY,
            workspace_validation_capability=False,
        )
        robot.safety_profile = profile
        values = np.zeros(robot.action_spec.state_dim, dtype=np.float32)
        values[list(robot.action_spec.gripper_indices)] = 0.5
        report = robot.validate_pose_target(_target(robot.action_spec, robot_name="piper", values=values))
        self.assertTrue(report.valid, report.issues)
        self.assertIn("workspace_validation_unavailable", {issue.code for issue in report.unavailable_checks})
        self.assertIn("joint_limits", {issue.code for issue in report.passed_checks})
        self.assertEqual(report.safety_profile_hash, profile.profile_hash)

    def test_strict_recorded_and_development_policies_are_distinct(self) -> None:
        robot = _piper_robot()
        strict = _profile(
            "piper",
            robot.action_spec,
            policy=SafetyPolicy.STRICT,
            workspace_validation_capability=False,
        )
        robot.safety_profile = strict
        strict_report = robot.validate_pose_target(_target(robot.action_spec, robot_name="piper"))
        self.assertFalse(strict_report.valid)
        self.assertIn("workspace_validation_unavailable", {issue.code for issue in strict_report.issues})

        override = dataclasses.replace(
            RobotSafetyProfile.from_action_spec(
                robot_name="piper",
                robot_model="piper",
                action_spec=robot.action_spec,
                stop_capability=True,
                policy=SafetyPolicy.DEVELOPMENT_OVERRIDE,
                development_override=DevelopmentOverride(True, "test-operator", "exercise unavailable checks"),
                hardware_motion_enabled=True,
            ),
            communication_timeout_s=1.0,
            workspace_validation_capability=False,
        )
        robot.safety_profile = override
        override_report = robot.validate_pose_target(_target(robot.action_spec, robot_name="piper"))
        self.assertTrue(override_report.valid, override_report.issues)
        self.assertTrue(override_report.development_override["enabled"])
        self.assertIn("development_override", {issue.code for issue in override_report.warnings})

    def test_rm2_health_errors_are_reported_per_arm(self) -> None:
        args = Rm2Args(robot_backend="mock")
        healthy = Rm2Robot(
            MockArm("left", args.joint_dof, 0.5, np.zeros(args.joint_dof, dtype=np.float32)),
            MockArm("right", args.joint_dof, 0.5, np.zeros(args.joint_dof, dtype=np.float32)),
            args,
        )
        profile = _profile("rm2", healthy.action_spec)
        healthy.safety_profile = profile
        target = _target(healthy.action_spec, robot_name="rm2")
        self.assertTrue(healthy.validate_pose_target(target).valid)
        health = healthy.read_state().health
        self.assertEqual(set(health["arms"]), {"left", "right"})

        cases = [
            ("disconnected", {"connected": False}, "robot_disconnected"),
            ("disabled", {"enabled": False}, "robot_disabled"),
            ("stale", {"last_feedback_age_s": 2.0}, "stale_feedback"),
            ("error", {"error_codes": ("E42",)}, "robot_error_code"),
        ]
        for label, changes, expected_code in cases:
            with self.subTest(label=label):
                left = MockArm("left", args.joint_dof, 0.5, np.zeros(args.joint_dof, dtype=np.float32))
                for key, value in changes.items():
                    setattr(left, key, value)
                robot = Rm2Robot(
                    left,
                    MockArm("right", args.joint_dof, 0.5, np.zeros(args.joint_dof, dtype=np.float32)),
                    args,
                    safety_profile=profile,
                )
                report = robot.validate_pose_target(target)
                self.assertIn(expected_code, {issue.code for issue in report.issues})
                self.assertFalse(report.valid)

    def test_joint_gripper_stop_and_profile_mismatch_are_enforced(self) -> None:
        args = Rm2Args(robot_backend="mock")
        robot = Rm2Robot(
            MockArm("left", args.joint_dof, 0.5, np.zeros(args.joint_dof, dtype=np.float32)),
            MockArm("right", args.joint_dof, 0.5, np.zeros(args.joint_dof, dtype=np.float32)),
            args,
        )
        profile = _profile("rm2", robot.action_spec)
        robot.safety_profile = profile
        below = np.zeros(robot.action_spec.state_dim, dtype=np.float32)
        below[0] = -2.0
        report = robot.validate_pose_target(_target(robot.action_spec, robot_name="rm2", values=below))
        self.assertIn(
            "joint_limit_exceeded",
            {issue.code for issue in report.issues},
        )
        bad_gripper = np.zeros(robot.action_spec.state_dim, dtype=np.float32)
        bad_gripper[list(robot.action_spec.gripper_indices)[0]] = 2.0
        report = robot.validate_pose_target(_target(robot.action_spec, robot_name="rm2", values=bad_gripper))
        self.assertIn(
            "gripper_range",
            {issue.code for issue in report.issues},
        )
        robot.safety_profile = dataclasses.replace(profile, stop_capability=False)
        self.assertIn(
            "stop_motion_unsupported",
            {issue.code for issue in robot.validate_pose_target(_target(robot.action_spec, robot_name="rm2")).issues},
        )
        robot.safety_profile = dataclasses.replace(profile, robot_model="other-rm2")
        self.assertIn(
            "profile_model_mismatch",
            {issue.code for issue in robot.validate_pose_target(_target(robot.action_spec, robot_name="rm2")).issues},
        )


class _ReplaySafetyRobot:
    def __init__(self, spec: ActionSpec, profile: RobotSafetyProfile) -> None:
        self.action_spec = spec
        self.safety_profile = profile
        self.state = np.asarray([0.0, 0.5], dtype=np.float32)
        self.commands: list[np.ndarray] = []

    def get_safety_profile(self) -> RobotSafetyProfile:
        return self.safety_profile

    def read_state(self) -> RobotState:
        now_ns = time.monotonic_ns()
        return RobotState(self.state.copy(), now_ns / 1e9, now_ns, health={"ok": True})

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        self.commands.append(target.copy())
        self.state = target.copy()
        return target.copy()

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_current_pose_state(self) -> RobotState:
        return self.read_state()

    def validate_pose_target(self, target: object) -> PoseValidationReport:
        del target
        return PoseValidationReport(
            safety_policy=self.safety_profile.policy.value,
            safety_profile_hash=self.safety_profile.profile_hash,
        )

    def plan_move_to_state(self, plan: MoveToRecordedStatePlan) -> MoveToRecordedStatePlan:
        return plan

    def execute_pose_plan(
        self,
        plan: MoveToRecordedStatePlan,
        *,
        stop_event: object,
        on_progress: object = None,
    ) -> PoseMoveResult:
        del stop_event, on_progress
        self.state = plan.target_state.copy()
        return PoseMoveResult(plan.plan_id, "reached", self.read_state(), 0.0)

    def stop_pose_motion(self) -> None:
        return None

    def verify_target_reached(self, plan: MoveToRecordedStatePlan) -> PoseMoveResult:
        return PoseMoveResult(plan.plan_id, "reached", self.read_state(), 0.0)


class PlanSafetyProfileTests(unittest.TestCase):
    def test_replay_rejects_old_plan_after_safety_profile_change(self) -> None:
        spec = ActionSpec(
            2,
            2,
            1,
            "rad",
            (),
            state_fields=(
                VectorField("joint_1", "rad", "joint_position"),
                VectorField("gripper", "normalized_0_open_1", "gripper_open_fraction"),
            ),
            action_fields=(
                VectorField("joint_1", "rad", "joint_position"),
                VectorField("gripper", "normalized_0_open_1", "gripper_open_fraction"),
            ),
        )
        samples = tuple(
            RecordedSample(
                episode_index=0,
                frame_index=index,
                index=index,
                timestamp=index * 0.02,
                task_index=0,
                state=np.asarray([index * 0.01, 0.5], dtype=np.float32),
                action=np.asarray([index * 0.01, 0.5], dtype=np.float32),
                images={},
                telemetry={},
            )
            for index in range(2)
        )
        plan = (
            ReplayPlanner(FakeRecordedEpisodeSource(spec, {0: samples}, robot_name="piper"))
            .plan(
                robot_name="piper",
                target_action_spec=spec,
                episode_index=0,
                constraints=ReplayConstraints(
                    min_interval_s=0.001,
                    max_interval_s=0.2,
                    max_step=0.1,
                    max_velocity=10.0,
                    max_acceleration=1_000.0,
                ),
            )
            .plan
        )
        assert plan is not None
        profile = _profile("piper", spec)
        plan = dataclasses.replace(
            plan,
            safety_profile_hash=profile.profile_hash,
            safety_policy=profile.policy.value,
            plan_hash="",
        )
        robot = _ReplaySafetyRobot(spec, profile)
        controller = RobotReplayController(robot, plan)
        controller.prepare()
        deadline = time.monotonic() + 2.0
        while controller.cursor().state.value not in {"armed", "error"} and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(controller.cursor().state.value, "armed", controller.cursor().message)
        robot.safety_profile = dataclasses.replace(profile, soft_margin=0.01)
        with self.assertRaises(ReplayPlanStaleError):
            controller.confirm_and_start(plan.plan_hash)
        controller.stop(wait=True, timeout=2.0)


if __name__ == "__main__":
    unittest.main()
