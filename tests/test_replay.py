from __future__ import annotations

import dataclasses
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mp_real.common.plan_integrity import canonical_hash, canonical_json
from mp_real.data.models import FakeRecordedEpisodeSource, RecordedSample
from mp_real.pose.models import PoseMoveProgress, PoseMoveResult, PoseValidationReport
from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import (
    ReplayAcknowledgementStrategy,
    ReplayConstraints,
    ReplayMode,
    ReplayPlan,
    ReplayPlanIntegrityError,
    ReplayPlanStaleError,
    ReplayState,
    ReplayStep,
    ReplayTimingMode,
    json_safe,
)
from mp_real.replay.planning import ReplayPlanner
from mp_real.replay.recording import ReplayRecordingConfig, ReplayRecordWriter
from mp_real.runtime.models import ActionSpec, RobotState, VectorField
from mp_real.web.resources import ResourceLeaseConflict, ResourceLeaseManager, ResourceRequest, ResourceType


def _spec() -> ActionSpec:
    fields = (
        VectorField("joint_1", "rad", "joint_position"),
        VectorField("gripper", "normalized_0_open_1", "gripper_open_fraction"),
    )
    return ActionSpec(2, 2, 1, "rad", (), state_fields=fields, action_fields=fields)


def _source(
    *,
    count: int = 4,
    info: dict | None = None,
    spec: ActionSpec | None = None,
    robot_name: str = "piper",
    samples: tuple[RecordedSample, ...] | None = None,
) -> FakeRecordedEpisodeSource:
    action_spec = spec or _spec()
    if samples is None:
        samples = tuple(
            RecordedSample(
                episode_index=0,
                frame_index=index,
                index=index,
                timestamp=index * 0.02,
                task_index=0,
                state=np.asarray([index * 0.01, 1.0], dtype=np.float32),
                action=np.asarray([index * 0.01, 1.0], dtype=np.float32),
                images={},
                telemetry={},
            )
            for index in range(count)
        )
    return FakeRecordedEpisodeSource(action_spec, {0: samples}, robot_name=robot_name, info=info)


def _constraints() -> ReplayConstraints:
    return ReplayConstraints(
        min_interval_s=0.001,
        max_interval_s=0.2,
        max_step=0.1,
        max_velocity=10.0,
        max_acceleration=1_000.0,
        tracking_tolerance=0.03,
        max_tracking_error=0.1,
        max_control_overrun_s=0.5,
    )


def _h5_constraints(**overrides: object) -> ReplayConstraints:
    values = {
        "min_interval_s": 0.001,
        "max_interval_s": 0.2,
        "max_step": 0.1,
        "max_velocity": 10.0,
        "max_acceleration": 1_000.0,
        "tracking_tolerance": 0.03,
        "max_tracking_error": 0.2,
        "max_control_overrun_s": 0.5,
        "feedback_poll_interval_s": 0.005,
        "acknowledgement_timeout_s": 1.0,
        "sustained_tracking_error_limit": 20,
        "extreme_tracking_error": 1.0,
    }
    values.update(overrides)
    return ReplayConstraints(**values)


class _FakeReplayRobot:
    def __init__(self, *, stall: bool = False) -> None:
        self.action_spec = _spec()
        self.state = np.zeros(2, dtype=np.float32)
        self.state[1] = 1.0
        self.stall = stall
        self.commands: list[np.ndarray] = []
        self.stops = 0
        self.closed = False

    def read_state(self) -> RobotState:
        now_ns = time.monotonic_ns()
        return RobotState(self.state.copy(), now_ns / 1e9, now_ns, health={"ok": True})

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        target = np.asarray(target, dtype=np.float32).copy()
        self.commands.append(target)
        if not self.stall:
            self.state = target
        return target

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def get_current_pose_state(self) -> RobotState:
        return self.read_state()

    def validate_pose_target(self, target: object) -> PoseValidationReport:
        del target
        return PoseValidationReport()

    def plan_move_to_state(self, plan: object) -> object:
        return plan

    def execute_pose_plan(
        self, plan: object, *, stop_event: threading.Event, on_progress: object = None
    ) -> PoseMoveResult:
        for waypoint in plan.waypoints:
            if stop_event.is_set():
                return PoseMoveResult(plan.plan_id, "aborted", self.read_state(), None, "stopped")
            self.state = waypoint.target.copy()
            if callable(on_progress):
                on_progress(
                    PoseMoveProgress(
                        plan.plan_id,
                        waypoint.index,
                        len(plan.waypoints),
                        self.state.copy(),
                        waypoint.target.copy(),
                        0.0,
                        time.monotonic_ns(),
                    )
                )
        return PoseMoveResult(plan.plan_id, "reached", self.read_state(), 0.0)

    def stop_pose_motion(self) -> None:
        self.stops += 1

    def verify_target_reached(self, plan: object) -> PoseMoveResult:
        error = float(np.max(np.abs(self.state - plan.target_state)))
        return PoseMoveResult(plan.plan_id, "reached" if error <= 0.03 else "failed", self.read_state(), error)


class _DelayedReplayRobot(_FakeReplayRobot):
    def __init__(
        self,
        *,
        feedback_cycles: int = 3,
        stale_feedback: bool = False,
        health_error: bool = False,
    ) -> None:
        super().__init__(stall=True)
        self.feedback_cycles = feedback_cycles
        self.stale_feedback = stale_feedback
        self.health_error = health_error
        self._target = self.state.copy()
        self._remaining_feedback_cycles = 0

    def read_state(self) -> RobotState:
        if self._remaining_feedback_cycles > 0:
            self.state = self.state + (self._target - self.state) / self._remaining_feedback_cycles
            self._remaining_feedback_cycles -= 1
        now_ns = time.monotonic_ns()
        health = {
            "robot_name": "piper",
            "connected": True,
            "enabled": True,
            "healthy": not self.health_error,
            "error_codes": ["fake_health_error"] if self.health_error else [],
            "stale_feedback": self.stale_feedback,
        }
        source_timestamp_ns = now_ns - 1_000_000_000 if self.stale_feedback else now_ns
        return RobotState(
            self.state.copy(),
            now_ns / 1e9,
            now_ns,
            source_timestamp_ns=source_timestamp_ns,
            health=health,
        )

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        target = np.asarray(target, dtype=np.float32).copy()
        self.commands.append(target)
        self._target = target
        self._remaining_feedback_cycles = max(1, self.feedback_cycles)
        return target


def _wait(predicate: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


class ReplayPlanningTests(unittest.TestCase):
    def _plan(self, *, robot_name: str = "piper", constraints: ReplayConstraints | None = None) -> ReplayPlan:
        plan = (
            ReplayPlanner(_source(robot_name=robot_name))
            .plan(
                robot_name=robot_name,
                target_action_spec=_spec(),
                episode_index=0,
                constraints=constraints or _constraints(),
            )
            .plan
        )
        assert plan is not None
        return plan

    def test_command_and_state_plans_are_explicit_and_hardware_free(self) -> None:
        source = _source()
        planner = ReplayPlanner(source)
        command = planner.plan(
            robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints()
        )
        self.assertTrue(command.report.valid, command.report.errors)
        self.assertEqual(command.plan.mode, ReplayMode.COMMAND_REPLAY)
        self.assertEqual(command.plan.steps[0].target_offset_ns, 0)
        state = planner.plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            mode=ReplayMode.STATE_TRAJECTORY_FOLLOWING,
            timing_mode=ReplayTimingMode.FIXED_FPS,
            fps=50,
            constraints=_constraints(),
        )
        self.assertTrue(state.report.valid, state.report.errors)
        self.assertTrue(any(item.code == "state_following" for item in state.report.warnings))

    def test_rejects_unknown_action_source_mismatch_nonfinite_and_invalid_range(self) -> None:
        missing_action_source = _source(info={"mp_real": {}})
        result = ReplayPlanner(missing_action_source).plan(
            robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints()
        )
        self.assertIsNone(result.plan)
        self.assertTrue(any(item.code == "action_source_unknown" for item in result.report.errors))

        mismatch = ActionSpec(
            2, 2, 1, "deg", (), state_fields=_spec().state_fields, action_fields=_spec().action_fields
        )
        result = ReplayPlanner(_source()).plan(
            robot_name="piper", target_action_spec=mismatch, episode_index=0, constraints=_constraints()
        )
        self.assertIsNone(result.plan)
        self.assertTrue(any(item.code == "joint_unit_mismatch" for item in result.report.errors))

        result = ReplayPlanner(_source()).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            start_sample=3,
            end_sample=9,
            constraints=_constraints(),
        )
        self.assertIsNone(result.plan)
        self.assertTrue(any(item.code == "sample_range" for item in result.report.errors))

        broken = _source()
        broken._episodes[0] = tuple(  # type: ignore[attr-defined]
            list(broken._episodes[0][:2])  # type: ignore[attr-defined]
            + [RecordedSample(0, 2, 2, 0.04, 0, np.asarray([np.nan, 1.0]), np.asarray([0.0, 1.0]), {}, {})]
        )
        result = ReplayPlanner(broken).plan(
            robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints()
        )
        self.assertTrue(any(item.code == "not_finite" for item in result.report.errors))

    def test_timing_and_kinematic_limits_are_reported(self) -> None:
        result = ReplayPlanner(_source()).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            speed_scale=0.1,
            constraints=ReplayConstraints(max_step=0.001, max_velocity=0.001, max_acceleration=0.001),
        )
        codes = {item.code for item in result.report.errors}
        self.assertIn("joint_max_step", codes)
        self.assertIn("joint_max_velocity", codes)

        limit_result = ReplayPlanner(_source()).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            constraints=ReplayConstraints(
                max_step=0.1,
                max_velocity=10.0,
                max_acceleration=1_000.0,
                lower_limits=(-0.1, 0.0),
                upper_limits=(0.1, 0.5),
            ),
        )
        self.assertTrue(any(item.code == "gripper_range" for item in limit_result.report.errors))

    def test_h5_gripper_constraints_are_independent_from_joint_limits(self) -> None:
        gripper_close = tuple(
            RecordedSample(
                episode_index=0,
                frame_index=index,
                index=index,
                timestamp=index * 0.02,
                task_index=0,
                state=np.asarray([0.0, float(1 - index)], dtype=np.float32),
                action=np.asarray([0.0, float(1 - index)], dtype=np.float32),
                images={},
                telemetry={},
            )
            for index in range(2)
        )
        result = ReplayPlanner(_source(samples=gripper_close)).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            speed_scale=1.0,
            constraints=ReplayConstraints(
                joint_max_step=0.05,
                joint_max_velocity=10.0,
                joint_max_acceleration=1_000.0,
                gripper_min=(0.0,),
                gripper_max=(1.0,),
            ),
        )
        self.assertTrue(result.report.valid, result.report.errors)
        self.assertEqual(result.report.maximum_observed_joint_delta, 0.0)
        self.assertEqual(result.report.maximum_observed_gripper_delta, 1.0)

        gripper_limited = ReplayPlanner(_source(samples=gripper_close)).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            speed_scale=1.0,
            constraints=ReplayConstraints(
                joint_max_step=0.05,
                joint_max_velocity=10.0,
                joint_max_acceleration=1_000.0,
                gripper_min=(0.0,),
                gripper_max=(1.0,),
                gripper_max_step=0.25,
            ),
        )
        self.assertTrue(any(item.code == "gripper_max_step" for item in gripper_limited.report.errors))

        gripper_out_of_range = tuple(
            dataclasses.replace(sample, action=np.asarray([0.0, 1.2], dtype=np.float32))
            for sample in gripper_close
        )
        result = ReplayPlanner(_source(samples=gripper_out_of_range)).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            constraints=ReplayConstraints(gripper_min=(0.0,), gripper_max=(1.0,)),
        )
        self.assertTrue(any(item.code == "gripper_range" for item in result.report.errors))

        joint_out_of_range = tuple(
            dataclasses.replace(sample, action=np.asarray([0.2, 1.0], dtype=np.float32))
            for sample in gripper_close
        )
        result = ReplayPlanner(_source(samples=joint_out_of_range)).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            constraints=ReplayConstraints(joint_lower_limits=(-0.1,), joint_upper_limits=(0.1,)),
        )
        self.assertTrue(any(item.code == "joint_limit_exceeded" for item in result.report.errors))

    def test_h3_replay_plan_arrays_are_readonly_and_inputs_are_copied(self) -> None:
        target = np.asarray([0.1, 1.0], dtype=np.float32)
        expected_state = np.asarray([0.2, 1.0], dtype=np.float32)
        step = ReplayStep(1, 1, 0.02, 20_000_000, target, expected_state)
        target[0] = 99.0
        expected_state[0] = 88.0
        np.testing.assert_allclose(step.target, [0.1, 1.0])
        np.testing.assert_allclose(step.expected_state, [0.2, 1.0])
        with self.assertRaises(ValueError):
            step.target[0] = 0.5
        with self.assertRaises(ValueError):
            step.expected_state[0] = 0.5

        result = ReplayPlanner(_source()).plan(
            robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints()
        )
        assert result.plan is not None
        self.assertFalse(result.plan.steps[0].target.flags.writeable)
        self.assertFalse(result.plan.steps[0].expected_state.flags.writeable)
        self.assertFalse(result.report.start_state.flags.writeable)
        self.assertFalse(result.report.end_state.flags.writeable)

    def test_h3_replay_plan_hash_covers_motion_fields_and_json_is_independent(self) -> None:
        def assert_tamper_rejected(mutator) -> None:
            plan = self._plan()
            original_hash = plan.plan_hash
            mutator(plan)
            self.assertNotEqual(plan.recompute_plan_hash(), original_hash)
            with self.assertRaises(ReplayPlanIntegrityError):
                plan.require_integrity()

        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan.steps[1], "target", plan.steps[1].target.copy() + np.asarray([0.01, 0.0], dtype=np.float32)
            )
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan.steps[1],
                "expected_state",
                plan.steps[1].expected_state.copy() + np.asarray([0.01, 0.0], dtype=np.float32),
            )
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(plan.steps[1], "target_offset_ns", plan.steps[1].target_offset_ns + 1)
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan,
                "action_spec",
                dataclasses.replace(plan.action_spec, action_mode="alternate_joint_position_target"),
            )
        )
        assert_tamper_rejected(lambda plan: object.__setattr__(plan, "dataset_hash", "replaced-source"))

        plan = self._plan(
            constraints=dataclasses.replace(_constraints(), lower_limits=(-1.0, 0.0), upper_limits=(1.0, 1.0))
        )
        changed_limits = dataclasses.replace(plan.constraints, lower_limits=(-0.5, 0.0), upper_limits=(0.5, 1.0))
        changed = dataclasses.replace(plan, constraints=changed_limits, plan_hash="")
        self.assertNotEqual(changed.plan_hash, plan.plan_hash)

        payload = json_safe(plan)
        payload["steps"][0]["target"][0] = 123.0
        np.testing.assert_allclose(plan.steps[0].target, [0.0, 1.0])
        plan.require_integrity()

        encoded = canonical_json(plan.canonical_payload())
        self.assertEqual(canonical_hash(json.loads(encoded)), plan.plan_hash)

    def test_h3_replay_source_dataset_replacement_changes_plan_identity(self) -> None:
        original = self._plan()
        replaced_samples = tuple(
            RecordedSample(
                episode_index=0,
                frame_index=index,
                index=index,
                timestamp=index * 0.02,
                task_index=0,
                state=np.asarray([index * 0.01, 1.0], dtype=np.float32),
                action=np.asarray([index * 0.02, 1.0], dtype=np.float32),
                images={},
                telemetry={},
            )
            for index in range(4)
        )
        replaced = (
            ReplayPlanner(FakeRecordedEpisodeSource(_spec(), {0: replaced_samples}, robot_name="piper"))
            .plan(robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints())
            .plan
        )
        assert replaced is not None
        self.assertNotEqual(replaced.dataset_hash, original.dataset_hash)

        object.__setattr__(original, "dataset_hash", replaced.dataset_hash)
        with self.assertRaises(ReplayPlanIntegrityError):
            original.require_integrity()


class ReplayControllerTests(unittest.TestCase):
    def _armed_controller(
        self,
        *,
        start: int = 0,
        robot: _FakeReplayRobot | None = None,
        count: int = 8,
        speed_scale: float = 0.1,
        constraints: ReplayConstraints | None = None,
    ) -> tuple[RobotReplayController, _FakeReplayRobot]:
        plan = (
            ReplayPlanner(_source(count=count))
            .plan(
                robot_name="piper",
                target_action_spec=_spec(),
                episode_index=0,
                start_sample=start,
                speed_scale=speed_scale,
                constraints=constraints or _constraints(),
            )
            .plan
        )
        assert plan is not None
        fake = robot or _FakeReplayRobot()
        controller = RobotReplayController(fake, plan)
        controller.prepare()
        _wait(lambda: controller.cursor().state in {ReplayState.ARMED, ReplayState.ERROR})
        self.assertEqual(controller.cursor().state, ReplayState.ARMED, controller.cursor().message)
        return controller, fake

    def test_move_to_start_and_command_replay(self) -> None:
        controller, robot = self._armed_controller(start=2)
        self.assertAlmostEqual(robot.state[0], 0.02, places=5)
        controller.confirm_and_start(controller.plan.plan_hash)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        self.assertEqual(controller.cursor().state, ReplayState.COMPLETED)
        self.assertEqual(len(robot.commands), len(controller.plan.steps))

    def test_stop_while_armed_is_terminal(self) -> None:
        controller, robot = self._armed_controller()
        self.assertTrue(controller.stop(emergency=True, wait=True, timeout=2.0))
        self.assertEqual(controller.cursor().state, ReplayState.ABORTED)
        self.assertGreater(robot.stops, 0)

    def test_pause_resume_stop_and_stale_lease(self) -> None:
        controller, robot = self._armed_controller()
        controller.confirm_and_start(controller.plan.plan_hash)
        _wait(lambda: len(robot.commands) >= 2)
        self.assertTrue(controller.pause())
        self.assertEqual(controller.cursor().state, ReplayState.PAUSED)
        self.assertGreater(robot.stops, 0)
        self.assertTrue(controller.resume())
        self.assertTrue(controller.stop(emergency=True, wait=True, timeout=2.0))
        self.assertEqual(controller.cursor().state, ReplayState.ABORTED)

        plan = (
            ReplayPlanner(_source())
            .plan(robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints())
            .plan
        )
        assert plan is not None
        stale = RobotReplayController(_FakeReplayRobot(), plan, lease_valid=lambda: False)
        stale.prepare()
        _wait(lambda: stale.cursor().state is ReplayState.ERROR)
        self.assertIn("stale", stale.cursor().message or "")

    def test_tracking_error_aborts(self) -> None:
        controller, _ = self._armed_controller(robot=_FakeReplayRobot(stall=True), count=20, speed_scale=1.0)
        controller.confirm_and_start(controller.plan.plan_hash)
        self.assertTrue(controller.join(timeout=2.0))
        self.assertEqual(controller.cursor().state, ReplayState.ABORTED)
        self.assertIn("tracking error", controller.cursor().message or "")

    def test_h5_send_feedback_acknowledged_cursors_are_separate_and_recorded(self) -> None:
        events: list[dict] = []
        plan = (
            ReplayPlanner(_source(count=4))
            .plan(
                robot_name="piper",
                target_action_spec=_spec(),
                episode_index=0,
                speed_scale=1.0,
                constraints=_h5_constraints(joint_tracking_error=0.0001, sustained_tracking_error_limit=50),
            )
            .plan
        )
        assert plan is not None
        controller = RobotReplayController(_DelayedReplayRobot(feedback_cycles=3), plan, record_callback=events.append)
        controller.prepare()
        _wait(lambda: controller.cursor().state is ReplayState.ARMED)
        controller.confirm_and_start(plan.plan_hash)
        _wait(lambda: any(event["type"] == "replay_command_sent" for event in events))
        command_event = next(event for event in events if event["type"] == "replay_command_sent")
        self.assertEqual(command_event["cursors"]["sent_sample_index"], plan.start_sample)
        self.assertIsNone(command_event["cursors"]["acknowledged_sample_index"])
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        cursor = controller.cursor()
        self.assertEqual(cursor.state, ReplayState.COMPLETED)
        self.assertEqual(cursor.sent_sample_index, plan.end_sample)
        self.assertEqual(cursor.feedback_sample_index, plan.end_sample)
        self.assertEqual(cursor.acknowledged_sample_index, plan.end_sample)
        self.assertEqual(cursor.displayed_sample_index, plan.end_sample)
        self.assertEqual(cursor.progress_ratio, cursor.acknowledged_progress_ratio)
        replay_step = next(event for event in events if event["type"] == "replay_step")
        self.assertIn("sent_sample_index", replay_step["cursors"])
        self.assertIn("feedback_sample_index", replay_step["cursors"])
        self.assertIn("acknowledged_sample_index", replay_step["cursors"])

    def test_h5_immediate_interface_ack_does_not_claim_state_arrival(self) -> None:
        plan = (
            ReplayPlanner(_source(count=2))
            .plan(
                robot_name="piper",
                target_action_spec=_spec(),
                episode_index=0,
                speed_scale=1.0,
                constraints=_h5_constraints(
                    acknowledgement_strategy=ReplayAcknowledgementStrategy.IMMEDIATE_INTERFACE_ACK
                ),
            )
            .plan
        )
        assert plan is not None
        controller = RobotReplayController(_DelayedReplayRobot(feedback_cycles=10), plan)
        controller.prepare()
        _wait(lambda: controller.cursor().state is ReplayState.ARMED)
        controller.confirm_and_start(plan.plan_hash)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        cursor = controller.cursor()
        self.assertEqual(cursor.state, ReplayState.COMPLETED)
        self.assertEqual(cursor.sent_sample_index, plan.end_sample)
        self.assertIsNone(cursor.feedback_sample_index)
        self.assertIsNone(cursor.acknowledged_sample_index)
        self.assertEqual(cursor.acknowledged_progress_ratio, 0.0)

    def test_h5_follower_window_and_state_settle_strategies_accept_delayed_feedback(self) -> None:
        for strategy in (
            ReplayAcknowledgementStrategy.FOLLOWER_WINDOW,
            ReplayAcknowledgementStrategy.STATE_TRAJECTORY_SETTLE,
        ):
            with self.subTest(strategy=strategy):
                controller, _ = self._armed_controller(
                    robot=_DelayedReplayRobot(feedback_cycles=2),
                    count=5,
                    speed_scale=0.1,
                    constraints=_h5_constraints(
                        acknowledgement_strategy=strategy,
                        follower_window_samples=2,
                        state_trajectory_settle_cycles=2,
                    ),
                )
                controller.confirm_and_start(controller.plan.plan_hash)
                self.assertTrue(controller.join(timeout=3.0, raise_on_error=True))
                self.assertEqual(controller.cursor().state, ReplayState.COMPLETED)

    def test_h5_tracking_stale_feedback_and_health_errors_abort(self) -> None:
        controller, _ = self._armed_controller(
            robot=_FakeReplayRobot(stall=True),
            count=6,
            speed_scale=1.0,
            constraints=_h5_constraints(joint_tracking_error=0.0001, sustained_tracking_error_limit=2),
        )
        controller.confirm_and_start(controller.plan.plan_hash)
        self.assertTrue(controller.join(timeout=2.0))
        self.assertEqual(controller.cursor().state, ReplayState.ABORTED)
        self.assertIn("sustained tracking error", controller.cursor().message or "")

        stale, _ = self._armed_controller(
            robot=_DelayedReplayRobot(stale_feedback=True),
            count=2,
            speed_scale=1.0,
            constraints=_h5_constraints(feedback_freshness_timeout_s=0.001),
        )
        stale.confirm_and_start(stale.plan.plan_hash)
        self.assertTrue(stale.join(timeout=2.0))
        self.assertEqual(stale.cursor().state, ReplayState.ABORTED)
        self.assertIn("stale", stale.cursor().message or "")

        unhealthy, _ = self._armed_controller(
            robot=_DelayedReplayRobot(health_error=True),
            count=2,
            speed_scale=1.0,
            constraints=_h5_constraints(),
        )
        unhealthy.confirm_and_start(unhealthy.plan.plan_hash)
        self.assertTrue(unhealthy.join(timeout=2.0))
        self.assertEqual(unhealthy.cursor().state, ReplayState.ABORTED)
        self.assertIn("health", unhealthy.cursor().message or "")

    def test_h5_pause_resume_uses_acknowledged_state_not_sent_state(self) -> None:
        controller, robot = self._armed_controller(
            robot=_DelayedReplayRobot(feedback_cycles=10),
            count=6,
            speed_scale=1.0,
            constraints=_h5_constraints(joint_tracking_error=0.0001, sustained_tracking_error_limit=50),
        )
        controller.confirm_and_start(controller.plan.plan_hash)
        _wait(
            lambda: (
                controller.cursor().sent_sample_index is not None
                and controller.cursor().sent_sample_index > controller.plan.start_sample
                and controller.cursor().sent_sample_index != controller.cursor().acknowledged_sample_index
            )
        )
        cursor_before_pause = controller.cursor()
        self.assertTrue(controller.pause())
        sent_position = cursor_before_pause.sent_sample_index - controller.plan.start_sample
        robot.state = controller.plan.steps[sent_position].expected_state.copy()
        self.assertFalse(controller.resume())
        self.assertEqual(controller.cursor().state, ReplayState.PAUSED)
        self.assertTrue(controller.stop(wait=True, timeout=2.0))

    def test_resume_state_mismatch_and_resource_lease_conflict(self) -> None:
        controller, robot = self._armed_controller()
        controller.confirm_and_start(controller.plan.plan_hash)
        _wait(lambda: len(robot.commands) >= 2)
        self.assertTrue(controller.pause())
        robot.state[0] += 1.0
        self.assertFalse(controller.resume())
        self.assertEqual(controller.cursor().state, ReplayState.PAUSED)
        self.assertTrue(controller.stop(wait=True, timeout=2.0))

        manager = ResourceLeaseManager()
        request = ResourceRequest(ResourceType.ROBOT_CONTROL, "piper")
        lease = manager.acquire("replay-a", (request,))
        with self.assertRaises(ResourceLeaseConflict):
            manager.acquire("replay-b", (request,))
        lease.release()

    def test_fake_piper_and_rm2_command_replays_use_the_same_controller(self) -> None:
        for robot_name in ("piper", "rm2"):
            with self.subTest(robot=robot_name):
                plan = (
                    ReplayPlanner(_source(count=2, robot_name=robot_name))
                    .plan(
                        robot_name=robot_name,
                        target_action_spec=_spec(),
                        episode_index=0,
                        speed_scale=1.0,
                        constraints=_constraints(),
                    )
                    .plan
                )
                assert plan is not None
                controller = RobotReplayController(_FakeReplayRobot(), plan)
                controller.prepare()
                _wait(lambda: controller.cursor().state is ReplayState.ARMED)
                controller.confirm_and_start(plan.plan_hash)
                self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
                self.assertEqual(controller.cursor().state, ReplayState.COMPLETED)

    def test_replay_record_is_written_by_a_background_worker(self) -> None:
        with TemporaryDirectory() as directory:
            plan = (
                ReplayPlanner(_source())
                .plan(robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints())
                .plan
            )
            assert plan is not None
            writer = ReplayRecordWriter(ReplayRecordingConfig(Path(directory)), plan)
            writer.start()
            controller = RobotReplayController(_FakeReplayRobot(), plan, record_callback=writer.emit)
            controller.prepare()
            _wait(lambda: controller.cursor().state is ReplayState.ARMED)
            controller.confirm_and_start(plan.plan_hash)
            self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
            self.assertTrue(writer.stop(result=controller.cursor().state.value, timeout=2.0))
            record = Path(directory) / f"replay-{plan.plan_id}"
            self.assertTrue(record.is_dir())
            manifest = json.loads((record / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["plan_hash"], plan.plan_hash)
            cursors = manifest["tracking_cursors"]
            self.assertIn("sent_sample_index", cursors)
            self.assertIn("feedback_sample_index", cursors)
            self.assertIn("acknowledged_sample_index", cursors)
            self.assertEqual(cursors["acknowledged_sample_index"], plan.end_sample)
            self.assertIn("replay_step", (record / "events.jsonl").read_text(encoding="utf-8"))

    def test_h3_replay_rehashes_before_arm_execute_resume_and_stale_identity(self) -> None:
        plan = (
            ReplayPlanner(_source())
            .plan(robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints())
            .plan
        )
        assert plan is not None
        arm_robot = _FakeReplayRobot()
        arm_controller = RobotReplayController(arm_robot, plan)
        object.__setattr__(plan, "speed_scale", 0.2)
        with self.assertRaises(ReplayPlanIntegrityError):
            arm_controller.prepare()
        self.assertEqual(arm_robot.commands, [])

        execute_controller, execute_robot = self._armed_controller()
        try:
            target = execute_controller.plan.steps[1].target.copy()
            target[0] += 0.25
            object.__setattr__(execute_controller.plan.steps[1], "target", target)
            with self.assertRaises(ReplayPlanIntegrityError):
                execute_controller.confirm_and_start(execute_controller.plan.plan_hash)
            self.assertEqual(execute_robot.commands, [])
        finally:
            execute_controller.stop(wait=True, timeout=2.0)

        generation = {"value": 1}
        stale_plan = (
            ReplayPlanner(_source())
            .plan(robot_name="piper", target_action_spec=_spec(), episode_index=0, constraints=_constraints())
            .plan
        )
        assert stale_plan is not None
        stale_controller = RobotReplayController(
            _FakeReplayRobot(), stale_plan, lease_valid=lambda: generation["value"] == stale_plan.generation_id
        )
        stale_controller.prepare()
        _wait(lambda: stale_controller.cursor().state is ReplayState.ARMED)
        generation["value"] = stale_plan.generation_id + 1
        with self.assertRaises(ReplayPlanStaleError):
            stale_controller.confirm_and_start(stale_plan.plan_hash)
        stale_controller.stop(wait=True, timeout=2.0)

        reconnect_controller, reconnect_robot = self._armed_controller()
        try:
            reconnect_robot.action_spec = dataclasses.replace(_spec(), action_mode="reconnected-layout")
            with self.assertRaises(ReplayPlanStaleError):
                reconnect_controller.confirm_and_start(reconnect_controller.plan.plan_hash)
            self.assertEqual(reconnect_robot.commands, [])
        finally:
            reconnect_controller.stop(wait=True, timeout=2.0)

        resume_controller, resume_robot = self._armed_controller(count=20, speed_scale=1.0)
        try:
            resume_controller.confirm_and_start(resume_controller.plan.plan_hash)
            _wait(lambda: len(resume_robot.commands) >= 2)
            self.assertTrue(resume_controller.pause())
            object.__setattr__(resume_controller.plan, "dataset_hash", "changed-during-pause")
            with self.assertRaises(ReplayPlanIntegrityError):
                resume_controller.resume()
        finally:
            resume_controller.stop(wait=True, timeout=2.0)


if __name__ == "__main__":
    unittest.main()
