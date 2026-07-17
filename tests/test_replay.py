from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mp_real.data.models import FakeRecordedEpisodeSource, RecordedSample
from mp_real.pose.models import PoseMoveProgress, PoseMoveResult, PoseValidationReport
from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import ReplayConstraints, ReplayMode, ReplayState, ReplayTimingMode
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
) -> FakeRecordedEpisodeSource:
    action_spec = spec or _spec()
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


def _wait(predicate: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


class ReplayPlanningTests(unittest.TestCase):
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
        self.assertIn("max_step", codes)
        self.assertIn("max_velocity", codes)

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
        self.assertTrue(any(item.code == "joint_limit_exceeded" for item in limit_result.report.errors))


class ReplayControllerTests(unittest.TestCase):
    def _armed_controller(
        self, *, start: int = 0, robot: _FakeReplayRobot | None = None, count: int = 8, speed_scale: float = 0.1
    ) -> tuple[RobotReplayController, _FakeReplayRobot]:
        plan = (
            ReplayPlanner(_source(count=count))
            .plan(
                robot_name="piper",
                target_action_spec=_spec(),
                episode_index=0,
                start_sample=start,
                speed_scale=speed_scale,
                constraints=_constraints(),
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
            manifest = (record / "manifest.json").read_text(encoding="utf-8")
            self.assertIn(plan.plan_hash, manifest)
            self.assertIn("replay_step", (record / "events.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
