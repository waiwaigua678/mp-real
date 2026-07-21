from __future__ import annotations

import dataclasses
import time
import unittest
from pathlib import Path

import numpy as np

from mp_real.data.models import DatasetMetadata, EpisodeMetadata, EpisodeStatus
from mp_real.data.pose import recorded_pose_target
from mp_real.pose.controller import PoseMoveController
from mp_real.pose.models import (
    MappingEntry,
    MoveToRecordedStatePlan,
    PoseMappingConfig,
    PoseMotionConstraints,
    PoseMoveProgress,
    PoseMoveResult,
    PosePlanIntegrityError,
    PosePlanStaleError,
    PoseSafetyLimits,
    RecordedPoseTarget,
)
from mp_real.pose.validation import MoveToStateValidator
from mp_real.runtime.models import ActionSpec, RobotState, VectorField
from mp_real.web.resources import ResourceLeaseConflict, ResourceLeaseManager, ResourceRequest, ResourceType


def _spec(*, unit: str = "rad", gripper: bool = True) -> ActionSpec:
    fields = [VectorField("joint_1", unit, "joint_position"), VectorField("joint_2", unit, "joint_position")]
    if gripper:
        fields.append(VectorField("gripper", "normalized_0_open_1", "gripper_open_fraction"))
    return ActionSpec(len(fields), len(fields), 2, unit, (), state_fields=tuple(fields), action_fields=tuple(fields))


def _target(
    *,
    spec: ActionSpec | None = None,
    values: np.ndarray | None = None,
    robot_name: str = "piper",
    status: str = "complete",
) -> RecordedPoseTarget:
    action_spec = spec or _spec()
    return RecordedPoseTarget(
        dataset_id="ds_test",
        episode_index=0,
        sample_index=3,
        robot_name=robot_name,
        state_schema=action_spec.state_field_names,
        state_values=np.asarray(values if values is not None else [0.1, -0.1, 1.0], dtype=np.float32),
        state_fields=action_spec.state_fields,
        joint_unit=action_spec.joint_unit,
        timestamp=1.25,
        source_metadata={"dataset_status": status},
        action_spec=action_spec,
    )


class _FakePoseRobot:
    def __init__(self, state: np.ndarray, *, stall: bool = False, delay_s: float = 0.0) -> None:
        self.state = np.asarray(state, dtype=np.float32).copy()
        self.stall = stall
        self.delay_s = delay_s
        self.commands: list[np.ndarray] = []
        self.stop_count = 0

    def get_current_pose_state(self) -> RobotState:
        now_ns = time.monotonic_ns()
        return RobotState(self.state.copy(), now_ns / 1e9, now_ns, health={"ok": True})

    def validate_pose_target(self, target: RecordedPoseTarget):
        return MoveToStateValidator("piper", target.action_spec).validate(target).report

    def plan_move_to_state(self, plan: MoveToRecordedStatePlan) -> MoveToRecordedStatePlan:
        return plan

    def execute_pose_plan(self, plan, *, stop_event, on_progress=None):
        for waypoint in plan.waypoints:
            if stop_event.is_set():
                return PoseMoveResult(plan.plan_id, "aborted", self.get_current_pose_state(), None, "stopped")
            if self.delay_s:
                time.sleep(self.delay_s)
            if not self.stall:
                self.state = waypoint.target.copy()
            self.commands.append(waypoint.target.copy())
            current = self.get_current_pose_state()
            error = float(np.max(np.abs(current.values - waypoint.target)))
            if error > plan.constraints.max_tracking_error:
                return PoseMoveResult(plan.plan_id, "failed", current, error, "tracking error")
            if on_progress is not None:
                on_progress(
                    PoseMoveProgress(
                        plan.plan_id,
                        waypoint.index,
                        len(plan.waypoints),
                        current.values,
                        waypoint.target,
                        error,
                        time.monotonic_ns(),
                    )
                )
        return PoseMoveResult(plan.plan_id, "reached", self.get_current_pose_state(), 0.0)

    def stop_pose_motion(self) -> None:
        self.stop_count += 1

    def verify_target_reached(self, plan: MoveToRecordedStatePlan) -> PoseMoveResult:
        current = self.get_current_pose_state()
        error = float(np.max(np.abs(current.values - plan.target_state)))
        return PoseMoveResult(
            plan.plan_id, "reached" if error <= plan.constraints.tracking_tolerance else "failed", current, error
        )


class _StateOnlyRecordedSource:
    """Fails the test if pose loading reaches the action-bearing sample API."""

    def __init__(self, spec: ActionSpec) -> None:
        self._spec = spec

    def get_dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            root=Path("state-only"),
            info={"robot_type": "piper"},
            status=EpisodeStatus.COMPLETE,
            is_mp_real=True,
            action_spec=self._spec,
            camera_roles=(),
        )

    def get_episode_metadata(self, episode_index: int) -> EpisodeMetadata:
        self.assert_index(episode_index)
        return EpisodeMetadata(episode_index, 1, (), EpisodeStatus.COMPLETE)

    def get_action_spec(self) -> ActionSpec:
        return self._spec

    def get_pose_state_sample(self, episode_index: int, index: int) -> tuple[np.ndarray, float]:
        self.assert_index(episode_index)
        self.assert_index(index)
        return np.asarray([0.25, -0.25, 0.75], dtype=np.float32), 12.5

    def get_sample(self, episode_index: int, index: int):
        del episode_index, index
        raise AssertionError("recorded pose loading must not read an action-bearing sample")

    @staticmethod
    def assert_index(index: int) -> None:
        if index != 0:
            raise IndexError(index)


def _plan(robot: _FakePoseRobot, target: RecordedPoseTarget, *, constraints: PoseMotionConstraints | None = None):
    validated = MoveToStateValidator(
        "piper",
        target.action_spec,
        safety_limits=PoseSafetyLimits(np.full(3, -2.0), np.full(3, 2.0)),
    ).validate(target)
    validated.report.require_valid()
    return MoveToRecordedStatePlan.build(
        target=target,
        current_state=robot.get_current_pose_state(),
        target_state=validated.values,
        gripper_indices=validated.gripper_indices,
        mapped_joint_names=validated.field_names,
        conversions=validated.mappings,
        constraints=constraints or PoseMotionConstraints(control_period_s=0.001, max_joint_step=0.05),
    )


class PoseValidationTests(unittest.TestCase):
    def test_state_target_never_uses_action_values(self) -> None:
        target = recorded_pose_target(
            _StateOnlyRecordedSource(_spec()), dataset_id="state_only", episode_index=0, sample_index=0
        )
        np.testing.assert_allclose(target.state_values, [0.25, -0.25, 0.75])
        self.assertEqual(target.sample_index, 0)

    def test_robot_dimension_name_unit_gripper_nonfinite_and_limit_mismatches_reject(self) -> None:
        target = _target()
        cases = [
            MoveToStateValidator("rm2", target.action_spec),
            MoveToStateValidator("piper", _spec(gripper=False)),
            MoveToStateValidator("piper", _spec(unit="deg")),
            MoveToStateValidator(
                "piper", target.action_spec, safety_limits=PoseSafetyLimits([-0.01, -2, -2], [0.01, 2, 2])
            ),
        ]
        for validator in cases:
            self.assertFalse(validator.validate(target).report.valid)
        with self.assertRaises(Exception):
            _target(values=np.asarray([0.0, np.nan, 0.0], dtype=np.float32))

    def test_explicit_mapping_requires_total_declared_conversion(self) -> None:
        target = _target(spec=_spec(unit="deg"))
        mapping = PoseMappingConfig(
            1,
            (
                # A deg/rad conversion with an explicit scale is valid.
                MappingEntry(
                    "joint_1", "joint_1", np.pi / 180, source_unit="deg", target_unit="rad", semantics="joint_position"
                ),
                MappingEntry(
                    "joint_2", "joint_2", np.pi / 180, source_unit="deg", target_unit="rad", semantics="joint_position"
                ),
                MappingEntry(
                    "gripper",
                    "gripper",
                    source_unit="normalized_0_open_1",
                    target_unit="normalized_0_open_1",
                    semantics="gripper_open_fraction",
                ),
            ),
            source_robot_name="piper",
            target_robot_name="piper",
        )
        validated = MoveToStateValidator("piper", _spec(), mapping_config=mapping).validate(target)
        self.assertTrue(validated.report.valid)
        self.assertAlmostEqual(validated.values[0], 0.1 * np.pi / 180, places=6)


class PoseMoveControllerTests(unittest.TestCase):
    def test_h3_pose_plan_arrays_are_readonly_and_inputs_are_copied(self) -> None:
        source_values = np.asarray([0.1, -0.1, 1.0], dtype=np.float32)
        target = _target(values=source_values)
        source_values[0] = 99.0
        np.testing.assert_allclose(target.state_values, [0.1, -0.1, 1.0])
        with self.assertRaises(ValueError):
            target.state_values[0] = 0.2

        limits = PoseSafetyLimits(np.full(3, -2.0), np.full(3, 2.0))
        with self.assertRaises(ValueError):
            limits.lower[0] = -3.0

        robot = _FakePoseRobot(np.zeros(3, dtype=np.float32))
        plan = _plan(robot, target)
        robot.state[:] = 42.0
        self.assertFalse(plan.current_state.values.flags.writeable)
        self.assertFalse(plan.target_state.flags.writeable)
        self.assertFalse(plan.per_dimension_delta.flags.writeable)
        self.assertFalse(plan.waypoints[0].target.flags.writeable)
        with self.assertRaises(ValueError):
            plan.target_state[0] = 0.3
        with self.assertRaises(ValueError):
            plan.current_state.values[0] = 0.3
        with self.assertRaises(ValueError):
            plan.waypoints[0].target[0] = 0.3

    def test_h3_pose_plan_hash_covers_motion_fields_and_json_is_independent(self) -> None:
        def make_plan() -> MoveToRecordedStatePlan:
            return _plan(_FakePoseRobot(np.zeros(3, dtype=np.float32)), _target())

        def assert_tamper_rejected(mutator) -> None:
            plan = make_plan()
            original_hash = plan.plan_hash
            mutator(plan)
            self.assertNotEqual(plan.recompute_plan_hash(), original_hash)
            with self.assertRaises(PosePlanIntegrityError):
                plan.require_integrity()

        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan.target,
                "state_values",
                plan.target.state_values.copy() + np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
            )
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan,
                "target_state",
                plan.target_state.copy() + np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
            )
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan.waypoints[-1],
                "target",
                plan.waypoints[-1].target.copy() + np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
            )
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan.waypoints[-1],
                "scheduled_at_monotonic_ns",
                plan.waypoints[-1].scheduled_at_monotonic_ns + 1,
            )
        )
        assert_tamper_rejected(
            lambda plan: object.__setattr__(
                plan.target,
                "action_spec",
                dataclasses.replace(plan.target.action_spec, action_mode="alternate_joint_position_target"),
            )
        )

        plan = make_plan()
        changed_gripper_constraints = dataclasses.replace(plan.constraints, max_gripper_step=0.01)
        changed = dataclasses.replace(plan, constraints=changed_gripper_constraints, plan_hash="")
        self.assertNotEqual(changed.plan_hash, plan.plan_hash)

        review_json = {
            "plan_hash": plan.plan_hash,
            "target_state": plan.target_state.tolist(),
            "waypoints": [waypoint.target.tolist() for waypoint in plan.waypoints],
        }
        review_json["target_state"][0] = 123.0
        review_json["waypoints"][0][0] = 123.0
        np.testing.assert_allclose(plan.target_state, [0.1, -0.1, 1.0])
        plan.require_integrity()

    def test_h3_pose_controller_rehashes_before_execute_and_rejects_expired_plan(self) -> None:
        robot = _FakePoseRobot(np.zeros(3, dtype=np.float32))
        plan = _plan(robot, _target())
        object.__setattr__(
            plan.waypoints[-1],
            "target",
            plan.waypoints[-1].target.copy() + np.asarray([0.25, 0.0, 0.0], dtype=np.float32),
        )
        with self.assertRaises(PosePlanIntegrityError):
            PoseMoveController(robot).start(plan)
        self.assertEqual(robot.commands, [])

        fresh = _plan(robot, _target(values=np.asarray([0.01, -0.01, 1.0], dtype=np.float32)))
        expired = dataclasses.replace(fresh, expires_at_monotonic_ns=time.monotonic_ns() - 1, plan_hash="")
        with self.assertRaises(PosePlanStaleError):
            PoseMoveController(robot).start(expired)
        self.assertEqual(robot.commands, [])

    def test_plan_obeys_joint_step_velocity_and_acceleration_constraints(self) -> None:
        robot = _FakePoseRobot(np.zeros(3, dtype=np.float32))
        constraints = PoseMotionConstraints(
            control_period_s=0.01,
            max_joint_velocity=0.2,
            max_joint_acceleration=0.4,
            max_joint_step=0.05,
            max_gripper_step=0.05,
        )
        plan = _plan(robot, _target(values=np.asarray([0.2, -0.1, 1.0], dtype=np.float32)), constraints=constraints)
        points = np.vstack((plan.current_state.values, *(waypoint.target for waypoint in plan.waypoints)))
        joint_steps = np.abs(np.diff(points[:, :2], axis=0))
        self.assertLessEqual(float(np.max(joint_steps)), constraints.max_joint_step + 1e-6)
        self.assertGreaterEqual(
            plan.expected_duration_s,
            np.sqrt(6.0 * 0.2 / constraints.max_joint_acceleration),
        )

    def test_fake_piper_and_rm2_style_pose_moves_reach_target(self) -> None:
        for name in ("fake-piper", "fake-rm2"):
            with self.subTest(name=name):
                robot = _FakePoseRobot(np.zeros(3, dtype=np.float32))
                controller = PoseMoveController(robot, thread_name=name)
                controller.start(_plan(robot, _target()))
                self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
                self.assertEqual(controller.result().status, "reached")
                self.assertGreater(len(robot.commands), 1)

        for robot_name in ("piper", "rm2"):
            with self.subTest(plan_robot=robot_name):
                robot = _FakePoseRobot(np.zeros(3, dtype=np.float32))
                target = _target(robot_name=robot_name)
                validated = MoveToStateValidator(
                    robot_name,
                    target.action_spec,
                    safety_limits=PoseSafetyLimits(np.full(3, -2.0), np.full(3, 2.0)),
                ).validate(target)
                validated.report.require_valid()
                plan = MoveToRecordedStatePlan.build(
                    target=target,
                    current_state=robot.get_current_pose_state(),
                    target_state=validated.values,
                    gripper_indices=validated.gripper_indices,
                    mapped_joint_names=validated.field_names,
                    conversions=validated.mappings,
                    constraints=PoseMotionConstraints(control_period_s=0.001, max_joint_step=0.05),
                )
                plan.require_integrity()

    def test_stop_tracking_error_and_stale_plan_abort(self) -> None:
        slow = _FakePoseRobot(np.zeros(3, dtype=np.float32), delay_s=0.02)
        controller = PoseMoveController(slow)
        controller.start(_plan(slow, _target()))
        time.sleep(0.01)
        self.assertTrue(controller.stop(wait=True, timeout=2.0))
        self.assertGreater(slow.stop_count, 0)

        stalled = _FakePoseRobot(np.zeros(3, dtype=np.float32), stall=True)
        failed = PoseMoveController(stalled)
        failed.start(_plan(stalled, _target()))
        self.assertTrue(failed.join(timeout=2.0))
        self.assertIsNotNone(failed.error())

        changed = _FakePoseRobot(np.zeros(3, dtype=np.float32))
        plan = _plan(changed, _target())
        changed.state[0] = 1.0
        with self.assertRaises(Exception):
            PoseMoveController(changed).start(plan)

    def test_pose_lease_conflict_and_atomic_handoff_replacement(self) -> None:
        manager = ResourceLeaseManager()
        robot = ResourceRequest(ResourceType.ROBOT_CONTROL, "piper")
        cameras = ResourceRequest(ResourceType.CAMERAS, "piper:head")
        first = manager.acquire("pose-session", (robot,))
        with self.assertRaises(ResourceLeaseConflict):
            manager.acquire("other", (robot,))
        handoff = first.replace((robot, cameras))
        self.assertEqual(manager.owner_of(robot), "pose-session")
        self.assertEqual(manager.owner_of(cameras), "pose-session")
        handoff.release()
        self.assertIsNone(manager.owner_of(robot))


if __name__ == "__main__":
    unittest.main()
