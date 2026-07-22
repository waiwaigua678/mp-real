from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from mp_real.robots.rm2 import infer as infer_rm2
from mp_real.runtime.models import RobotState


def _wait_until(predicate, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _target(args: infer_rm2.Args, left_gripper: float, right_gripper: float) -> np.ndarray:
    target = np.zeros(infer_rm2.action_dim(args), dtype=np.float32)
    target[2 * args.joint_dof] = left_gripper
    target[2 * args.joint_dof + 1] = right_gripper
    return target


class _TrackingArm(infer_rm2.MockArm):
    def __init__(self, name: str, args: infer_rm2.Args) -> None:
        super().__init__(
            name,
            args.joint_dof,
            0.0,
            np.zeros(args.joint_dof, dtype=np.float32),
        )
        self.gripper_commands: list[float] = []
        self.closed = False

    def command_gripper(self, value: float, args: infer_rm2.Args) -> None:
        self.gripper_commands.append(float(value))
        super().command_gripper(value, args)

    def close(self) -> None:
        self.closed = True


class _BlockingArm(_TrackingArm):
    def __init__(self, name: str, args: infer_rm2.Args) -> None:
        super().__init__(name, args)
        self.first_command_started = threading.Event()
        self.release_first_command = threading.Event()

    def command_gripper(self, value: float, args: infer_rm2.Args) -> None:
        self.gripper_commands.append(float(value))
        if len(self.gripper_commands) == 1:
            self.first_command_started.set()
            if not self.release_first_command.wait(2.0):
                raise TimeoutError("test gripper remained blocked")
        infer_rm2.MockArm.command_gripper(self, value, args)


class _FailingArm(_TrackingArm):
    def __init__(self, name: str, args: infer_rm2.Args) -> None:
        super().__init__(name, args)
        self.failure_started = threading.Event()

    def command_gripper(self, value: float, args: infer_rm2.Args) -> None:
        del value, args
        self.failure_started.set()
        raise ValueError("mock gripper failure")


class Rm2AsyncGripperTests(unittest.TestCase):
    def test_async_worker_is_non_daemon_and_coalesces_to_latest_pair(self) -> None:
        args = infer_rm2.Args(
            robot_backend="mock",
            interpolate_actions=False,
            gripper_command_rate_hz=1000.0,
            gripper_command_deadband=0.0,
            gripper_flush_timeout=1.0,
        )
        left = _BlockingArm("left", args)
        right = _TrackingArm("right", args)
        robot = infer_rm2.Rm2Robot(left, right, args)

        try:
            first = _target(args, 100.0, 110.0)
            robot.execute_transition(None, first)
            self.assertTrue(left.first_command_started.wait(1.0))

            commander = robot.gripper_commander
            self.assertIsInstance(commander, infer_rm2.AsyncGripperCommander)
            assert isinstance(commander, infer_rm2.AsyncGripperCommander)
            self.assertTrue(commander.is_alive)
            self.assertFalse(commander.thread_daemon)

            robot.execute_transition(first, _target(args, 200.0, 210.0))
            robot.execute_transition(first, _target(args, 300.0, 310.0))
            self.assertEqual(commander.coalesced_count, 1)

            left.release_first_command.set()
            self.assertTrue(_wait_until(lambda: len(right.gripper_commands) == 2))
            self.assertEqual(left.gripper_commands, [100.0, 300.0])
            self.assertEqual(right.gripper_commands, [110.0, 310.0])
        finally:
            left.release_first_command.set()
            robot.close()

        self.assertFalse(commander.is_alive)
        self.assertTrue(left.closed)
        self.assertTrue(right.closed)

    def test_worker_failure_keeps_original_type_and_cleans_up(self) -> None:
        args = infer_rm2.Args(
            robot_backend="mock",
            interpolate_actions=False,
            gripper_command_deadband=0.0,
            gripper_flush_timeout=1.0,
        )
        left = _FailingArm("left", args)
        right = _TrackingArm("right", args)
        robot = infer_rm2.Rm2Robot(left, right, args)

        robot.execute_transition(None, _target(args, 100.0, 110.0))
        self.assertTrue(left.failure_started.wait(1.0))
        commander = robot.gripper_commander
        assert isinstance(commander, infer_rm2.AsyncGripperCommander)

        def failed() -> bool:
            try:
                commander.raise_error()
            except ValueError:
                return True
            return False

        self.assertTrue(_wait_until(failed))
        with self.assertRaisesRegex(ValueError, "mock gripper failure"):
            robot.read_state()
        with self.assertRaisesRegex(ValueError, "mock gripper failure"):
            robot.close()

        self.assertFalse(commander.is_alive)
        self.assertTrue(left.closed)
        self.assertTrue(right.closed)

    def test_async_deadband_skips_small_changes(self) -> None:
        args = infer_rm2.Args(
            robot_backend="mock",
            interpolate_actions=False,
            gripper_command_rate_hz=1000.0,
            gripper_command_deadband=0.02,
            gripper_flush_timeout=1.0,
        )
        left = _TrackingArm("left", args)
        right = _TrackingArm("right", args)
        robot = infer_rm2.Rm2Robot(left, right, args)

        robot.execute_transition(None, _target(args, 100.0, 110.0))
        commander = robot.gripper_commander
        assert isinstance(commander, infer_rm2.AsyncGripperCommander)
        self.assertTrue(commander.flush(1, timeout=1.0))

        robot.execute_transition(None, _target(args, 110.0, 120.0))
        self.assertTrue(commander.flush(2, timeout=1.0))
        self.assertEqual(left.gripper_commands, [100.0])
        self.assertEqual(right.gripper_commands, [110.0])

        robot.execute_transition(None, _target(args, 130.0, 140.0))
        self.assertTrue(commander.flush(3, timeout=1.0))
        self.assertEqual(left.gripper_commands, [100.0, 130.0])
        self.assertEqual(right.gripper_commands, [110.0, 140.0])
        robot.close()

    def test_close_does_not_delete_arms_while_worker_is_alive(self) -> None:
        args = infer_rm2.Args(
            robot_backend="mock",
            interpolate_actions=False,
            gripper_command_deadband=0.0,
            gripper_flush_timeout=0.02,
        )
        left = _BlockingArm("left", args)
        right = _TrackingArm("right", args)
        robot = infer_rm2.Rm2Robot(left, right, args)
        robot.execute_transition(None, _target(args, 100.0, 110.0))
        self.assertTrue(left.first_command_started.wait(1.0))
        commander = robot.gripper_commander
        assert isinstance(commander, infer_rm2.AsyncGripperCommander)

        try:
            with self.assertRaises(TimeoutError):
                robot.close()
            self.assertTrue(commander.is_alive)
            self.assertFalse(left.closed)
            self.assertFalse(right.closed)
        finally:
            left.release_first_command.set()

        self.assertTrue(_wait_until(lambda: not commander.is_alive))
        robot.close()
        self.assertTrue(left.closed)
        self.assertTrue(right.closed)

    def test_zero_flush_timeout_still_joins_an_idle_worker(self) -> None:
        args = infer_rm2.Args(
            robot_backend="mock",
            interpolate_actions=False,
            gripper_flush_timeout=0.0,
        )
        left = _TrackingArm("left", args)
        right = _TrackingArm("right", args)
        robot = infer_rm2.Rm2Robot(left, right, args)

        robot.execute_transition(None, _target(args, 20.0, 30.0))
        commander = robot.gripper_commander
        assert isinstance(commander, infer_rm2.AsyncGripperCommander)
        self.assertTrue(commander.flush(1, timeout=1.0))

        robot.close()

        self.assertFalse(commander.is_alive)
        self.assertTrue(left.closed)
        self.assertTrue(right.closed)

    def test_sync_option_blocks_calling_thread_until_gripper_returns(self) -> None:
        args = infer_rm2.Args(
            robot_backend="mock",
            async_gripper=False,
            interpolate_actions=False,
        )
        left = _BlockingArm("left", args)
        right = _TrackingArm("right", args)
        robot = infer_rm2.Rm2Robot(left, right, args)
        errors: list[BaseException] = []

        def execute() -> None:
            try:
                robot.execute_transition(None, _target(args, 25.0, 35.0))
            except BaseException as exc:
                errors.append(exc)

        caller = threading.Thread(target=execute, name="test-sync-gripper-caller", daemon=False)
        caller.start()
        self.assertTrue(left.first_command_started.wait(1.0))
        self.assertTrue(caller.is_alive())
        left.release_first_command.set()
        caller.join(timeout=1.0)

        self.assertFalse(caller.is_alive())
        self.assertEqual(errors, [])
        self.assertIsInstance(robot.gripper_commander, infer_rm2.SyncGripperCommander)
        self.assertEqual(left.gripper_commands, [25.0])
        self.assertEqual(right.gripper_commands, [35.0])
        robot.close()

    def test_policy_state_transform_preserves_feedback_metadata_and_copies_values(self) -> None:
        health = {"status": "ok"}
        feedback = RobotState(
            values=np.arange(14, dtype=np.float32),
            timestamp_monotonic=12.5,
            timestamp_monotonic_ns=12_500_000_001,
            source_timestamp_ns=99,
            health=health,
        )
        static_args = infer_rm2.Args(use_static_left_state=True)
        policy_state = infer_rm2.policy_state_from_feedback(feedback, static_args)

        np.testing.assert_allclose(policy_state.values[:6], static_args.static_left_joints)
        np.testing.assert_array_equal(policy_state.values[6:], feedback.values[6:])
        self.assertFalse(np.shares_memory(policy_state.values, feedback.values))
        self.assertEqual(policy_state.timestamp_monotonic, feedback.timestamp_monotonic)
        self.assertEqual(policy_state.timestamp_monotonic_ns, feedback.timestamp_monotonic_ns)
        self.assertEqual(policy_state.source_timestamp_ns, feedback.source_timestamp_ns)
        self.assertIs(policy_state.health, health)

        dynamic_state = infer_rm2.policy_state_from_feedback(
            feedback,
            infer_rm2.Args(use_static_left_state=False),
        )
        np.testing.assert_array_equal(dynamic_state.values, feedback.values)
        self.assertFalse(np.shares_memory(dynamic_state.values, feedback.values))

    def test_legacy_deadband_conversion_is_preserved(self) -> None:
        raw = infer_rm2.Args(policy_gripper_unit="raw", gripper_command_deadband=0.02)
        normalized = infer_rm2.Args(policy_gripper_unit="normalized", gripper_command_deadband=0.02)
        raw_absolute = infer_rm2.Args(policy_gripper_unit="raw", gripper_command_deadband=2.0)

        self.assertAlmostEqual(infer_rm2.gripper_deadband_to_policy(raw), 19.98)
        self.assertEqual(infer_rm2.gripper_deadband_to_policy(normalized), 0.02)
        self.assertEqual(infer_rm2.gripper_deadband_to_policy(raw_absolute), 2.0)


if __name__ == "__main__":
    unittest.main()
