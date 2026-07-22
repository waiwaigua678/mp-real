from __future__ import annotations

import threading
import unittest

import numpy as np

from mp_real.robots.rm2 import infer as infer_rm2
from mp_real.web.profiles import RM2_WEB_PROFILE
from mp_real.web.server import ApiError, RobotWebRuntime


class _TrackingArm(infer_rm2.MockArm):
    def __init__(self, name: str, joint_dof: int, last_gripper: float, joints: np.ndarray) -> None:
        super().__init__(name, joint_dof, last_gripper, joints)
        self.joint_commands: list[np.ndarray] = []
        self.gripper_commands: list[float] = []

    def command_joints(self, joints: np.ndarray, args: infer_rm2.Args) -> None:
        self.joint_commands.append(np.asarray(joints, dtype=np.float32).copy())
        super().command_joints(joints, args)

    def command_gripper(self, value: float, args: infer_rm2.Args) -> None:
        self.gripper_commands.append(float(value))
        super().command_gripper(value, args)


class Rm2VerifiedProfileTests(unittest.TestCase):
    def test_cli_defaults_match_verified_right_arm_follow_profile(self) -> None:
        args = infer_rm2.Args()

        self.assertEqual(args.server_url, "ws://10.30.20.47:8000")
        self.assertEqual(args.camera_backend, "ros")
        self.assertEqual(args.max_steps, 1500)
        self.assertEqual(args.arm_command, "follow")
        self.assertEqual(args.fps, 10.0)
        self.assertEqual(args.replan_steps, 10)
        self.assertEqual(args.speed_percent, 35)
        self.assertEqual(args.max_action_step_deg, 0.0)
        self.assertEqual(args.max_joint_step_deg, 0.0)
        self.assertEqual(args.action_smoothing, 0.1)
        self.assertEqual(args.policy_gripper_unit, "raw")
        self.assertTrue(args.use_static_left_state)
        self.assertFalse(args.command_left_arm)
        self.assertTrue(args.command_right_arm)

    def test_static_left_state_and_raw_gripper_are_policy_observation_values(self) -> None:
        args = infer_rm2.Args(robot_backend="mock")
        left = infer_rm2.MockArm("left", args.joint_dof, 123.0, np.full(args.joint_dof, 99.0, dtype=np.float32))
        right = infer_rm2.MockArm("right", args.joint_dof, 456.0, np.full(args.joint_dof, 10.0, dtype=np.float32))

        state = infer_rm2.read_state(left, right, args)

        np.testing.assert_allclose(state[: args.joint_dof], args.static_left_joints, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            state[args.joint_dof : 2 * args.joint_dof],
            np.deg2rad(np.full(args.joint_dof, 10.0, dtype=np.float32)),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(float(state[2 * args.joint_dof]), 123.0)
        self.assertEqual(float(state[2 * args.joint_dof + 1]), 456.0)

    def test_default_execution_sends_right_arm_joints_without_left_arm_joints(self) -> None:
        args = infer_rm2.Args(robot_backend="mock")
        left = _TrackingArm("left", args.joint_dof, 0.0, np.zeros(args.joint_dof, dtype=np.float32))
        right = _TrackingArm("right", args.joint_dof, 0.0, np.zeros(args.joint_dof, dtype=np.float32))
        target = np.zeros(infer_rm2.action_dim(args), dtype=np.float32)
        target[args.joint_dof : 2 * args.joint_dof] = np.linspace(0.1, 0.6, args.joint_dof, dtype=np.float32)
        target[2 * args.joint_dof] = 42.0
        target[2 * args.joint_dof + 1] = 500.0

        executed = infer_rm2.execute_action_transition(
            None,
            target,
            left,
            right,
            args,
            robot_lock=threading.Lock(),
        )

        self.assertEqual(left.joint_commands, [])
        self.assertEqual(len(right.joint_commands), 1)
        np.testing.assert_allclose(right.joint_commands[0], np.rad2deg(target[args.joint_dof : 2 * args.joint_dof]))
        self.assertEqual(left.gripper_commands, [42.0])
        self.assertEqual(right.gripper_commands, [500.0])
        np.testing.assert_array_equal(executed, target)

    def test_gripper_unit_conversions_are_explicit(self) -> None:
        raw = infer_rm2.Args(policy_gripper_unit="raw")
        normalized = infer_rm2.Args(policy_gripper_unit="normalized")

        self.assertEqual(infer_rm2.policy_gripper_to_robot_position(42.0, raw), 42)
        self.assertEqual(infer_rm2.gripper_position_to_policy(42.0, raw), 42.0)
        self.assertEqual(infer_rm2.policy_gripper_to_robot_position(0.5, normalized), 500)
        self.assertAlmostEqual(infer_rm2.gripper_position_to_policy(500.5, normalized), 0.5, places=6)

    def test_web_policy_uses_static_left_state_without_hiding_live_robot_feedback(self) -> None:
        args = infer_rm2.Args(robot_backend="mock", camera_backend="black", use_static_left_state=True)
        left = infer_rm2.MockArm(
            "left",
            args.joint_dof,
            123.0,
            np.full(args.joint_dof, 99.0, dtype=np.float32),
        )
        right = infer_rm2.MockArm(
            "right",
            args.joint_dof,
            456.0,
            np.full(args.joint_dof, 10.0, dtype=np.float32),
        )
        robot = infer_rm2.Rm2Robot(left, right, args)
        self.addCleanup(robot.close)
        images = {
            role: np.zeros((6, 8, 3), dtype=np.uint8)
            for role in RM2_WEB_PROFILE.camera_roles_for_args(args)
        }
        adapter = RM2_WEB_PROFILE.make_adapter(
            robot,
            args,
            lambda: (images, None),
            lambda stage, elapsed_s: None,
        )

        live_state = robot.read_state()
        pose_state = robot.get_current_pose_state()
        observation_state = adapter.observe()["state"]
        initial_action = adapter.initial_action()
        expected_live_left = np.deg2rad(np.full(args.joint_dof, 99.0, dtype=np.float32))

        np.testing.assert_allclose(live_state.values[: args.joint_dof], expected_live_left, atol=1e-7)
        np.testing.assert_allclose(pose_state.values[: args.joint_dof], expected_live_left, atol=1e-7)
        np.testing.assert_allclose(observation_state[: args.joint_dof], args.static_left_joints, atol=1e-7)
        np.testing.assert_allclose(initial_action[: args.joint_dof], args.static_left_joints, atol=1e-7)
        np.testing.assert_allclose(observation_state[args.joint_dof :], live_state.values[args.joint_dof :])
        np.testing.assert_allclose(initial_action[args.joint_dof :], live_state.values[args.joint_dof :])

    def test_web_policy_uses_live_left_feedback_when_static_state_is_disabled(self) -> None:
        args = infer_rm2.Args(robot_backend="mock", camera_backend="black", use_static_left_state=False)
        left = infer_rm2.MockArm(
            "left",
            args.joint_dof,
            123.0,
            np.full(args.joint_dof, 27.0, dtype=np.float32),
        )
        right = infer_rm2.MockArm(
            "right",
            args.joint_dof,
            456.0,
            np.full(args.joint_dof, -10.0, dtype=np.float32),
        )
        robot = infer_rm2.Rm2Robot(left, right, args)
        self.addCleanup(robot.close)
        images = {
            role: np.zeros((6, 8, 3), dtype=np.uint8)
            for role in RM2_WEB_PROFILE.camera_roles_for_args(args)
        }
        adapter = RM2_WEB_PROFILE.make_adapter(
            robot,
            args,
            lambda: (images, None),
            lambda stage, elapsed_s: None,
        )

        live_state = robot.read_state().values

        np.testing.assert_allclose(adapter.observe()["state"], live_state, atol=1e-7)
        np.testing.assert_allclose(adapter.initial_action(), live_state, atol=1e-7)

    def test_web_rm2_config_exposes_and_updates_verified_fields(self) -> None:
        runtime = RobotWebRuntime(RM2_WEB_PROFILE.default_args(), profile=RM2_WEB_PROFILE)
        self.addCleanup(runtime.disconnect)

        config = runtime.get_config()
        self.assertEqual(config["robot"], "rm2")
        self.assertEqual(config["server_url"], "ws://10.30.20.47:8000")
        self.assertEqual(config["rm2_arm_command"], "follow")
        self.assertEqual(config["policy_gripper_unit"], "raw")
        self.assertFalse(config["command_left_arm"])
        self.assertTrue(config["use_static_left_state"])
        self.assertEqual(config["action_smoothing"], 0.1)
        self.assertEqual(config["async_gripper"], True)
        self.assertEqual(config["gripper_command_rate_hz"], 10.0)
        self.assertEqual(config["gripper_command_deadband"], 0.02)
        self.assertEqual(config["gripper_flush_timeout"], 2.0)

        updated = runtime.update_config(
            {
                "camera_backend": "black",
                "arm_command": "move_j",
                "rm2_arm_command": "canfd",
                "policy_gripper_unit": "normalized",
                "speed_percent": 20,
                "max_joint_step_deg": 3.0,
                "max_action_step_deg": 4.0,
                "action_smoothing": 0.25,
                "async_gripper": False,
                "gripper_command_rate_hz": 12.0,
                "gripper_command_deadband": 0.05,
                "gripper_flush_timeout": 1.5,
                "command_left_arm": True,
                "use_static_left_state": False,
            }
        )

        self.assertEqual(updated["camera_backend"], "black")
        self.assertEqual(updated["rm2_arm_command"], "canfd")
        self.assertEqual(updated["policy_gripper_unit"], "normalized")
        self.assertEqual(updated["speed_percent"], 20)
        self.assertEqual(updated["max_joint_step_deg"], 3.0)
        self.assertEqual(updated["max_action_step_deg"], 4.0)
        self.assertEqual(updated["action_smoothing"], 0.25)
        self.assertFalse(updated["async_gripper"])
        self.assertEqual(updated["gripper_command_rate_hz"], 12.0)
        self.assertEqual(updated["gripper_command_deadband"], 0.05)
        self.assertEqual(updated["gripper_flush_timeout"], 1.5)
        self.assertTrue(updated["command_left_arm"])
        self.assertFalse(updated["use_static_left_state"])

        with self.assertRaises(ApiError):
            runtime.update_config({"action_smoothing": None})

        self.assertEqual(runtime.get_config()["action_smoothing"], 0.25)


if __name__ == "__main__":
    unittest.main()
