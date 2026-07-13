from __future__ import annotations

import time
import unittest
from typing import Any

import numpy as np

from mp_real.common.camera import BlackCamera
from mp_real.runtime.models import ActionSpec, RobotState
from mp_real.web.server import CAMERA_NAMES, PiperWebRuntime, _default_args


class _FakeWebRobot:
    action_spec = ActionSpec(14, 14, 6, "rad", CAMERA_NAMES)

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []
        self.reset_count = 0
        self.closed = False
        self.runtime_configs: list[Any] = []

    def read_state(self) -> RobotState:
        return RobotState(np.zeros(14, dtype=np.float32), time.monotonic())

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        executed = np.asarray(target, dtype=np.float32).copy()
        self.executed.append(executed)
        return executed

    def reset(self) -> None:
        self.reset_count += 1

    def configure_runtime(self, config: object) -> None:
        self.runtime_configs.append(config)

    def close(self) -> None:
        self.closed = True


class _FakeWebPolicy:
    metadata = {"model": "fake"}

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.closed = False

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        return {"actions": np.zeros((1, 14), dtype=np.float32)}

    def close(self) -> None:
        self.closed = True


def _runtime(args: Any, robot: _FakeWebRobot, policy: _FakeWebPolicy) -> PiperWebRuntime:
    return PiperWebRuntime(
        args,
        robot_factory=lambda name, config: robot,
        policy_client_factory=lambda server_url, api_key, timeout: policy,
        camera_factory=lambda config: {
            name: BlackCamera(name, width=config.camera_width, height=config.camera_height) for name in CAMERA_NAMES
        },
    )


class WebRuntimeTests(unittest.TestCase):
    def test_piper_web_runs_once_with_fake_robot_and_policy(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = False
        args.replan_steps = 1
        args.max_steps = 1
        args.fps = 1000.0
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)

        runtime.start()
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        status = runtime.status()

        self.assertFalse(status["running"])
        self.assertEqual(status["phase"], "stopped")
        self.assertEqual(status["step"], 1)
        self.assertEqual(len(robot.executed), 1)
        self.assertEqual(len(policy.observations), 1)
        self.assertEqual(len(robot.runtime_configs), 1)
        self.assertEqual(set(policy.observations[0]["images"]), set(CAMERA_NAMES))

        runtime.disconnect()
        self.assertTrue(robot.closed)
        self.assertTrue(policy.closed)

    def test_piper_web_infer_only_uses_controller_without_executing(self) -> None:
        args = _default_args(camera_profile="black")
        args.infer_only = True
        args.infer_only_chunks = 3
        args.replan_steps = 1
        args.max_steps = 2
        args.fps = 1000.0
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)

        runtime.start()
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        status = runtime.status()

        self.assertEqual(status["step"], 2)
        self.assertEqual(len(policy.observations), 2)
        self.assertEqual(robot.executed, [])
        self.assertEqual(robot.reset_count, 0)
        runtime.disconnect()

    def test_piper_web_rtc_switch_uses_shared_runtime(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = True
        args.hold_last_action = False
        args.replan_steps = 1
        args.max_steps = 1
        args.fps = 100.0
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)

        runtime.start()
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=5.0, raise_on_error=True))

        self.assertEqual(runtime.status()["step"], 1)
        self.assertEqual(len(robot.executed), 1)
        runtime.disconnect()


if __name__ == "__main__":
    unittest.main()
