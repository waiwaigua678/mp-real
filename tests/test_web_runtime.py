from __future__ import annotations

import threading
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
        self.timeout = 3.0
        self.timeout_history: list[float] = []

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        return {"actions": np.zeros((1, 14), dtype=np.float32)}

    def set_timeout(self, timeout_s: float) -> None:
        self.timeout = timeout_s
        self.timeout_history.append(timeout_s)

    def close(self) -> None:
        self.closed = True


class _TrackedCamera(BlackCamera):
    def __init__(self, name: str, *, fail_reads: bool = False) -> None:
        super().__init__(name, width=8, height=6)
        self.fail_reads = fail_reads
        self.closed = False

    def read(self, *, timeout: float = 2.0) -> np.ndarray:
        if self.fail_reads:
            raise RuntimeError(f"{self.name} camera failed")
        return super().read(timeout=timeout)

    def close(self) -> None:
        self.closed = True


class _SlowFirstPolicy(_FakeWebPolicy):
    def __init__(self, delay_s: float, *, actions: list[float] | None = None) -> None:
        super().__init__()
        self.delay_s = delay_s
        self.actions = actions or [0.0]

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        call_index = len(self.observations) - 1
        if call_index == 0:
            time.sleep(self.delay_s)
            if self.delay_s > self.timeout:
                raise TimeoutError(f"fake policy exceeded {self.timeout:.3f}s")
        value = self.actions[min(call_index, len(self.actions) - 1)]
        return {"actions": np.full((1, 14), value, dtype=np.float32)}


class _BlockingWarmupPolicy(_FakeWebPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.released = threading.Event()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        self.entered.set()
        self.released.wait(timeout=2.0)
        if self.closed:
            raise TimeoutError("fake policy closed during warmup")
        return {"actions": np.zeros((1, 14), dtype=np.float32)}

    def close(self) -> None:
        super().close()
        self.released.set()


def _runtime(args: Any, robot: _FakeWebRobot, policy: _FakeWebPolicy) -> PiperWebRuntime:
    return PiperWebRuntime(
        args,
        robot_factory=lambda name, config: robot,
        policy_client_factory=lambda server_url, api_key, timeout: policy,
        camera_factory=lambda config: {
            name: BlackCamera(name, width=config.camera_width, height=config.camera_height) for name in CAMERA_NAMES
        },
    )


def _wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for runtime state")


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
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"})
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        status = runtime.status()

        self.assertFalse(status["running"])
        self.assertEqual(status["phase"], "stopped")
        self.assertEqual(status["step"], 1)
        self.assertEqual(len(robot.executed), 1)
        self.assertEqual(len(policy.observations), 2)
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
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"})
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        status = runtime.status()

        self.assertEqual(status["step"], 2)
        self.assertEqual(len(policy.observations), 3)
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
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"}, timeout=5.0)
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=5.0, raise_on_error=True))

        self.assertEqual(runtime.status()["step"], 1)
        self.assertEqual(len(robot.executed), 1)
        runtime.disconnect()

    def test_camera_preview_creates_only_cameras_and_can_stop_repeatedly(self) -> None:
        args = _default_args(camera_profile="black")
        calls = {"robot": 0, "policy": 0, "camera": 0}
        cameras = {name: _TrackedCamera(name) for name in CAMERA_NAMES}
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: calls.__setitem__("robot", calls["robot"] + 1),
            policy_client_factory=lambda server_url, api_key, timeout: calls.__setitem__("policy", calls["policy"] + 1),
            camera_factory=lambda config: calls.__setitem__("camera", calls["camera"] + 1) or cameras,
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "camera_preview"})

        status = runtime.start()
        self.assertEqual(status["phase"], "previewing")
        _wait_until(lambda: all(runtime.status()["frames"][name]["sequence"] > 0 for name in CAMERA_NAMES))
        self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 1})

        self.assertEqual(runtime.start()["phase"], "previewing")
        runtime.stop(wait=True)
        runtime.stop(wait=True)
        runtime.disconnect()
        self.assertTrue(all(camera.closed for camera in cameras.values()))

    def test_camera_preview_isolates_one_camera_read_failure(self) -> None:
        args = _default_args(camera_profile="black")
        cameras = {
            "cam_head": _TrackedCamera("cam_head", fail_reads=True),
            "cam_left_wrist": _TrackedCamera("cam_left_wrist"),
            "cam_right_wrist": _TrackedCamera("cam_right_wrist"),
        }
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: self.fail("camera preview must not create a robot"),
            policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                "camera preview must not create a policy"
            ),
            camera_factory=lambda config: cameras,
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "camera_preview"})

        runtime.start()
        _wait_until(
            lambda: runtime.status()["frames"]["cam_head"]["error"] is not None
            and runtime.status()["frames"]["cam_left_wrist"]["sequence"] > 0
            and runtime.status()["frames"]["cam_right_wrist"]["sequence"] > 0
        )
        status = runtime.status()
        self.assertGreater(status["frames"]["cam_left_wrist"]["sequence"], 0)
        self.assertGreater(status["frames"]["cam_right_wrist"]["sequence"], 0)
        runtime.disconnect()
        self.assertTrue(all(camera.closed for camera in cameras.values()))
        self.assertIsNone(runtime._camera_thread)

    def test_offline_replay_creates_no_resources(self) -> None:
        args = _default_args(camera_profile="black")
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: self.fail("offline replay must not create a robot"),
            policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                "offline replay must not create a policy"
            ),
            camera_factory=lambda config: self.fail("offline replay must not create cameras"),
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "offline_replay"})

        status = runtime.connect()
        self.assertEqual(status["phase"], "offline_replay")
        self.assertEqual(runtime.start()["phase"], "offline_replay")
        runtime.disconnect()

    def test_deployment_creates_robot_policy_and_cameras(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.max_steps = 1
        args.replan_steps = 1
        args.max_action_step = 0.0
        args.max_joint_step = 0.0
        args.action_smoothing = 0.0
        args.use_rtc = False
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        calls = {"robot": 0, "policy": 0, "camera": 0}
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: calls.__setitem__("robot", calls["robot"] + 1) or robot,
            policy_client_factory=lambda server_url, api_key, timeout: calls.__setitem__(
                "policy", calls["policy"] + 1
            )
            or policy,
            camera_factory=lambda config: calls.__setitem__("camera", calls["camera"] + 1)
            or {name: _TrackedCamera(name) for name in CAMERA_NAMES},
        )
        self.addCleanup(runtime.disconnect)

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] == "stopped")
        self.assertEqual(calls, {"robot": 1, "policy": 1, "camera": 1})

    def test_warmup_discards_its_action_and_prefetches_fresh_first_chunk(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = False
        args.max_steps = 1
        args.replan_steps = 1
        args.max_action_step = 0.0
        args.max_joint_step = 0.0
        args.action_smoothing = 0.0
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03, actions=[9.0, 2.0])
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"policy_inference_timeout_s": 0.01, "policy_warmup_timeout_s": 0.1})

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "warmup_failed"})
        self.assertEqual(runtime.status()["phase"], "stopped")
        np.testing.assert_array_equal(robot.executed, [np.full(14, 2.0, dtype=np.float32)])
        self.assertEqual(len(policy.observations), 2)
        self.assertGreaterEqual(policy.timeout_history[0], 0.1)
        self.assertIn(0.01, policy.timeout_history)
        self.assertIsNotNone(runtime.status()["metrics"]["cold_inference_latency_ms"])
        self.assertIsNotNone(runtime.status()["metrics"]["first_live_inference_latency_ms"])

    def test_warmup_timeout_reports_its_root_cause_without_motion(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.max_steps = 1
        args.replan_steps = 1
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03)
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"policy_warmup_timeout_s": 0.01})

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] == "warmup_failed")
        status = runtime.status()
        self.assertIn("PolicyWarmupTimeout", status["last_error"])
        self.assertEqual(robot.executed, [])

    def test_without_warmup_slow_first_rtc_request_reproduces_the_old_failure(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = True
        args.hold_last_action = False
        args.max_steps = 1
        args.replan_steps = 1
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03)
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config(
            {
                "policy_warmup_enabled": False,
                "policy_prefetch_first_chunk": False,
                "policy_inference_timeout_s": 0.01,
            }
        )

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] == "error", timeout=5.0)
        self.assertIn("TimeoutError: fake policy exceeded", runtime.status()["last_error"])
        self.assertEqual(robot.executed, [])

    def test_rtc_uses_prefetched_chunk_without_waiting_for_cold_inference(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = True
        args.hold_last_action = False
        args.max_steps = 1
        args.replan_steps = 1
        args.fps = 100.0
        args.max_action_step = 0.0
        args.max_joint_step = 0.0
        args.action_smoothing = 0.0
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03, actions=[8.0, 3.0])
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"policy_inference_timeout_s": 0.01, "policy_warmup_timeout_s": 0.1})

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "warmup_failed"}, timeout=5.0)
        self.assertEqual(runtime.status()["phase"], "stopped")
        np.testing.assert_array_equal(robot.executed, [np.full(14, 3.0, dtype=np.float32)])

    def test_stop_and_disconnect_during_warmup_close_the_policy_worker(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        robot = _FakeWebRobot()
        policy = _BlockingWarmupPolicy()
        runtime = _runtime(args, robot, policy)

        runtime.start()
        self.assertTrue(policy.entered.wait(timeout=1.0))
        runtime.stop(wait=True)
        _wait_until(lambda: not runtime._startup_active_locked())
        self.assertEqual(robot.executed, [])
        runtime.disconnect()
        self.assertTrue(policy.closed)

    def test_disconnect_during_warmup_releases_the_policy_worker(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        robot = _FakeWebRobot()
        policy = _BlockingWarmupPolicy()
        runtime = _runtime(args, robot, policy)

        runtime.start()
        self.assertTrue(policy.entered.wait(timeout=1.0))
        runtime.disconnect()
        self.assertTrue(policy.closed)
        self.assertFalse(runtime.status()["connected"])
        self.assertFalse(runtime._startup_active_locked())


if __name__ == "__main__":
    unittest.main()
