from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Mapping
from typing import Any

import numpy as np

from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.controller import ControllerAlreadyRunningError, RuntimeController, _GenerationHooks
from mp_real.runtime.inference import InferenceHooks
from mp_real.runtime.models import ActionSpec, RobotState


class _FakeRobot:
    action_spec = ActionSpec(2, 2, 1, "rad", ())

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []
        self.executed_event = threading.Event()
        self.closed = False

    def read_state(self) -> RobotState:
        return RobotState(np.zeros(2, dtype=np.float32), time.monotonic())

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        result = np.asarray(target, dtype=np.float32).copy()
        self.executed.append(result)
        self.executed_event.set()
        return result

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    name = "controller-fake"

    def __init__(self, robot: _FakeRobot) -> None:
        self.robot = robot

    def observe(self) -> dict[str, Any]:
        return {"state": self.robot.read_state().values, "prompt": "test"}

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        return self.robot.action_spec.validate_chunk(response["actions"])[:replan_steps]

    def initial_action(self) -> np.ndarray:
        return self.robot.read_state().values

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return np.asarray(action, dtype=np.float32)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return self.robot.execute_transition(previous, target)

    def infer_only_metadata(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


class _Policy:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.closed = False

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        if self.error is not None:
            raise self.error
        return {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)}

    def close(self) -> None:
        self.closed = True


class _BlockingPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        self.entered.set()
        self.release.wait(timeout=2.0)
        return {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)}


class _RecordingHooks(InferenceHooks):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.error: BaseException | None = None

    def on_loop_started(self, mode: str, config: InferenceLoopConfig) -> None:
        del config
        self.events.append(f"started:{mode}")

    def on_observation(self, observation: Mapping[str, Any]) -> None:
        del observation
        self.events.append("observation")

    def on_inference_started(self, observation: Mapping[str, Any]) -> None:
        del observation
        self.events.append("inference_started")

    def on_inference_finished(self, response: Mapping[str, Any], elapsed_s: float) -> None:
        del response, elapsed_s
        self.events.append("inference_finished")

    def on_chunk_received(self, chunk: np.ndarray) -> None:
        del chunk
        self.events.append("chunk")

    def on_action_selected(self, step: int, action: np.ndarray) -> None:
        del step, action
        self.events.append("selected")

    def on_action_stabilized(self, step: int, action: np.ndarray) -> None:
        del step, action
        self.events.append("stabilized")

    def on_action_executed(self, step: int, action: np.ndarray) -> None:
        del step, action
        self.events.append("executed")

    def on_loop_stopped(self, mode: str) -> None:
        self.events.append(f"stopped:{mode}")

    def on_error(self, error: BaseException) -> None:
        self.error = error


def _config(*, max_steps: int | None = None) -> InferenceLoopConfig:
    return InferenceLoopConfig(
        fps=1000.0,
        replan_steps=1,
        max_steps=max_steps,
        use_rtc=False,
        rtc_replan_stride=0,
        rtc_prefetch_steps=0,
        rtc_exp_weight=0.0,
        hold_last_action=True,
        infer_only=False,
        infer_only_chunks=1,
        infer_only_output=None,
        prompt="test",
        log_timing=False,
    )


def _controller(
    *, policy: _Policy | None = None, max_steps: int | None = None, hooks: InferenceHooks | None = None
) -> tuple[RuntimeController, _FakeRobot, _Policy]:
    robot = _FakeRobot()
    policy = policy or _Policy()
    controller = RuntimeController(robot, _FakeAdapter(robot), policy, _config(max_steps=max_steps), hooks=hooks)
    return controller, robot, policy


class RuntimeControllerTests(unittest.TestCase):
    def test_controller_starts_and_stops(self) -> None:
        controller, robot, policy = _controller()
        generation = controller.start()
        self.assertTrue(robot.executed_event.wait(timeout=1.0))

        self.assertTrue(controller.stop(wait=True, timeout=1.0))
        self.assertEqual(generation, 1)
        self.assertFalse(controller.status().running)
        controller.close()
        self.assertTrue(robot.closed)
        self.assertTrue(policy.closed)

    def test_repeated_start_is_rejected(self) -> None:
        policy = _BlockingPolicy()
        controller, _, _ = _controller(policy=policy)
        controller.start()
        self.assertTrue(policy.entered.wait(timeout=1.0))

        with self.assertRaisesRegex(ControllerAlreadyRunningError, "already running"):
            controller.start()

        policy.release.set()
        self.assertTrue(controller.stop(wait=True, timeout=1.0))
        controller.close()

    def test_repeated_stop_is_idempotent(self) -> None:
        controller, robot, _ = _controller()
        controller.start()
        self.assertTrue(robot.executed_event.wait(timeout=1.0))

        self.assertTrue(controller.stop(wait=True, timeout=1.0))
        self.assertTrue(controller.stop(wait=True, timeout=1.0))
        controller.close()

    def test_policy_error_is_propagated(self) -> None:
        hooks = _RecordingHooks()
        controller, _, _ = _controller(policy=_Policy(error=ValueError("policy failed")), max_steps=1, hooks=hooks)
        controller.start()

        with self.assertRaisesRegex(ValueError, "policy failed"):
            controller.join(timeout=1.0, raise_on_error=True)
        self.assertIsInstance(controller.status().error, ValueError)
        self.assertIsInstance(hooks.error, ValueError)
        controller.close()

    def test_stop_event_ends_control_loop(self) -> None:
        controller, robot, _ = _controller()
        controller.start()
        self.assertTrue(robot.executed_event.wait(timeout=1.0))

        controller.stop()

        self.assertTrue(controller.join(timeout=1.0))
        self.assertFalse(controller.status().running)
        controller.close()

    def test_late_old_generation_hook_result_is_dropped(self) -> None:
        controller, _, _ = _controller(max_steps=1)
        old_generation = controller.start()
        self.assertTrue(controller.join(timeout=1.0, raise_on_error=True))
        stale_delegate = _RecordingHooks()
        stale_hooks = _GenerationHooks(controller, old_generation, stale_delegate)

        new_generation = controller.start()
        self.assertTrue(controller.join(timeout=1.0, raise_on_error=True))
        stale_hooks.on_chunk_received(np.asarray([[9.0, 9.0]], dtype=np.float32))

        self.assertEqual(new_generation, old_generation + 1)
        self.assertEqual(stale_delegate.events, [])
        controller.close()

    def test_sync_hooks_run_in_order(self) -> None:
        hooks = _RecordingHooks()
        controller, _, _ = _controller(max_steps=1, hooks=hooks)
        controller.start()
        self.assertTrue(controller.join(timeout=1.0, raise_on_error=True))

        self.assertEqual(
            hooks.events,
            [
                "started:sync",
                "observation",
                "inference_started",
                "inference_finished",
                "chunk",
                "selected",
                "stabilized",
                "executed",
                "stopped:sync",
            ],
        )
        controller.close()


if __name__ == "__main__":
    unittest.main()
