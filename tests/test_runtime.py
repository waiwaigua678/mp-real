from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from mp_real.common.camera import BlackCamera, CameraFrame
from mp_real.common.runtime import RealTimeChunkingBuffer
from mp_real.robots import registry
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import run_rtc_loop, run_sync_loop
from mp_real.runtime.models import ActionSpec
from mp_real.runtime.observation import capture_observation
from mp_real.web.server import PiperWebRuntime, _default_args


class _Policy:
    def infer(self, observation: dict) -> dict:
        del observation
        return {"actions": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}


class _Adapter:
    name = "fake"

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []

    def observe(self) -> dict:
        return {"state": np.zeros(2, dtype=np.float32), "prompt": "test"}

    def decode_action_chunk(self, response: dict, replan_steps: int) -> np.ndarray:
        return ActionSpec(2, 2, 1, "rad", ()).validate_chunk(response["actions"])[:replan_steps]

    def initial_action(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return action

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        self.executed.append(target.copy())
        return target.copy()

    def infer_only_metadata(self, observation: dict) -> dict:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


class _BlockingPolicy:
    """Does not produce an RTC action until the controller requests stop."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event

    def infer(self, observation: dict) -> dict:
        del observation
        self.stop_event.wait(timeout=1.0)
        return {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)}


class _HoldLastAdapter(_Adapter):
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self.stop_event = stop_event

    def initial_action(self) -> np.ndarray:
        return np.asarray([10.0, 20.0], dtype=np.float32)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        result = super().execute_transition(previous, target)
        self.stop_event.set()
        return result


class _FixedCamera:
    def __init__(self) -> None:
        self.frame = CameraFrame(
            image=np.full((6, 8, 3), 17, dtype=np.uint8),
            timestamp_monotonic=123.5,
            camera_timestamp=456.25,
            info={"frame_id": "fixed-camera-frame"},
        )

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame:
        del timeout
        return self.frame

    def camera_info(self) -> dict[str, str]:
        return {"source": "fixed-camera"}


def _loop_config(*, use_rtc: bool, max_steps: int | None = 2, hold_last_action: bool = True) -> InferenceLoopConfig:
    return InferenceLoopConfig(
        fps=1000.0,
        replan_steps=2,
        max_steps=max_steps,
        use_rtc=use_rtc,
        rtc_replan_stride=0,
        rtc_prefetch_steps=0,
        rtc_exp_weight=0.0,
        hold_last_action=hold_last_action,
        infer_only=False,
        infer_only_chunks=1,
        infer_only_output=None,
        prompt="test",
        log_timing=False,
    )


class RuntimeTests(unittest.TestCase):
    def test_sync_loop_consumes_action_chunk(self) -> None:
        adapter = _Adapter()
        run_sync_loop(_Policy(), adapter, _loop_config(use_rtc=False))
        np.testing.assert_array_equal(adapter.executed, [[1.0, 2.0], [3.0, 4.0]])

    def test_rtc_buffer_fuses_overlapping_chunks(self) -> None:
        buffer = RealTimeChunkingBuffer(exp_weight=0.0)
        generation = buffer.get_generation()
        self.assertTrue(buffer.enqueue(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 0, generation))
        self.assertTrue(buffer.enqueue(np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32), 1, generation))

        np.testing.assert_array_equal(buffer.get_action(0), [1.0, 2.0])
        np.testing.assert_array_equal(buffer.get_action(1), [5.0, 6.0])

    def test_rtc_generation_update_rejects_old_result(self) -> None:
        buffer = RealTimeChunkingBuffer()
        old_generation = buffer.get_generation()
        buffer.clear()

        self.assertFalse(buffer.enqueue(np.asarray([[1.0, 2.0]], dtype=np.float32), 0, old_generation))
        self.assertIsNone(buffer.get_action(0))

    def test_rtc_holds_last_action_while_waiting_for_chunk(self) -> None:
        stop_event = threading.Event()
        adapter = _HoldLastAdapter(stop_event)

        run_rtc_loop(
            _BlockingPolicy(stop_event),
            adapter,
            _loop_config(use_rtc=True, max_steps=None, hold_last_action=True),
            stop_event=stop_event,
        )

        np.testing.assert_array_equal(adapter.executed, [[10.0, 20.0]])

    def test_observation_snapshot_keeps_timestamps(self) -> None:
        snapshot = capture_observation(
            {"head": BlackCamera("head", width=8, height=6)},
            read_state=lambda: np.asarray([1.0, 2.0], dtype=np.float32),
            prompt="test",
            resize_size=4,
            timeout=0.1,
            image_masks={"head": np.bool_(False)},
        )
        self.assertEqual(snapshot.images["head"].image.shape, (4, 4, 3))
        self.assertGreater(snapshot.images["head"].timestamp_monotonic, 0.0)
        self.assertGreater(snapshot.state.timestamp_monotonic, 0.0)
        self.assertEqual(snapshot.to_policy_observation()["prompt"], "test")

    def test_capture_observation_preserves_camera_and_state_timestamps(self) -> None:
        before_state_read = time.monotonic()
        snapshot = capture_observation(
            {"head": _FixedCamera()},
            read_state=lambda: np.asarray([3.0, 4.0], dtype=np.float32),
            prompt="timestamps",
            resize_size=4,
            timeout=0.1,
            image_masks={"head": np.bool_(True)},
            include_camera_params=True,
        )
        after_state_read = time.monotonic()

        sample = snapshot.images["head"]
        self.assertEqual(sample.timestamp_monotonic, 123.5)
        self.assertEqual(sample.camera_timestamp, 456.25)
        self.assertEqual(sample.info, {"frame_id": "fixed-camera-frame"})
        self.assertGreaterEqual(snapshot.state.timestamp_monotonic, before_state_read)
        self.assertLessEqual(snapshot.state.timestamp_monotonic, after_state_read)
        self.assertEqual(snapshot.camera_params, {"head": {"frame_id": "fixed-camera-frame"}})

    def test_robot_registry_registers_creates_and_rejects_duplicate_names(self) -> None:
        with patch.dict(registry._FACTORIES, {}, clear=True), patch.dict(registry._BUILTIN_MODULES, {}, clear=True):
            expected = object()
            registry.register_robot("fake", lambda config: (expected, config))

            self.assertEqual(registry.available_robots(), ("fake",))
            self.assertEqual(registry.create_robot("fake", {"name": "test"}), (expected, {"name": "test"}))
            with self.assertRaisesRegex(ValueError, "already registered"):
                registry.register_robot("fake", lambda config: config)
            with self.assertRaisesRegex(ValueError, "Unknown robot 'missing'; available: fake"):
                registry.create_robot("missing", None)

    def test_piper_web_config_json_round_trip(self) -> None:
        runtime = PiperWebRuntime(_default_args(camera_profile="black"), policy_timeout=4.0)
        serialized = json.loads(json.dumps(runtime.get_config()))

        updated = runtime.update_config(serialized)

        self.assertEqual(json.loads(json.dumps(updated)), serialized)
        self.assertEqual(runtime.get_config()["api_key"], "")
        self.assertEqual(runtime.get_config()["cam_head_backend"], "black")

    def test_piper_web_stop_is_idempotent_when_idle(self) -> None:
        runtime = PiperWebRuntime(_default_args(camera_profile="black"))

        first = runtime.stop()
        second = runtime.stop()

        self.assertFalse(first["running"])
        self.assertFalse(second["running"])
        self.assertEqual(second["phase"], "idle")


if __name__ == "__main__":
    unittest.main()
