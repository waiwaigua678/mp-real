from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from mp_real.common.camera import BlackCamera, CameraFrame, ROSImageCamera
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.controller import RuntimeController, _GenerationHooks
from mp_real.runtime.events import (
    ActionExecuted,
    ActionSelected,
    ActionStabilized,
    ChunkReceived,
    CompositeRuntimeEventSink,
    InMemoryRuntimeEventSink,
    ObservationCaptured,
    PolicyReady,
    PolicyWarmupFinished,
    PolicyWarmupStarted,
    RuntimeEvent,
    RuntimeEventHooks,
    RuntimeEventIdentity,
    RuntimeStarted,
    RuntimeStopped,
)
from mp_real.runtime.inference import run_rtc_loop, run_sync_loop
from mp_real.runtime.models import ActionSpec, CameraSample, RobotState
from mp_real.runtime.observation import capture_observation
from mp_real.runtime.startup import PolicyStartupConfig, PolicyStartupCoordinator


def _config(*, use_rtc: bool = False, max_steps: int | None = 1) -> InferenceLoopConfig:
    return InferenceLoopConfig(
        fps=1000.0,
        replan_steps=1,
        max_steps=max_steps,
        use_rtc=use_rtc,
        rtc_replan_stride=1,
        rtc_prefetch_steps=0,
        rtc_exp_weight=0.0,
        hold_last_action=True,
        infer_only=False,
        infer_only_chunks=1,
        infer_only_output=None,
        prompt="event test",
        log_timing=False,
    )


class _FixedCamera:
    def __init__(self, frame: CameraFrame) -> None:
        self.frame = frame

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame:
        del timeout
        return self.frame

    def camera_info(self) -> None:
        return None


class _Policy:
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        return {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)}


class _Adapter:
    name = "event-fake"

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []

    def observe(self) -> dict[str, Any]:
        return {"images": {}, "state": np.zeros(2, dtype=np.float32), "prompt": "event test"}

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        return ActionSpec(2, 2, 1, "rad", ()).validate_chunk(response["actions"])[:replan_steps]

    def initial_action(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return np.asarray(action, dtype=np.float32).copy()

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        result = np.asarray(target, dtype=np.float32).copy()
        self.executed.append(result)
        return result

    def infer_only_metadata(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


class _Robot:
    action_spec = ActionSpec(2, 2, 1, "rad", ())

    def __init__(self) -> None:
        self.closed = False

    def read_state(self) -> RobotState:
        return RobotState(np.zeros(2, dtype=np.float32), 10.0)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        return np.asarray(target, dtype=np.float32).copy()

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FailingSink:
    def emit(self, event: RuntimeEvent) -> None:
        del event
        raise RuntimeError("deliberate sink failure")


class RuntimeEventTests(unittest.TestCase):
    def test_black_camera_frame_ids_and_ns_timestamps_are_monotonic(self) -> None:
        camera = BlackCamera("black", width=4, height=3)
        first = camera.read_frame()
        second = camera.read_frame()

        self.assertEqual((first.frame_id, second.frame_id), (1, 2))
        self.assertLessEqual(first.timestamp_monotonic_ns, second.timestamp_monotonic_ns)
        self.assertEqual(first.timestamp_monotonic, first.timestamp_monotonic_ns / 1e9)

    def test_ros_cached_reads_do_not_invent_frame_ids(self) -> None:
        camera = object.__new__(ROSImageCamera)
        camera.name = "ros"
        camera.image_topic = "/fake/image_raw"
        camera._lock = threading.Lock()
        camera._cond = threading.Condition(camera._lock)
        camera._frame = None
        camera._frame_id = 0
        camera._info = None

        def message(sequence: int) -> SimpleNamespace:
            return SimpleNamespace(
                height=1,
                width=2,
                encoding="rgb8",
                data=bytes([1, 2, 3, 4, 5, 6]),
                header=SimpleNamespace(seq=sequence, stamp=SimpleNamespace(to_sec=lambda: 123.0)),
            )

        camera._image_cb(message(7))
        first = camera.read_frame()
        cached = camera.read_frame()
        camera._image_cb(message(8))
        next_frame = camera.read_frame()

        self.assertEqual(first.frame_id, cached.frame_id)
        self.assertEqual(first.source_sequence, 7)
        self.assertEqual(next_frame.frame_id, first.frame_id + 1)
        self.assertEqual(next_frame.source_sequence, 8)

    def test_observation_skew_and_legacy_timestamps_are_retained(self) -> None:
        left = _FixedCamera(
            CameraFrame(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                timestamp_monotonic=1.0,
                timestamp_monotonic_ns=1_000,
                frame_id=4,
            )
        )
        right = _FixedCamera(
            CameraFrame(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                timestamp_monotonic=1.0002,
                timestamp_monotonic_ns=1_200,
                frame_id=9,
            )
        )
        snapshot = capture_observation(
            {"left": left, "right": right},
            read_state=lambda: np.asarray([3.0, 4.0], dtype=np.float32),
            prompt="timestamps",
            resize_size=2,
            timeout=0.1,
            image_masks={"left": np.bool_(True), "right": np.bool_(True)},
        )

        self.assertEqual(snapshot.camera_frame_ids, {"left": 4, "right": 9})
        self.assertEqual(snapshot.camera_timestamps_ns, {"left": 1_000, "right": 1_200})
        self.assertEqual(snapshot.max_camera_skew_ns, 200)
        self.assertGreaterEqual(snapshot.capture_finished_ns, snapshot.capture_started_ns)
        self.assertGreaterEqual(snapshot.observation_age_ns, 0)
        self.assertEqual(snapshot.images["left"].timestamp_monotonic, 1.0)
        self.assertEqual(RobotState(np.zeros(1, dtype=np.float32), 12.25).timestamp_monotonic_ns, 12_250_000_000)
        self.assertEqual(CameraSample(np.zeros(1, dtype=np.uint8), 4.5).timestamp_monotonic_ns, 4_500_000_000)

    def test_warmup_events_reach_ready_in_order_and_copy_chunks(self) -> None:
        sink = InMemoryRuntimeEventSink()
        hooks = RuntimeEventHooks(sink, RuntimeEventIdentity("warmup-runtime", generation_id=3))
        coordinator = PolicyStartupCoordinator(
            _Policy(),
            _Adapter(),
            _config(),
            PolicyStartupConfig(warmup_enabled=True, warmup_requests=1, prefetch_first_chunk=True),
            hooks=hooks,
            stop_requested=lambda: False,
        )

        prepared = coordinator.prepare()
        events = sink.snapshot()
        event_types = [event.event_type for event in events]

        self.assertEqual(event_types[0], PolicyWarmupStarted.__name__)
        self.assertLess(
            event_types.index(PolicyWarmupStarted.__name__),
            event_types.index(PolicyWarmupFinished.__name__),
        )
        self.assertLess(event_types.index(PolicyWarmupFinished.__name__), event_types.index(PolicyReady.__name__))
        self.assertEqual(event_types.count(ObservationCaptured.__name__), 2)
        self.assertEqual(event_types.count(ChunkReceived.__name__), 2)
        self.assertIsNotNone(prepared.initial_chunk)
        timestamps = [event.monotonic_timestamp_ns for event in events]
        self.assertEqual(timestamps, sorted(timestamps))
        prepared.initial_chunk[0, 0] = 99.0
        first_chunk = [event for event in events if isinstance(event, ChunkReceived)][-1]
        self.assertEqual(first_chunk.payload["raw_action_chunk"][0, 0], 1.0)

    def test_sync_and_rtc_keep_action_event_order(self) -> None:
        for use_rtc in (False, True):
            with self.subTest(use_rtc=use_rtc):
                sink = InMemoryRuntimeEventSink()
                hooks = RuntimeEventHooks(sink, RuntimeEventIdentity("event-runtime", generation_id=int(use_rtc) + 1))
                adapter = _Adapter()
                if use_rtc:
                    run_rtc_loop(
                        _Policy(),
                        adapter,
                        _config(use_rtc=True),
                        hooks=hooks,
                        initial_chunk=np.asarray([[1.0, 2.0]], dtype=np.float32),
                    )
                else:
                    run_sync_loop(_Policy(), adapter, _config(), hooks=hooks)

                event_types = [event.event_type for event in sink.snapshot()]
                self.assertEqual(event_types[0], RuntimeStarted.__name__)
                selected = event_types.index(ActionSelected.__name__)
                stabilized = event_types.index(ActionStabilized.__name__)
                executed = event_types.index(ActionExecuted.__name__)
                self.assertLess(selected, stabilized)
                self.assertLess(stabilized, executed)
                self.assertEqual(event_types[-1], RuntimeStopped.__name__)
                events = sink.snapshot()
                selected_event = next(event for event in events if isinstance(event, ActionSelected))
                stabilized_event = next(event for event in events if isinstance(event, ActionStabilized))
                executed_event = next(event for event in events if isinstance(event, ActionExecuted))
                self.assertIn("selected_raw_action", selected_event.payload)
                self.assertIn(
                    "stabilized_target_action",
                    stabilized_event.payload,
                )
                self.assertIn("executed_action", executed_event.payload)

    def test_generation_gate_rejects_stale_event_hook(self) -> None:
        sink = InMemoryRuntimeEventSink()
        robot = _Robot()
        controller = RuntimeController(robot, _Adapter(), _Policy(), _config(), event_sink=sink)
        old_generation = controller.start()
        self.assertTrue(controller.join(timeout=1.0, raise_on_error=True))
        stale = _GenerationHooks(
            controller,
            old_generation,
            RuntimeEventHooks(sink, RuntimeEventIdentity(controller.runtime_id, generation_id=old_generation)),
        )
        controller.start()
        self.assertTrue(controller.join(timeout=1.0, raise_on_error=True))
        stale.on_action_selected(99, np.asarray([9.0, 9.0], dtype=np.float32))
        controller.close()

        stale_actions = [
            event
            for event in sink.snapshot()
            if isinstance(event, ActionSelected) and np.array_equal(event.payload["selected_raw_action"], [9.0, 9.0])
        ]
        self.assertEqual(stale_actions, [])

    def test_composite_sink_isolates_failures(self) -> None:
        memory = InMemoryRuntimeEventSink()
        composite = CompositeRuntimeEventSink(_FailingSink(), memory)
        event = RuntimeStarted(runtime_id="composite", generation_id=1)

        composite.emit(event)

        self.assertIs(memory.snapshot()[0], event)
        self.assertEqual(len(composite.failures), 1)
        self.assertEqual(composite.failures[0].error_type, "RuntimeError")


if __name__ == "__main__":
    unittest.main()
