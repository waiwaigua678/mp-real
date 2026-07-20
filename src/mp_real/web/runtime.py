from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from mp_real.robots.base import Robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import InferenceHooks
from mp_real.runtime.models import CameraSample, ObservationSnapshot, RobotState


class ObservationSource(Protocol):
    def observe(self) -> dict[str, Any]: ...


@dataclasses.dataclass
class CachedFrameObservationSource:
    """Build policy observations from Web preview frames and a Robot state."""

    robot: Robot
    read_images: Callable[[], tuple[Mapping[str, np.ndarray], Mapping[str, Any] | None]]
    image_masks: Mapping[str, np.bool_]
    prompt: str
    last_observation_snapshot: ObservationSnapshot | None = dataclasses.field(default=None, init=False)

    def observe(self) -> dict[str, Any]:
        return self.capture_observation_snapshot().to_policy_observation()

    def capture_observation_snapshot(self) -> ObservationSnapshot:
        result = self.read_images()
        images, camera_params = result[0], result[1]
        frame_metadata = result[2] if len(result) > 2 else {}
        now_ns = time.monotonic_ns()
        state = self.robot.read_state()
        if not isinstance(state, RobotState):
            state = RobotState(np.asarray(state, dtype=np.float32), now_ns / 1e9, now_ns)
        samples: dict[str, CameraSample] = {}
        for name, image in images.items():
            metadata = frame_metadata.get(name) if isinstance(frame_metadata, Mapping) else None
            timestamp_ns = int(getattr(metadata, "timestamp_monotonic_ns", 0) or now_ns)
            frame_id = int(getattr(metadata, "frame_id", 0) or 0)
            samples[name] = CameraSample(
                image=np.asarray(image).copy(),
                timestamp_monotonic=timestamp_ns / 1e9,
                frame_id=frame_id,
                timestamp_monotonic_ns=timestamp_ns,
                source_sequence=getattr(metadata, "source_sequence", None),
                capture_latency_ns=getattr(metadata, "capture_latency_ns", None),
            )
        self.last_observation_snapshot = ObservationSnapshot(
            images=samples,
            image_masks=dict(self.image_masks),
            state=state,
            prompt=self.prompt,
            camera_params=camera_params,
            captured_at_monotonic=now_ns / 1e9,
            capture_started_ns=now_ns,
            capture_finished_ns=now_ns,
            state_timestamp_ns=state.timestamp_monotonic_ns,
            camera_frame_ids={name: sample.frame_id for name, sample in samples.items()},
            camera_timestamps_ns={name: sample.timestamp_monotonic_ns for name, sample in samples.items()},
        )
        return self.last_observation_snapshot


@dataclasses.dataclass
class WebInferenceAdapter:
    name: str
    robot: Robot
    observation_source: ObservationSource
    decode_chunk: Callable[[dict[str, Any], int], np.ndarray]
    stabilize: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    metadata_keys: tuple[str, ...] = ()
    infer_interval_s: float = 0.0
    profile_callback: Callable[[str, float], None] | None = None

    def observe(self) -> dict[str, Any]:
        return self.observation_source.observe()

    @property
    def last_observation_snapshot(self) -> ObservationSnapshot | None:
        return getattr(self.observation_source, "last_observation_snapshot", None)

    def capture_observation_snapshot(self) -> ObservationSnapshot:
        capture_snapshot = getattr(self.observation_source, "capture_observation_snapshot", None)
        if callable(capture_snapshot):
            snapshot = capture_snapshot()
            if isinstance(snapshot, ObservationSnapshot):
                return snapshot
        self.observe()
        snapshot = self.last_observation_snapshot
        if snapshot is None:
            raise RuntimeError("Web observation source did not provide an ObservationSnapshot")
        return snapshot

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        return self.decode_chunk(response, replan_steps)

    def initial_action(self) -> np.ndarray:
        return self.robot.read_state().values

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        return self.stabilize(action, previous)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return self.robot.execute_transition(previous, target)

    def infer_only_metadata(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return {key: observation[key] for key in self.metadata_keys if key in observation}

    def profile(self, stage: str, elapsed_s: float) -> None:
        if self.profile_callback is not None:
            self.profile_callback(stage, elapsed_s)

    def infer_only_interval_s(self) -> float:
        return self.infer_interval_s


@dataclasses.dataclass(frozen=True)
class WebLoopSnapshot:
    running: bool
    step: int
    action_queue_len: int
    error: BaseException | None
    started_at_monotonic_ns: int | None
    infer_latency_ms: float | None
    infer_hz: float | None
    loop_ms: float | None
    control_hz: float | None


class WebLoopHooks(InferenceHooks):
    """Thread-safe, in-memory hook implementation for the existing Web status."""

    def __init__(
        self,
        *,
        error_callback: Callable[[BaseException], None] | None = None,
        stopped_callback: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._error_callback = error_callback
        self._stopped_callback = stopped_callback
        self._running = False
        self._step = 0
        self._action_queue_len = 0
        self._error: BaseException | None = None
        self._started_at_monotonic_ns: int | None = None
        self._action_started_at_monotonic_ns: int | None = None
        self._infer_latency_ms: float | None = None
        self._infer_hz: float | None = None
        self._loop_ms: float | None = None
        self._control_hz: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._running = False
            self._step = 0
            self._action_queue_len = 0
            self._error = None
            self._started_at_monotonic_ns = None
            self._action_started_at_monotonic_ns = None
            self._infer_latency_ms = None
            self._infer_hz = None
            self._loop_ms = None
            self._control_hz = None

    def on_loop_started(self, mode: str, config: InferenceLoopConfig) -> None:
        del mode, config
        with self._lock:
            self._running = True
            self._started_at_monotonic_ns = time.monotonic_ns()

    def on_inference_finished(self, response: Mapping[str, Any], elapsed_s: float) -> None:
        del response
        with self._lock:
            self._infer_latency_ms = elapsed_s * 1000.0
            self._infer_hz = 1.0 / elapsed_s if elapsed_s > 0 else None

    def on_chunk_received(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._action_queue_len = len(chunk)

    def on_action_selected(self, step: int, action: np.ndarray) -> None:
        del step, action
        with self._lock:
            self._action_started_at_monotonic_ns = time.monotonic_ns()

    def on_action_executed(self, step: int, action: np.ndarray) -> None:
        del step, action
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._action_started_at_monotonic_ns is None:
                return
            elapsed_s = (now_ns - self._action_started_at_monotonic_ns) / 1e9
            self._loop_ms = elapsed_s * 1000.0
            self._control_hz = 1.0 / elapsed_s if elapsed_s > 0 else None

    def on_loop_stopped(self, mode: str) -> None:
        del mode
        with self._lock:
            self._running = False
            self._action_queue_len = 0
        if self._stopped_callback is not None:
            self._stopped_callback()

    def on_error(self, error: BaseException) -> None:
        with self._lock:
            self._error = error
        if self._error_callback is not None:
            self._error_callback(error)

    def on_step(self, step: int, action_queue_len: int) -> None:
        with self._lock:
            self._step = step
            self._action_queue_len = action_queue_len

    def snapshot(self) -> WebLoopSnapshot:
        with self._lock:
            return WebLoopSnapshot(
                running=self._running,
                step=self._step,
                action_queue_len=self._action_queue_len,
                error=self._error,
                started_at_monotonic_ns=self._started_at_monotonic_ns,
                infer_latency_ms=self._infer_latency_ms,
                infer_hz=self._infer_hz,
                loop_ms=self._loop_ms,
                control_hz=self._control_hz,
            )
