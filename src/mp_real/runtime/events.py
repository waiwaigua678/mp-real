from __future__ import annotations

import copy
import dataclasses
import datetime
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import InferenceHooks
from mp_real.runtime.models import ActionProvenance, ControlStepRecord, ObservationSnapshot


def _wall_timestamp_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def copy_event_payload(value: Any) -> Any:
    """Copy mutable payload data before an event leaves the control path.

    Event payloads own their ndarray values. This prevents a later action
    stabilization, buffer fusion, or vendor SDK call from mutating recorded
    in-memory telemetry retroactively.
    """
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: copy_event_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(copy_event_payload(item) for item in value)
    if isinstance(value, list):
        return [copy_event_payload(item) for item in value]
    if isinstance(value, set):
        return {copy_event_payload(item) for item in value}
    return copy.copy(value)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RuntimeEvent:
    event_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    runtime_id: str = ""
    session_id: str | None = None
    episode_id: str | None = None
    generation_id: int = 0
    request_id: int | None = None
    chunk_id: int | None = None
    step: int | None = None
    monotonic_timestamp_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    wall_timestamp_iso: str = dataclasses.field(default_factory=_wall_timestamp_iso)
    payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", copy_event_payload(self.payload))

    @property
    def event_type(self) -> str:
        return type(self).__name__


class RuntimeStarted(RuntimeEvent):
    pass


class ObservationCaptured(RuntimeEvent):
    pass


class PolicyWarmupStarted(RuntimeEvent):
    pass


class PolicyWarmupFinished(RuntimeEvent):
    pass


class PolicyWarmupFailed(RuntimeEvent):
    pass


class PolicyReady(RuntimeEvent):
    pass


class InferenceStarted(RuntimeEvent):
    pass


class InferenceFinished(RuntimeEvent):
    pass


class ChunkReceived(RuntimeEvent):
    pass


class ActionSelected(RuntimeEvent):
    pass


class ActionStabilized(RuntimeEvent):
    pass


class ActionExecuted(RuntimeEvent):
    pass


class ControlStepRecorded(RuntimeEvent):
    pass


class SafetyRejected(RuntimeEvent):
    pass


class RuntimeStopped(RuntimeEvent):
    pass


class RuntimeFailed(RuntimeEvent):
    pass


class RuntimeEventSink(Protocol):
    """A sink must return quickly; control loops never wait for disk or network I/O."""

    def emit(self, event: RuntimeEvent) -> None: ...


class NoOpRuntimeEventSink:
    def emit(self, event: RuntimeEvent) -> None:
        del event


@dataclasses.dataclass(frozen=True)
class RuntimeEventSinkFailure:
    sink_index: int
    event_id: str
    error_type: str
    message: str


class CompositeRuntimeEventSink:
    """Fan out events and isolate a failing child sink from the remaining sinks."""

    def __init__(self, *sinks: RuntimeEventSink) -> None:
        self._sinks = tuple(sinks)
        self._lock = threading.Lock()
        self._failures: list[RuntimeEventSinkFailure] = []

    @property
    def failures(self) -> tuple[RuntimeEventSinkFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    @property
    def requires_observation_images(self) -> bool:
        return any(bool(getattr(sink, "requires_observation_images", False)) for sink in self._sinks)

    def emit(self, event: RuntimeEvent) -> None:
        for index, sink in enumerate(self._sinks):
            try:
                sink.emit(event)
            except BaseException as exc:
                with self._lock:
                    self._failures.append(
                        RuntimeEventSinkFailure(
                            sink_index=index,
                            event_id=event.event_id,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )


class InMemoryRuntimeEventSink:
    """A bounded, thread-safe event store for status and future recording tests."""

    def __init__(self, *, max_events: int = 4096) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._events: deque[RuntimeEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped_events

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self._dropped_events += 1
            self._events.append(event)

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped_events = 0


class RuntimeEventDispatcher:
    """Bounded asynchronous bridge from a control loop to potentially slow sinks."""

    _STOP = object()

    def __init__(
        self,
        sink: RuntimeEventSink,
        *,
        queue_size: int = 1024,
        thread_name: str = "runtime-event-dispatcher",
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._sink = sink
        self._queue: queue.Queue[RuntimeEvent | object] = queue.Queue(maxsize=queue_size)
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._dropped_events = 0
        self._failures: list[RuntimeEventSinkFailure] = []

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped_events

    @property
    def failures(self) -> tuple[RuntimeEventSinkFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    @property
    def requires_observation_images(self) -> bool:
        return bool(getattr(self._sink, "requires_observation_images", False))

    def start(self) -> None:
        with self._lock:
            if self._stopping:
                raise RuntimeError("Runtime event dispatcher is stopping")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=False)
            self._thread.start()

    def emit(self, event: RuntimeEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._dropped_events += 1

    def stop(self, *, timeout: float | None = 5.0) -> bool:
        with self._lock:
            self._stopping = True
            thread = self._thread
        if thread is None:
            return True
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            # The worker will observe the sentinel once it drains an event.
            try:
                self._queue.put(self._STOP, timeout=timeout)
            except queue.Full:
                return False
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, RuntimeEvent)
                try:
                    self._sink.emit(item)
                except BaseException as exc:
                    with self._lock:
                        self._failures.append(
                            RuntimeEventSinkFailure(
                                sink_index=0,
                                event_id=item.event_id,
                                error_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )
            finally:
                self._queue.task_done()


@dataclasses.dataclass(frozen=True)
class RuntimeEventIdentity:
    runtime_id: str
    generation_id: int
    session_id: str | None = None
    episode_id: str | None = None


class RuntimeEventHooks(InferenceHooks):
    """Translate the stable InferenceHooks API into copied, typed runtime events."""

    def __init__(
        self,
        sink: RuntimeEventSink,
        identity: RuntimeEventIdentity,
        *,
        include_observation_images: bool | None = None,
    ) -> None:
        self._sink = sink
        self._identity = identity
        self._lock = threading.Lock()
        self._last_timestamp_ns = 0
        self._request_id = 0
        self._chunk_id = 0
        self._local = threading.local()
        self._observation_context: dict[int, tuple[int | None, int | None]] = {}
        self._include_observation_images = (
            bool(getattr(sink, "requires_observation_images", False))
            if include_observation_images is None
            else include_observation_images
        )

    def on_loop_started(self, mode: str, config: InferenceLoopConfig) -> None:
        self._emit(RuntimeStarted, payload={"mode": mode, "config": dataclasses.asdict(config)})

    def on_observation_captured(self, snapshot: ObservationSnapshot) -> None:
        self._local.snapshot_captured = True
        self._local.observation_id = snapshot.observation_id
        payload: dict[str, Any] = {
            "observation_id": snapshot.observation_id,
            "capture_started_ns": snapshot.capture_started_ns,
            "capture_finished_ns": snapshot.capture_finished_ns,
            "state_timestamp_ns": snapshot.state_timestamp_ns,
            "camera_frame_ids": snapshot.camera_frame_ids,
            "camera_timestamps_ns": snapshot.camera_timestamps_ns,
            "max_camera_skew_ns": snapshot.max_camera_skew_ns,
            "observation_age_ns": snapshot.observation_age_ns,
            "state": snapshot.state.values,
        }
        if self._include_observation_images:
            payload["images"] = {name: sample.image for name, sample in snapshot.images.items()}
        self._emit(
            ObservationCaptured,
            payload=payload,
        )

    def on_observation(self, observation: Mapping[str, Any]) -> None:
        if getattr(self._local, "snapshot_captured", False):
            self._local.snapshot_captured = False
            return
        self._local.observation_id = None
        self._emit(
            ObservationCaptured,
            payload={
                "observation_id": None,
                "camera_names": tuple(observation.get("images", {}).keys()),
                "state": observation.get("state"),
                "prompt": observation.get("prompt"),
            },
        )

    def on_inference_started_context(self, observation: Mapping[str, Any], stage: str) -> None:
        del observation
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            self._local.request_id = request_id
        self._emit(InferenceStarted, request_id=request_id, payload={"stage": stage})

    def on_inference_finished_context(self, response: Mapping[str, Any], elapsed_s: float, stage: str) -> None:
        del response
        self._emit(
            InferenceFinished,
            request_id=getattr(self._local, "request_id", None),
            payload={
                "stage": stage,
                "inference_latency_ns": round(elapsed_s * 1e9),
                "observation_id": getattr(self._local, "observation_id", None),
            },
        )

    def on_chunk_received_context(self, chunk: np.ndarray, stage: str) -> None:
        with self._lock:
            self._chunk_id += 1
            chunk_id = self._chunk_id
        self._local.chunk_id = chunk_id
        observation_id = getattr(self._local, "observation_id", None)
        with self._lock:
            if observation_id is not None:
                self._observation_context[int(observation_id)] = (getattr(self._local, "request_id", None), chunk_id)
        self._emit(
            ChunkReceived,
            request_id=getattr(self._local, "request_id", None),
            chunk_id=chunk_id,
            payload={"stage": stage, "raw_action_chunk": chunk, "observation_id": observation_id},
        )

    def on_action_selected(self, step: int, action: np.ndarray) -> None:
        self.on_action_selected_context(step, action, None)

    def on_action_selected_context(self, step: int, action: np.ndarray, provenance: ActionProvenance | None) -> None:
        request_id, chunk_id, payload = self._provenance_context(provenance)
        self._emit(
            ActionSelected,
            request_id=request_id,
            chunk_id=chunk_id,
            step=step,
            payload={"selected_raw_action": action, **payload},
        )

    def on_action_stabilized(self, step: int, action: np.ndarray) -> None:
        self.on_action_stabilized_context(step, action, None)

    def on_action_stabilized_context(self, step: int, action: np.ndarray, provenance: ActionProvenance | None) -> None:
        request_id, chunk_id, payload = self._provenance_context(provenance)
        self._emit(
            ActionStabilized,
            request_id=request_id,
            chunk_id=chunk_id,
            step=step,
            payload={"stabilized_target_action": action, **payload},
        )

    def on_action_executed(self, step: int, action: np.ndarray) -> None:
        self.on_action_executed_context(step, action, None)

    def on_action_executed_context(self, step: int, action: np.ndarray, provenance: ActionProvenance | None) -> None:
        request_id, chunk_id, payload = self._provenance_context(provenance)
        self._emit(
            ActionExecuted,
            request_id=request_id,
            chunk_id=chunk_id,
            step=step,
            payload={"executed_action": action, **payload},
        )

    def on_control_step_recorded(self, record: ControlStepRecord) -> None:
        provenance = ActionProvenance(
            observation_id=record.policy_observation_id,
            chunk_cursor=record.chunk_cursor,
            source_observation_ids=record.source_observation_ids,
            control_cycle_ns=record.control_cycle_ns,
        )
        request_id, chunk_id, _ = self._provenance_context(provenance)
        if record.policy_request_id is not None:
            request_id = record.policy_request_id
        if record.chunk_id is not None:
            chunk_id = record.chunk_id
        payload: dict[str, Any] = {
            "control_step_id": record.control_step_id,
            "observation_id": record.control_observation_id,
            "policy_observation_id": record.policy_observation_id,
            "policy_request_id": request_id,
            "chunk_id": chunk_id,
            "chunk_cursor": record.chunk_cursor,
            "source_observation_ids": list(record.source_observation_ids),
            "control_step_monotonic_timestamp_ns": record.monotonic_timestamp_ns,
            "scheduled_timestamp_ns": record.scheduled_timestamp_ns,
            "observation_capture_started_ns": record.observation_capture_started_ns,
            "observation_capture_finished_ns": record.observation_capture_finished_ns,
            "state_timestamp_ns": record.robot_state_timestamp_ns,
            "camera_frame_ids": dict(record.camera_frame_ids),
            "camera_timestamps_ns": dict(record.camera_timestamps_ns),
            "camera_age_ns": dict(record.camera_age_ns),
            "max_camera_skew_ns": record.max_camera_skew_ns,
            "state": record.robot_state_before_action,
            "selected_raw_action": record.selected_raw_action,
            "stabilized_target_action": record.stabilized_action,
            "executed_action": record.executed_action,
            "action_sent_timestamp_ns": record.action_sent_timestamp_ns,
            "control_cycle_ns": record.control_cycle_ns,
            "safety_flags": list(record.safety_flags),
        }
        if record.feedback_state is not None:
            payload["feedback_state"] = record.feedback_state
            payload["feedback_state_timestamp_ns"] = record.feedback_state_timestamp_ns
        if self._include_observation_images:
            payload["images"] = {name: sample.image for name, sample in record.camera_samples.items()}
        self._emit(
            ControlStepRecorded,
            request_id=request_id,
            chunk_id=chunk_id,
            step=record.control_step_id,
            payload=payload,
        )

    def on_safety_rejected(self, step: int | None, action: np.ndarray | None, error: BaseException) -> None:
        self._emit(
            SafetyRejected,
            step=step,
            payload={
                "candidate_action": action,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    def on_policy_warmup_started(self, requests: int) -> None:
        self._emit(PolicyWarmupStarted, payload={"requests": requests})

    def on_policy_warmup_finished(self, elapsed_s: float) -> None:
        self._emit(PolicyWarmupFinished, payload={"warmup_latency_ns": round(elapsed_s * 1e9)})

    def on_policy_warmup_failed(self, error: BaseException) -> None:
        self._emit(
            PolicyWarmupFailed,
            payload={"error_type": type(error).__name__, "message": str(error)},
        )

    def on_policy_ready(self, initial_chunk: np.ndarray | None) -> None:
        self._emit(
            PolicyReady,
            payload={"initial_chunk": initial_chunk, "has_initial_chunk": initial_chunk is not None},
        )

    def on_loop_stopped(self, mode: str) -> None:
        self._emit(RuntimeStopped, payload={"mode": mode})

    def on_error(self, error: BaseException) -> None:
        self._emit(RuntimeFailed, payload={"error_type": type(error).__name__, "message": str(error)})

    def _provenance_context(
        self, provenance: ActionProvenance | None
    ) -> tuple[int | None, int | None, dict[str, object]]:
        if provenance is None:
            return None, None, {"observation_id": None, "chunk_cursor": None, "source_observation_ids": []}
        with self._lock:
            request_id, chunk_id = self._observation_context.get(provenance.observation_id or -1, (None, None))
        return (
            request_id,
            chunk_id,
            {
                "observation_id": provenance.observation_id,
                "chunk_cursor": provenance.chunk_cursor,
                "source_observation_ids": list(provenance.source_observation_ids),
                "control_cycle_ns": provenance.control_cycle_ns,
            },
        )

    def _emit(
        self,
        event_type: type[RuntimeEvent],
        *,
        request_id: int | None = None,
        chunk_id: int | None = None,
        step: int | None = None,
        payload: Mapping[str, Any],
    ) -> None:
        with self._lock:
            timestamp_ns = max(time.monotonic_ns(), self._last_timestamp_ns + 1)
            self._last_timestamp_ns = timestamp_ns
        event = event_type(
            runtime_id=self._identity.runtime_id,
            session_id=self._identity.session_id,
            episode_id=self._identity.episode_id,
            generation_id=self._identity.generation_id,
            request_id=request_id,
            chunk_id=chunk_id,
            step=step,
            monotonic_timestamp_ns=timestamp_ns,
            payload=payload,
        )
        try:
            self._sink.emit(event)
        except BaseException:
            # Event collection must not fail or stall a robot control loop.
            return
