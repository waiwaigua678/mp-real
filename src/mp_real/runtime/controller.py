from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mp_real.robots.base import Robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import InferenceAdapter, InferenceHooks, PolicyClient, run_policy_loop


class ControllerAlreadyRunningError(RuntimeError):
    pass


class ControllerClosedError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class RuntimeControllerStatus:
    generation_id: int
    running: bool
    stop_requested: bool
    closed: bool
    started_at_monotonic_ns: int | None
    stopped_at_monotonic_ns: int | None
    error: BaseException | None


class _GenerationHooks(InferenceHooks):
    """Drop delayed hook events that belong to an older controller run."""

    def __init__(self, controller: RuntimeController, generation_id: int, delegate: InferenceHooks) -> None:
        self._controller = controller
        self._generation_id = generation_id
        self._delegate = delegate

    def _is_current(self) -> bool:
        return self._controller._is_current_generation(self._generation_id)

    def on_loop_started(self, mode: str, config: InferenceLoopConfig) -> None:
        if self._is_current():
            self._delegate.on_loop_started(mode, config)

    def on_observation(self, observation: Mapping[str, Any]) -> None:
        if self._is_current():
            self._delegate.on_observation(observation)

    def on_inference_started(self, observation: Mapping[str, Any]) -> None:
        if self._is_current():
            self._delegate.on_inference_started(observation)

    def on_inference_finished(self, response: Mapping[str, Any], elapsed_s: float) -> None:
        if self._is_current():
            self._delegate.on_inference_finished(response, elapsed_s)

    def on_chunk_received(self, chunk: np.ndarray) -> None:
        if self._is_current():
            self._delegate.on_chunk_received(chunk)

    def on_action_selected(self, step: int, action: np.ndarray) -> None:
        if self._is_current():
            self._delegate.on_action_selected(step, action)

    def on_action_stabilized(self, step: int, action: np.ndarray) -> None:
        if self._is_current():
            self._delegate.on_action_stabilized(step, action)

    def on_action_executed(self, step: int, action: np.ndarray) -> None:
        if self._is_current():
            self._delegate.on_action_executed(step, action)

    def on_loop_stopped(self, mode: str) -> None:
        if self._is_current():
            self._delegate.on_loop_stopped(mode)

    def on_error(self, error: BaseException) -> None:
        if self._is_current():
            self._delegate.on_error(error)


class RuntimeController:
    """Own one robot policy-loop lifecycle without knowing a vendor SDK."""

    def __init__(
        self,
        robot: Robot,
        adapter: InferenceAdapter,
        policy_client: PolicyClient,
        config: InferenceLoopConfig,
        *,
        hooks: InferenceHooks | None = None,
        on_step: Callable[[int, int], None] | None = None,
        thread_name: str | None = None,
        print_infer_only_chunks: bool = True,
        rtc_producer_daemon: bool = False,
    ) -> None:
        config.validate()
        self._robot = robot
        self._adapter = adapter
        self._policy_client = policy_client
        self._config = config
        self._hooks = hooks
        self._on_step = on_step
        self._thread_name = thread_name or f"{adapter.name}-runtime-controller"
        self._print_infer_only_chunks = print_infer_only_chunks
        self._rtc_producer_daemon = rtc_producer_daemon

        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._generation_id = 0
        self._running = False
        self._stop_requested = False
        self._closed = False
        self._started_at_monotonic_ns: int | None = None
        self._stopped_at_monotonic_ns: int | None = None
        self._error: BaseException | None = None

    @property
    def robot(self) -> Robot:
        return self._robot

    def configure(
        self,
        adapter: InferenceAdapter,
        config: InferenceLoopConfig,
        *,
        hooks: InferenceHooks | None = None,
        on_step: Callable[[int, int], None] | None = None,
    ) -> None:
        config.validate()
        with self._lock:
            self._require_open_locked()
            if self._running or (self._thread is not None and self._thread.is_alive()):
                raise ControllerAlreadyRunningError("Cannot configure a running controller")
            self._adapter = adapter
            self._config = config
            self._hooks = hooks
            self._on_step = on_step

    def start(self) -> int:
        with self._lock:
            self._require_open_locked()
            if self._running or (self._thread is not None and self._thread.is_alive()):
                raise ControllerAlreadyRunningError("Runtime controller is already running")

            self._generation_id += 1
            generation_id = self._generation_id
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._running = True
            self._stop_requested = False
            self._error = None
            self._started_at_monotonic_ns = time.monotonic_ns()
            self._stopped_at_monotonic_ns = None
            self._thread = threading.Thread(
                target=self._run,
                args=(generation_id, stop_event),
                name=f"{self._thread_name}-g{generation_id}",
                daemon=False,
            )
            self._thread.start()
            return generation_id

    def stop(self, *, wait: bool = False, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
            if not self._running and (thread is None or not thread.is_alive()):
                return True
            self._stop_requested = True
            self._stop_event.set()

        if wait:
            return self.join(timeout=timeout, raise_on_error=False)
        return thread is None or not thread.is_alive()

    def join(self, *, timeout: float | None = None, raise_on_error: bool = False) -> bool:
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        finished = thread is None or not thread.is_alive()
        if finished and raise_on_error:
            self.raise_if_failed()
        return finished

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def reset_robot(self) -> None:
        with self._lock:
            self._require_open_locked()
            if self._running or (self._thread is not None and self._thread.is_alive()):
                raise ControllerAlreadyRunningError("Cannot reset the robot while the controller is running")
        self._robot.reset()

    def configure_robot(self, config: object) -> bool:
        """Apply optional robot-specific runtime settings through the robot boundary."""
        with self._lock:
            self._require_open_locked()
            if self._running or (self._thread is not None and self._thread.is_alive()):
                raise ControllerAlreadyRunningError("Cannot configure the robot while the controller is running")
            configure_runtime = getattr(self._robot, "configure_runtime", None)
        if not callable(configure_runtime):
            return False
        configure_runtime(config)
        return True

    def close(self, *, timeout: float | None = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
        if not self.stop(wait=True, timeout=timeout):
            raise TimeoutError("Runtime controller did not stop before close")

        errors: list[BaseException] = []
        try:
            self._robot.close()
        except BaseException as exc:
            errors.append(exc)
        close_client = getattr(self._policy_client, "close", None)
        if callable(close_client):
            try:
                close_client()
            except BaseException as exc:
                errors.append(exc)
        with self._lock:
            self._closed = True
        if errors:
            raise RuntimeError("Failed to close runtime controller resources") from errors[0]

    def status(self) -> RuntimeControllerStatus:
        with self._lock:
            return RuntimeControllerStatus(
                generation_id=self._generation_id,
                running=self._running,
                stop_requested=self._stop_requested,
                closed=self._closed,
                started_at_monotonic_ns=self._started_at_monotonic_ns,
                stopped_at_monotonic_ns=self._stopped_at_monotonic_ns,
                error=self._error,
            )

    def _run(self, generation_id: int, stop_event: threading.Event) -> None:
        with self._lock:
            adapter = self._adapter
            client = self._policy_client
            config = self._config
            hooks = self._hooks
            on_step = self._on_step
        guarded_hooks = _GenerationHooks(self, generation_id, hooks) if hooks is not None else None

        def guarded_on_step(step: int, action_queue_len: int) -> None:
            if on_step is not None and self._is_current_generation(generation_id):
                on_step(step, action_queue_len)

        error: BaseException | None = None
        try:
            run_policy_loop(
                client,
                adapter,
                config,
                stop_event=stop_event,
                on_step=guarded_on_step if on_step is not None else None,
                hooks=guarded_hooks,
                print_infer_only_chunks=self._print_infer_only_chunks,
                rtc_producer_daemon=self._rtc_producer_daemon,
            )
        except BaseException as exc:
            error = exc
        finally:
            stopped_at_ns = time.monotonic_ns()
            with self._lock:
                if generation_id == self._generation_id:
                    self._running = False
                    self._stop_requested = False
                    self._stopped_at_monotonic_ns = stopped_at_ns
                    self._error = error

    def _is_current_generation(self, generation_id: int) -> bool:
        with self._lock:
            return generation_id == self._generation_id and not self._closed

    def _require_open_locked(self) -> None:
        if self._closed:
            raise ControllerClosedError("Runtime controller is closed")
