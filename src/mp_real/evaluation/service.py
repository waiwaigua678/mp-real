from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from mp_real.evaluation.models import EvaluationConfig, EvaluationResult, EvaluationState, FailureReason
from mp_real.evaluation.session import EvaluationSession, EvaluationStateConflict
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.controller import RuntimeController
from mp_real.runtime.events import (
    PolicyWarmupFailed,
    RuntimeEvent,
    RuntimeEventHooks,
    RuntimeEventIdentity,
    RuntimeEventSink,
    RuntimeFailed,
    RuntimeStopped,
    SafetyRejected,
)
from mp_real.runtime.inference import CompositeInferenceHooks, InferenceAdapter
from mp_real.runtime.startup import (
    PolicyStartupCancelled,
    PolicyStartupConfig,
    PolicyStartupCoordinator,
)


class EvaluationConflict(RuntimeError):
    """A 409-worthy orchestration conflict with actionable legal operations."""

    def __init__(self, message: str, *, legal_operations: tuple[str, ...] = ()) -> None:
        self.legal_operations = legal_operations
        allowed = ", ".join(legal_operations) or "none"
        super().__init__(f"{message}; legal operations: {allowed}")


@dataclasses.dataclass(frozen=True)
class EvaluationRuntimeLease:
    """Opaque deployment resources reserved by one evaluation session."""

    controller: RuntimeController
    runtime_config_snapshot: Mapping[str, Any]
    action_spec_snapshot: Mapping[str, Any]
    robot_name: str
    make_adapter: Callable[[str], InferenceAdapter]
    make_loop_config: Callable[[str], InferenceLoopConfig]
    make_startup_config: Callable[[], PolicyStartupConfig]
    release: Callable[[], None]


class EvaluationRuntimeBroker(Protocol):
    def acquire_evaluation_control(self, evaluation_id: str) -> EvaluationRuntimeLease: ...


class EvaluationService(RuntimeEventSink):
    """Coordinates a manual-label evaluation through shared runtime events.

    It only receives an opaque runtime lease from its host.  In particular it
    does not import a Web handler or any robot SDK, and the RuntimeController
    remains unaware that an evaluation session exists.
    """

    def __init__(self, broker: EvaluationRuntimeBroker) -> None:
        self._broker = broker
        self._lock = threading.RLock()
        self._session: EvaluationSession | None = None
        self._lease: EvaluationRuntimeLease | None = None
        self._warmup_thread: threading.Thread | None = None
        self._start_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._warmup_stop_event = threading.Event()
        self._start_stop_event = threading.Event()
        self._watchdog_stop_event = threading.Event()

    def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        requested_id = str(payload.get("evaluation_id") or uuid.uuid4().hex)
        config_payload = dict(payload)
        config_payload["evaluation_id"] = requested_id
        with self._lock:
            if self._session is not None and not self._session.is_terminal:
                raise EvaluationConflict(
                    "An active evaluation already owns robot control",
                    legal_operations=self._session.legal_operations(),
                )

        # The broker performs the resource-mode and normal-control checks
        # atomically with reservation of the existing deployment controller.
        lease = self._broker.acquire_evaluation_control(requested_id)
        try:
            config = EvaluationConfig.from_payload(
                config_payload,
                runtime_config_snapshot=lease.runtime_config_snapshot,
                action_spec_snapshot=lease.action_spec_snapshot,
                default_robot_name=lease.robot_name,
            )
            if requested_id and config.evaluation_id != requested_id:
                raise RuntimeError("evaluation_id changed while creating the session")
            if config.robot_name != lease.robot_name:
                raise ValueError(f"robot_name must match the active deployment robot: {lease.robot_name}")
            session = EvaluationSession(config)
        except BaseException:
            lease.release()
            raise

        with self._lock:
            self._session = session
            self._lease = lease
            self._warmup_stop_event = threading.Event()
            self._start_stop_event = threading.Event()
            self._watchdog_stop_event = threading.Event()
            return session.snapshot()

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            return self._session.snapshot() if self._session is not None else None

    def warmup(self) -> dict[str, Any]:
        with self._lock:
            session, lease = self._require_session_and_lease()
            self._raise_as_conflict(session.start_warmup)
            if self._warmup_thread is not None and self._warmup_thread.is_alive():
                raise EvaluationConflict(
                    "Policy warmup is already in progress",
                    legal_operations=session.legal_operations(),
                )
            self._warmup_stop_event.clear()
            self._warmup_thread = threading.Thread(
                target=self._run_warmup,
                args=(session.session_id, lease, self._warmup_stop_event),
                name=f"evaluation-warmup-{session.session_id}",
                daemon=False,
            )
            self._warmup_thread.start()
            return session.snapshot()

    def reset_ready(self) -> dict[str, Any]:
        with self._lock:
            session, _ = self._require_session_and_lease()
            self._raise_as_conflict(session.reset_ready)
            return session.snapshot()

    def start_episode(self) -> dict[str, Any]:
        with self._lock:
            session, lease = self._require_session_and_lease()
            episode = self._raise_as_conflict(session.begin_episode_preparation)
            self._start_stop_event.clear()
            self._start_thread = threading.Thread(
                target=self._run_episode_start,
                args=(session.session_id, episode.episode_id, lease, self._start_stop_event),
                name=f"evaluation-start-{session.session_id}-{episode.episode_index}",
                daemon=False,
            )
            self._start_thread.start()
            return session.snapshot()

    def stop_episode(self) -> dict[str, Any]:
        with self._lock:
            session, lease = self._require_session_and_lease()
            self._raise_as_conflict(lambda: session.request_stop("MANUAL"))
            self._watchdog_stop_event.set()
            controller = lease.controller
        controller.stop(wait=False)
        return self.current_or_raise()

    def label(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            result = EvaluationResult(str(payload.get("result", "")))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in EvaluationResult)
            raise ValueError(f"result must be one of: {allowed}") from exc
        failure_value = payload.get("failure_reason")
        try:
            failure_reason = FailureReason(str(failure_value)) if failure_value not in (None, "") else None
        except ValueError as exc:
            allowed = ", ".join(item.value for item in FailureReason)
            raise ValueError(f"failure_reason must be one of: {allowed}") from exc

        with self._lock:
            session, _ = self._require_session_and_lease()
            self._raise_as_conflict(
                lambda: session.label(result, failure_reason=failure_reason, notes=str(payload.get("notes", "")))
            )
            self._release_if_terminal_locked()
            return session.snapshot()

    def abort(self) -> dict[str, Any]:
        with self._lock:
            session, lease = self._require_session_and_lease()
            try:
                stop_running_loop = session.request_abort()
            except EvaluationStateConflict as exc:
                raise self._conflict_from_state(exc) from exc
            self._warmup_stop_event.set()
            self._start_stop_event.set()
            self._watchdog_stop_event.set()
            if not stop_running_loop:
                session.finish_abort()
                self._release_if_terminal_locked()
            controller = lease.controller
        if stop_running_loop:
            controller.stop(wait=False)
        return self.current_or_raise()

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop owned work and join service workers during host shutdown."""
        with self._lock:
            session = self._session
            lease = self._lease
        if session is not None and not session.is_terminal:
            try:
                self.abort()
            except EvaluationConflict:
                pass
        if lease is not None and not lease.controller.stop(wait=True, timeout=timeout):
            raise TimeoutError("Evaluation runtime controller did not stop during shutdown")
        with self._lock:
            if self._session is not None and self._session.abort_requested:
                self._session.finish_abort()
                self._release_if_terminal_locked(force=True)
        incomplete = [
            attribute
            for attribute in ("_warmup_thread", "_start_thread", "_watchdog_thread")
            if not self._join_worker(attribute, timeout)
        ]
        if incomplete:
            raise TimeoutError("Evaluation workers did not stop during shutdown: " + ", ".join(incomplete))

    def emit(self, event: RuntimeEvent) -> None:
        """Receive copied runtime events through the controller event sink."""
        controller_to_stop: RuntimeController | None = None
        with self._lock:
            session = self._session
            lease = self._lease
            if session is None or lease is None or event.session_id != session.session_id:
                return
            active_id = session.active_episode_id
            if event.episode_id is not None and event.episode_id != active_id and not session.is_terminal:
                return
            active_generation_id = session.active_episode_generation_id
            if (
                active_generation_id is not None
                and event.generation_id != active_generation_id
            ):
                return

            if isinstance(event, PolicyWarmupFailed) and session.state is EvaluationState.WARMING_UP:
                error_type = event.payload.get("error_type", "PolicyWarmupFailed")
                message = event.payload.get("message", "policy warmup failed")
                session.fail(f"{error_type}: {message}")
                self._release_if_terminal_locked()
                return

            if isinstance(event, SafetyRejected) and session.state is EvaluationState.RUNNING:
                try:
                    session.request_stop(EvaluationResult.SAFETY_ABORT.value)
                    self._watchdog_stop_event.set()
                    controller_to_stop = lease.controller
                except EvaluationStateConflict:
                    return
            elif isinstance(event, RuntimeFailed):
                error_type = event.payload.get("error_type", "RuntimeFailed")
                message = event.payload.get("message", "runtime failed")
                session.fail(f"{error_type}: {message}")
            elif isinstance(event, RuntimeStopped):
                self._watchdog_stop_event.set()
                if session.abort_requested:
                    session.finish_abort()
                else:
                    session.runtime_stopped(stopped_at_monotonic_ns=event.monotonic_timestamp_ns)
                # RuntimeStopped is emitted after the policy loop has left its
                # action path, even though RuntimeController may update its
                # status field a few instructions later.
                self._release_if_terminal_locked(force=True)

        if controller_to_stop is not None:
            controller_to_stop.stop(wait=False)

    def current_or_raise(self) -> dict[str, Any]:
        status = self.current()
        if status is None:
            raise RuntimeError("Evaluation session disappeared")
        return status

    def _run_warmup(
        self,
        session_id: str,
        lease: EvaluationRuntimeLease,
        stop_event: threading.Event,
    ) -> None:
        try:
            with self._lock:
                session = self._session
                if session is None or session.session_id != session_id:
                    return
                config = session.config
            adapter = lease.make_adapter(config.prompt)
            loop_config = lease.make_loop_config(config.prompt)
            if loop_config.infer_only:
                raise ValueError("Evaluation cannot use infer_only runtime configuration")
            # Evaluation warmup must always occur, but its chunks are discarded.
            startup_config = dataclasses.replace(
                lease.make_startup_config(),
                warmup_enabled=True,
                prefetch_first_chunk=False,
            )
            event_hooks = RuntimeEventHooks(
                lease.controller.event_sink,
                RuntimeEventIdentity(
                    runtime_id=lease.controller.runtime_id,
                    generation_id=lease.controller.status().generation_id + 1,
                    session_id=session_id,
                ),
            )
            coordinator = PolicyStartupCoordinator(
                lease.controller.policy_client,
                adapter,
                loop_config,
                startup_config,
                hooks=CompositeInferenceHooks(event_hooks),
                stop_requested=stop_event.is_set,
            )
            coordinator.prepare()
            with self._lock:
                session = self._session
                if session is None or session.session_id != session_id:
                    return
                if session.abort_requested:
                    session.finish_abort()
                elif session.state is EvaluationState.WARMING_UP:
                    session.warmup_succeeded()
                self._release_if_terminal_locked()
        except PolicyStartupCancelled:
            with self._lock:
                session = self._session
                if session is not None and session.session_id == session_id and session.abort_requested:
                    session.finish_abort()
                    self._release_if_terminal_locked()
        except BaseException as exc:
            with self._lock:
                session = self._session
                if session is not None and session.session_id == session_id:
                    session.fail(exc)
                    self._release_if_terminal_locked()
        finally:
            with self._lock:
                if self._warmup_thread is threading.current_thread():
                    self._warmup_thread = None
                self._release_if_terminal_locked()

    def _run_episode_start(
        self,
        session_id: str,
        episode_id: str,
        lease: EvaluationRuntimeLease,
        stop_event: threading.Event,
    ) -> None:
        try:
            with self._lock:
                session = self._session
                if session is None or session.session_id != session_id:
                    return
                config = session.config
            adapter = lease.make_adapter(config.prompt)
            loop_config = lease.make_loop_config(config.prompt)
            if loop_config.infer_only:
                raise ValueError("Evaluation cannot use infer_only runtime configuration")
            loop_config = dataclasses.replace(loop_config, max_steps=None)
            startup_config = dataclasses.replace(
                lease.make_startup_config(),
                warmup_enabled=False,
                prefetch_first_chunk=True,
            )
            event_hooks = RuntimeEventHooks(
                lease.controller.event_sink,
                RuntimeEventIdentity(
                    runtime_id=lease.controller.runtime_id,
                    generation_id=lease.controller.status().generation_id + 1,
                    session_id=session_id,
                    episode_id=episode_id,
                ),
            )
            coordinator = PolicyStartupCoordinator(
                lease.controller.policy_client,
                adapter,
                loop_config,
                startup_config,
                hooks=CompositeInferenceHooks(event_hooks),
                stop_requested=stop_event.is_set,
            )
            prepared = coordinator.prepare()
            if stop_event.is_set():
                return

            controller = lease.controller
            generation_id = controller.status().generation_id + 1
            controller.configure_event_identity(session_id=session_id, episode_id=episode_id)
            controller.configure(adapter, loop_config, initial_chunk=prepared.initial_chunk)
            started_at_ns = time.monotonic_ns()
            with self._lock:
                session = self._session
                if session is None or session.session_id != session_id:
                    return
                if session.abort_requested:
                    session.finish_abort()
                    self._release_if_terminal_locked()
                    return
                session.episode_started(generation_id, started_at_monotonic_ns=started_at_ns)
            controller.start()
            self._start_watchdog(
                session_id,
                episode_id,
                generation_id,
                started_at_ns + round(config.max_episode_seconds * 1e9),
            )
        except PolicyStartupCancelled:
            with self._lock:
                session = self._session
                if session is not None and session.session_id == session_id and session.abort_requested:
                    session.finish_abort()
                    self._release_if_terminal_locked()
        except BaseException as exc:
            with self._lock:
                session = self._session
                if session is not None and session.session_id == session_id:
                    session.fail(exc)
                    self._release_if_terminal_locked()
        finally:
            with self._lock:
                if self._start_thread is threading.current_thread():
                    self._start_thread = None
                self._release_if_terminal_locked()

    def _start_watchdog(
        self,
        session_id: str,
        episode_id: str,
        generation_id: int,
        deadline_ns: int,
    ) -> None:
        with self._lock:
            self._watchdog_stop_event = threading.Event()
            stop_event = self._watchdog_stop_event
            self._watchdog_thread = threading.Thread(
                target=self._run_watchdog,
                args=(session_id, episode_id, generation_id, deadline_ns, stop_event),
                name=f"evaluation-watchdog-{session_id}-{generation_id}",
                daemon=False,
            )
            self._watchdog_thread.start()

    def _run_watchdog(
        self,
        session_id: str,
        episode_id: str,
        generation_id: int,
        deadline_ns: int,
        stop_event: threading.Event,
    ) -> None:
        try:
            while not stop_event.is_set():
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                stop_event.wait(min(remaining_ns / 1e9, 0.1))
            if stop_event.is_set():
                return
            with self._lock:
                session = self._session
                lease = self._lease
                if session is None or lease is None or session.session_id != session_id:
                    return
                active_id = session.active_episode_id
                if active_id != episode_id or session.state is not EvaluationState.RUNNING:
                    return
                active = session.request_stop(EvaluationResult.TIMEOUT.value)
                if active.generation_id != generation_id:
                    return
                controller = lease.controller
            controller.stop(wait=False)
        finally:
            with self._lock:
                if self._watchdog_thread is threading.current_thread():
                    self._watchdog_thread = None

    def _require_session_and_lease(self) -> tuple[EvaluationSession, EvaluationRuntimeLease]:
        if self._session is not None and self._lease is None:
            raise EvaluationConflict(
                f"Evaluation is {self._session.state.value}",
                legal_operations=self._session.legal_operations(),
            )
        if self._session is None or self._lease is None:
            raise EvaluationConflict("No current evaluation session", legal_operations=("create",))
        return self._session, self._lease

    def _release_if_terminal_locked(self, *, force: bool = False) -> None:
        if self._session is None or self._lease is None or not self._session.is_terminal:
            return
        workers = (self._warmup_thread, self._start_thread)
        if any(worker is not None and worker.is_alive() for worker in workers):
            return
        controller_status = self._lease.controller.status()
        if controller_status.running and not force:
            return
        lease = self._lease
        self._lease = None
        lease.release()

    @staticmethod
    def _conflict_from_state(exc: EvaluationStateConflict) -> EvaluationConflict:
        return EvaluationConflict(str(exc), legal_operations=exc.legal_operations)

    def _raise_as_conflict(self, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except EvaluationStateConflict as exc:
            raise self._conflict_from_state(exc) from exc

    def _join_worker(self, attribute: str, timeout: float) -> bool:
        with self._lock:
            thread = getattr(self, attribute)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return thread is None or not thread.is_alive()
