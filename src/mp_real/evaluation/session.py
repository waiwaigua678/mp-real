from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from collections.abc import Iterable
from typing import Any

from mp_real.evaluation.models import EpisodeRecord, EvaluationConfig, EvaluationResult, EvaluationState, FailureReason
from mp_real.evaluation.summary import build_summary


class EvaluationStateConflict(RuntimeError):
    """A requested state transition is unavailable for the current session."""

    def __init__(self, operation: str, state: EvaluationState, legal_operations: Iterable[str]) -> None:
        self.operation = operation
        self.state = state
        self.legal_operations = tuple(legal_operations)
        allowed = ", ".join(self.legal_operations) or "none"
        super().__init__(f"Cannot {operation} while evaluation is {state.value}; legal operations: {allowed}")


@dataclasses.dataclass(frozen=True)
class StateTransition:
    from_state: EvaluationState
    to_state: EvaluationState
    monotonic_timestamp_ns: int
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "monotonic_timestamp_ns": self.monotonic_timestamp_ns,
            "reason": self.reason,
        }


class EvaluationSession:
    """Thread-safe state machine; it has no dependency on the Web layer."""

    _TERMINAL_STATES = frozenset({EvaluationState.COMPLETED, EvaluationState.ABORTED, EvaluationState.ERROR})

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._state = EvaluationState.IDLE
        self._episodes: list[EpisodeRecord] = []
        self._active_episode: EpisodeRecord | None = None
        self._episode_start_pending = False
        self._abort_requested = False
        self._error: str | None = None
        self._created_at_monotonic_ns = time.monotonic_ns()
        self._updated_at_monotonic_ns = self._created_at_monotonic_ns
        self._transitions: list[StateTransition] = []
        self._transition(EvaluationState.PREPARING, "configuration snapshots frozen")

    @property
    def session_id(self) -> str:
        return self.config.evaluation_id

    @property
    def state(self) -> EvaluationState:
        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._state in self._TERMINAL_STATES

    @property
    def active_episode_id(self) -> str | None:
        with self._lock:
            return self._active_episode.episode_id if self._active_episode is not None else None

    @property
    def active_episode_generation_id(self) -> int | None:
        with self._lock:
            return self._active_episode.generation_id if self._active_episode is not None else None

    @property
    def episode_start_pending(self) -> bool:
        with self._lock:
            return self._episode_start_pending

    @property
    def abort_requested(self) -> bool:
        with self._lock:
            return self._abort_requested

    def legal_operations(self) -> tuple[str, ...]:
        with self._lock:
            if self._state is EvaluationState.PREPARING:
                return ("warmup", "abort")
            if self._state is EvaluationState.WARMING_UP:
                return ("abort",)
            if self._state is EvaluationState.WAITING_RESET:
                return ("reset-ready", "abort")
            if self._state is EvaluationState.READY:
                return ("abort",) if self._episode_start_pending else ("start-episode", "abort")
            if self._state is EvaluationState.RUNNING:
                return ("stop-episode", "abort")
            if self._state is EvaluationState.STOPPING:
                return ("abort",)
            if self._state is EvaluationState.WAITING_RESULT:
                return ("label", "abort")
            if self._state is EvaluationState.SAVING:
                return ("abort",)
            return ()

    def start_warmup(self) -> None:
        with self._lock:
            self._require("warmup", EvaluationState.PREPARING)
            self._transition(EvaluationState.WARMING_UP)

    def warmup_succeeded(self) -> None:
        with self._lock:
            self._require_internal(EvaluationState.WARMING_UP)
            self._transition(EvaluationState.WAITING_RESET)

    def reset_ready(self) -> None:
        with self._lock:
            self._require("reset-ready", EvaluationState.WAITING_RESET)
            self._transition(EvaluationState.READY)

    def begin_episode_preparation(self) -> EpisodeRecord:
        with self._lock:
            self._require("start-episode", EvaluationState.READY)
            if self._episode_start_pending:
                self._raise_conflict("start-episode")
            episode = EpisodeRecord(
                episode_id=uuid.uuid4().hex,
                episode_index=len(self._episodes) + 1,
            )
            self._active_episode = episode
            self._episode_start_pending = True
            self._touch()
            return dataclasses.replace(episode)

    def episode_started(self, generation_id: int, *, started_at_monotonic_ns: int | None = None) -> None:
        with self._lock:
            self._require_internal(EvaluationState.READY)
            if self._active_episode is None or not self._episode_start_pending:
                raise RuntimeError("No prepared episode is available to start")
            self._active_episode.generation_id = generation_id
            self._active_episode.started_at_monotonic_ns = started_at_monotonic_ns or time.monotonic_ns()
            self._episode_start_pending = False
            self._transition(EvaluationState.RUNNING)

    def request_stop(self, trigger: str) -> EpisodeRecord:
        with self._lock:
            self._require("stop-episode", EvaluationState.RUNNING)
            assert self._active_episode is not None
            self._active_episode.stop_trigger = trigger
            self._transition(EvaluationState.STOPPING, trigger)
            return dataclasses.replace(self._active_episode)

    def runtime_stopped(self, *, stopped_at_monotonic_ns: int | None = None) -> None:
        with self._lock:
            if self._state is not EvaluationState.STOPPING:
                return
            if self._active_episode is not None:
                self._active_episode.stopped_at_monotonic_ns = stopped_at_monotonic_ns or time.monotonic_ns()
            self._transition(EvaluationState.WAITING_RESULT)

    def label(
        self,
        result: EvaluationResult,
        *,
        failure_reason: FailureReason | None = None,
        notes: str = "",
    ) -> None:
        with self._lock:
            self._require("label", EvaluationState.WAITING_RESULT)
            if self._active_episode is None:
                raise RuntimeError("No stopped episode is awaiting a result")
            if result is EvaluationResult.FAILURE and failure_reason is None:
                raise ValueError("failure_reason is required when result is FAILURE")
            if result is not EvaluationResult.FAILURE and failure_reason is not None:
                raise ValueError("failure_reason is only valid when result is FAILURE")
            self._transition(EvaluationState.SAVING)
            self._active_episode.result = result
            self._active_episode.failure_reason = failure_reason
            self._active_episode.notes = notes
            if self._active_episode.stopped_at_monotonic_ns is None:
                self._active_episode.stopped_at_monotonic_ns = time.monotonic_ns()
            self._episodes.append(self._active_episode)
            self._active_episode = None
            self._episode_start_pending = False
            if len(self._episodes) >= self.config.planned_episodes:
                self._transition(EvaluationState.COMPLETED)
            else:
                # Both auto and operator-driven advance require a fresh manual
                # reset acknowledgement before the next action loop can begin.
                reason = "auto advance" if self.config.auto_advance else "await operator reset acknowledgement"
                self._transition(EvaluationState.WAITING_RESET, reason)

    def request_abort(self) -> bool:
        """Request an abort and return whether a running loop must be stopped."""
        with self._lock:
            if self._state in self._TERMINAL_STATES:
                self._raise_conflict("abort")
            self._abort_requested = True
            if self._state is EvaluationState.RUNNING:
                assert self._active_episode is not None
                self._active_episode.stop_trigger = EvaluationResult.OPERATOR_ABORT.value
                self._transition(EvaluationState.STOPPING, EvaluationResult.OPERATOR_ABORT.value)
                return True
            if self._state is EvaluationState.STOPPING:
                return True
            return False

    def finish_abort(self) -> None:
        with self._lock:
            if self._state in self._TERMINAL_STATES:
                return
            if self._active_episode is not None:
                self._active_episode.result = EvaluationResult.OPERATOR_ABORT
                self._active_episode.failure_reason = None
                self._active_episode.stopped_at_monotonic_ns = (
                    self._active_episode.stopped_at_monotonic_ns or time.monotonic_ns()
                )
                self._episodes.append(self._active_episode)
                self._active_episode = None
            self._episode_start_pending = False
            self._transition(EvaluationState.ABORTED, EvaluationResult.OPERATOR_ABORT.value)

    def fail(self, error: BaseException | str) -> None:
        with self._lock:
            if self._state in self._TERMINAL_STATES:
                return
            self._error = str(error)
            if self._active_episode is not None:
                self._active_episode.result = EvaluationResult.SYSTEM_ERROR
                self._active_episode.stopped_at_monotonic_ns = (
                    self._active_episode.stopped_at_monotonic_ns or time.monotonic_ns()
                )
                self._episodes.append(self._active_episode)
                self._active_episode = None
            self._episode_start_pending = False
            self._transition(EvaluationState.ERROR, self._error)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            completed = tuple(dataclasses.replace(episode) for episode in self._episodes)
            active = dataclasses.replace(self._active_episode) if self._active_episode is not None else None
            return {
                "evaluation_id": self.config.evaluation_id,
                "state": self._state.value,
                "config": self.config.to_dict(),
                "created_at_monotonic_ns": self._created_at_monotonic_ns,
                "updated_at_monotonic_ns": self._updated_at_monotonic_ns,
                "error": self._error,
                "legal_operations": list(self.legal_operations()),
                "episode_start_pending": self._episode_start_pending,
                "abort_requested": self._abort_requested,
                "active_episode": active.to_dict() if active is not None else None,
                "episodes": [episode.to_dict() for episode in completed],
                "summary": build_summary(completed, planned_episodes=self.config.planned_episodes),
                "transitions": [transition.to_dict() for transition in self._transitions],
            }

    def _transition(self, target: EvaluationState, reason: str | None = None) -> None:
        now_ns = time.monotonic_ns()
        self._transitions.append(StateTransition(self._state, target, now_ns, reason))
        self._state = target
        self._updated_at_monotonic_ns = now_ns

    def _touch(self) -> None:
        self._updated_at_monotonic_ns = time.monotonic_ns()

    def _require(self, operation: str, expected: EvaluationState) -> None:
        if self._state is not expected:
            self._raise_conflict(operation)

    def _require_internal(self, expected: EvaluationState) -> None:
        if self._state is not expected:
            raise RuntimeError(f"Expected evaluation state {expected.value}, got {self._state.value}")

    def _raise_conflict(self, operation: str) -> None:
        raise EvaluationStateConflict(operation, self._state, self.legal_operations())
