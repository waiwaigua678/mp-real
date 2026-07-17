"""Bounded, stoppable background orchestration for offline open-loop jobs."""

from __future__ import annotations

import dataclasses
import enum
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource
from mp_real.data.models import RecordedEpisodeSource
from mp_real.evaluation.open_loop.evaluator import OpenLoopEvaluator
from mp_real.evaluation.open_loop.models import OpenLoopEvaluationConfig


class OpenLoopJobState(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL_ERROR = "partial_error"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclasses.dataclass
class _Job:
    job_id: str
    config: OpenLoopEvaluationConfig
    stop_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    state: OpenLoopJobState = OpenLoopJobState.QUEUED
    submitted_ns: int = dataclasses.field(default_factory=time.monotonic_ns)
    started_ns: int | None = None
    finished_ns: int | None = None
    progress: dict[str, Any] = dataclasses.field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class OpenLoopEvaluationJobManager:
    """One explicit non-daemon worker and a bounded submission queue.

    The manager stores job identity separately from a source dataset and opens
    a new read-only LeRobot source in the worker.  That prevents a browser
    request from doing policy work or sharing mutable reader state.
    """

    def __init__(
        self,
        output_root: Path | str,
        *,
        queue_size: int = 8,
        source_factory: Callable[[Path], RecordedEpisodeSource] = LeRobotV21EpisodeSource,
        policy_factory: Callable[[str, str | None, float, float], Any] | None = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.output_root = Path(output_root).expanduser().resolve()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_size)
        self._jobs: dict[str, _Job] = {}
        self._source_factory = source_factory
        self._policy_factory = policy_factory
        self._lock = threading.RLock()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="open-loop-evaluation-worker", daemon=False)
        self._worker.start()

    def submit(self, config: OpenLoopEvaluationConfig) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("open-loop job manager is closed")
            job_id = f"open-loop-{uuid.uuid4().hex[:12]}"
            isolated = dataclasses.replace(
                config,
                evaluation_id=job_id,
                output_dir=self.output_root / job_id,
                resume=False,
            )
            job = _Job(job_id=job_id, config=isolated)
            self._jobs[job_id] = job
            try:
                self._queue.put_nowait(job_id)
            except queue.Full as exc:
                self._jobs.pop(job_id, None)
                raise RuntimeError("open-loop evaluation queue is full") from exc
            return self._status_locked(job)

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._get(job_id)
            if job.state in {OpenLoopJobState.COMPLETE, OpenLoopJobState.PARTIAL_ERROR, OpenLoopJobState.ERROR}:
                raise RuntimeError(f"open-loop job {job_id} is already terminal ({job.state.value})")
            job.stop_event.set()
            if job.state is OpenLoopJobState.QUEUED:
                job.state = OpenLoopJobState.CANCELLED
                job.finished_ns = time.monotonic_ns()
            return self._status_locked(job)

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._status_locked(self._get(job_id))

    def list_status(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._jobs.values(), key=lambda item: item.submitted_ns, reverse=True)
            return [self._status_locked(job) for job in ordered]

    def report_path(self, job_id: str, episode_index: int) -> Path:
        with self._lock:
            job = self._get(job_id)
            path = job.config.output_dir / "reports" / f"episode_{episode_index:06d}.json"
            if not path.is_file():
                raise FileNotFoundError("open-loop episode report is not available")
            return path

    def prediction_path(self, job_id: str, episode_index: int) -> Path:
        with self._lock:
            job = self._get(job_id)
            path = job.config.output_dir / "predictions" / f"episode_{episode_index:06d}.npz"
            if not path.is_file():
                raise FileNotFoundError("open-loop episode predictions are not available")
            return path

    def close(self, *, timeout: float = 10.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in self._jobs.values():
                if job.state in {OpenLoopJobState.QUEUED, OpenLoopJobState.RUNNING}:
                    job.stop_event.set()
            self._queue.put(None)
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("open-loop evaluation worker did not stop")

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None or job.stop_event.is_set():
                        continue
                    job.state = OpenLoopJobState.RUNNING
                    job.started_ns = time.monotonic_ns()
                source = self._source_factory(job.config.dataset)
                try:
                    result = OpenLoopEvaluator(
                        job.config,
                        source=source,
                        policy_factory=self._policy_factory,
                        stop_event=job.stop_event,
                        progress_callback=lambda payload: self._set_progress(job_id, payload),
                    ).run()
                finally:
                    source.close()
                with self._lock:
                    job.finished_ns = time.monotonic_ns()
                    job.state = OpenLoopJobState(result.status)
            except BaseException as exc:
                with self._lock:
                    job = self._jobs.get(job_id) if job_id is not None else None
                    if job is not None:
                        job.finished_ns = time.monotonic_ns()
                        job.state = OpenLoopJobState.CANCELLED if job.stop_event.is_set() else OpenLoopJobState.ERROR
                        job.error_type = type(exc).__name__
                        job.error_message = str(exc)
            finally:
                self._queue.task_done()

    def _set_progress(self, job_id: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.progress = dict(payload)

    def _get(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown open-loop job {job_id}") from exc

    @staticmethod
    def _status_locked(job: _Job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "submitted_monotonic_ns": job.submitted_ns,
            "started_monotonic_ns": job.started_ns,
            "finished_monotonic_ns": job.finished_ns,
            "progress": dict(job.progress),
            "error_type": job.error_type,
            "error_message": job.error_message,
            "output_dir": str(job.config.output_dir),
            "policy_label": job.config.policy_label,
        }
