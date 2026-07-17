"""Bounded asynchronous writer for compact Baseline run references."""

from __future__ import annotations

import dataclasses
import queue
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mp_real.evaluation.baseline.service import BaselineService


class BaselineReferenceWriter:
    """Keep all Baseline JSON writes off Web handlers and robot control paths."""

    def __init__(self, service: BaselineService, *, queue_size: int = 16) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._service = service
        self._queue: queue.Queue[_Job | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._closed = False
        self._error: str | None = None
        self._dropped = 0
        self._worker = threading.Thread(target=self._run, name="baseline-reference-writer", daemon=False)
        self._worker.start()

    def submit_evaluation(self, snapshot: Mapping[str, Any]) -> bool:
        """Best-effort non-blocking enqueue; never blocks an evaluation transition."""
        config = snapshot.get("config")
        if not isinstance(config, Mapping) or not config.get("baseline_id"):
            return True
        with self._lock:
            if self._closed:
                return False
        return self._enqueue("attach_evaluation", {"snapshot": dict(snapshot)}) is not None

    def submit_create(
        self, payload: Mapping[str, Any], *, runtime_config: Mapping[str, Any], git_commit: str | None
    ) -> dict[str, Any]:
        return self._submit(
            "create",
            {"payload": dict(payload), "runtime_config": dict(runtime_config), "git_commit": git_commit},
        )

    def submit_clone(self, baseline_id: str, patch: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
        return self._submit("clone", {"baseline_id": baseline_id, "patch": dict(patch), "reason": reason})

    def submit_create_from_evaluation(self, snapshot: Mapping[str, Any], *, name: str | None = None) -> dict[str, Any]:
        return self._submit("create_from_evaluation", {"snapshot": dict(snapshot), "name": name})

    def submit_open_loop(self, baseline_id: str, result_dir: Path | str) -> dict[str, Any]:
        return self._submit("attach_open_loop", {"baseline_id": baseline_id, "result_dir": str(result_dir)})

    def job_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"unknown Baseline job {job_id}") from exc
            return job.to_dict()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "dropped": self._dropped,
                "error": self._error,
                "jobs": [item.to_dict() for item in self._jobs.values()],
            }

    def close(self, *, timeout: float = 10.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full as exc:
            raise TimeoutError("Baseline reference writer queue did not drain during shutdown") from exc
        self._worker.join(timeout)
        if self._worker.is_alive():
            raise TimeoutError("Baseline reference writer did not stop")

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._run_job(job)
            except BaseException as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def _submit(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        job = self._enqueue(kind, payload)
        if job is None:
            raise RuntimeError("Baseline writer queue is full")
        return job.to_dict()

    def _enqueue(self, kind: str, payload: Mapping[str, Any]) -> _Job | None:
        job = _Job(job_id=f"baseline-{uuid.uuid4().hex[:12]}", kind=kind, payload=dict(payload))
        with self._lock:
            if self._closed:
                return None
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._dropped += 1
                return None
            self._jobs[job.job_id] = job
        return job

    def _run_job(self, job: _Job) -> None:
        with self._lock:
            job.state = "running"
        try:
            if job.kind == "attach_evaluation":
                snapshot = job.payload["snapshot"]
                config = snapshot.get("config") if isinstance(snapshot, Mapping) else None
                baseline_id = config.get("baseline_id") if isinstance(config, Mapping) else None
                if baseline_id:
                    baseline = self._service.attach_evaluation(str(baseline_id), snapshot)
                    result: Any = baseline.to_dict()
                else:
                    result = {"ignored": "evaluation has no baseline_id"}
            elif job.kind == "create":
                baseline = self._service.create_from_runtime(
                    job.payload["payload"],
                    runtime_config=job.payload["runtime_config"],
                    git_commit=job.payload["git_commit"],
                )
                result = baseline.to_dict()
            elif job.kind == "clone":
                baseline = self._service.clone(
                    str(job.payload["baseline_id"]),
                    job.payload["patch"],
                    derived_reason=str(job.payload["reason"]),
                )
                result = baseline.to_dict()
            elif job.kind == "create_from_evaluation":
                baseline = self._service.create_from_evaluation(job.payload["snapshot"], name=job.payload["name"])
                result = baseline.to_dict()
            elif job.kind == "attach_open_loop":
                baseline = self._service.attach_open_loop(
                    str(job.payload["baseline_id"]), str(job.payload["result_dir"])
                )
                result = baseline.to_dict()
            else:  # pragma: no cover - all submitters are local.
                raise RuntimeError(f"unsupported Baseline job {job.kind}")
        except BaseException as exc:
            with self._lock:
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = True
            return
        with self._lock:
            job.state = "complete"
            job.result = result
            job.finished = True


@dataclasses.dataclass
class _Job:
    job_id: str
    kind: str
    payload: Mapping[str, Any]
    state: str = "queued"
    result: Any = None
    error: str | None = None
    finished: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "finished": self.finished,
            "result": self.result,
            "error": self.error,
        }
