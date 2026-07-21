"""Bounded background writer for an explicit replay record."""

from __future__ import annotations

import dataclasses
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from mp_real.replay.models import ReplayPlan, json_safe


class ReplayRecordingError(RuntimeError):
    pass


_STOP = object()


@dataclasses.dataclass(frozen=True)
class ReplayRecordingConfig:
    root: Path
    queue_size: int = 256

    def __post_init__(self) -> None:
        if self.queue_size <= 0:
            raise ValueError("replay recording queue_size must be positive")


class ReplayRecordWriter:
    """Write an explicit, versioned replay record without blocking control."""

    def __init__(self, config: ReplayRecordingConfig, plan: ReplayPlan) -> None:
        self._config = config
        self._plan = plan
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_size)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._dropped = 0
        self._started = False
        self._stopped = False
        self._result: str | None = None

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"replay-record-writer-{self._plan.plan_id[:8]}",
                daemon=False,
            )
            self._thread.start()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self._error is not None:
                raise ReplayRecordingError(f"replay recorder failed: {self._error}") from self._error
            if not self._started or self._stopped:
                raise ReplayRecordingError("replay recorder is not active")
        try:
            self._queue.put_nowait(dict(event))
        except queue.Full as exc:
            with self._lock:
                self._dropped += 1
            raise ReplayRecordingError("replay recorder queue is full; safety abort required") from exc

    def stop(self, *, result: str, timeout: float = 10.0) -> bool:
        with self._lock:
            if not self._started:
                return True
            if not self._stopped:
                self._stopped = True
                self._result = result
                self._queue.put(_STOP)
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def _run(self) -> None:
        root = self._config.root
        work = root / f"replay-{self._plan.plan_id}.inprogress"
        final = root / f"replay-{self._plan.plan_id}"
        try:
            work.mkdir(parents=True, exist_ok=False)
            with (work / "events.jsonl").open("w", encoding="utf-8") as stream:
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        break
                    stream.write(json.dumps(json_safe(item), ensure_ascii=False, separators=(",", ":")))
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            manifest = {
                "schema_version": 2,
                "kind": "mp_real_robot_trajectory_replay",
                "source_dataset": self._plan.dataset_id,
                "source_episode_index": self._plan.episode_index,
                "source_start_sample": self._plan.start_sample,
                "source_end_sample": self._plan.end_sample,
                "replay_mode": self._plan.mode.value,
                "timing_mode": self._plan.timing_mode.value,
                "speed_scale": self._plan.speed_scale,
                "acknowledgement_strategy": self._plan.constraints.acknowledgement_strategy.value,
                "plan_id": self._plan.plan_id,
                "plan_hash": self._plan.plan_hash,
                "session_id": self._plan.session_id,
                "generation_id": self._plan.generation_id,
                "dropped_event_count": self._dropped,
                "result": self._result,
                "finalized_at_monotonic_ns": time.monotonic_ns(),
            }
            with (work / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(work, final)
        except BaseException as exc:
            with self._lock:
                self._error = exc
