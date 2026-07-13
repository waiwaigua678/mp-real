from __future__ import annotations

import queue
import threading
import time
import urllib.parse
from typing import Any

import numpy as np


class RealTimeChunkingBuffer:
    """Thread-safe online fusion buffer for overlapping action chunks."""

    def __init__(self, *, exp_weight: float = 0.0) -> None:
        self._exp_weight = exp_weight
        self._control_t = 0
        self._chunks: dict[int, np.ndarray] = {}
        self._generation = 0
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._control_t = 0
            self._chunks.clear()
            self._generation += 1

    def set_control_time(self, control_t: int) -> None:
        with self._lock:
            self._control_t = control_t

    def get_control_time(self) -> int:
        with self._lock:
            return self._control_t

    def get_generation(self) -> int:
        with self._lock:
            return self._generation

    def has_chunk(self, cursor: int) -> bool:
        with self._lock:
            return cursor in self._chunks

    def enqueue(self, chunk: np.ndarray, cursor: int, generation: int) -> bool:
        chunk = np.asarray(chunk, dtype=np.float32)
        with self._lock:
            if generation != self._generation:
                return False
            self._chunks[cursor] = chunk
            return True

    def get_action(self, current_time: int) -> np.ndarray | None:
        with self._lock:
            relevant: list[tuple[int, np.ndarray]] = []
            expired: list[int] = []
            for cursor, chunk in self._chunks.items():
                end = cursor + len(chunk)
                if cursor <= current_time < end:
                    relevant.append((cursor, chunk[current_time - cursor]))
                elif end <= current_time:
                    expired.append(cursor)

            for cursor in expired:
                del self._chunks[cursor]

            if not relevant:
                return None

            relevant.sort(key=lambda item: item[0])
            candidate_actions = np.asarray([action for _, action in relevant], dtype=np.float32)

        weights = np.exp(self._exp_weight * np.arange(len(candidate_actions), dtype=np.float32))
        weights = (weights / weights.sum())[:, None]
        return np.sum(candidate_actions * weights, axis=0).astype(np.float32)


def parse_server_url(url: str) -> tuple[str, int | None]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme:
        return parsed.hostname or "127.0.0.1", parsed.port
    return url, None


def sleep_until(next_t: float) -> None:
    sleep_s = next_t - time.monotonic()
    if sleep_s > 0:
        time.sleep(sleep_s)


def sleep_remaining(loop_t0: float, dt: float) -> None:
    sleep_s = dt - (time.monotonic() - loop_t0)
    if sleep_s > 0:
        time.sleep(sleep_s)


def rtc_replan_stride(args: Any) -> int:
    return args.replan_steps if args.rtc_replan_stride <= 0 else args.rtc_replan_stride


def rtc_prefetch_steps(args: Any) -> int:
    stride = rtc_replan_stride(args)
    return max(1, stride // 2) if args.rtc_prefetch_steps <= 0 else args.rtc_prefetch_steps


def select_rtc_cursor(rtc: RealTimeChunkingBuffer, args: Any) -> int | None:
    stride = rtc_replan_stride(args)
    prefetch = rtc_prefetch_steps(args)
    control_t = rtc.get_control_time()
    cursor = (control_t // stride) * stride
    if not rtc.has_chunk(cursor):
        return cursor
    next_cursor = cursor + stride
    if control_t + prefetch >= next_cursor and not rtc.has_chunk(next_cursor):
        return next_cursor
    return None


def raise_rtc_producer_error(errors: queue.Queue[BaseException]) -> None:
    try:
        exc = errors.get_nowait()
    except queue.Empty:
        return
    raise exc
