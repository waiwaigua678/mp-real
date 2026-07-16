"""Streaming, hardware-free episode metrics for recorded data viewers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


class _Reservoir:
    """Deterministic bounded sample used for display percentiles."""

    def __init__(self, capacity: int = 4096) -> None:
        self._capacity = capacity
        self._values: list[float] = []
        self._seen = 0
        self._sum = 0.0

    def add(self, value: object) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        self._seen += 1
        if len(self._values) < self._capacity:
            self._values.append(number)
            self._sum += number
            return
        # Evenly replace positions across the stream; this is deterministic,
        # bounded, and avoids retaining a complete long episode.
        position = (self._seen - 1) % self._capacity
        self._sum -= self._values[position]
        self._values[position] = number
        self._sum += number

    def percentile(self, value: float) -> float | None:
        if not self._values:
            return None
        return float(np.percentile(np.asarray(self._values, dtype=np.float64), value))

    def mean(self) -> float | None:
        return self._sum / len(self._values) if self._values else None


class EpisodeMetricAccumulator:
    """One-pass metrics accumulator for an action/state time series.

    Rows are ordinary decoded Parquet mappings.  Optional mp-real fields can
    be absent; standard LeRobot datasets then still get duration, action and
    motion metrics without invented telemetry values.
    """

    def __init__(self) -> None:
        self.frame_count = 0
        self.first_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self._latency = _Reservoir()
        self._cycle = _Reservoir()
        self._skew = _Reservoir()
        self._jump_sum = 0.0
        self._jump_max = 0.0
        self._jump_count = 0
        self._velocity_sq_sum = 0.0
        self._velocity_count = 0
        self._acceleration_sq_sum = 0.0
        self._acceleration_count = 0
        self._jerk_sq_sum = 0.0
        self._jerk_count = 0
        self._raw_to_stabilized = _Reservoir()
        self._stabilized_to_executed = _Reservoir()
        self._safety_modification_count = 0
        self._control_overrun_count = 0
        self._previous_action: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None
        self._previous_acceleration: np.ndarray | None = None
        self._previous_timestamp: float | None = None
        self._previous_row_timestamp: float | None = None

    def add(self, row: Mapping[str, Any]) -> None:
        timestamp = _number(row.get("timestamp"))
        if timestamp is not None:
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.last_timestamp = timestamp
        self.frame_count += 1
        self._latency.add(_number(row.get("mp_real.inference_latency_ns")))
        cycle_ns = _number(row.get("mp_real.control_cycle_ns"))
        self._cycle.add(cycle_ns)
        self._skew.add(_number(row.get("mp_real.camera_skew_ns")))
        if timestamp is not None and self._previous_row_timestamp is not None and cycle_ns is not None:
            expected_cycle_ns = (timestamp - self._previous_row_timestamp) * 1e9
            if expected_cycle_ns > 0 and cycle_ns > expected_cycle_ns * 1.05:
                self._control_overrun_count += 1
        if timestamp is not None:
            self._previous_row_timestamp = timestamp

        executed = _vector(row.get("action"))
        selected = _vector(row.get("mp_real.selected_raw_action"))
        stabilized = _vector(row.get("mp_real.stabilized_action"))
        if executed is None:
            return
        if selected is not None and stabilized is not None:
            self._raw_to_stabilized.add(float(np.linalg.norm(selected - stabilized)))
        if stabilized is not None:
            modification = float(np.linalg.norm(stabilized - executed))
            self._stabilized_to_executed.add(modification)
            if modification > 1e-9:
                self._safety_modification_count += 1

        if self._previous_action is not None:
            jump = float(np.linalg.norm(executed - self._previous_action))
            self._jump_sum += jump
            self._jump_max = max(self._jump_max, jump)
            self._jump_count += 1
            if timestamp is not None and self._previous_timestamp is not None:
                dt = timestamp - self._previous_timestamp
                if dt > 0:
                    velocity = (executed - self._previous_action) / dt
                    self._velocity_sq_sum += float(np.dot(velocity, velocity))
                    self._velocity_count += velocity.size
                    if self._previous_velocity is not None:
                        acceleration = (velocity - self._previous_velocity) / dt
                        self._acceleration_sq_sum += float(np.dot(acceleration, acceleration))
                        self._acceleration_count += acceleration.size
                        if self._previous_acceleration is not None:
                            jerk = (acceleration - self._previous_acceleration) / dt
                            self._jerk_sq_sum += float(np.dot(jerk, jerk))
                            self._jerk_count += jerk.size
                        self._previous_acceleration = acceleration
                    self._previous_velocity = velocity
        self._previous_action = executed
        self._previous_timestamp = timestamp

    def finish(self, *, dropped_frame_count: int = 0, dropped_event_count: int = 0) -> dict[str, float | int | None]:
        duration = None
        if self.first_timestamp is not None and self.last_timestamp is not None:
            duration = max(0.0, self.last_timestamp - self.first_timestamp)
        cycle_mean_ns = self._cycle.mean()
        return {
            "episode_duration_s": duration,
            "frame_count": self.frame_count,
            "inference_latency_p50_ms": _nanoseconds_to_ms(self._latency.percentile(50)),
            "inference_latency_p95_ms": _nanoseconds_to_ms(self._latency.percentile(95)),
            "inference_latency_p99_ms": _nanoseconds_to_ms(self._latency.percentile(99)),
            "control_frequency_mean_hz": 1e9 / cycle_mean_ns if cycle_mean_ns and cycle_mean_ns > 0 else None,
            "control_overrun_count": self._control_overrun_count,
            "camera_skew_p50_ms": _nanoseconds_to_ms(self._skew.percentile(50)),
            "camera_skew_p95_ms": _nanoseconds_to_ms(self._skew.percentile(95)),
            "action_jump_mean": self._jump_sum / self._jump_count if self._jump_count else None,
            "action_jump_max": self._jump_max if self._jump_count else None,
            "velocity_rms": _rms(self._velocity_sq_sum, self._velocity_count),
            "acceleration_rms": _rms(self._acceleration_sq_sum, self._acceleration_count),
            "jerk_rms": _rms(self._jerk_sq_sum, self._jerk_count),
            "raw_to_stabilized_mean": self._raw_to_stabilized.mean(),
            "stabilized_to_executed_mean": self._stabilized_to_executed.mean(),
            "safety_modification_count": self._safety_modification_count,
            "dropped_frame_count": dropped_frame_count,
            "dropped_event_count": dropped_event_count,
        }


def compute_episode_metrics(
    rows: Iterable[Mapping[str, Any]], *, dropped_frame_count: int = 0, dropped_event_count: int = 0
) -> dict[str, float | int | None]:
    accumulator = EpisodeMetricAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.finish(
        dropped_frame_count=dropped_frame_count,
        dropped_event_count=dropped_event_count,
    )


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    return result if result.ndim == 1 and np.all(np.isfinite(result)) else None


def _nanoseconds_to_ms(value: float | None) -> float | None:
    return value / 1e6 if value is not None else None


def _rms(total: float, count: int) -> float | None:
    return math.sqrt(total / count) if count else None
