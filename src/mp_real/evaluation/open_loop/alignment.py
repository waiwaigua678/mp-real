"""Explicit target alignment for teacher-forced action chunks."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

from mp_real.data.models import RecordedSample
from mp_real.evaluation.open_loop.models import (
    AlignedAction,
    AlignmentMode,
    OpenLoopEvaluationConfig,
)
from mp_real.evaluation.open_loop.source import OpenLoopInputError


class ActionAlignment:
    """Map one chunk horizon to a recorded target without hidden offsets."""

    def __init__(self, samples: Sequence[RecordedSample], config: OpenLoopEvaluationConfig, *, fps: float) -> None:
        if fps <= 0:
            raise ValueError("dataset fps must be positive")
        self._samples = tuple(samples)
        self._config = config
        self._fps = float(fps)
        self._timestamps = tuple(float(sample.timestamp) for sample in samples)
        self._control_to_local: dict[int, int] = {}
        if config.alignment_mode is AlignmentMode.ABSOLUTE_CONTROL_STEP_ALIGNMENT:
            if not config.allow_frame_index_as_control_step:
                raise OpenLoopInputError(
                    "absolute_control_step alignment requires explicit allow_frame_index_as_control_step"
                )
            for local_index, sample in enumerate(samples):
                cursor = _optional_integer(sample.telemetry.get("chunk_cursor"))
                if cursor is None or cursor < 0:
                    raise OpenLoopInputError(
                        "absolute_control_step alignment requires recorded non-negative mp_real.chunk_cursor telemetry"
                    )
                if sample.frame_index in self._control_to_local:
                    raise OpenLoopInputError(f"duplicate control step/frame_index {sample.frame_index}")
                self._control_to_local[sample.frame_index] = local_index

    def align(self, source_local_index: int, horizon_index: int) -> AlignedAction:
        if source_local_index < 0 or source_local_index >= len(self._samples) or horizon_index < 0:
            raise IndexError("source_local_index and horizon_index must be non-negative and in range")
        if self._config.alignment_mode is AlignmentMode.SAMPLE_INDEX_ALIGNMENT:
            return self._sample_alignment(source_local_index, horizon_index)
        if self._config.alignment_mode is AlignmentMode.TIMESTAMP_ALIGNMENT:
            return self._timestamp_alignment(source_local_index, horizon_index)
        return self._control_alignment(source_local_index, horizon_index)

    def _sample_alignment(self, index: int, horizon: int) -> AlignedAction:
        target = index + horizon
        if target >= len(self._samples):
            return self._invalid(index, horizon, "episode_tail")
        return self._valid(index, horizon, target, 0.0)

    def _timestamp_alignment(self, index: int, horizon: int) -> AlignedAction:
        expected = self._timestamps[index] + horizon / self._fps
        right = bisect.bisect_left(self._timestamps, expected)
        candidates = [candidate for candidate in (right - 1, right) if 0 <= candidate < len(self._samples)]
        if not candidates:
            return self._invalid(index, horizon, "episode_empty")
        target = min(candidates, key=lambda candidate: abs(self._timestamps[candidate] - expected))
        error = self._timestamps[target] - expected
        if abs(error) > self._config.max_timestamp_error_s:
            return self._invalid(index, horizon, "timestamp_tolerance", target=target, error=error)
        return self._valid(index, horizon, target, error)

    def _control_alignment(self, index: int, horizon: int) -> AlignedAction:
        source = self._samples[index]
        control_step = source.frame_index + horizon
        target = self._control_to_local.get(control_step)
        if target is None:
            return self._invalid(index, horizon, "missing_control_step", control_step=control_step)
        expected = source.timestamp + horizon / self._fps
        return self._valid(
            index,
            horizon,
            target,
            self._samples[target].timestamp - expected,
            control_step=control_step,
            chunk_cursor=_optional_integer(source.telemetry.get("chunk_cursor")),
        )

    def _valid(
        self,
        source: int,
        horizon: int,
        target: int,
        error: float,
        *,
        control_step: int | None = None,
        chunk_cursor: int | None = None,
    ) -> AlignedAction:
        return AlignedAction(
            mode=self._config.alignment_mode,
            source_sample_index=source,
            horizon_index=horizon,
            target_sample_index=target,
            target_timestamp=self._samples[target].timestamp,
            alignment_error_s=error,
            valid=True,
            control_step=control_step,
            chunk_cursor=chunk_cursor,
        )

    def _invalid(
        self,
        source: int,
        horizon: int,
        reason: str,
        *,
        target: int | None = None,
        error: float | None = None,
        control_step: int | None = None,
    ) -> AlignedAction:
        return AlignedAction(
            mode=self._config.alignment_mode,
            source_sample_index=source,
            horizon_index=horizon,
            target_sample_index=target,
            target_timestamp=self._samples[target].timestamp if target is not None else None,
            alignment_error_s=error,
            valid=False,
            reason=reason,
            control_step=control_step,
            chunk_cursor=_optional_integer(self._samples[source].telemetry.get("chunk_cursor")),
        )


def _optional_integer(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
