from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from mp_real.evaluation.models import EpisodeRecord, EvaluationResult


def build_summary(episodes: Iterable[EpisodeRecord], *, planned_episodes: int) -> dict[str, Any]:
    """Build a JSON-ready summary with explicit evaluation result semantics."""
    completed = tuple(episode for episode in episodes if episode.result is not None)
    result_counts = Counter(episode.result.value for episode in completed if episode.result is not None)
    failure_counts = Counter(
        episode.failure_reason.value for episode in completed if episode.failure_reason is not None
    )
    # A system error is retained for auditability but is not an operator-valid
    # trial by default.  Operator abort terminates the session rather than
    # becoming a scored result.  TIMEOUT and SAFETY_ABORT remain valid failed
    # trials, matching the evaluation product semantics.
    excluded_from_rate = {
        EvaluationResult.INVALID,
        EvaluationResult.SYSTEM_ERROR,
        EvaluationResult.OPERATOR_ABORT,
    }
    valid_for_rate = tuple(episode for episode in completed if episode.result not in excluded_from_rate)
    successes = sum(episode.result is EvaluationResult.SUCCESS for episode in valid_for_rate)
    denominator = len(valid_for_rate)
    stop_counts = Counter(episode.stop_trigger for episode in completed if episode.stop_trigger)
    return {
        "planned_episodes": planned_episodes,
        "completed_episodes": len(completed),
        "remaining_episodes": max(0, planned_episodes - len(completed)),
        "result_counts": dict(sorted(result_counts.items())),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "invalid_count": result_counts[EvaluationResult.INVALID.value],
        "timeout_count": stop_counts[EvaluationResult.TIMEOUT.value],
        "safety_abort_count": stop_counts[EvaluationResult.SAFETY_ABORT.value],
        "system_error_count": result_counts[EvaluationResult.SYSTEM_ERROR.value],
        "successes": successes,
        "success_rate_denominator": denominator,
        "success_rate": successes / denominator if denominator else None,
    }
