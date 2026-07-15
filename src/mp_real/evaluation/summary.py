from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from mp_real.evaluation.models import EpisodeRecord, EvaluationResult


def build_summary(episodes: Iterable[EpisodeRecord], *, planned_episodes: int) -> dict[str, Any]:
    """Build a JSON-ready summary while keeping INVALID out of success rate."""
    completed = tuple(episode for episode in episodes if episode.result is not None)
    result_counts = Counter(episode.result.value for episode in completed if episode.result is not None)
    failure_counts = Counter(
        episode.failure_reason.value for episode in completed if episode.failure_reason is not None
    )
    valid_for_rate = tuple(episode for episode in completed if episode.result is not EvaluationResult.INVALID)
    successes = sum(episode.result is EvaluationResult.SUCCESS for episode in valid_for_rate)
    denominator = len(valid_for_rate)
    return {
        "planned_episodes": planned_episodes,
        "completed_episodes": len(completed),
        "remaining_episodes": max(0, planned_episodes - len(completed)),
        "result_counts": dict(sorted(result_counts.items())),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "successes": successes,
        "success_rate_denominator": denominator,
        "success_rate": successes / denominator if denominator else None,
    }
