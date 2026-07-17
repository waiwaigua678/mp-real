"""Read-only target and prompt resolution for recorded episodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mp_real.data.models import RecordedEpisodeSource, RecordedSample
from mp_real.evaluation.open_loop.models import (
    OpenLoopEvaluationConfig,
    PredictionResultSource,
)


class OpenLoopInputError(ValueError):
    """A dataset cannot supply a complete, formal open-loop input."""


def resolve_prompt(
    source: RecordedEpisodeSource,
    episode_index: int,
    samples: Sequence[RecordedSample],
    override: str | None,
) -> str:
    if override is not None:
        if not override.strip():
            raise OpenLoopInputError("prompt_override cannot be empty")
        return override
    metadata = source.get_episode_metadata(episode_index)
    if len(metadata.tasks) != 1:
        raise OpenLoopInputError(
            f"episode {episode_index} has {len(metadata.tasks)} tasks; prompt_override is required"
        )
    if not samples:
        raise OpenLoopInputError(f"episode {episode_index} has no readable samples")
    prompt = source.get_task(episode_index, samples[0].task_index)
    if not prompt.strip():
        raise OpenLoopInputError(f"episode {episode_index} has an empty recorded task")
    if any(source.get_task(episode_index, sample.task_index) != prompt for sample in samples):
        raise OpenLoopInputError(f"episode {episode_index} changes task; prompt_override is required")
    return prompt


def resolve_image_masks(config: OpenLoopEvaluationConfig, roles: tuple[str, ...]) -> dict[str, np.bool_]:
    """Use an explicit recording/evaluation mask; never infer camera roles from action dimension."""

    configured = config.image_masks or {}
    unknown = set(configured) - set(roles)
    if unknown:
        raise OpenLoopInputError("image_masks contains unselected roles: " + ", ".join(sorted(unknown)))
    # LeRobot v2.1 does not standardize image masks.  Existing recordings have
    # complete selected image inputs, so the conservative default is visible.
    # The chosen values are serialized in config.json and every report.
    return {role: np.bool_(bool(configured.get(role, True))) for role in roles}


def resolve_target(
    source: RecordedEpisodeSource,
    sample: RecordedSample,
    config: OpenLoopEvaluationConfig,
) -> tuple[np.ndarray, str]:
    """Return one target vector and the physical source field used."""

    spec = source.get_action_spec()
    source_kind = config.target_source
    if source_kind is PredictionResultSource.ACTION:
        return _validated(sample.action, spec.action_dim, "action"), "action"
    if source_kind is PredictionResultSource.EXECUTED_ACTION:
        replay = source.get_dataset_metadata().info.get("mp_real", {})
        if isinstance(replay, dict):
            replay = replay.get("replay", {})
        if isinstance(replay, dict) and replay.get("action_source") == "executed_action":
            return _validated(sample.action, spec.action_dim, "action"), "action (declared executed_action provenance)"
        return _extension_target(source, sample, ("executed_action", "mp_real.executed_action"), spec.action_dim)
    if source_kind is PredictionResultSource.EXPERT_ACTION:
        return _extension_target(source, sample, ("expert_action", "mp_real.expert_action"), spec.action_dim)
    if source_kind is PredictionResultSource.STATE_DERIVED:
        derived = config.state_derived
        assert derived is not None
        if len(derived.state_indices) != spec.action_dim:
            raise OpenLoopInputError(
                "state-derived conversion must explicitly produce every ActionSpec action dimension"
            )
        state = np.asarray(sample.state, dtype=np.float32)
        if max(derived.state_indices) >= len(state):
            raise OpenLoopInputError("state-derived conversion references a missing observation.state index")
        values = state[list(derived.state_indices)] * np.asarray(derived.scale) + np.asarray(derived.offset)
        return _validated(values, spec.action_dim, "state-derived target"), f"state-derived:{derived.converter_id}"
    raise AssertionError(source_kind)


def _extension_target(
    source: RecordedEpisodeSource,
    sample: RecordedSample,
    fields: tuple[str, ...],
    action_dim: int,
) -> tuple[np.ndarray, str]:
    get_row = getattr(source, "get_row", None)
    if not callable(get_row):
        raise OpenLoopInputError(f"target source {fields[0]!r} is unavailable from this episode source")
    columns = set(getattr(source, "get_column_names")(sample.episode_index))
    field = next((candidate for candidate in fields if candidate in columns), None)
    if field is None:
        raise OpenLoopInputError(f"target source is missing; expected one of: {', '.join(fields)}")
    row = get_row(sample.episode_index, sample.frame_index, columns=(field,))
    return _validated(row.get(field), action_dim, field), field


def _validated(value: Any, action_dim: int, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (action_dim,):
        raise OpenLoopInputError(f"{field} shape must be ({action_dim},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise OpenLoopInputError(f"{field} contains non-finite values")
    return vector.copy()
