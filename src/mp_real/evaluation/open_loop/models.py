"""Immutable contracts for teacher-forced open-loop evaluation."""

from __future__ import annotations

import dataclasses
import enum
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class AlignmentMode(enum.StrEnum):
    SAMPLE_INDEX_ALIGNMENT = "sample_index"
    TIMESTAMP_ALIGNMENT = "timestamp"
    ABSOLUTE_CONTROL_STEP_ALIGNMENT = "absolute_control_step"


class PredictionResultSource(enum.StrEnum):
    ACTION = "action"
    EXECUTED_ACTION = "executed_action"
    EXPERT_ACTION = "expert_action"
    STATE_DERIVED = "state_derived"


class EvaluationRequestMode(enum.StrEnum):
    SEQUENTIAL = "sequential"
    BATCH = "batch"


@dataclasses.dataclass(frozen=True)
class OpenLoopWarmupConfig:
    enabled: bool = True
    requests: int = 1
    timeout_s: float = 60.0
    inference_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        if self.requests <= 0:
            raise ValueError("warmup requests must be positive")
        if self.timeout_s <= 0 or self.inference_timeout_s <= 0:
            raise ValueError("warmup timeouts must be positive")


@dataclasses.dataclass(frozen=True)
class StateDerivedTargetConfig:
    """An explicit state-to-action conversion; never inferred from next state."""

    converter_id: str
    state_indices: tuple[int, ...]
    scale: tuple[float, ...]
    offset: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.converter_id.strip():
            raise ValueError("state-derived converter_id cannot be empty")
        if not self.state_indices:
            raise ValueError("state-derived state_indices cannot be empty")
        if not (len(self.state_indices) == len(self.scale) == len(self.offset)):
            raise ValueError("state-derived indices, scale, and offset must have equal lengths")
        if min(self.state_indices) < 0:
            raise ValueError("state-derived indices must be non-negative")


@dataclasses.dataclass(frozen=True)
class OpenLoopEvaluationConfig:
    """All inputs that affect a reproducible teacher-forced evaluation."""

    dataset: Path
    episode_indices: tuple[int, ...] | None
    policy_url: str
    policy_label: str
    output_dir: Path
    prompt_override: str | None = None
    policy_api_key: str | None = None
    connection_timeout_s: float = 10.0
    metadata_timeout_s: float = 10.0
    warmup: OpenLoopWarmupConfig = dataclasses.field(default_factory=OpenLoopWarmupConfig)
    target_source: PredictionResultSource = PredictionResultSource.ACTION
    alignment_mode: AlignmentMode = AlignmentMode.SAMPLE_INDEX_ALIGNMENT
    max_timestamp_error_s: float = 0.05
    selected_camera_roles: tuple[str, ...] | None = None
    image_masks: Mapping[str, bool] | None = None
    request_mode: EvaluationRequestMode = EvaluationRequestMode.SEQUENTIAL
    batch_size: int = 1
    deterministic_seed: int | None = None
    resize_size: int = 224
    replan_steps: int = 5
    allow_frame_index_as_control_step: bool = False
    state_derived: StateDerivedTargetConfig | None = None
    evaluation_id: str = dataclasses.field(default_factory=lambda: f"open-loop-{uuid.uuid4().hex[:12]}")
    resume: bool = False
    limit: int | None = None
    top_error_count: int = 20

    def __post_init__(self) -> None:
        if not str(self.dataset):
            raise ValueError("dataset cannot be empty")
        if self.episode_indices is not None and (not self.episode_indices or min(self.episode_indices) < 0):
            raise ValueError("episode_indices must contain non-negative indexes")
        if not self.policy_url.strip() or not self.policy_label.strip():
            raise ValueError("policy_url and policy_label cannot be empty")
        if self.max_timestamp_error_s < 0:
            raise ValueError("max_timestamp_error_s must be non-negative")
        if self.connection_timeout_s <= 0 or self.metadata_timeout_s <= 0:
            raise ValueError("connection and metadata timeouts must be positive")
        if self.resize_size <= 0 or self.replan_steps <= 0:
            raise ValueError("resize_size and replan_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when supplied")
        if self.top_error_count <= 0:
            raise ValueError("top_error_count must be positive")
        if self.target_source is PredictionResultSource.STATE_DERIVED and self.state_derived is None:
            raise ValueError("state_derived target source requires state_derived configuration")
        if self.target_source is not PredictionResultSource.STATE_DERIVED and self.state_derived is not None:
            raise ValueError("state_derived configuration is only valid with target_source=state_derived")


@dataclasses.dataclass(frozen=True)
class AlignedAction:
    mode: AlignmentMode
    source_sample_index: int
    horizon_index: int
    target_sample_index: int | None
    target_timestamp: float | None
    alignment_error_s: float | None
    valid: bool
    reason: str | None = None
    control_step: int | None = None
    chunk_cursor: int | None = None


@dataclasses.dataclass(frozen=True)
class OpenLoopMetrics:
    """JSON-ready metric payload, kept separate from prediction artifacts."""

    values: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class OpenLoopReport:
    evaluation_id: str
    episode_index: int
    status: str
    teacher_forced: bool
    target_source: PredictionResultSource
    alignment_mode: AlignmentMode
    metrics: OpenLoopMetrics
    completed_samples: int
    valid_prediction_count: int
    errors: tuple[Mapping[str, Any], ...] = ()
    incomplete_observation: bool = False


def config_json(config: OpenLoopEvaluationConfig) -> dict[str, Any]:
    """Make configuration serializable without exposing the API key."""

    result = dataclasses.asdict(config)
    result["dataset"] = str(config.dataset)
    result["output_dir"] = str(config.output_dir)
    result["policy_api_key"] = None
    return result
