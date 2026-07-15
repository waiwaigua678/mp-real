from __future__ import annotations

import copy
import dataclasses
import enum
import uuid
from collections.abc import Mapping
from typing import Any


class EvaluationState(enum.StrEnum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    WARMING_UP = "WARMING_UP"
    WAITING_RESET = "WAITING_RESET"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    WAITING_RESULT = "WAITING_RESULT"
    SAVING = "SAVING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


class EvaluationResult(enum.StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INVALID = "INVALID"
    TIMEOUT = "TIMEOUT"
    SAFETY_ABORT = "SAFETY_ABORT"
    OPERATOR_ABORT = "OPERATOR_ABORT"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class FailureReason(enum.StrEnum):
    NOT_REACHED = "NOT_REACHED"
    GRASP_FAILED = "GRASP_FAILED"
    LIFT_FAILED = "LIFT_FAILED"
    DROPPED_DURING_TRANSFER = "DROPPED_DURING_TRANSFER"
    PLACE_FAILED = "PLACE_FAILED"
    GRIPPER_ERROR = "GRIPPER_ERROR"
    VISION_ERROR = "VISION_ERROR"
    ACTION_ERROR = "ACTION_ERROR"
    POLICY_ERROR = "POLICY_ERROR"
    SAFETY_STOP = "SAFETY_STOP"
    OTHER = "OTHER"


def _copied_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Frozen configuration attached to a single in-process evaluation."""

    evaluation_id: str
    name: str
    robot_name: str
    task_name: str
    prompt: str
    planned_episodes: int
    max_episode_seconds: float
    auto_advance: bool
    reset_mode: str
    result_mode: str
    save_data: bool
    save_video: bool
    policy_label: str | None
    operator: str | None
    tags: tuple[str, ...]
    notes: str
    storage_root: str | None
    runtime_config_snapshot: Mapping[str, Any]
    action_spec_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.evaluation_id:
            raise ValueError("evaluation_id cannot be empty")
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if not self.robot_name.strip():
            raise ValueError("robot_name cannot be empty")
        if not self.task_name.strip():
            raise ValueError("task_name cannot be empty")
        if self.planned_episodes <= 0:
            raise ValueError("planned_episodes must be positive")
        if self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        if self.reset_mode != "manual":
            raise ValueError("reset_mode currently supports only 'manual'")
        if self.result_mode != "manual":
            raise ValueError("result_mode currently supports only 'manual'")
        object.__setattr__(self, "runtime_config_snapshot", _copied_mapping(self.runtime_config_snapshot))
        object.__setattr__(self, "action_spec_snapshot", _copied_mapping(self.action_spec_snapshot))

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        runtime_config_snapshot: Mapping[str, Any],
        action_spec_snapshot: Mapping[str, Any],
        default_robot_name: str,
    ) -> EvaluationConfig:
        evaluation_id = str(payload.get("evaluation_id") or uuid.uuid4().hex)
        tags_value = payload.get("tags", ())
        if isinstance(tags_value, str):
            tags = tuple(tag.strip() for tag in tags_value.split(",") if tag.strip())
        elif isinstance(tags_value, list | tuple):
            tags = tuple(str(tag) for tag in tags_value)
        else:
            raise ValueError("tags must be a list or comma-separated string")

        def optional_text(name: str) -> str | None:
            value = payload.get(name)
            if value is None or value == "":
                return None
            return str(value)

        return cls(
            evaluation_id=evaluation_id,
            name=str(payload.get("name", "evaluation")),
            robot_name=str(payload.get("robot_name", default_robot_name)),
            task_name=str(payload.get("task_name", "unnamed-task")),
            prompt=str(payload.get("prompt", runtime_config_snapshot.get("prompt", ""))),
            planned_episodes=int(payload.get("planned_episodes", 1)),
            max_episode_seconds=float(payload.get("max_episode_seconds", 60.0)),
            auto_advance=bool(payload.get("auto_advance", False)),
            reset_mode=str(payload.get("reset_mode", "manual")),
            result_mode=str(payload.get("result_mode", "manual")),
            save_data=bool(payload.get("save_data", False)),
            save_video=bool(payload.get("save_video", False)),
            policy_label=optional_text("policy_label"),
            operator=optional_text("operator"),
            tags=tags,
            notes=str(payload.get("notes", "")),
            storage_root=optional_text("storage_root"),
            runtime_config_snapshot=runtime_config_snapshot,
            action_spec_snapshot=action_spec_snapshot,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["tags"] = list(self.tags)
        return result


@dataclasses.dataclass
class EpisodeRecord:
    episode_id: str
    episode_index: int
    generation_id: int | None = None
    started_at_monotonic_ns: int | None = None
    stopped_at_monotonic_ns: int | None = None
    result: EvaluationResult | None = None
    failure_reason: FailureReason | None = None
    notes: str = ""
    stop_trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "generation_id": self.generation_id,
            "started_at_monotonic_ns": self.started_at_monotonic_ns,
            "stopped_at_monotonic_ns": self.stopped_at_monotonic_ns,
            "result": self.result.value if self.result is not None else None,
            "failure_reason": self.failure_reason.value if self.failure_reason is not None else None,
            "notes": self.notes,
            "stop_trigger": self.stop_trigger,
        }
