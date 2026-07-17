from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mp_real.runtime.models import ActionSpec


class EpisodeStatus(enum.StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CORRUPTED = "corrupted"


@dataclasses.dataclass(frozen=True)
class RecorderConfig:
    """Static recording configuration for one LeRobot dataset/session."""

    dataset_root: Path
    dataset_name: str
    robot_name: str
    fps: float
    action_spec: ActionSpec
    save_video: bool = True
    queue_size: int = 128
    max_camera_age_ns: int = 500_000_000
    session_id: str | None = None
    operator: str | None = None
    policy_label: str | None = None
    runtime_config: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if self.max_camera_age_ns < 0:
            raise ValueError("max_camera_age_ns must be non-negative")


@dataclasses.dataclass(frozen=True)
class EpisodeRecordingContext:
    episode_index: int
    episode_id: str
    task: str
    session_id: str | None = None
    generation_id: int | None = None

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if not self.task:
            raise ValueError("task cannot be empty")


@dataclasses.dataclass(frozen=True)
class DatasetMetadata:
    root: Path
    info: Mapping[str, Any]
    status: EpisodeStatus
    is_mp_real: bool
    action_spec: ActionSpec
    camera_roles: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class EpisodeMetadata:
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    status: EpisodeStatus
    labels: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class RecordedSample:
    episode_index: int
    frame_index: int
    index: int
    timestamp: float
    task_index: int
    state: np.ndarray
    action: np.ndarray
    images: Mapping[str, np.ndarray | None]
    telemetry: Mapping[str, Any]


class RecordedEpisodeSource(Protocol):
    """Read-only source consumed by future playback and evaluation features."""

    def list_episodes(self) -> tuple[EpisodeMetadata, ...]: ...

    def get_dataset_metadata(self) -> DatasetMetadata: ...

    def get_episode_metadata(self, episode_index: int) -> EpisodeMetadata: ...

    def get_action_spec(self) -> ActionSpec: ...

    def get_state_schema(self) -> tuple[str, ...]: ...

    def get_camera_roles(self) -> tuple[str, ...]: ...

    def get_length(self, episode_index: int) -> int: ...

    def get_task(self, episode_index: int, task_index: int) -> str: ...

    def get_pose_state_sample(self, episode_index: int, index: int) -> tuple[np.ndarray, float]: ...

    def get_sample(self, episode_index: int, index: int) -> RecordedSample: ...

    def get_sample_at_timestamp(self, episode_index: int, timestamp: float) -> RecordedSample: ...

    def iter_samples(self, episode_index: int) -> Iterator[RecordedSample]: ...

    def close(self) -> None: ...


class FakeRecordedEpisodeSource:
    """In-memory fake for hardware-free playback and data-layer tests."""

    def __init__(
        self,
        action_spec: ActionSpec,
        episodes: Mapping[int, Sequence[RecordedSample]],
        *,
        robot_name: str = "fake",
        info: Mapping[str, Any] | None = None,
    ) -> None:
        self._action_spec = action_spec
        self._episodes = {index: tuple(samples) for index, samples in episodes.items()}
        self._metadata = DatasetMetadata(
            root=Path("<fake>"),
            info={
                "codebase_version": "v2.1",
                "robot_type": robot_name,
                "fps": 1.0,
                "mp_real": {"replay": {"action_source": "standard_action", "action_mode": "joint_position_target"}},
                **dict(info or {}),
            },
            status=EpisodeStatus.COMPLETE,
            is_mp_real=True,
            action_spec=action_spec,
            camera_roles=action_spec.camera_roles,
        )

    def list_episodes(self) -> tuple[EpisodeMetadata, ...]:
        return tuple(
            EpisodeMetadata(index, len(samples), (), EpisodeStatus.COMPLETE)
            for index, samples in sorted(self._episodes.items())
        )

    def get_dataset_metadata(self) -> DatasetMetadata:
        return self._metadata

    def get_episode_metadata(self, episode_index: int) -> EpisodeMetadata:
        return EpisodeMetadata(episode_index, self.get_length(episode_index), (), EpisodeStatus.COMPLETE)

    def get_action_spec(self) -> ActionSpec:
        return self._action_spec

    def get_state_schema(self) -> tuple[str, ...]:
        return self._action_spec.state_field_names

    def get_camera_roles(self) -> tuple[str, ...]:
        return self._action_spec.camera_roles

    def get_length(self, episode_index: int) -> int:
        return len(self._episodes[episode_index])

    def get_task(self, episode_index: int, task_index: int) -> str:
        metadata = self.get_episode_metadata(episode_index)
        if len(metadata.tasks) != 1:
            raise ValueError(
                f"fake episode {episode_index} does not have one unambiguous task; use an explicit prompt override"
            )
        del task_index
        return metadata.tasks[0]

    def get_pose_state_sample(self, episode_index: int, index: int) -> tuple[np.ndarray, float]:
        sample = self._episodes[episode_index][index]
        return np.asarray(sample.state, dtype=np.float32).copy(), float(sample.timestamp)

    def get_sample(self, episode_index: int, index: int) -> RecordedSample:
        return self._episodes[episode_index][index]

    def get_sample_at_timestamp(self, episode_index: int, timestamp: float) -> RecordedSample:
        samples = self._episodes[episode_index]
        return min(samples, key=lambda sample: abs(sample.timestamp - timestamp))

    def iter_samples(self, episode_index: int) -> Iterator[RecordedSample]:
        return iter(self._episodes[episode_index])

    def close(self) -> None:
        return None
