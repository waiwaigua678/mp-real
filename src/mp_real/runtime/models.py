from __future__ import annotations

import dataclasses
import itertools
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

_observation_ids = itertools.count(1)
_observation_id_lock = threading.Lock()


def _next_observation_id() -> int:
    with _observation_id_lock:
        return next(_observation_ids)


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    """Policy-facing action/state contract for a concrete robot."""

    action_dim: int
    state_dim: int
    joint_dof_per_arm: int
    joint_unit: str
    camera_roles: tuple[str, ...]
    supports_rtc: bool = True
    supports_interpolation: bool = True

    def validate_chunk(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise RuntimeError(f"Expected action chunk [T, >= {self.action_dim}], got {actions.shape}")
        return actions[:, : self.action_dim].copy()


@dataclasses.dataclass(frozen=True)
class RobotState:
    values: np.ndarray
    timestamp_monotonic: float
    timestamp_monotonic_ns: int = 0
    source_timestamp_ns: int | None = None
    health: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Keep the legacy float timestamp while making nanoseconds canonical."""
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", int(self.timestamp_monotonic * 1e9))


@dataclasses.dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    timestamp_monotonic: float
    camera_timestamp: float | None = None
    info: Mapping[str, Any] | None = None
    frame_id: int = 0
    timestamp_monotonic_ns: int = 0
    source_sequence: int | None = None
    capture_latency_ns: int | None = None

    def __post_init__(self) -> None:
        """Older camera implementations can still provide only the float timestamp."""
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", int(self.timestamp_monotonic * 1e9))


@dataclasses.dataclass(frozen=True)
class ObservationSnapshot:
    """An observation with timestamps retained outside the policy wire schema."""

    images: Mapping[str, CameraSample]
    image_masks: Mapping[str, np.bool_]
    state: RobotState
    prompt: str
    camera_params: Mapping[str, Mapping[str, Any] | None] | None = None
    captured_at_monotonic: float = dataclasses.field(default_factory=time.monotonic)
    observation_id: int = 0
    capture_started_ns: int = 0
    capture_finished_ns: int = 0
    state_timestamp_ns: int = 0
    camera_frame_ids: Mapping[str, int] = dataclasses.field(default_factory=dict)
    camera_timestamps_ns: Mapping[str, int] = dataclasses.field(default_factory=dict)
    max_camera_skew_ns: int = 0
    observation_age_ns: int = 0

    def __post_init__(self) -> None:
        """Fill derived timing metadata without changing the policy wire schema.

        ``max_camera_skew_ns`` is the range of camera timestamps in this
        snapshot. ``observation_age_ns`` is measured at capture completion
        against the oldest included state or camera source timestamp.
        """
        if self.observation_id <= 0:
            object.__setattr__(self, "observation_id", _next_observation_id())

        finished_ns = self.capture_finished_ns or int(self.captured_at_monotonic * 1e9)
        if finished_ns <= 0:
            finished_ns = time.monotonic_ns()
        started_ns = self.capture_started_ns or finished_ns
        state_timestamp_ns = self.state_timestamp_ns or self.state.timestamp_monotonic_ns
        camera_frame_ids = self.camera_frame_ids or {name: sample.frame_id for name, sample in self.images.items()}
        camera_timestamps_ns = self.camera_timestamps_ns or {
            name: sample.timestamp_monotonic_ns for name, sample in self.images.items()
        }
        timestamps = tuple(camera_timestamps_ns.values())
        max_camera_skew_ns = self.max_camera_skew_ns
        if max_camera_skew_ns == 0 and len(timestamps) > 1:
            max_camera_skew_ns = max(timestamps) - min(timestamps)
        source_timestamps = (state_timestamp_ns, *timestamps)
        observation_age_ns = self.observation_age_ns
        if observation_age_ns == 0 and source_timestamps:
            observation_age_ns = max(0, finished_ns - min(source_timestamps))

        object.__setattr__(self, "capture_started_ns", started_ns)
        object.__setattr__(self, "capture_finished_ns", finished_ns)
        object.__setattr__(self, "state_timestamp_ns", state_timestamp_ns)
        object.__setattr__(self, "camera_frame_ids", dict(camera_frame_ids))
        object.__setattr__(self, "camera_timestamps_ns", dict(camera_timestamps_ns))
        object.__setattr__(self, "max_camera_skew_ns", max_camera_skew_ns)
        object.__setattr__(self, "observation_age_ns", observation_age_ns)

    def to_policy_observation(self) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "images": {name: sample.image for name, sample in self.images.items()},
            "image_masks": dict(self.image_masks),
            "state": self.state.values,
            "prompt": self.prompt,
        }
        if self.camera_params is not None:
            observation["camera_params"] = dict(self.camera_params)
        return observation
