from __future__ import annotations

import dataclasses
import time
from typing import Any, Mapping

import numpy as np


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


@dataclasses.dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    timestamp_monotonic: float
    camera_timestamp: float | None = None
    info: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class ObservationSnapshot:
    """An observation with timestamps retained outside the policy wire schema."""

    images: Mapping[str, CameraSample]
    image_masks: Mapping[str, np.bool_]
    state: RobotState
    prompt: str
    camera_params: Mapping[str, Mapping[str, Any] | None] | None = None
    captured_at_monotonic: float = dataclasses.field(default_factory=time.monotonic)

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
