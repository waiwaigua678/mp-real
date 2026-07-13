from __future__ import annotations

from typing import Protocol

import numpy as np

from mp_real.runtime.models import ActionSpec, RobotState


class Robot(Protocol):
    """Hardware boundary used by robot-independent inference runtime."""

    action_spec: ActionSpec

    def read_state(self) -> RobotState: ...

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
