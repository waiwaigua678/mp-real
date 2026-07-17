"""Safe, policy-free real-robot trajectory replay.

The package deliberately owns no vendor SDK calls.  It consumes the existing
``Robot`` and optional ``PoseControlCapability`` boundaries, so Piper and RM2
remain the only modules that talk to their SDKs.
"""

from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import (
    ReplayPlan,
    ReplaySafetyReport,
    ReplayState,
    ReplayTimingMode,
    RobotReplayCursor,
)
from mp_real.replay.planning import ReplayPlanner

__all__ = [
    "ReplayPlan",
    "ReplayPlanner",
    "ReplaySafetyReport",
    "ReplayState",
    "ReplayTimingMode",
    "RobotReplayController",
    "RobotReplayCursor",
]
