from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMoveProgress,
    PoseMoveResult,
    PoseValidationReport,
    RecordedPoseTarget,
)
from mp_real.runtime.models import RobotState


@runtime_checkable
class PoseControlCapability(Protocol):
    """Optional high-risk position-control capability for one concrete robot.

    ``Robot`` remains intentionally small; callers must explicitly discover
    this capability before planning or executing a recorded-state move.
    """

    def get_current_pose_state(self) -> RobotState: ...

    def validate_pose_target(self, target: RecordedPoseTarget) -> PoseValidationReport: ...

    def plan_move_to_state(self, plan: MoveToRecordedStatePlan) -> MoveToRecordedStatePlan: ...

    def execute_pose_plan(
        self,
        plan: MoveToRecordedStatePlan,
        *,
        stop_event: object,
        on_progress: Callable[[PoseMoveProgress], None] | None = None,
    ) -> PoseMoveResult: ...

    def stop_pose_motion(self) -> None: ...

    def verify_target_reached(self, plan: MoveToRecordedStatePlan) -> PoseMoveResult: ...
