from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMoveAborted,
    PoseMoveProgress,
    PoseMoveResult,
    PosePlanStaleError,
)
from mp_real.robots.pose import PoseControlCapability


class PoseMoveController:
    """Own one bounded, joinable recorded-pose motion lifecycle.

    It intentionally delegates all SDK calls to a capability.  The controller
    adds stale-plan checks and provides a common worker lifecycle for Web/CLI.
    """

    def __init__(self, capability: PoseControlCapability, *, thread_name: str = "pose-move-controller") -> None:
        self._capability = capability
        self._thread_name = thread_name
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._result: PoseMoveResult | None = None
        self._error: BaseException | None = None
        self._plan: MoveToRecordedStatePlan | None = None

    def revalidate(self, plan: MoveToRecordedStatePlan) -> None:
        current = self._capability.get_current_pose_state()
        values = np.asarray(current.values, dtype=np.float32)
        expected = np.asarray(plan.current_state.values, dtype=np.float32)
        if values.shape != expected.shape:
            raise PosePlanStaleError("robot state dimension changed after plan generation")
        drift = float(np.max(np.abs(values - expected))) if len(values) else 0.0
        if drift > plan.constraints.max_joint_step:
            raise PosePlanStaleError(
                "robot state changed by "
                f"{drift:.6f}, exceeding revalidation threshold "
                f"{plan.constraints.max_joint_step:.6f}"
            )

    def start(
        self,
        plan: MoveToRecordedStatePlan,
        *,
        on_progress: Callable[[PoseMoveProgress], None] | None = None,
    ) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("pose move is already running")
            self.revalidate(plan)
            self._plan = plan
            self._result = None
            self._error = None
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                args=(plan, on_progress),
                name=f"{self._thread_name}-{plan.plan_id[:8]}",
                daemon=False,
            )
            self._thread.start()

    def stop(self, *, wait: bool = False, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        try:
            self._capability.stop_pose_motion()
        except BaseException as exc:
            with self._lock:
                if self._error is None:
                    self._error = exc
        if wait:
            return self.join(timeout=timeout)
        return thread is None or not thread.is_alive()

    def join(self, *, timeout: float | None = None, raise_on_error: bool = False) -> bool:
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        finished = thread is None or not thread.is_alive()
        if finished and raise_on_error and self._error is not None:
            raise self._error
        return finished

    def result(self) -> PoseMoveResult | None:
        with self._lock:
            return self._result

    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def _run(self, plan: MoveToRecordedStatePlan, on_progress: Callable[[PoseMoveProgress], None] | None) -> None:
        try:
            result = self._capability.execute_pose_plan(plan, stop_event=self._stop_event, on_progress=on_progress)
            if self._stop_event.is_set() and result.status not in {"aborted", "failed"}:
                raise PoseMoveAborted("pose move stopped before verification")
            if result.status != "reached":
                raise PoseMoveAborted(result.message or f"pose move ended with {result.status}")
            deadline = time.monotonic() + plan.constraints.verify_timeout_s
            while True:
                verified = self._capability.verify_target_reached(plan)
                if verified.status in {"reached", "reached_with_warning"}:
                    break
                if self._stop_event.is_set():
                    raise PoseMoveAborted("pose move stopped during final verification")
                if time.monotonic() >= deadline:
                    raise PoseMoveAborted(verified.message or "target verification timed out")
                self._stop_event.wait(min(plan.constraints.control_period_s, 0.05))
            with self._lock:
                self._result = verified
        except BaseException as exc:
            try:
                self._capability.stop_pose_motion()
            except BaseException:
                pass
            with self._lock:
                self._error = exc
