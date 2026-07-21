"""Joinable, policy-free execution lifecycle for a validated replay plan."""

from __future__ import annotations

import dataclasses
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mp_real.common.plan_integrity import PlanIntegrityError
from mp_real.pose.controller import PoseMoveController
from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMotionConstraints,
    RecordedPoseTarget,
)
from mp_real.pose.validation import MoveToStateValidator
from mp_real.replay.models import (
    ReplayPlan,
    ReplayPlanStaleError,
    ReplayState,
    RobotReplayCursor,
)
from mp_real.robots.base import Robot
from mp_real.robots.pose import PoseControlCapability
from mp_real.runtime.models import RobotState

ReplayRecordCallback = Callable[[Mapping[str, Any]], None]


class RobotReplayController:
    """Own one explicit, bounded replay worker lifecycle.

    The controller has no policy, camera, or vendor-SDK dependency.  A caller
    creates the robot only after a validated offline plan, then gives the
    controller a lease-valid predicate that rejects stale Web/CLI lifecycles.
    """

    def __init__(
        self,
        robot: Robot,
        plan: ReplayPlan,
        *,
        lease_valid: Callable[[], bool] | None = None,
        record_callback: ReplayRecordCallback | None = None,
        thread_name: str = "robot-replay-controller",
    ) -> None:
        plan.require_integrity()
        self._robot = robot
        self._plan = plan
        self._lease_valid = lease_valid or (lambda: True)
        self._record_callback = record_callback
        self._thread_name = thread_name
        self._lock = threading.RLock()
        self._pause_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pose_controller: PoseMoveController | None = None
        self._cursor = self._cursor_for(ReplayState.IDLE)
        self._error: BaseException | None = None
        self._last_expected_state: np.ndarray | None = None
        self._pause_started_ns: int | None = None
        self._schedule_shift_ns = 0

    @property
    def plan(self) -> ReplayPlan:
        return self._plan

    def cursor(self) -> RobotReplayCursor:
        with self._lock:
            return dataclasses.replace(self._cursor)

    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def prepare(self) -> None:
        """Start a background connect/revalidate/move-to-start lifecycle."""
        with self._lock:
            self._require_state(ReplayState.IDLE)
            self._require_plan_integrity(check_expiration=True)
            self._set_cursor_locked(ReplayState.CONNECTING)
            self._thread = threading.Thread(
                target=self._run_prepare,
                name=f"{self._thread_name}-prepare-{self._plan.plan_id[:8]}",
                daemon=False,
            )
            self._thread.start()

    def confirm_and_start(self, plan_hash: str) -> None:
        """Run only after a fresh user confirmation of the immutable plan."""
        with self._lock:
            self._require_state(ReplayState.ARMED)
            self._require_plan_integrity(check_expiration=True)
            self._require_action_spec()
            if not secrets.compare_digest(self._plan.plan_hash, str(plan_hash)):
                raise ReplayPlanStaleError("confirmation plan hash does not match the armed plan")
            self._require_live_identity()
            self._set_cursor_locked(ReplayState.RUNNING)
            self._thread = threading.Thread(
                target=self._run_replay,
                name=f"{self._thread_name}-run-{self._plan.plan_id[:8]}",
                daemon=False,
            )
            self._thread.start()

    def pause(self) -> bool:
        with self._pause_condition:
            if self._cursor.state is not ReplayState.RUNNING:
                return False
            self._pause_started_ns = time.monotonic_ns()
            self._set_cursor_locked(ReplayState.PAUSED, message="operator pause")
        self._safe_stop_motion()
        self._record_event("pause")
        return True

    def resume(self) -> bool:
        with self._pause_condition:
            if self._cursor.state is not ReplayState.PAUSED:
                return False
            expected = self._last_expected_state
            if expected is None:
                self._set_cursor_locked(ReplayState.ERROR, message="pause has no expected state")
                return False
            self._require_plan_integrity(check_expiration=True)
            self._require_action_spec()
        actual = self._robot.read_state()
        error = self._tracking_error(actual, expected)
        if error > self._plan.constraints.tracking_tolerance:
            with self._pause_condition:
                self._set_cursor_locked(
                    ReplayState.PAUSED,
                    tracking_error=error,
                    message="resume rejected: move to the recorded pause state first",
                )
            return False
        with self._pause_condition:
            if self._pause_started_ns is not None:
                self._schedule_shift_ns += time.monotonic_ns() - self._pause_started_ns
            self._pause_started_ns = None
            self._set_cursor_locked(ReplayState.RUNNING, tracking_error=error, message="resumed")
            self._pause_condition.notify_all()
        self._record_event("resume", tracking_error=error)
        return True

    def stop(self, *, emergency: bool = False, wait: bool = False, timeout: float | None = None) -> bool:
        """Request an operator or emergency stop and optionally join safely."""
        with self._pause_condition:
            if self._cursor.state in {ReplayState.COMPLETED, ReplayState.ABORTED, ReplayState.ERROR, ReplayState.IDLE}:
                return self.join(timeout=timeout)
            self._stop_event.set()
            self._set_cursor_locked(ReplayState.STOPPING, message="emergency stop" if emergency else "operator stop")
            self._pause_condition.notify_all()
            pose_controller = self._pose_controller
        if pose_controller is not None:
            pose_controller.stop(wait=False)
        self._safe_stop_motion()
        self._record_event("emergency_stop" if emergency else "stop")
        return self.join(timeout=timeout) if wait else False

    def join(self, *, timeout: float | None = None, raise_on_error: bool = False) -> bool:
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        complete = thread is None or not thread.is_alive()
        if complete and raise_on_error and self.error() is not None:
            raise self.error()  # type: ignore[misc]
        return complete

    def _run_prepare(self) -> None:
        try:
            self._require_plan_integrity(check_expiration=True)
            self._require_live_identity()
            if not isinstance(self._robot, PoseControlCapability):
                raise RuntimeError("robot lacks required PoseControlCapability")
            self._require_action_spec()
            with self._lock:
                self._set_cursor_locked(ReplayState.MOVING_TO_START)
            self._move_to_start(self._robot)
            self._require_live_identity()
            current = self._robot.get_current_pose_state()
            error = self._tracking_error(current, self._plan.start_state)
            if error > self._plan.constraints.tracking_tolerance:
                raise RuntimeError(f"move-to-start tracking error {error:.6f} exceeds tolerance")
            with self._lock:
                self._last_expected_state = self._plan.start_state
                self._set_cursor_locked(ReplayState.ARMED, tracking_error=error, message="move-to-start verified")
        except BaseException as exc:
            self._finish_error(_ReplayStopped() if self._stop_event.is_set() else exc)

    def _move_to_start(self, capability: PoseControlCapability) -> None:
        self._require_plan_integrity(check_expiration=True)
        current = capability.get_current_pose_state()
        target = RecordedPoseTarget(
            dataset_id=self._plan.dataset_id,
            episode_index=self._plan.episode_index,
            sample_index=self._plan.start_sample,
            robot_name=self._plan.robot_name,
            state_schema=self._plan.action_spec.state_field_names,
            state_values=self._plan.start_state,
            state_fields=self._plan.action_spec.state_fields,
            joint_unit=self._plan.action_spec.joint_unit,
            timestamp=self._plan.steps[0].source_timestamp_s,
            source_metadata={"dataset_status": "complete", "replay_plan_hash": self._plan.plan_hash},
            action_spec=self._plan.action_spec,
        )
        validated = MoveToStateValidator(self._plan.robot_name, capability.action_spec).validate(target)
        validated.report.require_valid()
        capability.validate_pose_target(target).require_valid()
        pose_constraints = self._plan.constraints.move_to_start_constraints or PoseMotionConstraints()
        move_plan = MoveToRecordedStatePlan.build(
            target=target,
            current_state=current,
            target_state=validated.values,
            gripper_indices=validated.gripper_indices,
            mapped_joint_names=validated.field_names,
            conversions=validated.mappings,
            constraints=pose_constraints,
            safety_warnings=("trajectory replay move-to-start",),
            required_confirmations=("move_to_replay_start",),
            session_id=self._plan.session_id,
            generation_id=self._plan.generation_id,
        )
        move_plan = capability.plan_move_to_state(move_plan)
        controller = PoseMoveController(capability, thread_name=f"{self._thread_name}-move-to-start")
        with self._lock:
            self._pose_controller = controller
        controller.start(move_plan)
        while not controller.join(timeout=0.1):
            self._require_live_identity()
            if self._stop_event.is_set():
                controller.stop(wait=True, timeout=2.0)
                raise RuntimeError("move-to-start stopped")
        with self._lock:
            self._pose_controller = None
        if controller.error() is not None:
            raise controller.error()  # type: ignore[misc]
        result = controller.result()
        if result is None or result.status not in {"reached", "reached_with_warning"}:
            raise RuntimeError(
                result.message if result is not None and result.message else "move-to-start was not verified"
            )

    def _run_replay(self) -> None:
        started_ns = time.monotonic_ns()
        previous_action: np.ndarray | None = None
        try:
            self._require_plan_integrity(check_expiration=True)
            self._require_action_spec()
            for step_number, step in enumerate(self._plan.steps):
                self._wait_until(started_ns + step.target_offset_ns)
                self._require_live_identity()
                if self._stop_event.is_set():
                    raise _ReplayStopped()
                cycle_started_ns = time.monotonic_ns()
                target = step.target.copy()
                if not np.isfinite(target).all():
                    raise RuntimeError("planned target became non-finite")
                executed = self._robot.execute_transition(previous_action, target)
                previous_action = np.asarray(executed, dtype=np.float32).copy()
                actual = self._robot.read_state()
                self._check_health(actual)
                tracking_error = self._tracking_error(actual, step.expected_state)
                if tracking_error > self._plan.constraints.max_tracking_error:
                    raise RuntimeError(f"tracking error {tracking_error:.6f} exceeds replay limit")
                elapsed_ns = time.monotonic_ns() - cycle_started_ns
                if elapsed_ns > int(self._plan.constraints.max_control_overrun_s * 1e9):
                    raise RuntimeError("replay control cycle overrun")
                self._last_expected_state = step.expected_state.copy()
                self._record_step(step, previous_action, actual, tracking_error)
                with self._lock:
                    self._set_cursor_locked(
                        ReplayState.RUNNING,
                        source_sample_index=step.source_sample_index,
                        sent_sample_index=step.source_sample_index,
                        acknowledged_sample_index=step.source_sample_index,
                        progress_ratio=(step_number + 1) / len(self._plan.steps),
                        elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
                        tracking_error=tracking_error,
                    )
            with self._lock:
                self._set_cursor_locked(ReplayState.COMPLETED, progress_ratio=1.0, message="replay completed")
        except _ReplayStopped:
            with self._lock:
                self._set_cursor_locked(ReplayState.ABORTED, message="replay stopped")
        except BaseException as exc:
            self._finish_error(exc, abort=True)

    def _wait_until(self, target_ns: int) -> None:
        while True:
            if self._stop_event.is_set():
                raise _ReplayStopped()
            with self._pause_condition:
                while self._cursor.state is ReplayState.PAUSED and not self._stop_event.is_set():
                    self._pause_condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    raise _ReplayStopped()
                shifted_target_ns = target_ns + self._schedule_shift_ns
            remaining_ns = shifted_target_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            self._stop_event.wait(min(remaining_ns / 1e9, 0.05))

    def _record_step(self, step: Any, sent: np.ndarray, actual: RobotState, tracking_error: float) -> None:
        if self._record_callback is None:
            return
        self._record_callback(
            {
                "type": "replay_step",
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "source_sample_index": step.source_sample_index,
                "sent_action": sent.copy(),
                "actual_state": actual.values.copy(),
                "tracking_error": tracking_error,
                "plan_hash": self._plan.plan_hash,
            }
        )

    def _record_event(self, event_type: str, **payload: Any) -> None:
        if self._record_callback is None:
            return
        event = {
            "type": event_type,
            "monotonic_timestamp_ns": time.monotonic_ns(),
            "plan_hash": self._plan.plan_hash,
            **payload,
        }
        self._record_callback(event)

    def _require_live_identity(self) -> None:
        if self._stop_event.is_set():
            raise _ReplayStopped()
        if not self._lease_valid():
            raise ReplayPlanStaleError("replay resource lease or generation is stale")

    def _require_plan_integrity(self, *, check_expiration: bool = False) -> None:
        self._plan.require_integrity(check_expiration=check_expiration)

    def _require_action_spec(self) -> None:
        if self._robot.action_spec != self._plan.action_spec:
            raise ReplayPlanStaleError("connected robot ActionSpec does not match the reviewed replay plan")

    def _check_health(self, state: RobotState) -> None:
        health = state.health
        if not isinstance(health, Mapping):
            return
        if health.get("ok") is False or health.get("error"):
            raise RuntimeError(f"robot health check failed: {dict(health)}")

    @staticmethod
    def _tracking_error(actual: RobotState, expected: np.ndarray) -> float:
        values = np.asarray(actual.values, dtype=np.float32)
        target = np.asarray(expected, dtype=np.float32)
        if values.shape != target.shape:
            raise RuntimeError("robot state dimension changed during replay")
        if not np.isfinite(values).all():
            raise RuntimeError("robot returned non-finite state")
        return float(np.max(np.abs(values - target))) if len(values) else 0.0

    def _safe_stop_motion(self) -> None:
        stop = getattr(self._robot, "stop_pose_motion", None)
        if not callable(stop):
            return
        try:
            stop()
        except BaseException as exc:
            with self._lock:
                if self._error is None:
                    self._error = exc

    def _finish_error(self, error: BaseException, *, abort: bool = False) -> None:
        self._safe_stop_motion()
        if isinstance(error, PlanIntegrityError):
            self._record_event("plan_integrity_error", error=str(error))
        with self._lock:
            self._error = error
            state = ReplayState.ABORTED if abort or isinstance(error, _ReplayStopped) else ReplayState.ERROR
            self._set_cursor_locked(state, message=f"{type(error).__name__}: {error}")

    def _require_state(self, state: ReplayState) -> None:
        if self._cursor.state is not state:
            raise RuntimeError(f"replay state is {self._cursor.state.value}, expected {state.value}")

    def _cursor_for(self, state: ReplayState) -> RobotReplayCursor:
        return RobotReplayCursor(
            state=state,
            session_id=self._plan.session_id,
            generation_id=self._plan.generation_id,
            plan_hash=self._plan.plan_hash,
        )

    def _set_cursor_locked(self, state: ReplayState, **changes: Any) -> None:
        self._cursor = dataclasses.replace(
            self._cursor_for(state),
            source_sample_index=changes.get("source_sample_index", self._cursor.source_sample_index),
            sent_sample_index=changes.get("sent_sample_index", self._cursor.sent_sample_index),
            acknowledged_sample_index=changes.get("acknowledged_sample_index", self._cursor.acknowledged_sample_index),
            progress_ratio=changes.get("progress_ratio", self._cursor.progress_ratio),
            elapsed_s=changes.get("elapsed_s", self._cursor.elapsed_s),
            tracking_error=changes.get("tracking_error", self._cursor.tracking_error),
            message=changes.get("message", self._cursor.message),
            timestamp_monotonic_ns=time.monotonic_ns(),
        )


class _ReplayStopped(RuntimeError):
    pass
