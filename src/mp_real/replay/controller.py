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
    ReplayAcknowledgementStrategy,
    ReplayCommandRecord,
    ReplayFeedbackRecord,
    ReplayPlan,
    ReplayPlanStaleError,
    ReplayState,
    RobotReplayCursor,
)
from mp_real.robots.base import Robot
from mp_real.robots.pose import PoseControlCapability
from mp_real.runtime.models import RobotState
from mp_real.safety.models import health_from_state_mapping, profile_for_robot

ReplayRecordCallback = Callable[[Mapping[str, Any]], None]


@dataclasses.dataclass(frozen=True)
class _FeedbackEvaluation:
    actual: RobotState
    record: ReplayFeedbackRecord
    matched_command: ReplayCommandRecord | None
    matched_step_number: int | None
    within_threshold: bool


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
        self._last_acknowledged_step_number = -1
        self._last_sent_step_number = -1
        self._last_feedback_step_number = -1
        self._max_tracking_error_seen: float | None = None
        self._sustained_tracking_error_count = 0
        self._settle_counts: dict[str, int] = {}
        self._sent_actions: dict[str, np.ndarray] = {}
        self._step_positions = {step.source_sample_index: index for index, step in enumerate(self._plan.steps)}
        self._rewind_to_ack_requested = False
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
            expected = self._resume_reference_state_locked()
            if expected is None:
                self._set_cursor_locked(ReplayState.ERROR, message="pause has no expected state")
                return False
            self._require_plan_integrity(check_expiration=True)
            self._require_action_spec()
        actual = self._robot.read_state()
        self._check_health(actual)
        self._require_fresh_feedback(actual, self._feedback_age_s(actual))
        error, within_threshold = self._tracking_error_values(np.asarray(actual.values, dtype=np.float32), expected)
        if not within_threshold:
            with self._pause_condition:
                self._set_cursor_locked(
                    ReplayState.PAUSED,
                    tracking_error=error,
                    lag_adjusted_tracking_error=error,
                    message="resume rejected: move to the acknowledged replay state or replan first",
                )
            return False
        with self._pause_condition:
            if self._pause_started_ns is not None:
                self._schedule_shift_ns += time.monotonic_ns() - self._pause_started_ns
            self._pause_started_ns = None
            self._rewind_to_ack_requested = True
            self._set_cursor_locked(ReplayState.RUNNING, tracking_error=error, message="resumed")
            self._pause_condition.notify_all()
        self._record_event("resume", tracking_error=error)
        return True

    def stop(self, *, emergency: bool = False, wait: bool = False, timeout: float | None = None) -> bool:
        """Request an operator or emergency stop and optionally join safely."""
        with self._pause_condition:
            state = self._cursor.state
            if state in {ReplayState.COMPLETED, ReplayState.ABORTED, ReplayState.ERROR, ReplayState.IDLE}:
                return self.join(timeout=timeout)
            self._stop_event.set()
            # Once move-to-start has completed, the prepare thread has no more
            # work to observe the stop event.  Keep this cancellation terminal
            # instead of leaving an ARMED controller forever in STOPPING with
            # a robot lease still held by its owner.
            terminal_without_replay_worker = state is ReplayState.ARMED
            self._set_cursor_locked(
                ReplayState.ABORTED if terminal_without_replay_worker else ReplayState.STOPPING,
                message="emergency stop" if emergency else "operator stop",
            )
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
        capability_report = capability.validate_pose_target(target)
        capability_report.require_valid()
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
            safety_profile_hash=capability_report.safety_profile_hash,
            safety_policy=capability_report.safety_policy,
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
        commands: list[ReplayCommandRecord] = []
        next_step_number = 0
        last_feedback_poll_ns = 0
        strategy = self._plan.constraints.acknowledgement_strategy
        try:
            self._require_plan_integrity(check_expiration=True)
            self._require_action_spec()
            with self._lock:
                self._last_acknowledged_step_number = -1
                self._last_sent_step_number = -1
                self._last_feedback_step_number = -1
                self._max_tracking_error_seen = None
                self._sustained_tracking_error_count = 0
                self._settle_counts = {}
                self._sent_actions = {}
            while True:
                paused = self._wait_if_paused()
                with self._lock:
                    if self._rewind_to_ack_requested:
                        next_step_number = max(0, self._last_acknowledged_step_number + 1)
                        commands = [
                            command
                            for command in commands
                            if self._step_positions[command.source_sample_index] <= self._last_acknowledged_step_number
                        ]
                        previous_action = (
                            None if self._last_expected_state is None else self._last_expected_state.copy()
                        )
                        self._rewind_to_ack_requested = False
                self._require_live_identity()
                if self._stop_event.is_set():
                    raise _ReplayStopped()
                now_ns = time.monotonic_ns()
                while not paused and next_step_number < len(self._plan.steps):
                    step = self._plan.steps[next_step_number]
                    target_ns = started_ns + step.target_offset_ns + self._schedule_shift_ns
                    if now_ns < target_ns:
                        break
                    previous_action, command = self._send_replay_step(
                        step_number=next_step_number,
                        previous_action=previous_action,
                        started_ns=started_ns,
                    )
                    commands.append(command)
                    next_step_number += 1
                    now_ns = time.monotonic_ns()
                if strategy is ReplayAcknowledgementStrategy.IMMEDIATE_INTERFACE_ACK:
                    if next_step_number >= len(self._plan.steps):
                        with self._lock:
                            self._set_cursor_locked(
                                ReplayState.COMPLETED,
                                elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
                                message="all commands sent; no state-arrival acknowledgement configured",
                            )
                        return
                    self._wait_for_next_replay_tick(
                        started_ns,
                        next_step_number,
                        commands,
                        last_feedback_poll_ns,
                        paused,
                    )
                    continue
                elif self._last_acknowledged_step_number >= len(self._plan.steps) - 1:
                    with self._lock:
                        self._set_cursor_locked(
                            ReplayState.COMPLETED,
                            progress_ratio=1.0,
                            acknowledged_progress_ratio=1.0,
                            displayed_sample_index=self._plan.steps[-1].source_sample_index,
                            elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
                            message="replay completed",
                        )
                    return
                if commands and now_ns - last_feedback_poll_ns >= int(
                    self._plan.constraints.feedback_poll_interval_s * 1e9
                ):
                    feedback = self._read_feedback(commands)
                    last_feedback_poll_ns = time.monotonic_ns()
                    self._process_feedback(feedback, commands, started_ns)
                    self._check_acknowledgement_deadlines(commands)
                    continue
                self._wait_for_next_replay_tick(started_ns, next_step_number, commands, last_feedback_poll_ns, paused)
        except _ReplayStopped:
            with self._lock:
                self._set_cursor_locked(ReplayState.ABORTED, message="replay stopped")
        except BaseException as exc:
            self._finish_error(exc, abort=True)

    def _wait_if_paused(self) -> bool:
        with self._pause_condition:
            while self._cursor.state is ReplayState.PAUSED and not self._stop_event.is_set():
                if self._plan.constraints.poll_feedback_while_paused:
                    return True
                self._pause_condition.wait(timeout=0.1)
            if self._stop_event.is_set():
                raise _ReplayStopped()
            return self._cursor.state is ReplayState.PAUSED

    def _wait_for_next_replay_tick(
        self,
        started_ns: int,
        next_step_number: int,
        commands: list[ReplayCommandRecord],
        last_feedback_poll_ns: int,
        paused: bool,
    ) -> None:
        deadlines: list[int] = []
        if not paused and next_step_number < len(self._plan.steps):
            deadlines.append(started_ns + self._plan.steps[next_step_number].target_offset_ns + self._schedule_shift_ns)
        if (
            commands
            and self._plan.constraints.acknowledgement_strategy
            is not ReplayAcknowledgementStrategy.IMMEDIATE_INTERFACE_ACK
        ):
            deadlines.append(last_feedback_poll_ns + int(self._plan.constraints.feedback_poll_interval_s * 1e9))
        if not deadlines:
            self._stop_event.wait(0.01)
            return
        remaining_s = max(0.0, (min(deadlines) - time.monotonic_ns()) / 1e9)
        self._stop_event.wait(min(remaining_s, 0.05))

    def _send_replay_step(
        self,
        *,
        step_number: int,
        previous_action: np.ndarray | None,
        started_ns: int,
    ) -> tuple[np.ndarray, ReplayCommandRecord]:
        step = self._plan.steps[step_number]
        cycle_started_ns = time.monotonic_ns()
        target = step.target.copy()
        if not np.isfinite(target).all():
            raise RuntimeError("planned target became non-finite")
        executed = self._robot.execute_transition(previous_action, target)
        sent_action = np.asarray(executed, dtype=np.float32).copy()
        sent_ns = time.monotonic_ns()
        elapsed_ns = sent_ns - cycle_started_ns
        if elapsed_ns > int(self._plan.constraints.max_control_overrun_s * 1e9):
            raise RuntimeError("replay control cycle overrun")
        command_id = f"{self._plan.session_id}:{step_number}:{step.source_sample_index}"
        command = ReplayCommandRecord(
            command_id=command_id,
            source_sample_index=step.source_sample_index,
            sent_timestamp_ns=sent_ns,
            target=target,
            expected_state=step.expected_state,
            acknowledgement_deadline_ns=sent_ns + int(self._acknowledgement_timeout_s(step_number) * 1e9),
            joint_tracking_threshold=self._joint_tracking_threshold(),
            gripper_tracking_threshold=self._gripper_tracking_threshold(),
        )
        with self._lock:
            self._sent_actions[command_id] = sent_action.copy()
            self._last_sent_step_number = step_number
            self._set_cursor_locked(
                ReplayState.RUNNING,
                planned_sample_index=step.source_sample_index,
                source_sample_index=step.source_sample_index,
                sent_sample_index=step.source_sample_index,
                sent_progress_ratio=self._progress_for_step_number(step_number),
                elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
            )
        self._record_event(
            "replay_command_sent",
            command_id=command.command_id,
            source_sample_index=command.source_sample_index,
            sent_timestamp_ns=command.sent_timestamp_ns,
            target=command.target.copy(),
            sent_action=sent_action.copy(),
            expected_state=command.expected_state.copy(),
            acknowledgement_deadline_ns=command.acknowledgement_deadline_ns,
            cursors=self._cursor_payload(),
        )
        return sent_action, command

    def _read_feedback(self, commands: list[ReplayCommandRecord]) -> _FeedbackEvaluation:
        actual = self._robot.read_state()
        self._check_health(actual)
        feedback_age_s = self._feedback_age_s(actual)
        self._require_fresh_feedback(actual, feedback_age_s)
        evaluation = self._evaluate_feedback(actual, commands, feedback_age_s)
        self._record_event("replay_feedback", **dataclasses.asdict(evaluation.record), cursors=self._cursor_payload())
        return evaluation

    def _evaluate_feedback(
        self,
        actual: RobotState,
        commands: list[ReplayCommandRecord],
        feedback_age_s: float | None,
    ) -> _FeedbackEvaluation:
        values = np.asarray(actual.values, dtype=np.float32)
        if values.shape != self._plan.steps[0].expected_state.shape:
            raise RuntimeError("robot state dimension changed during replay")
        if not np.isfinite(values).all():
            raise RuntimeError("robot returned non-finite state")
        candidates = self._feedback_candidates(actual.timestamp_monotonic_ns, commands)
        instantaneous = None
        if commands:
            instantaneous = self._tracking_error_values(values, commands[-1].expected_state)[0]
        best: tuple[float, ReplayCommandRecord, int, bool] | None = None
        for command in candidates:
            total_error, within_threshold = self._tracking_error_values(values, command.expected_state)
            step_number = self._step_positions[command.source_sample_index]
            if best is None or total_error < best[0]:
                best = (total_error, command, step_number, within_threshold)
        matched_command = best[1] if best is not None else None
        matched_step_number = best[2] if best is not None else None
        lag_adjusted = best[0] if best is not None else instantaneous
        record = ReplayFeedbackRecord(
            feedback_timestamp_ns=actual.timestamp_monotonic_ns,
            robot_state=values.copy(),
            feedback_age_s=feedback_age_s,
            matched_command_id=matched_command.command_id if matched_command is not None else None,
            instantaneous_tracking_error=instantaneous,
            lag_adjusted_tracking_error=lag_adjusted,
            acknowledged=False,
        )
        return _FeedbackEvaluation(
            actual=actual,
            record=record,
            matched_command=matched_command,
            matched_step_number=matched_step_number,
            within_threshold=bool(best[3]) if best is not None else False,
        )

    def _process_feedback(
        self,
        evaluation: _FeedbackEvaluation,
        commands: list[ReplayCommandRecord],
        started_ns: int,
    ) -> None:
        del commands
        if evaluation.matched_command is None:
            with self._lock:
                self._set_cursor_locked(
                    ReplayState.RUNNING,
                    elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
                    tracking_error=evaluation.record.lag_adjusted_tracking_error,
                    instantaneous_tracking_error=evaluation.record.instantaneous_tracking_error,
                    lag_adjusted_tracking_error=evaluation.record.lag_adjusted_tracking_error,
                    max_tracking_error=self._max_tracking_error_seen,
                    sustained_tracking_error_count=self._sustained_tracking_error_count,
                )
            return
        lag_error = evaluation.record.lag_adjusted_tracking_error
        self._update_tracking_error_state(lag_error, evaluation.within_threshold)
        ack_step_number: int | None = None
        if (
            evaluation.matched_command is not None
            and evaluation.matched_step_number is not None
            and evaluation.within_threshold
        ):
            if self._plan.constraints.acknowledgement_strategy is ReplayAcknowledgementStrategy.STATE_TRAJECTORY_SETTLE:
                count = self._settle_counts.get(evaluation.matched_command.command_id, 0) + 1
                self._settle_counts[evaluation.matched_command.command_id] = count
                if count >= self._plan.constraints.state_trajectory_settle_cycles:
                    ack_step_number = evaluation.matched_step_number
            else:
                ack_step_number = evaluation.matched_step_number
        if ack_step_number is not None:
            self._acknowledge_feedback(evaluation, ack_step_number, started_ns)
            return
        with self._lock:
            if evaluation.matched_step_number is not None:
                self._last_feedback_step_number = max(self._last_feedback_step_number, evaluation.matched_step_number)
            self._set_cursor_locked(
                ReplayState.RUNNING,
                feedback_sample_index=(
                    evaluation.matched_command.source_sample_index
                    if evaluation.matched_command is not None
                    else self._cursor.feedback_sample_index
                ),
                feedback_progress_ratio=self._progress_for_step_number(self._last_feedback_step_number),
                elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
                tracking_error=lag_error,
                instantaneous_tracking_error=evaluation.record.instantaneous_tracking_error,
                lag_adjusted_tracking_error=lag_error,
                max_tracking_error=self._max_tracking_error_seen,
                sustained_tracking_error_count=self._sustained_tracking_error_count,
            )

    def _acknowledge_feedback(
        self,
        evaluation: _FeedbackEvaluation,
        ack_step_number: int,
        started_ns: int,
    ) -> None:
        ack_step_number = max(self._last_acknowledged_step_number, ack_step_number)
        command = evaluation.matched_command
        if command is None:
            return
        acknowledged_record = dataclasses.replace(evaluation.record, acknowledged=True)
        with self._lock:
            self._last_acknowledged_step_number = ack_step_number
            self._last_feedback_step_number = max(self._last_feedback_step_number, ack_step_number)
            self._last_expected_state = self._plan.steps[ack_step_number].expected_state.copy()
            self._set_cursor_locked(
                ReplayState.RUNNING,
                feedback_sample_index=command.source_sample_index,
                acknowledged_sample_index=self._plan.steps[ack_step_number].source_sample_index,
                displayed_sample_index=self._plan.steps[ack_step_number].source_sample_index,
                progress_ratio=self._progress_for_step_number(ack_step_number),
                feedback_progress_ratio=self._progress_for_step_number(self._last_feedback_step_number),
                acknowledged_progress_ratio=self._progress_for_step_number(ack_step_number),
                elapsed_s=(time.monotonic_ns() - started_ns) / 1e9,
                tracking_error=acknowledged_record.lag_adjusted_tracking_error,
                instantaneous_tracking_error=acknowledged_record.instantaneous_tracking_error,
                lag_adjusted_tracking_error=acknowledged_record.lag_adjusted_tracking_error,
                max_tracking_error=self._max_tracking_error_seen,
                sustained_tracking_error_count=self._sustained_tracking_error_count,
            )
        self._record_step(command, acknowledged_record, evaluation.actual)

    def _feedback_candidates(
        self,
        feedback_timestamp_ns: int,
        commands: list[ReplayCommandRecord],
    ) -> list[ReplayCommandRecord]:
        pending = [
            command
            for command in commands
            if self._step_positions[command.source_sample_index] > self._last_acknowledged_step_number
            and feedback_timestamp_ns >= command.sent_timestamp_ns
        ]
        if not pending:
            return []
        strategy = self._plan.constraints.acknowledgement_strategy
        if strategy is ReplayAcknowledgementStrategy.FOLLOWER_WINDOW:
            return pending[: self._plan.constraints.follower_window_samples + 1]
        return pending[:1]

    def _check_acknowledgement_deadlines(self, commands: list[ReplayCommandRecord]) -> None:
        if self._plan.constraints.acknowledgement_strategy is ReplayAcknowledgementStrategy.IMMEDIATE_INTERFACE_ACK:
            return
        next_step_number = self._last_acknowledged_step_number + 1
        for command in commands:
            if self._step_positions[command.source_sample_index] != next_step_number:
                continue
            if time.monotonic_ns() > command.acknowledgement_deadline_ns:
                raise RuntimeError(
                    "feedback acknowledgement timeout for "
                    f"sample {command.source_sample_index} after command {command.command_id}"
                )
            return

    def _update_tracking_error_state(self, tracking_error: float | None, within_threshold: bool) -> None:
        if tracking_error is None:
            return
        with self._lock:
            self._max_tracking_error_seen = (
                tracking_error
                if self._max_tracking_error_seen is None
                else max(self._max_tracking_error_seen, tracking_error)
            )
            if within_threshold:
                self._sustained_tracking_error_count = 0
                return
            if tracking_error > self._plan.constraints.effective_extreme_tracking_error:
                raise RuntimeError(f"tracking error {tracking_error:.6f} exceeds replay extreme limit")
            self._sustained_tracking_error_count += 1
            count = self._sustained_tracking_error_count
        self._record_event(
            "replay_tracking_warning",
            lag_adjusted_tracking_error=tracking_error,
            sustained_tracking_error_count=count,
            cursors=self._cursor_payload(),
        )
        if count >= self._plan.constraints.sustained_tracking_error_limit:
            raise RuntimeError(
                "sustained tracking error "
                f"{tracking_error:.6f} exceeded tolerance for {count} feedback cycles"
            )

    def _tracking_error_values(self, actual_values: np.ndarray, expected: np.ndarray) -> tuple[float, bool]:
        values = np.asarray(actual_values, dtype=np.float32)
        target = np.asarray(expected, dtype=np.float32)
        if values.shape != target.shape:
            raise RuntimeError("robot state dimension changed during replay")
        if not np.isfinite(values).all():
            raise RuntimeError("robot returned non-finite state")
        errors = np.abs(values - target)
        joint_indices = self._joint_indices()
        gripper_indices = self._gripper_indices()
        joint_set = set(joint_indices)
        gripper_set = set(gripper_indices)
        other_indices = tuple(
            index for index in range(len(errors)) if index not in joint_set and index not in gripper_set
        )
        joint_error = _max_indexed_error(errors, joint_indices)
        gripper_error = _max_indexed_error(errors, gripper_indices)
        other_error = _max_indexed_error(errors, other_indices)
        total = max(joint_error, gripper_error, other_error)
        within = (
            joint_error <= self._joint_tracking_threshold()
            and gripper_error <= self._gripper_tracking_threshold()
            and other_error <= self._joint_tracking_threshold()
        )
        return total, within

    def _joint_indices(self) -> tuple[int, ...]:
        fields = (
            self._plan.action_spec.action_fields
            if self._plan.mode.value == "command"
            else self._plan.action_spec.state_fields
        )
        gripper_set = set(self._gripper_indices())
        if fields:
            return tuple(
                index
                for index, field in enumerate(fields)
                if field.semantics == "joint_position" and index not in gripper_set
            )
        return tuple(index for index in range(self._plan.steps[0].expected_state.shape[0]) if index not in gripper_set)

    def _gripper_indices(self) -> tuple[int, ...]:
        if self._plan.constraints.gripper_indices is not None:
            return self._plan.constraints.gripper_indices
        return self._plan.source.gripper_indices

    def _joint_tracking_threshold(self) -> float:
        if self._plan.constraints.joint_tracking_error is not None:
            return self._plan.constraints.joint_tracking_error
        profile = profile_for_robot(self._robot)
        if profile is not None and profile.tracking_error_threshold is not None:
            return profile.tracking_error_threshold
        return self._plan.constraints.effective_joint_tracking_error

    def _gripper_tracking_threshold(self) -> float:
        if self._plan.constraints.gripper_tracking_threshold is not None:
            return self._plan.constraints.gripper_tracking_threshold
        return self._plan.constraints.effective_gripper_tracking_threshold

    def _acknowledgement_timeout_s(self, step_number: int) -> float:
        timeout = self._plan.constraints.acknowledgement_timeout_s
        if self._gripper_indices() and step_number > 0:
            previous = self._plan.steps[step_number - 1].expected_state
            current = self._plan.steps[step_number].expected_state
            if np.any(np.abs(current[list(self._gripper_indices())] - previous[list(self._gripper_indices())]) > 0):
                timeout = max(timeout, self._plan.constraints.gripper_settle_timeout_s)
        return timeout

    def _feedback_age_s(self, state: RobotState) -> float | None:
        now_ns = time.monotonic_ns()
        if state.source_timestamp_ns is not None:
            return max(0.0, (now_ns - state.source_timestamp_ns) / 1e9)
        health = health_from_state_mapping(state.health)
        if health is not None and health.last_feedback_age_s is not None:
            return health.last_feedback_age_s
        return None

    def _require_fresh_feedback(self, state: RobotState, feedback_age_s: float | None) -> None:
        health = health_from_state_mapping(state.health)
        if health is not None and health.stale_feedback is True:
            raise RuntimeError("robot feedback is stale")
        timeout = self._plan.constraints.feedback_freshness_timeout_s
        profile = profile_for_robot(self._robot)
        if timeout is None and profile is not None:
            timeout = profile.communication_timeout_s
        if timeout is not None and feedback_age_s is not None and feedback_age_s > timeout:
            raise RuntimeError(f"robot feedback age {feedback_age_s:.6f}s exceeds {timeout:.6f}s")

    def _progress_for_step_number(self, step_number: int | None) -> float:
        if step_number is None or step_number < 0:
            return 0.0
        return min(1.0, (step_number + 1) / len(self._plan.steps))

    def _cursor_payload(self) -> dict[str, Any]:
        with self._lock:
            return dataclasses.asdict(self._cursor)

    def _record_step(self, command: ReplayCommandRecord, feedback: ReplayFeedbackRecord, actual: RobotState) -> None:
        if self._record_callback is None:
            return
        sent_action = self._sent_actions.get(command.command_id, command.target)
        self._record_callback(
            {
                "type": "replay_step",
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "command_id": command.command_id,
                "source_sample_index": command.source_sample_index,
                "sent_timestamp_ns": command.sent_timestamp_ns,
                "feedback_timestamp_ns": feedback.feedback_timestamp_ns,
                "acknowledged": feedback.acknowledged,
                "sent_action": sent_action.copy(),
                "target": command.target.copy(),
                "expected_state": command.expected_state.copy(),
                "actual_state": actual.values.copy(),
                "feedback_age_s": feedback.feedback_age_s,
                "tracking_error": feedback.lag_adjusted_tracking_error,
                "instantaneous_tracking_error": feedback.instantaneous_tracking_error,
                "lag_adjusted_tracking_error": feedback.lag_adjusted_tracking_error,
                "cursors": self._cursor_payload(),
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

    def _resume_reference_state_locked(self) -> np.ndarray | None:
        if self._last_acknowledged_step_number >= 0:
            return self._plan.steps[self._last_acknowledged_step_number].expected_state.copy()
        if self._last_expected_state is not None:
            return self._last_expected_state.copy()
        return self._plan.start_state

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
        self._require_safety_profile()
        self._require_acknowledgement_strategy()

    def _require_safety_profile(self) -> None:
        if self._plan.safety_profile_hash is None:
            return
        profile = profile_for_robot(self._robot)
        if profile is None:
            raise ReplayPlanStaleError("connected robot does not expose the reviewed safety profile")
        if profile.profile_hash != self._plan.safety_profile_hash:
            raise ReplayPlanStaleError("connected robot safety profile changed after replay plan generation")
        if self._plan.safety_policy is not None and profile.policy.value != self._plan.safety_policy:
            raise ReplayPlanStaleError("connected robot safety policy changed after replay plan generation")

    def _require_acknowledgement_strategy(self) -> None:
        if self._plan.constraints.acknowledgement_strategy is not ReplayAcknowledgementStrategy.IMMEDIATE_INTERFACE_ACK:
            return
        profile = profile_for_robot(self._robot)
        if profile is not None and profile.hardware_motion_enabled:
            raise ReplayPlanStaleError(
                "immediate interface acknowledgement cannot be used as real hardware arrival confirmation"
            )

    def _check_health(self, state: RobotState) -> None:
        health = state.health
        if not isinstance(health, Mapping):
            return
        snapshot = health_from_state_mapping(health)
        if snapshot is not None:
            failures: list[str] = []
            if snapshot.connected is False:
                failures.append("disconnected")
            if snapshot.enabled is False:
                failures.append("disabled")
            if snapshot.healthy is False:
                failures.append("unhealthy")
            if snapshot.error_codes:
                failures.append(f"error_codes={list(snapshot.error_codes)}")
            for name, arm in snapshot.arms.items():
                if arm.connected is False:
                    failures.append(f"{name}:disconnected")
                if arm.enabled is False:
                    failures.append(f"{name}:disabled")
                if arm.healthy is False:
                    failures.append(f"{name}:unhealthy")
                if arm.error_codes:
                    failures.append(f"{name}:error_codes={list(arm.error_codes)}")
            if failures:
                raise RuntimeError(f"robot health check failed: {', '.join(failures)}")
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
            acknowledgement_strategy=self._plan.constraints.acknowledgement_strategy.value,
        )

    def _set_cursor_locked(self, state: ReplayState, **changes: Any) -> None:
        planned_sample_index = changes.get("planned_sample_index", self._cursor.planned_sample_index)
        source_sample_index = changes.get("source_sample_index", planned_sample_index)
        sent_sample_index = changes.get("sent_sample_index", self._cursor.sent_sample_index)
        feedback_sample_index = changes.get("feedback_sample_index", self._cursor.feedback_sample_index)
        acknowledged_sample_index = changes.get(
            "acknowledged_sample_index", self._cursor.acknowledged_sample_index
        )
        displayed_sample_index = changes.get(
            "displayed_sample_index",
            acknowledged_sample_index
            if acknowledged_sample_index is not None
            else self._cursor.displayed_sample_index,
        )
        sent_step_number = self._step_positions.get(sent_sample_index, -1) if sent_sample_index is not None else -1
        feedback_step_number = (
            self._step_positions.get(feedback_sample_index, -1) if feedback_sample_index is not None else -1
        )
        acknowledged_step_number = (
            self._step_positions.get(acknowledged_sample_index, -1)
            if acknowledged_sample_index is not None
            else -1
        )
        self._cursor = dataclasses.replace(
            self._cursor_for(state),
            planned_sample_index=planned_sample_index,
            source_sample_index=source_sample_index,
            sent_sample_index=sent_sample_index,
            feedback_sample_index=feedback_sample_index,
            acknowledged_sample_index=acknowledged_sample_index,
            displayed_sample_index=displayed_sample_index,
            progress_ratio=changes.get("progress_ratio", self._progress_for_step_number(acknowledged_step_number)),
            sent_progress_ratio=changes.get("sent_progress_ratio", self._progress_for_step_number(sent_step_number)),
            feedback_progress_ratio=changes.get(
                "feedback_progress_ratio", self._progress_for_step_number(feedback_step_number)
            ),
            acknowledged_progress_ratio=changes.get(
                "acknowledged_progress_ratio", self._progress_for_step_number(acknowledged_step_number)
            ),
            lag_samples=max(0, sent_step_number - acknowledged_step_number),
            elapsed_s=changes.get("elapsed_s", self._cursor.elapsed_s),
            tracking_error=changes.get("tracking_error", self._cursor.tracking_error),
            instantaneous_tracking_error=changes.get(
                "instantaneous_tracking_error", self._cursor.instantaneous_tracking_error
            ),
            lag_adjusted_tracking_error=changes.get(
                "lag_adjusted_tracking_error", self._cursor.lag_adjusted_tracking_error
            ),
            max_tracking_error=changes.get("max_tracking_error", self._cursor.max_tracking_error),
            sustained_tracking_error_count=changes.get(
                "sustained_tracking_error_count", self._cursor.sustained_tracking_error_count
            ),
            message=changes.get("message", self._cursor.message),
            timestamp_monotonic_ns=time.monotonic_ns(),
        )


class _ReplayStopped(RuntimeError):
    pass


def _max_indexed_error(errors: np.ndarray, indices: tuple[int, ...]) -> float:
    if not indices:
        return 0.0
    return float(np.max(errors[list(indices)]))
