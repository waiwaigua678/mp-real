from __future__ import annotations

import gc
import json
import re
import tempfile
import threading
import time
import tomllib
import unittest
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mp_real.data.lerobot_v21 import LeRobotV21EpisodeRecorder, LeRobotV21EpisodeSource, validate_lerobot_v21_dataset
from mp_real.data.models import EpisodeRecordingContext, FakeRecordedEpisodeSource, RecordedSample, RecorderConfig
from mp_real.pose.controller import PoseMoveController
from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMotionConstraints,
    PoseMoveProgress,
    PoseMoveResult,
    PosePlanStaleError,
    PoseValidationError,
    RecordedPoseTarget,
)
from mp_real.pose.validation import MoveToStateValidator
from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import ReplayConstraints, ReplayPlanStaleError, ReplayState
from mp_real.replay.planning import ReplayPlanner
from mp_real.robots.piper.infer import Args as PiperArgs
from mp_real.robots.piper.infer import PiperArm, PiperRobot
from mp_real.robots.rm2.infer import Args as Rm2Args
from mp_real.robots.rm2.infer import MockArm, Rm2Robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.events import (
    ActionExecuted,
    ActionSelected,
    ActionStabilized,
    ControlStepRecorded,
    InMemoryRuntimeEventSink,
    ObservationCaptured,
    RuntimeEventHooks,
    RuntimeEventIdentity,
)
from mp_real.runtime.inference import run_rtc_loop, run_sync_loop
from mp_real.runtime.models import (
    ActionProvenance,
    ActionSpec,
    CameraSample,
    ObservationSnapshot,
    RobotState,
    VectorField,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(
    *,
    action_mode: str = "joint_position_target",
    camera_roles: tuple[str, ...] = ("head",),
    arm_count: int | None = None,
) -> ActionSpec:
    fields = (
        VectorField("joint_1", "rad", "joint_position"),
        VectorField("gripper", "normalized_0_open_1", "gripper_open_fraction"),
    )
    return ActionSpec(
        2,
        2,
        1,
        "rad",
        camera_roles,
        state_fields=fields,
        action_fields=fields,
        action_mode=action_mode,
        arm_count=arm_count,
    )


def _dual_arm_spec() -> ActionSpec:
    fields = (
        VectorField("left_joint_1", "rad", "joint_position"),
        VectorField("right_joint_1", "rad", "joint_position"),
        VectorField("left_gripper", "normalized_0_open_1", "gripper_open_fraction"),
        VectorField("right_gripper", "normalized_0_open_1", "gripper_open_fraction"),
    )
    return ActionSpec(
        4,
        4,
        1,
        "rad",
        (),
        state_fields=fields,
        action_fields=fields,
    )


def _state(values: np.ndarray) -> RobotState:
    now_ns = time.monotonic_ns()
    return RobotState(np.asarray(values, dtype=np.float32).copy(), now_ns / 1e9, now_ns, health={"ok": True})


def _target(spec: ActionSpec, values: np.ndarray | None = None, *, robot_name: str = "piper") -> RecordedPoseTarget:
    return RecordedPoseTarget(
        dataset_id="hardening",
        episode_index=0,
        sample_index=0,
        robot_name=robot_name,
        state_schema=spec.state_field_names,
        state_values=np.asarray(values if values is not None else np.zeros(spec.state_dim), dtype=np.float32),
        state_fields=spec.state_fields,
        joint_unit=spec.joint_unit,
        timestamp=0.0,
        source_metadata={"dataset_status": "complete"},
        action_spec=spec,
    )


class _Policy:
    def __init__(self, action_dim: int, *, chunk_len: int = 3) -> None:
        self._action_dim = action_dim
        self._chunk_len = chunk_len

    def infer(self, observation: dict) -> dict:
        base = float(np.asarray(observation["state"], dtype=np.float32)[0])
        actions = np.zeros((self._chunk_len, self._action_dim), dtype=np.float32)
        for cursor in range(self._chunk_len):
            actions[cursor, 0] = base + 0.1 * (cursor + 1)
            if self._action_dim > 1:
                actions[cursor, 1] = 1.0
        return {"actions": actions}


class _RuntimeAdapter:
    name = "hardening-fake"

    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec
        self.observe_count = 0
        self.execute_count = 0
        self.last_observation_snapshot: ObservationSnapshot | None = None

    def observe(self) -> dict:
        now_ns = 2_000_000_000 + self.observe_count * 100_000_000
        image = np.full((8, 10, 3), self.observe_count, dtype=np.uint8)
        state = np.asarray([float(self.observe_count), 1.0], dtype=np.float32)
        self.last_observation_snapshot = ObservationSnapshot(
            images={
                role: CameraSample(
                    image,
                    now_ns / 1e9,
                    frame_id=self.observe_count + 1,
                    timestamp_monotonic_ns=now_ns,
                )
                for role in self.spec.camera_roles
            },
            image_masks={role: np.bool_(True) for role in self.spec.camera_roles},
            state=RobotState(state, now_ns / 1e9, now_ns),
            prompt="hardening",
            capture_started_ns=now_ns,
            capture_finished_ns=now_ns,
        )
        self.observe_count += 1
        return self.last_observation_snapshot.to_policy_observation()

    def decode_action_chunk(self, response: dict, replan_steps: int) -> np.ndarray:
        return self.spec.validate_chunk(np.asarray(response["actions"], dtype=np.float32))[:replan_steps]

    def initial_action(self) -> np.ndarray:
        return np.zeros(self.spec.action_dim, dtype=np.float32)

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return np.asarray(action, dtype=np.float32)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        self.execute_count += 1
        return np.asarray(target, dtype=np.float32).copy()

    def infer_only_metadata(self, observation: dict) -> dict:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


@dataclass(frozen=True)
class LeRobotSemanticAudit:
    same_chunk_reused_observation_pairs: tuple[tuple[int, int, int], ...]
    repeated_camera_frame_pairs: tuple[tuple[int, int, str, int], ...]
    repeated_robot_state_pairs: tuple[tuple[int, int], ...]

    @property
    def has_risk_signal(self) -> bool:
        return bool(
            self.same_chunk_reused_observation_pairs
            or self.repeated_camera_frame_pairs
            or self.repeated_robot_state_pairs
        )


def audit_lerobot_episode_semantics(source: LeRobotV21EpisodeSource, episode_index: int) -> LeRobotSemanticAudit:
    """Risk-signal fixture only; repeated frames or states can be legitimate for a still robot."""

    samples = [
        source.get_sample(episode_index, index, include_images=False)
        for index in range(source.get_length(episode_index))
    ]
    same_observations: list[tuple[int, int, int]] = []
    repeated_states: list[tuple[int, int]] = []
    for previous, current in zip(samples, samples[1:], strict=False):
        previous_observation_id = int(previous.telemetry.get("observation_id", -1))
        current_observation_id = int(current.telemetry.get("observation_id", -1))
        previous_cursor = int(previous.telemetry.get("chunk_cursor", -1))
        current_cursor = int(current.telemetry.get("chunk_cursor", -1))
        if previous_observation_id >= 0 and previous_observation_id == current_observation_id:
            if previous_cursor >= 0 and current_cursor == previous_cursor + 1:
                same_observations.append((previous.index, current.index, current_observation_id))
        if np.array_equal(previous.state, current.state):
            repeated_states.append((previous.index, current.index))

    repeated_frames: list[tuple[int, int, str, int]] = []
    telemetry = source.get_episode_telemetry(episode_index, keys=("camera_roles", "camera_frame_ids"))
    if "camera_roles" in telemetry and "camera_frame_ids" in telemetry:
        roles = [str(role) for role in telemetry["camera_roles"]]
        frame_ids = np.asarray(telemetry["camera_frame_ids"], dtype=np.int64)
        for row_index in range(1, len(frame_ids)):
            for role_index, role in enumerate(roles):
                if frame_ids[row_index, role_index] == frame_ids[row_index - 1, role_index]:
                    repeated_frames.append(
                        (row_index - 1, row_index, role, int(frame_ids[row_index, role_index]))
                    )
    return LeRobotSemanticAudit(tuple(same_observations), tuple(repeated_frames), tuple(repeated_states))


def _record_sync_dataset(root: Path, *, max_steps: int = 3, replan_steps: int = 3) -> tuple[Path, _RuntimeAdapter]:
    spec = _spec()
    adapter = _RuntimeAdapter(spec)
    recorder = LeRobotV21EpisodeRecorder(
        RecorderConfig(root, root.name, "fake", 10.0, spec, queue_size=256, session_id="session")
    )
    recorder.start()
    recorder.begin_episode(EpisodeRecordingContext(0, "episode", "task", "session", 1))
    run_sync_loop(
        _Policy(spec.action_dim, chunk_len=replan_steps),
        adapter,
        InferenceLoopConfig(10.0, replan_steps, max_steps, False, 0, 0, 0.0, False, False, 1, None, "task", False),
        hooks=RuntimeEventHooks(recorder, RuntimeEventIdentity("runtime", 1, "session", "episode")),
    )
    recorder.end_episode(labels={"result": "SUCCESS"})
    assert recorder.stop(timeout=20)
    return root, adapter


def _emit_one_recorded_frame(
    recorder: LeRobotV21EpisodeRecorder,
    *,
    episode_id: str,
    spec: ActionSpec,
    observation_id: int,
    step: int,
) -> None:
    now_ns = 2_000_000_000 + step * 100_000_000
    state = np.full(spec.state_dim, step, dtype=np.float32)
    images = {role: np.full((8, 10, 3), step, dtype=np.uint8) for role in spec.camera_roles}
    action = np.full(spec.action_dim, 0.25 + step, dtype=np.float32)
    identity = {
        "runtime_id": "runtime",
        "session_id": "session",
        "episode_id": episode_id,
        "generation_id": 1,
    }
    recorder.emit(
        ObservationCaptured(
            **identity,
            monotonic_timestamp_ns=now_ns,
            payload={
                "observation_id": observation_id,
                "state": state,
                "images": images,
                "camera_frame_ids": {role: step + 1 for role in spec.camera_roles},
                "camera_timestamps_ns": {role: now_ns for role in spec.camera_roles},
                "max_camera_skew_ns": 0,
            },
        )
    )
    _emit_action_events(
        recorder,
        episode_id=episode_id,
        observation_id=observation_id,
        step=step,
        action=action,
        now_ns=now_ns,
    )


def _emit_action_events(
    recorder: LeRobotV21EpisodeRecorder,
    *,
    episode_id: str,
    observation_id: int,
    step: int,
    action: np.ndarray,
    now_ns: int,
    chunk_cursor: int | None = None,
) -> None:
    identity = {
        "runtime_id": "runtime",
        "session_id": "session",
        "episode_id": episode_id,
        "generation_id": 1,
    }
    selected_payload: dict[str, object] = {"observation_id": observation_id, "selected_raw_action": action}
    stabilized_payload: dict[str, object] = {"observation_id": observation_id, "stabilized_target_action": action}
    executed_payload: dict[str, object] = {"observation_id": observation_id, "executed_action": action}
    if chunk_cursor is not None:
        selected_payload["chunk_cursor"] = chunk_cursor
        stabilized_payload["chunk_cursor"] = chunk_cursor
        executed_payload["chunk_cursor"] = chunk_cursor
    recorder.emit(
        ActionSelected(
            **identity,
            step=step,
            monotonic_timestamp_ns=now_ns + 1,
            payload=selected_payload,
        )
    )
    recorder.emit(
        ActionStabilized(
            **identity,
            step=step,
            monotonic_timestamp_ns=now_ns + 2,
            payload=stabilized_payload,
        )
    )
    recorder.emit(
        ActionExecuted(
            **identity,
            step=step,
            monotonic_timestamp_ns=now_ns + 3,
            payload=executed_payload,
        )
    )


def _source(
    *,
    count: int = 4,
    spec: ActionSpec | None = None,
    robot_name: str = "piper",
    samples: tuple[RecordedSample, ...] | None = None,
) -> FakeRecordedEpisodeSource:
    action_spec = spec or _spec()
    episode_samples = samples or tuple(
        RecordedSample(
            episode_index=0,
            frame_index=index,
            index=index,
            timestamp=index * 0.02,
            task_index=0,
            state=np.asarray([index * 0.01, 1.0], dtype=np.float32),
            action=np.asarray([index * 0.01, 1.0], dtype=np.float32),
            images={},
            telemetry={},
        )
        for index in range(count)
    )
    return FakeRecordedEpisodeSource(action_spec, {0: episode_samples}, robot_name=robot_name)


def _constraints() -> ReplayConstraints:
    return ReplayConstraints(
        min_interval_s=0.001,
        max_interval_s=0.2,
        max_step=0.1,
        max_velocity=10.0,
        max_acceleration=1_000.0,
        tracking_tolerance=0.03,
        max_tracking_error=0.1,
        max_control_overrun_s=0.5,
    )


class _FakeReplayRobot:
    def __init__(self, *, state_bias: float = 0.0) -> None:
        self.action_spec = _spec()
        self.state = np.asarray([0.0, 1.0], dtype=np.float32)
        self.state_bias = state_bias
        self.commands: list[np.ndarray] = []
        self.stops = 0

    def read_state(self) -> RobotState:
        return _state(self.state + np.asarray([self.state_bias, 0.0], dtype=np.float32))

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        target = np.asarray(target, dtype=np.float32).copy()
        self.commands.append(target)
        self.state = target
        return target

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_current_pose_state(self) -> RobotState:
        return self.read_state()

    def validate_pose_target(self, target: RecordedPoseTarget):
        return MoveToStateValidator("piper", target.action_spec).validate(target).report

    def plan_move_to_state(self, plan: MoveToRecordedStatePlan) -> MoveToRecordedStatePlan:
        return plan

    def execute_pose_plan(
        self,
        plan: MoveToRecordedStatePlan,
        *,
        stop_event: threading.Event,
        on_progress: object = None,
    ) -> PoseMoveResult:
        for waypoint in plan.waypoints:
            if stop_event.is_set():
                return PoseMoveResult(plan.plan_id, "aborted", self.read_state(), None, "stopped")
            self.state = waypoint.target.copy()
            if callable(on_progress):
                on_progress(
                    PoseMoveProgress(
                        plan.plan_id,
                        waypoint.index,
                        len(plan.waypoints),
                        self.state.copy(),
                        waypoint.target.copy(),
                        0.0,
                        time.monotonic_ns(),
                    )
                )
        return PoseMoveResult(plan.plan_id, "reached", self.read_state(), 0.0)

    def stop_pose_motion(self) -> None:
        self.stops += 1

    def verify_target_reached(self, plan: MoveToRecordedStatePlan) -> PoseMoveResult:
        error = float(np.max(np.abs(self.state - plan.target_state)))
        return PoseMoveResult(plan.plan_id, "reached" if error <= 0.03 else "failed", self.read_state(), error)


def _wait(predicate: object, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


def _replay_plan(*, count: int = 4, speed_scale: float = 1.0):
    result = ReplayPlanner(_source(count=count)).plan(
        robot_name="piper",
        target_action_spec=_spec(),
        episode_index=0,
        speed_scale=speed_scale,
        constraints=_constraints(),
    )
    assert result.plan is not None, result.report.errors
    return result.plan


def _pose_plan(robot: _FakeReplayRobot, target: RecordedPoseTarget | None = None) -> MoveToRecordedStatePlan:
    target = target or _target(_spec(), np.asarray([0.2, 1.0], dtype=np.float32))
    validated = MoveToStateValidator("piper", target.action_spec).validate(target)
    validated.report.require_valid()
    return MoveToRecordedStatePlan.build(
        target=target,
        current_state=robot.get_current_pose_state(),
        target_state=validated.values,
        gripper_indices=validated.gripper_indices,
        mapped_joint_names=validated.field_names,
        conversions=validated.mappings,
        constraints=PoseMotionConstraints(control_period_s=0.001, max_joint_step=0.05, max_gripper_step=0.05),
    )


class _StopArm:
    def stop(self) -> None:
        return None


class HardeningLeRobotSemanticsTests(unittest.TestCase):
    def test_h1_policy_chunk_actions_should_not_share_one_recorded_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _ = _record_sync_dataset(Path(directory) / "chunk-reuse", max_steps=5, replan_steps=5)
            source = LeRobotV21EpisodeSource(root)
            try:
                samples = [source.get_sample(0, index, include_images=False) for index in range(5)]
                observation_ids = [sample.telemetry["observation_id"] for sample in samples]
                policy_observation_ids = [sample.telemetry["policy_observation_id"] for sample in samples]
                self.assertEqual(len(set(observation_ids)), len(observation_ids))
                self.assertEqual(len(set(policy_observation_ids)), 1)
                self.assertFalse(np.array_equal(samples[0].state, samples[1].state))
                audit = audit_lerobot_episode_semantics(source, 0)
                self.assertFalse(audit.same_chunk_reused_observation_pairs)
                self.assertFalse(audit.repeated_camera_frame_pairs)
                self.assertFalse(audit.repeated_robot_state_pairs)
            finally:
                source.close()

    def test_h1_control_loop_should_capture_before_each_executed_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, adapter = _record_sync_dataset(Path(directory) / "capture-count", max_steps=5, replan_steps=5)
            self.assertEqual(adapter.execute_count, 5)
            self.assertEqual(adapter.observe_count, adapter.execute_count + 1)
            source = LeRobotV21EpisodeSource(root)
            try:
                states = [source.get_sample(0, index, include_images=False).state[0] for index in range(5)]
                self.assertEqual(states, [1.0, 2.0, 3.0, 4.0, 5.0])
            finally:
                source.close()

    def test_h1_rtc_records_control_step_observations_for_prefetched_chunk(self) -> None:
        spec = _spec()
        sink = InMemoryRuntimeEventSink()
        initial_chunk = np.asarray(
            [[0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0]],
            dtype=np.float32,
        )
        run_rtc_loop(
            _Policy(spec.action_dim, chunk_len=5),
            _RuntimeAdapter(spec),
            InferenceLoopConfig(1000.0, 5, 3, True, 0, 0, 0.0, False, False, 1, None, "task", False),
            hooks=RuntimeEventHooks(sink, RuntimeEventIdentity("runtime", 1, "session", "episode")),
            initial_chunk=initial_chunk,
            initial_provenance=ActionProvenance(observation_id=123),
        )
        records = [event for event in sink.snapshot() if isinstance(event, ControlStepRecorded)]
        self.assertEqual([record.step for record in records], [0, 1, 2])
        self.assertEqual({record.payload["policy_observation_id"] for record in records}, {123})
        self.assertEqual(
            [record.payload["control_step_id"] for record in records],
            [0, 1, 2],
        )

    def test_h1_observation_and_actions_should_join_by_control_step(self) -> None:
        spec = _spec()
        sink = InMemoryRuntimeEventSink()
        run_sync_loop(
            _Policy(spec.action_dim, chunk_len=3),
            _RuntimeAdapter(spec),
            InferenceLoopConfig(10.0, 3, 3, False, 0, 0, 0.0, False, False, 1, None, "task", False),
            hooks=RuntimeEventHooks(sink, RuntimeEventIdentity("runtime", 1, "session", "episode")),
        )
        events = sink.snapshot()
        records_by_step = {event.step: event for event in events if isinstance(event, ControlStepRecorded)}
        executed_steps = [event.step for event in events if isinstance(event, ActionExecuted)]
        self.assertEqual(set(executed_steps), set(records_by_step))
        for event in records_by_step.values():
            self.assertIn("state", event.payload)
            self.assertIn("selected_raw_action", event.payload)
            self.assertIn("executed_action", event.payload)

    def test_h0_lerobot_semantic_audit_fixture_reports_risk_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = _spec()
            root = Path(directory) / "audit"
            recorder = LeRobotV21EpisodeRecorder(
                RecorderConfig(root, "audit", "fake", 10.0, spec, queue_size=256, session_id="session")
            )
            recorder.start()
            recorder.begin_episode(EpisodeRecordingContext(0, "episode", "task", "session", 1))
            now_ns = 2_000_000_000
            for step in range(3):
                action = np.asarray([0.25 + step, 1.0], dtype=np.float32)
                recorder.emit(
                    ControlStepRecorded(
                        runtime_id="runtime",
                        session_id="session",
                        episode_id="episode",
                        generation_id=1,
                        step=step,
                        monotonic_timestamp_ns=now_ns + step * 100,
                        payload={
                            "control_step_id": step,
                            "observation_id": 1,
                            "policy_observation_id": 1,
                            "state": np.asarray([1.0, 1.0], dtype=np.float32),
                            "images": {"head": np.full((8, 10, 3), 7, dtype=np.uint8)},
                            "camera_frame_ids": {"head": 1},
                            "camera_timestamps_ns": {"head": now_ns},
                            "max_camera_skew_ns": 0,
                            "chunk_cursor": step,
                            "selected_raw_action": action,
                            "stabilized_target_action": action,
                            "executed_action": action,
                        },
                    )
                )
            recorder.end_episode(labels={"result": "SUCCESS"})
            self.assertTrue(recorder.stop(timeout=20))
            source = LeRobotV21EpisodeSource(root)
            try:
                audit = audit_lerobot_episode_semantics(source, 0)
                self.assertTrue(audit.has_risk_signal)
                self.assertEqual(audit.same_chunk_reused_observation_pairs[0][:2], (0, 1))
                self.assertGreater(audit.same_chunk_reused_observation_pairs[0][2], 0)
                self.assertTrue(audit.repeated_camera_frame_pairs)
                self.assertTrue(audit.repeated_robot_state_pairs)
                report = validate_lerobot_v21_dataset(root)
                self.assertTrue(report.valid, report.errors)
            finally:
                source.close()

    def test_h2_telemetry_should_be_durable_before_episode_end(self) -> None:
        """H2 streams telemetry parts instead of writing one dense NPZ only at close."""

        with tempfile.TemporaryDirectory() as directory:
            spec = _spec()
            recorder = LeRobotV21EpisodeRecorder(
                RecorderConfig(Path(directory) / "telemetry", "telemetry", "fake", 10.0, spec, session_id="session")
            )
            recorder.start()
            try:
                recorder.begin_episode(EpisodeRecordingContext(0, "episode", "task", "session", 1))
                _emit_one_recorded_frame(recorder, episode_id="episode", spec=spec, observation_id=1, step=0)
                self.assertTrue(recorder.flush(timeout=5.0))
                telemetry_path = (
                    recorder.work_root
                    / "telemetry"
                    / "chunk-000"
                    / "episode_000000"
                    / "part_000000.npz"
                )
                self.assertTrue(telemetry_path.is_file())
            finally:
                recorder.stop(finalize=False, timeout=20)

    def test_h2_recorder_should_release_matched_observation_images(self) -> None:
        """H2 prunes observation cache entries once actions are recorded."""

        with tempfile.TemporaryDirectory() as directory:
            spec = _spec()
            recorder = LeRobotV21EpisodeRecorder(
                RecorderConfig(Path(directory) / "cache", "cache", "fake", 10.0, spec, session_id="session")
            )
            image_refs: list[weakref.ReferenceType[np.ndarray]] = []
            recorder.start()
            try:
                recorder.begin_episode(EpisodeRecordingContext(0, "episode", "task", "session", 1))
                for step in range(3):
                    now_ns = 2_000_000_000 + step * 100_000_000
                    image = np.full((32, 32, 3), step, dtype=np.uint8)
                    observation = ObservationCaptured(
                        runtime_id="runtime",
                        session_id="session",
                        episode_id="episode",
                        generation_id=1,
                        monotonic_timestamp_ns=now_ns,
                        payload={
                            "observation_id": step + 1,
                            "state": np.full(spec.state_dim, step, dtype=np.float32),
                            "images": {"head": image},
                            "camera_frame_ids": {"head": step + 1},
                            "camera_timestamps_ns": {"head": now_ns},
                            "max_camera_skew_ns": 0,
                        },
                    )
                    image_refs.append(weakref.ref(observation.payload["images"]["head"]))
                    recorder.emit(observation)
                    observation = None  # type: ignore[assignment]
                    _emit_action_events(
                        recorder,
                        episode_id="episode",
                        observation_id=step + 1,
                        step=step,
                        action=np.full(spec.action_dim, 0.25 + step, dtype=np.float32),
                        now_ns=now_ns,
                    )
                self.assertTrue(recorder.flush(timeout=5.0))
                gc.collect()
                self.assertTrue(all(ref() is None for ref in image_refs))
            finally:
                recorder.stop(finalize=False, timeout=20)

    def test_h1_recorder_metadata_should_preserve_action_spec_action_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = _spec(action_mode="velocity_target")
            recorder = LeRobotV21EpisodeRecorder(
                RecorderConfig(Path(directory) / "action-mode", "action-mode", "fake", 10.0, spec, session_id="s")
            )
            recorder.start()
            recorder.begin_episode(EpisodeRecordingContext(0, "episode", "task", "s", 1))
            _emit_one_recorded_frame(recorder, episode_id="episode", spec=spec, observation_id=1, step=0)
            recorder.end_episode()
            self.assertTrue(recorder.stop(timeout=20))
            info = json.loads((Path(directory) / "action-mode" / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["mp_real"]["replay"]["action_mode"], spec.action_mode)


class HardeningPoseReplaySafetyTests(unittest.TestCase):
    def test_h0_piper_and_rm2_pose_validation_currently_block_move_to_state(self) -> None:
        piper = PiperRobot(
            PiperArm("left", _StopArm(), object(), 1.0),
            PiperArm("right", _StopArm(), object(), 1.0),
            PiperArgs(),
        )
        rm_args = Rm2Args(robot_backend="mock")
        rm2 = Rm2Robot(
            MockArm("left", rm_args.joint_dof, 1.0, np.zeros(rm_args.joint_dof, dtype=np.float32)),
            MockArm("right", rm_args.joint_dof, 1.0, np.zeros(rm_args.joint_dof, dtype=np.float32)),
            rm_args,
        )
        for robot_name, robot in (("piper", piper), ("rm2", rm2)):
            with self.subTest(robot=robot_name):
                report = robot.validate_pose_target(_target(robot.action_spec, robot_name=robot_name))
                codes = {issue.code for issue in report.issues}
                self.assertIn("workspace_validation_unavailable", codes)
                self.assertIn("joint_limit_validation_unavailable", codes)
                self.assertIn("health_validation_unavailable", codes)
                self.assertFalse(report.valid)
                with self.assertRaises(PoseValidationError):
                    report.require_valid()

    def test_h3_replay_plan_hash_should_reject_step_array_mutation(self) -> None:
        """H3 recomputes payload hashes before arming/running."""

        plan = _replay_plan()
        robot = _FakeReplayRobot()
        controller = RobotReplayController(robot, plan)
        try:
            controller.prepare()
            _wait(lambda: controller.cursor().state in {ReplayState.ARMED, ReplayState.ERROR})
            self.assertEqual(controller.cursor().state, ReplayState.ARMED, controller.cursor().message)
            with self.assertRaises(ValueError):
                plan.steps[1].target[0] += 0.25
            replacement = plan.steps[1].target.copy()
            replacement[0] += 0.25
            object.__setattr__(plan.steps[1], "target", replacement)
            with self.assertRaises(ReplayPlanStaleError):
                controller.confirm_and_start(plan.plan_hash)
        finally:
            controller.stop(wait=True, timeout=2.0)

    def test_h3_pose_plan_hash_should_reject_waypoint_array_mutation(self) -> None:
        """H3 revalidates MoveToRecordedStatePlan payload hashes."""

        robot = _FakeReplayRobot()
        plan = _pose_plan(robot)
        with self.assertRaises(ValueError):
            plan.waypoints[-1].target[0] += 0.25
        replacement = plan.waypoints[-1].target.copy()
        replacement[0] += 0.25
        object.__setattr__(plan.waypoints[-1], "target", replacement)
        controller = PoseMoveController(robot)
        try:
            with self.assertRaises(PosePlanStaleError):
                controller.start(plan)
        finally:
            controller.stop(wait=True, timeout=2.0)

    def test_h5_replay_controller_should_not_acknowledge_sent_with_nonzero_tracking_error(self) -> None:
        """H5 separates sent, feedback, and acknowledged replay states."""

        plan = _replay_plan(count=2)
        robot = _FakeReplayRobot()
        controller = RobotReplayController(robot, plan)
        controller.prepare()
        _wait(lambda: controller.cursor().state in {ReplayState.ARMED, ReplayState.ERROR})
        self.assertEqual(controller.cursor().state, ReplayState.ARMED, controller.cursor().message)
        robot.state_bias = 0.2
        controller.confirm_and_start(plan.plan_hash)
        self.assertTrue(controller.join(timeout=2.0))
        cursor = controller.cursor()
        self.assertEqual(cursor.state, ReplayState.ABORTED)
        self.assertEqual(cursor.sent_sample_index, plan.start_sample)
        self.assertIsNone(cursor.acknowledged_sample_index)

    def test_h5_replay_planner_should_not_apply_arm_limits_to_gripper_only_changes(self) -> None:
        """H5 splits arm and gripper kinematic safety semantics."""

        samples = tuple(
            RecordedSample(
                episode_index=0,
                frame_index=index,
                index=index,
                timestamp=index * 0.02,
                task_index=0,
                state=np.asarray([0.0, float(index)], dtype=np.float32),
                action=np.asarray([0.0, float(index)], dtype=np.float32),
                images={},
                telemetry={},
            )
            for index in range(2)
        )
        result = ReplayPlanner(_source(samples=samples)).plan(
            robot_name="piper",
            target_action_spec=_spec(),
            episode_index=0,
            speed_scale=1.0,
            constraints=ReplayConstraints(max_step=0.05, max_velocity=10.0, max_acceleration=1_000.0),
        )
        self.assertTrue(result.report.valid, result.report.errors)

    def test_h0_dual_arm_replay_plan_preserves_declared_arm_count(self) -> None:
        spec = _dual_arm_spec()
        samples = tuple(
            RecordedSample(
                episode_index=0,
                frame_index=index,
                index=index,
                timestamp=index * 0.02,
                task_index=0,
                state=np.asarray([index * 0.01, index * 0.01, 1.0, 1.0], dtype=np.float32),
                action=np.asarray([index * 0.01, index * 0.01, 1.0, 1.0], dtype=np.float32),
                images={},
                telemetry={},
            )
            for index in range(2)
        )
        result = ReplayPlanner(_source(spec=spec, samples=samples)).plan(
            robot_name="piper",
            target_action_spec=spec,
            episode_index=0,
            speed_scale=1.0,
            constraints=ReplayConstraints(max_step=0.1, max_velocity=10.0, max_acceleration=1_000.0),
        )
        self.assertTrue(result.report.valid, result.report.errors)
        assert result.plan is not None
        self.assertEqual(result.plan.source.arm_count, 2)


class HardeningDocsPackagingTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_h6_readme_local_doc_links_should_exist(self) -> None:
        """Expected to pass after H6 fixes README doc paths or restores linked documents."""

        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        linked_paths = {
            match.group(1)
            for match in re.finditer(r"\((docs/[^)#]+)(?:#[^)]+)?\)", readme)
        }
        missing = sorted(path for path in linked_paths if not (REPO_ROOT / path).exists())
        self.assertEqual(missing, [])

    @unittest.expectedFailure
    def test_h6_av_and_pyarrow_should_not_be_core_deployment_dependencies(self) -> None:
        """Expected to pass after H6 moves data/video extras out of core deployment deps."""

        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = tuple(
            str(item).split(">=", 1)[0].split("[", 1)[0] for item in pyproject["project"]["dependencies"]
        )
        self.assertNotIn("av", dependencies)
        self.assertNotIn("pyarrow", dependencies)


if __name__ == "__main__":
    unittest.main()
