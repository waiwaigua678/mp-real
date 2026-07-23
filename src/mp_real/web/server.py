from __future__ import annotations

# ruff: noqa: I001, N802

import argparse
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
import copy
from contextlib import contextmanager
import dataclasses
import enum
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import logging
import os
import pathlib
import secrets
import subprocess
import threading
import time
from typing import Any
import urllib.parse

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from mp_real.common.image import preprocess_image
from mp_real.common.runtime import rtc_replan_stride
from mp_real.data.view import DataViewError, DataViewSession, downsample_series
from mp_real.evaluation.baseline import (
    BaselineConfigurationConflict,
    BaselineReferenceWriter,
    BaselineService,
    BaselineStore,
)
from mp_real.evaluation.service import EvaluationConflict, EvaluationRuntimeLease, EvaluationService
from mp_real.evaluation.open_loop.jobs import OpenLoopEvaluationJobManager, OpenLoopJobState
from mp_real.evaluation.open_loop.models import (
    AlignmentMode,
    EvaluationRequestMode,
    OpenLoopEvaluationConfig,
    PredictionResultSource,
)
from mp_real.policy_client.client import PolicyClient
from mp_real.robots.base import Robot
from mp_real.robots.pose import PoseControlCapability
from mp_real.robots.piper import infer as infer_piper
from mp_real.robots.rm2 import infer as infer_rm2
from mp_real.robots.registry import create_robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.controller import ControllerAlreadyRunningError, RuntimeController
from mp_real.runtime.events import (
    CompositeRuntimeEventSink,
    InMemoryRuntimeEventSink,
    RuntimeEventHooks,
    RuntimeEventIdentity,
)
from mp_real.runtime.inference import CompositeInferenceHooks
from mp_real.runtime.startup import (
    PolicyStartupCancelled,
    PolicyStartupConfig,
    PolicyStartupCoordinator,
)
from mp_real.pose.config import load_pose_mapping_config
from mp_real.pose.controller import PoseMoveController
from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMappingConfig,
    PoseMotionConstraints,
    PoseMoveProgress,
    PoseValidationReport,
)
from mp_real.pose.validation import MoveToStateValidator
from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import (
    ReplayAcknowledgementStrategy,
    ReplayConstraints,
    ReplayMode,
    ReplayPlan,
    ReplayPlanStaleError,
    ReplaySafetyReport,
    ReplayState,
    ReplayTimingMode,
    RobotReplayCursor,
    json_safe as _replay_json_safe,
)
from mp_real.replay.planning import ReplayPlanner, build_replay_source_hash
from mp_real.replay.recording import ReplayRecordWriter, ReplayRecordingConfig
from mp_real.web.runtime import CachedFrameObservationSource, WebInferenceAdapter, WebLoopHooks
from mp_real.web.profiles import (
    PIPER_WEB_PROFILE,
    RM2_WEB_PROFILE,
    RobotWebProfile,
    clone_default_args,
    close_profile_cameras,
    get_web_profile,
)
from mp_real.web.resources import (
    ResourceLease,
    ResourceLeaseConflict,
    ResourceLeaseManager,
    ResourceRequest,
    ResourceType,
)


CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_BACKENDS = ("realsense", "v4l2", "black")
ARM_COMMAND_MODES = ("move_j", "move_js", "auto")
RM2_ARM_COMMAND_MODES = ("canfd", "follow", "movej")
RM2_CAMERA_BACKENDS = ("ros", "realsense", "black")
RM2_POLICY_JOINT_UNITS = ("rad", "deg")
RM2_POLICY_GRIPPER_UNITS = ("raw", "normalized")
RESET_LEFT_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RESET_RIGHT_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RESET_LEFT_GRIPPER = 1.0
RESET_RIGHT_GRIPPER = 1.0
DATA_VIEW_MAX_IMPORTED_ROOTS = 16
DATA_VIEW_MAX_PATH_CHARS = 4096


def _git_commit() -> str | None:
    """Best-effort source revision for immutable evaluation metadata."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


CONNECTION_CONFIG_FIELDS = {
    "runtime_mode",
    "server_url",
    "api_key",
    "left_can",
    "right_can",
    "cam_head_backend",
    "cam_left_wrist_backend",
    "cam_right_wrist_backend",
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist",
    "camera_width",
    "camera_height",
    "camera_fps",
    "camera_timeout",
    "camera_stream_fps",
    "policy_timeout",
    "policy_connect_timeout_s",
    "policy_metadata_timeout_s",
    "policy_warmup_timeout_s",
    "policy_inference_timeout_s",
    "policy_warmup_enabled",
    "policy_warmup_requests",
    "policy_prefetch_first_chunk",
    "resize_size",
    "enable_on_start",
    "enable_timeout_s",
    "reset_on_start",
    "reset_timeout_s",
    "init_left_joints",
    "init_right_joints",
    "init_left_gripper",
    "init_right_gripper",
}


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


class RuntimeMode(enum.StrEnum):
    DEPLOYMENT = "deployment"
    CAMERA_PREVIEW = "camera_preview"
    OFFLINE_REPLAY = "offline_replay"
    DATA_VIEW = "data_view"


def _pose_validation_json(report: PoseValidationReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "valid": report.valid,
        "mapping_fingerprint": report.mapping_fingerprint,
        "safety_policy": report.safety_policy,
        "safety_profile_hash": report.safety_profile_hash,
        "safety_profile": _replay_json_safe(report.safety_profile),
        "development_override": _replay_json_safe(report.development_override),
        "errors": [dataclasses.asdict(issue) for issue in report.errors],
        "issues": [dataclasses.asdict(issue) for issue in report.issues],
        "warnings": [dataclasses.asdict(issue) for issue in report.warnings],
        "unavailable_checks": [dataclasses.asdict(issue) for issue in report.unavailable_checks],
        "passed_checks": [dataclasses.asdict(issue) for issue in report.passed_checks],
    }


def _pose_plan_json(plan: MoveToRecordedStatePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    plan.require_integrity()
    return {
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "generation_id": plan.generation_id,
        "expires_at_monotonic_ns": plan.expires_at_monotonic_ns,
        "current_state": plan.current_state.values.tolist(),
        "target_state": plan.target_state.tolist(),
        "per_dimension_delta": plan.per_dimension_delta.tolist(),
        "mapped_joint_names": list(plan.mapped_joint_names),
        "unit_conversions": [dataclasses.asdict(item) for item in plan.unit_conversions],
        "expected_duration_s": plan.expected_duration_s,
        "waypoint_count": len(plan.waypoints),
        "safety_warnings": list(plan.safety_warnings),
        "safety_policy": plan.safety_policy,
        "safety_profile_hash": plan.safety_profile_hash,
        "required_confirmations": list(plan.required_confirmations),
    }


def _replay_report_json(report: ReplaySafetyReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return _replay_json_safe(report)


def _replay_plan_json(plan: ReplayPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    plan.require_integrity()
    return {
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "generation_id": plan.generation_id,
        "expires_at_monotonic_ns": plan.expires_at_monotonic_ns,
        "dataset_id": plan.dataset_id,
        "episode_index": plan.episode_index,
        "start_sample": plan.start_sample,
        "end_sample": plan.end_sample,
        "mode": plan.mode.value,
        "timing_mode": plan.timing_mode.value,
        "speed_scale": plan.speed_scale,
        "expected_duration_s": plan.expected_duration_s,
        "step_count": len(plan.steps),
        "safety_policy": plan.safety_policy,
        "safety_profile_hash": plan.safety_profile_hash,
    }


def _replay_cursor_json(cursor: RobotReplayCursor | None) -> dict[str, Any] | None:
    return _replay_json_safe(cursor) if cursor is not None else None


def _replay_progress_json(cursor: RobotReplayCursor | None) -> dict[str, Any] | None:
    if cursor is None:
        return None
    return {
        "sent": cursor.sent_progress_ratio,
        "feedback": cursor.feedback_progress_ratio,
        "acknowledged": cursor.acknowledged_progress_ratio,
        "displayed": cursor.progress_ratio,
        "planned_sample_index": cursor.planned_sample_index,
        "sent_sample_index": cursor.sent_sample_index,
        "feedback_sample_index": cursor.feedback_sample_index,
        "acknowledged_sample_index": cursor.acknowledged_sample_index,
        "displayed_sample_index": cursor.displayed_sample_index,
        "lag_samples": cursor.lag_samples,
        "acknowledgement_strategy": cursor.acknowledgement_strategy,
    }


def _replay_constraints_from_payload(payload: Mapping[str, Any]) -> ReplayConstraints:
    kwargs: dict[str, Any] = {}
    for field in (
        "min_interval_s",
        "max_interval_s",
        "max_step",
        "max_velocity",
        "max_acceleration",
        "tracking_tolerance",
        "max_tracking_error",
        "max_control_overrun_s",
        "joint_max_step",
        "joint_max_velocity",
        "joint_max_acceleration",
        "joint_tracking_error",
        "gripper_max_step",
        "gripper_settle_timeout_s",
        "gripper_tracking_threshold",
        "gripper_transition_hysteresis",
        "feedback_poll_interval_s",
        "acknowledgement_timeout_s",
        "feedback_freshness_timeout_s",
        "extreme_tracking_error",
        "plan_expiration_s",
    ):
        value = payload.get(field)
        if value is not None:
            kwargs[field] = float(value)
    for field in (
        "lower_limits",
        "upper_limits",
        "joint_lower_limits",
        "joint_upper_limits",
        "gripper_min",
        "gripper_max",
    ):
        value = payload.get(field)
        if value is not None:
            kwargs[field] = tuple(float(item) for item in value)
    for field in ("gripper_indices",):
        value = payload.get(field)
        if value is not None:
            kwargs[field] = tuple(int(item) for item in value)
    for field in ("follower_window_samples", "state_trajectory_settle_cycles", "sustained_tracking_error_limit"):
        value = payload.get(field)
        if value is not None:
            kwargs[field] = int(value)
    if payload.get("acknowledgement_strategy") is not None:
        kwargs["acknowledgement_strategy"] = ReplayAcknowledgementStrategy(str(payload["acknowledgement_strategy"]))
    if payload.get("gripper_command_mode") is not None:
        kwargs["gripper_command_mode"] = str(payload["gripper_command_mode"])
    if payload.get("poll_feedback_while_paused") is not None:
        kwargs["poll_feedback_while_paused"] = bool(payload["poll_feedback_while_paused"])
    return ReplayConstraints(**kwargs)


def _runtime_mode(value: Any) -> RuntimeMode:
    try:
        return RuntimeMode(str(value))
    except ValueError as exc:
        available = ", ".join(mode.value for mode in RuntimeMode)
        raise ApiError(f"runtime_mode must be one of: {available}") from exc


@dataclasses.dataclass(frozen=True)
class FrameSnapshot:
    image: np.ndarray | None = None
    jpeg: bytes = b""
    sequence: int = 0
    updated_at: float = 0.0
    error: str | None = None
    frame_id: int = 0
    timestamp_monotonic_ns: int = 0
    source_sequence: int | None = None
    capture_latency_ns: int | None = None


@dataclasses.dataclass
class PolicyMetrics:
    connect_latency_ms: float | None = None
    metadata_latency_ms: float | None = None
    cold_inference_latency_ms: float | None = None
    warmup_latency_ms: float | None = None
    first_live_inference_latency_ms: float | None = None
    steady_inference_latency_ms: float | None = None


@dataclasses.dataclass
class _DeploymentResources:
    """Retryable ownership for resources that must close as one deployment."""

    cameras: dict[str, infer_piper.Camera] = dataclasses.field(default_factory=dict)
    controller: RuntimeController | None = None
    robot: Robot | None = None
    client: PolicyClient | None = None
    _close_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def complete(self) -> bool:
        return not self.cameras and self.controller is None and self.robot is None and self.client is None

    def close(self) -> None:
        # ThreadingHTTPServer may dispatch repeated Disconnect requests in
        # parallel. Keep one owner in the close path so a robot SDK handle is
        # never closed twice concurrently; a later caller can still retry any
        # resource whose first close did not complete.
        with self._close_lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        errors: list[BaseException] = []
        if self.cameras:
            try:
                close_profile_cameras(self.cameras)
            except BaseException as exc:
                errors.append(exc)
            else:
                self.cameras = {}

        if self.controller is not None:
            try:
                self.controller.close()
            except BaseException as exc:
                errors.append(exc)
                if self.controller.status().closed:
                    self.controller = None
            else:
                self.controller = None
        else:
            if self.robot is not None:
                try:
                    self.robot.close()
                except BaseException as exc:
                    errors.append(exc)
                    if bool(getattr(self.robot, "close_complete", False)):
                        self.robot = None
                else:
                    self.robot = None
            if self.client is not None:
                try:
                    self.client.close()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    self.client = None

        if errors:
            raise errors[0]


class _LeasedPolicyClient:
    """Release an in-process policy lease exactly when an evaluator closes it."""

    def __init__(self, client: Any, lease: ResourceLease) -> None:
        self._client = client
        self._lease = lease
        self._closed = False
        self._close_lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        finally:
            self._lease.release()


def _default_args(*, camera_profile: str = "hardware") -> infer_piper.Args:
    args = infer_piper.Args()
    args.init_left_joints = RESET_LEFT_JOINTS
    args.init_right_joints = RESET_RIGHT_JOINTS
    args.init_left_gripper = RESET_LEFT_GRIPPER
    args.init_right_gripper = RESET_RIGHT_GRIPPER
    if camera_profile == "black":
        args.cam_head_backend = "black"
        args.cam_left_wrist_backend = "black"
        args.cam_right_wrist_backend = "black"
    return args


def _camera_masks(args: infer_piper.Args) -> dict[str, np.bool_]:
    return {
        "cam_head": np.bool_(args.cam_head_backend != "black"),
        "cam_left_wrist": np.bool_(args.cam_left_wrist_backend != "black"),
        "cam_right_wrist": np.bool_(args.cam_right_wrist_backend != "black"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _encode_jpeg_rgb(image: np.ndarray, *, quality: int = 95) -> bytes:
    image = np.asarray(image, dtype=np.uint8)
    if cv2 is not None:
        bgr = image[:, :, ::-1]
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG frame")
        return bytes(encoded)

    from PIL import Image

    with io.BytesIO() as buf:
        Image.fromarray(image).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _black_frame(size: int) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _config_to_dict(
    args: Any,
    *,
    camera_stream_fps: float,
    policy_timeout: float,
    runtime_mode: RuntimeMode = RuntimeMode.DEPLOYMENT,
    policy_connect_timeout_s: float | None = None,
    policy_metadata_timeout_s: float | None = None,
    policy_warmup_timeout_s: float = 60.0,
    policy_inference_timeout_s: float | None = None,
    policy_warmup_enabled: bool = True,
    policy_warmup_requests: int = 1,
    policy_prefetch_first_chunk: bool = True,
) -> dict[str, Any]:
    if isinstance(args, infer_rm2.Args):
        return {
            "robot": "rm2",
            "runtime_mode": runtime_mode.value,
            "server_url": args.server_url,
            "api_key": args.api_key or "",
            "prompt": args.prompt,
            "fps": args.fps,
            "replan_steps": args.replan_steps,
            "max_steps": args.max_steps,
            "dry_run": args.dry_run,
            "infer_only": args.infer_only,
            "infer_only_chunks": args.infer_only_chunks,
            "use_rtc": args.use_rtc,
            "rtc_replan_stride": args.rtc_replan_stride,
            "rtc_prefetch_steps": args.rtc_prefetch_steps,
            "rtc_exp_weight": args.rtc_exp_weight,
            "hold_last_action": args.hold_last_action,
            "log_timing": args.log_timing,
            "reset_on_start": args.reset_on_start,
            "camera_backend": args.camera_backend,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "camera_fps": args.camera_fps,
            "camera_timeout": args.camera_timeout,
            "camera_stream_fps": camera_stream_fps,
            "resize_size": args.resize_size,
            "rm_config": args.rm_config,
            "rm_sdk_lib": args.rm_sdk_lib,
            "left_ip": args.left_ip or "",
            "right_ip": args.right_ip or "",
            "arm_port": args.arm_port,
            "joint_dof": args.joint_dof,
            "policy_joint_unit": args.policy_joint_unit,
            "policy_gripper_unit": args.policy_gripper_unit,
            "speed_percent": args.speed_percent,
            "arm_command": args.arm_command,
            "rm2_arm_command": args.arm_command,
            "max_joint_step_deg": args.max_joint_step_deg,
            "max_action_step_deg": args.max_action_step_deg,
            "action_smoothing": args.action_smoothing,
            "gripper_smoothing": args.gripper_smoothing,
            "gripper_min": args.gripper_min,
            "gripper_max": args.gripper_max,
            "gripper_timeout": args.gripper_timeout,
            "async_gripper": args.async_gripper,
            "gripper_command_rate_hz": args.gripper_command_rate_hz,
            "gripper_command_deadband": args.gripper_command_deadband,
            "gripper_flush_timeout": args.gripper_flush_timeout,
            "init_left_joints": args.init_left_joints,
            "init_right_joints": args.init_right_joints,
            "init_left_gripper": args.init_left_gripper,
            "init_right_gripper": args.init_right_gripper,
            "read_gripper_state": args.read_gripper_state,
            "use_static_left_state": args.use_static_left_state,
            "static_left_joints": args.static_left_joints,
            "command_left_arm": args.command_left_arm,
            "command_right_arm": args.command_right_arm,
            "command_gripper": args.command_gripper,
            "interpolate_actions": args.interpolate_actions,
            "command_rate_hz": args.command_rate_hz,
            "command_gripper_every_step": args.command_gripper_every_step,
            "cam_left_topic": args.cam_left_topic,
            "cam_right_topic": args.cam_right_topic,
            "cam_head_topic": args.cam_head_topic,
            "cam_left_serial": args.cam_left_serial,
            "cam_right_serial": args.cam_right_serial,
            "cam_head_serial": args.cam_head_serial,
            "policy_timeout": policy_timeout,
            "policy_connect_timeout_s": policy_connect_timeout_s
            if policy_connect_timeout_s is not None
            else policy_timeout,
            "policy_metadata_timeout_s": policy_metadata_timeout_s
            if policy_metadata_timeout_s is not None
            else policy_timeout,
            "policy_warmup_timeout_s": policy_warmup_timeout_s,
            "policy_inference_timeout_s": policy_inference_timeout_s
            if policy_inference_timeout_s is not None
            else policy_timeout,
            "policy_warmup_enabled": policy_warmup_enabled,
            "policy_warmup_requests": policy_warmup_requests,
            "policy_prefetch_first_chunk": policy_prefetch_first_chunk,
        }
    return {
        "robot": "piper",
        "runtime_mode": runtime_mode.value,
        "server_url": args.server_url,
        "api_key": args.api_key or "",
        "prompt": args.prompt,
        "left_can": args.left_can,
        "right_can": args.right_can,
        "cam_head_backend": args.cam_head_backend,
        "cam_left_wrist_backend": args.cam_left_wrist_backend,
        "cam_right_wrist_backend": args.cam_right_wrist_backend,
        "cam_head": args.cam_head,
        "cam_left_wrist": args.cam_left_wrist,
        "cam_right_wrist": args.cam_right_wrist,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "camera_fps": args.camera_fps,
        "camera_timeout": args.camera_timeout,
        "camera_stream_fps": camera_stream_fps,
        "policy_timeout": policy_timeout,
        "policy_connect_timeout_s": policy_connect_timeout_s
        if policy_connect_timeout_s is not None
        else policy_timeout,
        "policy_metadata_timeout_s": policy_metadata_timeout_s
        if policy_metadata_timeout_s is not None
        else policy_timeout,
        "policy_warmup_timeout_s": policy_warmup_timeout_s,
        "policy_inference_timeout_s": policy_inference_timeout_s
        if policy_inference_timeout_s is not None
        else policy_timeout,
        "policy_warmup_enabled": policy_warmup_enabled,
        "policy_warmup_requests": policy_warmup_requests,
        "policy_prefetch_first_chunk": policy_prefetch_first_chunk,
        "fps": args.fps,
        "replan_steps": args.replan_steps,
        "max_steps": args.max_steps,
        "resize_size": args.resize_size,
        "dry_run": args.dry_run,
        "infer_only": args.infer_only,
        "infer_only_chunks": args.infer_only_chunks,
        "enable_on_start": args.enable_on_start,
        "enable_timeout_s": args.enable_timeout_s,
        "reset_on_start": args.reset_on_start,
        "reset_timeout_s": args.reset_timeout_s,
        "speed_percent": args.speed_percent,
        "arm_command": args.arm_command,
        "init_left_joints": args.init_left_joints,
        "init_right_joints": args.init_right_joints,
        "init_left_gripper": args.init_left_gripper,
        "init_right_gripper": args.init_right_gripper,
        "gripper_closed_deg": args.gripper_closed_deg,
        "gripper_open_deg": args.gripper_open_deg,
        "gripper_force": args.gripper_force,
        "max_joint_step": args.max_joint_step,
        "use_rtc": args.use_rtc,
        "rtc_replan_stride": args.rtc_replan_stride,
        "rtc_prefetch_steps": args.rtc_prefetch_steps,
        "rtc_exp_weight": args.rtc_exp_weight,
        "action_smoothing": args.action_smoothing,
        "gripper_smoothing": args.gripper_smoothing,
        "joint_deadband": args.joint_deadband,
        "max_action_step": args.max_action_step,
        "interpolate_actions": args.interpolate_actions,
        "command_rate_hz": args.command_rate_hz,
        "command_gripper_every_step": args.command_gripper_every_step,
        "hold_last_action": args.hold_last_action,
        "log_timing": args.log_timing,
    }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("max_steps must be empty or positive")
    return parsed


def _coerce_required_float(value: Any, *, field: str) -> float:
    if value is None or value == "":
        raise ApiError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{field} must be a finite number") from exc
    if not np.isfinite(parsed):
        raise ApiError(f"{field} must be a finite number")
    return parsed


def _coerce_required_int(value: Any, *, field: str) -> int:
    if value is None or value == "":
        raise ApiError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{field} must be an integer") from exc


def _coerce_float_tuple(value: Any, *, length: int, field: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    elif isinstance(value, list | tuple):
        parts = list(value)
    else:
        raise ValueError(f"{field} must be a list or space/comma separated string")

    if len(parts) != length:
        raise ValueError(f"{field} must contain {length} values")
    return tuple(float(part) for part in parts)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result.strip() else None


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None or value == "":
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _normalized_data_view_roots(roots: tuple[pathlib.Path | str, ...]) -> tuple[pathlib.Path, ...]:
    """Canonicalize and deduplicate local recording roots without requiring them to exist.

    Startup roots preserve the existing behavior: a missing mount can become
    available after the Web process starts.  Paths supplied from the browser
    are validated separately with ``strict=True`` before reaching this helper.
    """

    result: list[pathlib.Path] = []
    for root in roots:
        path = pathlib.Path(root).expanduser().resolve(strict=False)
        if path not in result:
            result.append(path)
    return tuple(result)


def _data_view_root_id(root: pathlib.Path) -> str:
    """Return a stable opaque identifier without disclosing an absolute path."""

    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
    return f"root_{digest}"


class RobotWebRuntime:
    def __init__(
        self,
        initial_args: Any | None = None,
        *,
        profile: RobotWebProfile = PIPER_WEB_PROFILE,
        policy_timeout: float = 3.0,
        robot_factory: Callable[[str, Any], Robot] = create_robot,
        policy_client_factory: Callable[[str, str | None, float], PolicyClient] | None = None,
        camera_factory: Callable[[Any], dict[str, infer_piper.Camera]] | None = None,
        resource_manager: ResourceLeaseManager | None = None,
        recorded_data_roots: tuple[pathlib.Path | str, ...] = (),
        data_view_web_roots: tuple[pathlib.Path | str, ...] = (),
        pose_mapping_config: PoseMappingConfig | None = None,
        replay_record_root: pathlib.Path | str | None = None,
        baseline_root: pathlib.Path | str = "recordings/baselines",
        open_loop_output_root: pathlib.Path | str = "open_loop_results",
    ) -> None:
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._profile = profile
        self._args = copy.deepcopy(initial_args) if initial_args is not None else clone_default_args(profile)
        self._camera_stream_fps = 10.0
        self._policy_timeout = policy_timeout
        self._policy_connect_timeout_s = policy_timeout
        self._policy_metadata_timeout_s = policy_timeout
        self._policy_warmup_timeout_s = 60.0
        self._policy_inference_timeout_s = policy_timeout
        self._policy_warmup_enabled = True
        self._policy_warmup_requests = 1
        self._policy_prefetch_first_chunk = True
        self._robot_factory = robot_factory
        self._policy_client_factory = policy_client_factory
        self._camera_factory = camera_factory or profile.create_cameras
        self._resource_manager = resource_manager or ResourceLeaseManager()
        self._resource_owner_id = f"web-{profile.robot_name}-{secrets.token_hex(8)}"
        self._resource_lease: ResourceLease | None = None
        self._evaluation_owner: str | None = None
        self._baseline_root = pathlib.Path(baseline_root).expanduser().resolve(strict=False)
        self._baseline_service = BaselineService(BaselineStore(self._baseline_root))
        self._baseline_writer: BaselineReferenceWriter | None = None
        self._evaluation_service = EvaluationService(self, terminal_sink=self._submit_baseline_reference)
        self._startup_recorded_data_roots = _normalized_data_view_roots(recorded_data_roots)
        self._data_view_web_roots = tuple(
            root
            for root in _normalized_data_view_roots(data_view_web_roots)
            if root not in self._startup_recorded_data_roots
        )
        self._recorded_data_roots = (
            *self._startup_recorded_data_roots,
            *self._data_view_web_roots,
        )
        self._recorded_data_view = DataViewSession(self._recorded_data_roots) if self._recorded_data_roots else None
        # A browser request can keep a session open after it leaves the
        # runtime lock.  Root replacement therefore retires (rather than
        # immediately closes) old sessions until their bounded viewer lease
        # count reaches zero.
        self._data_view_viewer_leases: dict[int, int] = {}
        self._retired_data_view_sessions: list[DataViewSession] = []
        self._data_view_root_dataset_counts: dict[pathlib.Path, int | None] = {
            root: None for root in self._recorded_data_roots
        }
        self._open_loop_output_root = pathlib.Path(open_loop_output_root).expanduser().resolve(strict=False)
        self._data_view_open_loop_jobs: OpenLoopEvaluationJobManager | None = None
        self._data_view_generation_id = 0
        self._data_view_session_id: str | None = None
        self._pose_mapping_config = pose_mapping_config
        self._replay_record_root = pathlib.Path(replay_record_root) if replay_record_root is not None else None
        self._pose_target = None
        self._pose_validation: PoseValidationReport | None = None
        self._pose_live_validation: PoseValidationReport | None = None
        self._pose_plan: MoveToRecordedStatePlan | None = None
        self._pose_robot: Robot | None = None
        self._pose_controller: PoseMoveController | None = None
        self._pose_lease: ResourceLease | None = None
        self._pose_connect_thread: threading.Thread | None = None
        self._pose_watch_thread: threading.Thread | None = None
        self._pose_handoff_thread: threading.Thread | None = None
        self._pose_deploy_thread: threading.Thread | None = None
        self._pose_stop_event = threading.Event()
        self._pose_generation_id = 0
        self._pose_phase = "idle"
        self._pose_error: str | None = None
        self._pose_progress: PoseMoveProgress | None = None
        self._pose_prepared = None
        self._recorded_start_context: dict[str, Any] | None = None

        # Stage-10 replay is intentionally separate from policy deployment and
        # recorded-state handoff.  Its worker owns the only motion path.
        self._replay_plan: ReplayPlan | None = None
        self._replay_report: ReplaySafetyReport | None = None
        self._replay_controller: RobotReplayController | None = None
        self._replay_robot: Robot | None = None
        self._replay_lease: ResourceLease | None = None
        self._replay_plan_thread: threading.Thread | None = None
        self._replay_connect_thread: threading.Thread | None = None
        self._replay_watch_thread: threading.Thread | None = None
        self._replay_stop_event = threading.Event()
        self._replay_generation_id = 0
        self._replay_phase = ReplayState.IDLE
        self._replay_error: str | None = None
        self._replay_view_locked = False
        self._replay_recorder: ReplayRecordWriter | None = None

        self._controller: RuntimeController | None = None
        self._pending_deployment_cleanup: _DeploymentResources | None = None
        self._loop_hooks = WebLoopHooks(
            error_callback=self._record_loop_error,
            stopped_callback=self._record_loop_stopped,
        )
        self._cameras: dict[str, infer_piper.Camera] = {}
        self._frames: dict[str, FrameSnapshot] = {
            name: FrameSnapshot(
                image=_black_frame(self._args.resize_size), jpeg=_encode_jpeg_rgb(_black_frame(self._args.resize_size))
            )
            for name in self._profile.camera_roles_for_args(self._args)
        }

        self._camera_thread: threading.Thread | None = None
        self._camera_stop_event = threading.Event()
        self._connect_thread: threading.Thread | None = None
        self._connect_stop_event = threading.Event()
        self._connect_generation_id = 0
        self._start_after_connect = False
        self._startup_thread: threading.Thread | None = None
        self._startup_stop_event = threading.Event()
        self._startup_generation_id = 0
        self._logs: deque[str] = deque(maxlen=200)

        self._runtime_mode = RuntimeMode.DEPLOYMENT
        self._policy_state = "DISCONNECTED"
        self._policy_metrics = PolicyMetrics()
        self._connected = False
        self._policy_connected = False
        self._running = False
        self._stop_requested = False
        self._phase = "idle"
        self._last_error: str | None = None
        self._server_metadata: dict[str, Any] | None = None

    @property
    def evaluation_service(self) -> EvaluationService:
        return self._evaluation_service

    @property
    def baseline_service(self) -> BaselineService:
        return self._baseline_service

    def baseline_status(self) -> dict[str, Any]:
        with self._lock:
            writer = self._baseline_writer
        return writer.status() if writer is not None else {"state": "idle", "queue_depth": 0, "queue_capacity": 16}

    def baseline_job_status(self, job_id: str) -> dict[str, Any]:
        return self._baseline_writer_or_start().job_status(job_id)

    def create_baseline(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        runtime = self._baseline_runtime_snapshot()
        return self._baseline_writer_or_start().submit_create(payload, runtime_config=runtime, git_commit=_git_commit())

    def clone_baseline(self, baseline_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        patch = payload.get("patch", {})
        if not isinstance(patch, Mapping):
            raise ApiError("patch must be a JSON object")
        return self._baseline_writer_or_start().submit_clone(
            baseline_id,
            patch,
            reason=str(payload.get("derived_reason", "")),
        )

    def create_baseline_from_evaluation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = self._evaluation_service.current()
        if snapshot is None:
            raise ApiError("No current evaluation is available to snapshot", HTTPStatus.CONFLICT)
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ApiError("name must be a string")
        return self._baseline_writer_or_start().submit_create_from_evaluation(snapshot, name=name)

    def baseline_diff(self, baseline_a: str, baseline_b: str) -> dict[str, Any]:
        return self._baseline_service.diff(baseline_a, baseline_b).to_dict()

    def compare_baselines(self, baseline_ids: tuple[str, ...]) -> dict[str, Any]:
        return self._baseline_service.compare(baseline_ids)

    def run_baseline(self, baseline_id: str) -> dict[str, Any]:
        payload = self._baseline_service.prepare_evaluation_run(
            baseline_id,
            runtime_config=self._baseline_runtime_snapshot(),
            git_commit=_git_commit(),
        )
        return self._evaluation_service.create(payload)

    def attach_open_loop_baseline(self, baseline_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result_dir = payload.get("result_dir")
        if not result_dir:
            raise ApiError("result_dir is required")
        return self._baseline_writer_or_start().submit_open_loop(baseline_id, str(result_dir))

    def _baseline_runtime_snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self.get_config()
            snapshot.update(_json_safe(self._profile.baseline_config_for_args(self._args)))
            snapshot["policy_metadata"] = _json_safe(self._server_metadata or {})
            snapshot["git_commit"] = _git_commit()
            if self._recorded_start_context is not None:
                snapshot["recorded_start"] = copy.deepcopy(self._recorded_start_context)
            return snapshot

    def shutdown_baselines(self) -> None:
        with self._lock:
            writer = self._baseline_writer
            self._baseline_writer = None
        if writer is not None:
            writer.close()

    def _baseline_writer_or_start(self) -> BaselineReferenceWriter:
        with self._lock:
            if self._baseline_writer is None:
                self._baseline_writer = BaselineReferenceWriter(self._baseline_service)
            return self._baseline_writer

    def _submit_baseline_reference(self, snapshot: Mapping[str, Any]) -> None:
        config = snapshot.get("config")
        if not isinstance(config, Mapping) or not config.get("baseline_id"):
            return
        self._baseline_writer_or_start().submit_evaluation(snapshot)

    @property
    def resource_manager(self) -> ResourceLeaseManager:
        return self._resource_manager

    def acquire_evaluation_control(self, evaluation_id: str) -> EvaluationRuntimeLease:
        """Reserve this deployment's existing controller for one evaluation."""
        with self._lock:
            self._refresh_controller_state_locked()
            if self._runtime_mode is not RuntimeMode.DEPLOYMENT:
                raise EvaluationConflict(
                    "Real-robot evaluation requires DEPLOYMENT mode; CAMERA_PREVIEW and OFFLINE_REPLAY cannot run it",
                    legal_operations=("connect",),
                )
            if self._evaluation_owner is not None:
                raise EvaluationConflict("Another evaluation already owns this deployment controller")
            if not self._connected or self._controller is None:
                raise EvaluationConflict("Connect the deployment runtime before creating an evaluation")
            if self._running or self._startup_active_locked() or self._connection_active_locked():
                raise EvaluationConflict("Stop normal deployment control before creating an evaluation")
            if self._args.infer_only:
                raise EvaluationConflict(
                    "Evaluation requires a motion-capable (non-infer-only) deployment configuration"
                )

            controller = self._controller
            args = copy.deepcopy(self._args)
            runtime_snapshot = self.get_config()
            runtime_snapshot["policy_metadata"] = _json_safe(self._server_metadata or {})
            runtime_snapshot["git_commit"] = _git_commit()
            if self._recorded_start_context is not None:
                runtime_snapshot["recorded_start"] = copy.deepcopy(self._recorded_start_context)
            action_spec_snapshot = dataclasses.asdict(controller.robot.action_spec)
            startup_config = self._policy_startup_config_locked()
            self._evaluation_owner = evaluation_id

        def make_args(prompt: str) -> Any:
            evaluation_args = copy.deepcopy(args)
            evaluation_args.prompt = prompt
            return evaluation_args

        def release() -> None:
            with self._lock:
                if self._evaluation_owner == evaluation_id:
                    self._evaluation_owner = None

        return EvaluationRuntimeLease(
            controller=controller,
            runtime_config_snapshot=runtime_snapshot,
            action_spec_snapshot=action_spec_snapshot,
            robot_name=self._profile.robot_name,
            make_adapter=lambda prompt: self._make_inference_adapter(controller.robot, make_args(prompt)),
            make_loop_config=lambda prompt: self._loop_config(make_args(prompt)),
            make_startup_config=lambda: startup_config,
            release=release,
        )

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self._lock:
            self._logs.append(line)
        logging.info(message)

    def _record_loop_error(self, error: BaseException) -> None:
        with self._lock:
            self._policy_connected = False
            self._policy_state = "ERROR"
            self._last_error = f"{type(error).__name__}: {error}"
            self._phase = "error"
            self._logs.append(f"{time.strftime('%H:%M:%S')} Inference loop failed: {self._last_error}")

    def _record_loop_stopped(self) -> None:
        with self._lock:
            self._running = False
            self._stop_requested = False
            if self._phase != "error":
                self._phase = "stopped"
                self._policy_state = "READY"
            self._logs.append(f"{time.strftime('%H:%M:%S')} Inference loop stopped")

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            config = _config_to_dict(
                copy.deepcopy(self._args),
                camera_stream_fps=self._camera_stream_fps,
                policy_timeout=self._policy_timeout,
                runtime_mode=self._runtime_mode,
                policy_connect_timeout_s=self._policy_connect_timeout_s,
                policy_metadata_timeout_s=self._policy_metadata_timeout_s,
                policy_warmup_timeout_s=self._policy_warmup_timeout_s,
                policy_inference_timeout_s=self._policy_inference_timeout_s,
                policy_warmup_enabled=self._policy_warmup_enabled,
                policy_warmup_requests=self._policy_warmup_requests,
                policy_prefetch_first_chunk=self._policy_prefetch_first_chunk,
            )
            config.update(
                {
                    "robot": self._profile.robot_name,
                    "camera_roles": list(self._profile.camera_roles_for_args(self._args)),
                    "action_spec": dataclasses.asdict(self._profile.action_spec_for_args(self._args)),
                    "capabilities": dataclasses.asdict(self._profile.capabilities),
                }
            )
            return config

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._profile.robot_name == "rm2":
            return self._update_rm2_config(payload)
        with self._lock:
            self._refresh_controller_state_locked()
            if self._pending_deployment_cleanup is not None:
                raise ApiError(
                    "Previous deployment cleanup is incomplete; retry Disconnect before changing parameters",
                    HTTPStatus.CONFLICT,
                )
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns this deployment; abort or complete it before changing configuration",
                    HTTPStatus.CONFLICT,
                )
            if not self._can_edit_config_locked():
                raise ApiError("Parameters can only be changed while the robot is not running")
            args = copy.deepcopy(self._args)
            camera_stream_fps = self._camera_stream_fps
            policy_timeout = self._policy_timeout
            runtime_mode = self._runtime_mode
            policy_connect_timeout_s = self._policy_connect_timeout_s
            policy_metadata_timeout_s = self._policy_metadata_timeout_s
            policy_warmup_timeout_s = self._policy_warmup_timeout_s
            policy_inference_timeout_s = self._policy_inference_timeout_s
            policy_warmup_enabled = self._policy_warmup_enabled
            policy_warmup_requests = self._policy_warmup_requests
            policy_prefetch_first_chunk = self._policy_prefetch_first_chunk
            protected_before = _config_to_dict(
                args,
                camera_stream_fps=camera_stream_fps,
                policy_timeout=policy_timeout,
                runtime_mode=runtime_mode,
                policy_connect_timeout_s=policy_connect_timeout_s,
                policy_metadata_timeout_s=policy_metadata_timeout_s,
                policy_warmup_timeout_s=policy_warmup_timeout_s,
                policy_inference_timeout_s=policy_inference_timeout_s,
                policy_warmup_enabled=policy_warmup_enabled,
                policy_warmup_requests=policy_warmup_requests,
                policy_prefetch_first_chunk=policy_prefetch_first_chunk,
            )

            string_fields = [
                "server_url",
                "api_key",
                "prompt",
                "left_can",
                "right_can",
                "cam_head",
                "cam_left_wrist",
                "cam_right_wrist",
            ]
            for field in string_fields:
                if field in payload:
                    value = str(payload[field])
                    setattr(args, field, value if field != "api_key" or value else None)

            for field in ("cam_head_backend", "cam_left_wrist_backend", "cam_right_wrist_backend"):
                if field in payload:
                    value = str(payload[field])
                    if value not in CAMERA_BACKENDS:
                        raise ApiError(f"{field} must be one of {CAMERA_BACKENDS}")
                    setattr(args, field, value)

            if "arm_command" in payload:
                value = str(payload["arm_command"])
                if value not in ARM_COMMAND_MODES:
                    raise ApiError(f"arm_command must be one of {ARM_COMMAND_MODES}")
                args.arm_command = value

            int_fields = [
                "camera_width",
                "camera_height",
                "camera_fps",
                "replan_steps",
                "resize_size",
                "speed_percent",
                "rtc_replan_stride",
                "rtc_prefetch_steps",
            ]
            for field in int_fields:
                if field in payload:
                    value = int(payload[field])
                    if field in {"rtc_replan_stride", "rtc_prefetch_steps"}:
                        if value < 0:
                            raise ApiError(f"{field} must be non-negative")
                    elif value <= 0:
                        raise ApiError(f"{field} must be positive")
                    setattr(args, field, value)

            float_fields = [
                "camera_timeout",
                "fps",
                "enable_timeout_s",
                "reset_timeout_s",
                "gripper_closed_deg",
                "gripper_open_deg",
                "gripper_force",
                "max_joint_step",
                "rtc_exp_weight",
                "action_smoothing",
                "gripper_smoothing",
                "joint_deadband",
                "max_action_step",
                "command_rate_hz",
            ]
            for field in float_fields:
                if field in payload:
                    value = float(payload[field])
                    if field in {"camera_timeout", "fps", "command_rate_hz"} and value <= 0:
                        raise ApiError(f"{field} must be positive")
                    if field in {"rtc_exp_weight", "joint_deadband"} and value < 0:
                        raise ApiError(f"{field} must be non-negative")
                    setattr(args, field, value)

            if "max_steps" in payload:
                args.max_steps = _coerce_optional_int(payload["max_steps"])
            if "infer_only_chunks" in payload:
                args.infer_only_chunks = int(payload["infer_only_chunks"])
                if args.infer_only_chunks <= 0:
                    raise ApiError("infer_only_chunks must be positive")

            for field in (
                "dry_run",
                "infer_only",
                "enable_on_start",
                "reset_on_start",
                "use_rtc",
                "interpolate_actions",
                "command_gripper_every_step",
                "hold_last_action",
                "log_timing",
            ):
                if field in payload:
                    setattr(args, field, _coerce_bool(payload[field]))

            for field in ("init_left_joints", "init_right_joints"):
                if field in payload:
                    setattr(args, field, _coerce_float_tuple(payload[field], length=6, field=field))
            for field in ("init_left_gripper", "init_right_gripper"):
                if field in payload:
                    setattr(args, field, float(payload[field]))

            if "camera_stream_fps" in payload:
                camera_stream_fps = float(payload["camera_stream_fps"])
                if camera_stream_fps <= 0:
                    raise ApiError("camera_stream_fps must be positive")
            if "policy_timeout" in payload:
                policy_timeout = float(payload["policy_timeout"])
                if policy_timeout <= 0:
                    raise ApiError("policy_timeout must be positive")
                policy_inference_timeout_s = policy_timeout
            if "runtime_mode" in payload:
                runtime_mode = _runtime_mode(payload["runtime_mode"])
            for field in (
                "policy_connect_timeout_s",
                "policy_metadata_timeout_s",
                "policy_warmup_timeout_s",
                "policy_inference_timeout_s",
            ):
                if field in payload:
                    value = float(payload[field])
                    if value <= 0:
                        raise ApiError(f"{field} must be positive")
                    if field == "policy_connect_timeout_s":
                        policy_connect_timeout_s = value
                    elif field == "policy_metadata_timeout_s":
                        policy_metadata_timeout_s = value
                    elif field == "policy_warmup_timeout_s":
                        policy_warmup_timeout_s = value
                    else:
                        policy_inference_timeout_s = value
                        policy_timeout = value
            if "policy_warmup_requests" in payload:
                policy_warmup_requests = int(payload["policy_warmup_requests"])
                if policy_warmup_requests <= 0:
                    raise ApiError("policy_warmup_requests must be positive")
            if "policy_warmup_enabled" in payload:
                policy_warmup_enabled = _coerce_bool(payload["policy_warmup_enabled"])
            if "policy_prefetch_first_chunk" in payload:
                policy_prefetch_first_chunk = _coerce_bool(payload["policy_prefetch_first_chunk"])

            protected_after = _config_to_dict(
                args,
                camera_stream_fps=camera_stream_fps,
                policy_timeout=policy_timeout,
                runtime_mode=runtime_mode,
                policy_connect_timeout_s=policy_connect_timeout_s,
                policy_metadata_timeout_s=policy_metadata_timeout_s,
                policy_warmup_timeout_s=policy_warmup_timeout_s,
                policy_inference_timeout_s=policy_inference_timeout_s,
                policy_warmup_enabled=policy_warmup_enabled,
                policy_warmup_requests=policy_warmup_requests,
                policy_prefetch_first_chunk=policy_prefetch_first_chunk,
            )
            if self._connected:
                changed_protected = [
                    field
                    for field in sorted(CONNECTION_CONFIG_FIELDS)
                    if protected_before.get(field) != protected_after.get(field)
                ]
                if changed_protected:
                    raise ApiError(
                        "These parameters require disconnecting first: " + ", ".join(changed_protected),
                        HTTPStatus.CONFLICT,
                    )

            if rtc_replan_stride(args) > args.replan_steps:
                raise ApiError("rtc_replan_stride must be <= replan_steps")

            self._args = args
            self._camera_stream_fps = camera_stream_fps
            self._policy_timeout = policy_timeout
            self._runtime_mode = runtime_mode
            self._policy_connect_timeout_s = policy_connect_timeout_s
            self._policy_metadata_timeout_s = policy_metadata_timeout_s
            self._policy_warmup_timeout_s = policy_warmup_timeout_s
            self._policy_inference_timeout_s = policy_inference_timeout_s
            self._policy_warmup_enabled = policy_warmup_enabled
            self._policy_warmup_requests = policy_warmup_requests
            self._policy_prefetch_first_chunk = policy_prefetch_first_chunk
            if not self._connected:
                self._reset_placeholder_frames_locked(args.resize_size)
            self.log("Updated runtime parameters")
            return self.get_config()

    def _update_rm2_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply RM2's existing CLI-shaped settings without Piper field assumptions."""
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns this deployment; abort or complete it before changing configuration",
                    HTTPStatus.CONFLICT,
                )
            if not self._can_edit_connection_config_locked():
                raise ApiError("Disconnect RM2 before changing parameters", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            for field in (
                "server_url",
                "api_key",
                "prompt",
                "left_ip",
                "right_ip",
                "cam_left_topic",
                "cam_right_topic",
                "cam_head_topic",
                "cam_left_serial",
                "cam_right_serial",
                "cam_head_serial",
            ):
                if field in payload:
                    value = str(payload[field])
                    setattr(args, field, value if field != "api_key" or value else None)
            for field in ("rm_config", "rm_sdk_lib"):
                if field in payload:
                    value = str(payload[field]).strip()
                    setattr(args, field, pathlib.Path(value).expanduser() if value else None)
            for field in (
                "fps",
                "camera_timeout",
                "action_smoothing",
                "gripper_smoothing",
                "max_joint_step_deg",
                "max_action_step_deg",
                "command_rate_hz",
                "rtc_exp_weight",
                "gripper_command_rate_hz",
                "gripper_command_deadband",
                "gripper_flush_timeout",
            ):
                if field in payload:
                    value = _coerce_required_float(payload[field], field=field)
                    if field in {
                        "fps",
                        "camera_timeout",
                        "command_rate_hz",
                    } and value <= 0:
                        raise ApiError(f"{field} must be positive")
                    if field in {
                        "max_joint_step_deg",
                        "max_action_step_deg",
                        "rtc_exp_weight",
                        "gripper_command_deadband",
                        "gripper_flush_timeout",
                    } and value < 0:
                        raise ApiError(f"{field} must be non-negative")
                    setattr(args, field, value)
            for field in (
                "replan_steps",
                "camera_width",
                "camera_height",
                "camera_fps",
                "resize_size",
                "joint_dof",
                "rtc_replan_stride",
                "rtc_prefetch_steps",
                "speed_percent",
                "gripper_min",
                "gripper_max",
                "gripper_timeout",
            ):
                if field in payload:
                    value = _coerce_required_int(payload[field], field=field)
                    if value < 0 or (
                        field not in {"rtc_replan_stride", "rtc_prefetch_steps", "gripper_timeout"}
                        and value == 0
                    ):
                        raise ApiError(f"{field} must be positive")
                    setattr(args, field, value)
            if "max_steps" in payload:
                args.max_steps = _coerce_optional_int(payload["max_steps"])
            if "infer_only_chunks" in payload:
                args.infer_only_chunks = _coerce_required_int(payload["infer_only_chunks"], field="infer_only_chunks")
                if args.infer_only_chunks <= 0:
                    raise ApiError("infer_only_chunks must be positive")
            if "camera_backend" in payload:
                value = str(payload["camera_backend"])
                if value not in RM2_CAMERA_BACKENDS:
                    raise ApiError(f"camera_backend must be one of {RM2_CAMERA_BACKENDS}")
                args.camera_backend = value
            if "policy_joint_unit" in payload:
                value = str(payload["policy_joint_unit"])
                if value not in RM2_POLICY_JOINT_UNITS:
                    raise ApiError(f"policy_joint_unit must be one of {RM2_POLICY_JOINT_UNITS}")
                args.policy_joint_unit = value
            if "policy_gripper_unit" in payload:
                value = str(payload["policy_gripper_unit"])
                if value not in RM2_POLICY_GRIPPER_UNITS:
                    raise ApiError(f"policy_gripper_unit must be one of {RM2_POLICY_GRIPPER_UNITS}")
                args.policy_gripper_unit = value
            arm_command_value = payload.get("rm2_arm_command", payload.get("arm_command"))
            if arm_command_value is not None and str(arm_command_value):
                value = str(arm_command_value)
                if value not in RM2_ARM_COMMAND_MODES:
                    raise ApiError(f"arm_command must be one of {RM2_ARM_COMMAND_MODES}")
                args.arm_command = value
            if "arm_port" in payload:
                args.arm_port = (
                    _coerce_required_int(payload["arm_port"], field="arm_port")
                    if payload["arm_port"] not in (None, "")
                    else None
                )
            for field in (
                "dry_run",
                "infer_only",
                "use_rtc",
                "reset_on_start",
                "hold_last_action",
                "log_timing",
                "read_gripper_state",
                "use_static_left_state",
                "command_left_arm",
                "command_right_arm",
                "command_gripper",
                "async_gripper",
                "interpolate_actions",
                "command_gripper_every_step",
                "profile_timing",
            ):
                if field in payload:
                    setattr(args, field, _coerce_bool(payload[field]))
            for field in ("init_left_joints", "init_right_joints", "static_left_joints"):
                if field in payload:
                    setattr(args, field, _coerce_float_tuple(payload[field], length=args.joint_dof, field=field))
            for field in ("init_left_gripper", "init_right_gripper"):
                if field in payload:
                    setattr(args, field, _coerce_required_float(payload[field], field=field))
            for field in (
                "policy_connect_timeout_s",
                "policy_metadata_timeout_s",
                "policy_warmup_timeout_s",
                "policy_inference_timeout_s",
                "policy_timeout",
                "camera_stream_fps",
            ):
                if field in payload:
                    value = _coerce_required_float(payload[field], field=field)
                    if value <= 0:
                        raise ApiError(f"{field} must be positive")
                    if field == "policy_connect_timeout_s":
                        self._policy_connect_timeout_s = value
                    elif field == "policy_metadata_timeout_s":
                        self._policy_metadata_timeout_s = value
                    elif field == "policy_warmup_timeout_s":
                        self._policy_warmup_timeout_s = value
                    elif field in {"policy_inference_timeout_s", "policy_timeout"}:
                        self._policy_inference_timeout_s = value
                        self._policy_timeout = value
                    else:
                        self._camera_stream_fps = value
            if "runtime_mode" in payload:
                self._runtime_mode = _runtime_mode(payload["runtime_mode"])
            if "policy_warmup_enabled" in payload:
                self._policy_warmup_enabled = _coerce_bool(payload["policy_warmup_enabled"])
            if "policy_warmup_requests" in payload:
                self._policy_warmup_requests = _coerce_required_int(
                    payload["policy_warmup_requests"], field="policy_warmup_requests"
                )
                if self._policy_warmup_requests <= 0:
                    raise ApiError("policy_warmup_requests must be positive")
            if "policy_prefetch_first_chunk" in payload:
                self._policy_prefetch_first_chunk = _coerce_bool(payload["policy_prefetch_first_chunk"])
            try:
                self._profile.validate_args(args)
            except ValueError as exc:
                raise ApiError(str(exc)) from exc
            self._args = args
            self._reset_placeholder_frames_locked(args.resize_size)
            self.log("Updated RM2 runtime parameters")
            return self.get_config()

    def _reset_placeholder_frames_locked(self, resize_size: int) -> None:
        placeholder = _black_frame(resize_size)
        jpeg = _encode_jpeg_rgb(placeholder)
        now = time.monotonic()
        self._frames = {
            name: FrameSnapshot(image=placeholder.copy(), jpeg=jpeg, sequence=0, updated_at=now)
            for name in self._profile.camera_roles_for_args(self._args)
        }
        self._frame_condition.notify_all()

    def _can_edit_config_locked(self) -> bool:
        return (
            not self._running
            and not self._stop_requested
            and self._pending_deployment_cleanup is None
            and not self._connection_active_locked()
            and not self._startup_active_locked()
            and self._phase not in {"connecting", "connecting_cameras", "stopping"}
        )

    def _can_edit_connection_config_locked(self) -> bool:
        return self._can_edit_config_locked() and not self._connected

    def _make_inference_adapter(self, robot: Robot, args: Any) -> WebInferenceAdapter:
        return self._profile.make_adapter(
            robot,
            args,
            lambda: self._latest_images_for_inference(args),
            self._record_inference_profile,
        )

    def _loop_config(self, args: Any) -> InferenceLoopConfig:
        return self._profile.make_loop_config(args)

    def _refresh_controller_state_locked(self) -> None:
        if self._controller is None:
            return
        controller_status = self._controller.status()
        self._running = controller_status.running
        self._stop_requested = controller_status.stop_requested
        if controller_status.error is not None:
            self._policy_connected = False
            self._policy_state = "ERROR"
            self._last_error = f"{type(controller_status.error).__name__}: {controller_status.error}"
            self._phase = "error"
        elif not controller_status.running and self._phase == "running":
            self._phase = "stopped"
            self._policy_state = "READY"

    def _startup_active_locked(self) -> bool:
        return self._startup_thread is not None and self._startup_thread.is_alive()

    def _connection_active_locked(self) -> bool:
        return self._connect_thread is not None and self._connect_thread.is_alive()

    def _record_inference_profile(self, stage: str, elapsed_s: float) -> None:
        if stage != "inference":
            return
        with self._lock:
            self._policy_metrics.steady_inference_latency_ms = elapsed_s * 1000.0

    def _create_policy_client(self, args: Any) -> PolicyClient:
        if self._policy_client_factory is not None:
            return self._policy_client_factory(args.server_url, args.api_key, self._policy_connect_timeout_s)
        return PolicyClient(
            args.server_url,
            args.api_key,
            timeout=self._policy_connect_timeout_s,
            metadata_timeout=self._policy_metadata_timeout_s,
        )

    def _policy_startup_config_locked(self) -> PolicyStartupConfig:
        return PolicyStartupConfig(
            warmup_enabled=self._policy_warmup_enabled,
            warmup_requests=self._policy_warmup_requests,
            warmup_timeout_s=self._policy_warmup_timeout_s,
            inference_timeout_s=self._policy_inference_timeout_s,
            prefetch_first_chunk=self._policy_prefetch_first_chunk,
        )

    # DATA_VIEW deliberately owns only recorded-data browsing and the
    # background open-loop worker.  It has no Robot, Camera, or PolicyClient
    # construction path.  A policy client is created only inside the existing
    # OpenLoopEvaluationJobManager worker after an explicit user submission.
    def _data_view_ready_locked(self) -> bool:
        return (
            self._runtime_mode is RuntimeMode.DATA_VIEW
            and self._connected
            and self._phase == "data_view"
            and self._recorded_data_view is not None
            and self._data_view_open_loop_jobs is not None
        )

    def _require_data_view_viewer(self) -> DataViewSession:
        """Return the read-only viewer before or after DATA_VIEW Connect.

        The iframe loads its dataset catalog immediately after the mode is
        selected.  Browsing recorded files does not need a resource lease or
        create any hardware/policy object, so it intentionally remains usable
        before the virtual DATA_VIEW connection is opened.
        """
        with self._lock:
            if self._runtime_mode is not RuntimeMode.DATA_VIEW:
                raise ApiError("Data view APIs require DATA_VIEW mode", HTTPStatus.CONFLICT)
            if self._recorded_data_view is None:
                raise ApiError(
                    "Import a recording directory in DATA_VIEW or configure --recorded-data-root at Web server startup",
                    HTTPStatus.CONFLICT,
                )
            return self._recorded_data_view

    @contextmanager
    def _data_view_viewer_lease(self) -> Iterator[DataViewSession]:
        """Keep a DataViewSession alive while one API request reads from it."""

        with self._lock:
            if self._runtime_mode is not RuntimeMode.DATA_VIEW:
                raise ApiError("Data view APIs require DATA_VIEW mode", HTTPStatus.CONFLICT)
            viewer = self._recorded_data_view
            if viewer is None:
                raise ApiError(
                    "Import a recording directory in DATA_VIEW or configure --recorded-data-root at Web server startup",
                    HTTPStatus.CONFLICT,
                )
            viewer_key = self._retain_data_view_viewer_locked(viewer)
        try:
            yield viewer
        finally:
            self._release_data_view_viewer_lease(viewer_key)

    def _retain_data_view_viewer_locked(self, viewer: DataViewSession) -> int:
        viewer_key = id(viewer)
        self._data_view_viewer_leases[viewer_key] = self._data_view_viewer_leases.get(viewer_key, 0) + 1
        return viewer_key

    def _release_data_view_viewer_lease(self, viewer_key: int) -> None:
        closable: list[DataViewSession] = []
        with self._lock:
            remaining = self._data_view_viewer_leases.get(viewer_key, 0) - 1
            if remaining > 0:
                self._data_view_viewer_leases[viewer_key] = remaining
            else:
                self._data_view_viewer_leases.pop(viewer_key, None)
            closable = self._collect_retired_data_view_sessions_locked()
        self._close_replaced_data_views(closable)

    def _collect_retired_data_view_sessions_locked(self) -> list[DataViewSession]:
        closable: list[DataViewSession] = []
        retained: list[DataViewSession] = []
        for viewer in self._retired_data_view_sessions:
            if self._data_view_viewer_leases.get(id(viewer), 0) == 0:
                closable.append(viewer)
            else:
                retained.append(viewer)
        self._retired_data_view_sessions = retained
        return closable

    def _require_data_view_ready(self) -> tuple[DataViewSession, OpenLoopEvaluationJobManager]:
        with self._lock:
            if self._runtime_mode is not RuntimeMode.DATA_VIEW:
                raise ApiError("Data view APIs require DATA_VIEW mode", HTTPStatus.CONFLICT)
            if not self._data_view_ready_locked():
                raise ApiError("Connect DATA_VIEW before browsing recorded data", HTTPStatus.CONFLICT)
            assert self._recorded_data_view is not None
            assert self._data_view_open_loop_jobs is not None
            return self._recorded_data_view, self._data_view_open_loop_jobs

    @staticmethod
    def _validate_data_view_import_path(payload: Mapping[str, Any]) -> tuple[pathlib.Path, int]:
        """Validate one explicitly submitted recording root without browsing elsewhere.

        There is intentionally no filesystem listing endpoint.  The only
        local path a browser can cause us to inspect is the exact directory it
        submits here, and validation reads only LeRobot metadata through the
        existing read-only catalog/source abstractions.
        """

        raw_path = payload.get("path")
        if not isinstance(raw_path, str):
            raise ApiError("path must be a local directory string")
        raw_path = raw_path.strip()
        if not raw_path or len(raw_path) > DATA_VIEW_MAX_PATH_CHARS or "\x00" in raw_path:
            raise ApiError("path must be a non-empty local directory")
        try:
            requested = pathlib.Path(raw_path).expanduser()
        except (OSError, RuntimeError) as exc:
            raise ApiError("path is not a valid local directory") from exc
        if not requested.is_absolute():
            raise ApiError("path must be absolute (or begin with ~)")
        try:
            root = requested.resolve(strict=True)
        except OSError as exc:
            raise ApiError("path is not an accessible local directory") from exc
        if not root.is_dir() or root == root.parent:
            raise ApiError("path must be a recording directory, not a filesystem root")

        session: DataViewSession | None = None
        try:
            session = DataViewSession((root,))
            datasets = session.datasets()
            if not datasets:
                raise ApiError("No readable LeRobot v2.1 dataset was found in the submitted directory")
            # Catalog scanning intentionally remains metadata-only.  Probe one
            # catalog entry through the source constructor as well so a stale
            # or incompatible info.json cannot be imported and fail later on
            # the first sample request.
            for dataset in datasets:
                try:
                    session.replay_source(str(dataset["dataset_id"])).get_dataset_metadata()
                except Exception:
                    continue
                return root, len(datasets)
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError("The submitted directory is not a readable LeRobot v2.1 recording root") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    logging.exception("Failed to close temporary DATA_VIEW validation session")
        raise ApiError("No compatible LeRobot v2.1 dataset was found in the submitted directory")

    def _data_view_root_summaries_locked(self) -> list[dict[str, Any]]:
        web_roots = set(self._data_view_web_roots)
        return [
            {
                "root_id": _data_view_root_id(root),
                # Do not serialize the absolute path.  A caller that imported
                # it already knows it; other Web clients only need an opaque
                # selector for removal.
                "label": root.name or "filesystem-root",
                "origin": "web" if root in web_roots else "startup",
                # Startup roots are deliberately not scanned on every status
                # poll; null means the count has not been indexed in this Web
                # process yet.  A Web-imported root has a validated count.
                "dataset_count": self._data_view_root_dataset_counts.get(root),
            }
            for root in self._recorded_data_roots
        ]

    def _data_view_open_loop_active_locked(self) -> bool:
        manager = self._data_view_open_loop_jobs
        if manager is None:
            return False
        active_states = {OpenLoopJobState.QUEUED.value, OpenLoopJobState.RUNNING.value}
        return any(job["state"] in active_states for job in manager.list_status())

    def _check_data_view_catalog_mutation_locked(self) -> None:
        if self._runtime_mode is not RuntimeMode.DATA_VIEW:
            raise ApiError("Recording directories can only be changed in DATA_VIEW mode", HTTPStatus.CONFLICT)
        if self._data_view_open_loop_active_locked():
            raise ApiError(
                "Stop the active open-loop evaluation before changing recording directories",
                HTTPStatus.CONFLICT,
            )
        if self._replay_active_locked() or self._replay_view_locked:
            raise ApiError("Stop robot trajectory replay before changing recording directories", HTTPStatus.CONFLICT)
        if self._pose_worker_active_locked() or self._pose_phase in {
            "moving",
            "stopping",
            "awaiting_move_confirmation",
        }:
            raise ApiError(
                "Finish the recorded-state pose session before changing recording directories",
                HTTPStatus.CONFLICT,
            )

    def _invalidate_data_view_replay_plan_locked(self) -> bool:
        """Discard an idle plan whose source catalog may just have changed."""

        if self._replay_plan is None:
            return False
        self._replay_plan = None
        self._replay_report = None
        self._replay_error = None
        self._replay_phase = ReplayState.IDLE
        return True

    def _replace_data_view_roots_locked(
        self, web_roots: tuple[pathlib.Path, ...]
    ) -> list[DataViewSession]:
        """Install a fresh catalog and retire its predecessor safely."""

        self._data_view_web_roots = web_roots
        self._recorded_data_roots = (
            *self._startup_recorded_data_roots,
            *self._data_view_web_roots,
        )
        old_viewer = self._recorded_data_view
        self._recorded_data_view = (
            DataViewSession(self._recorded_data_roots) if self._recorded_data_roots else None
        )
        if old_viewer is not None:
            self._retired_data_view_sessions.append(old_viewer)
        self._invalidate_data_view_replay_plan_locked()
        return self._collect_retired_data_view_sessions_locked()

    @staticmethod
    def _close_replaced_data_views(viewers: Iterable[DataViewSession]) -> None:
        for viewer in viewers:
            try:
                viewer.close()
            except Exception:
                logging.exception("Failed to close replaced DATA_VIEW catalog")

    def data_view_import_root(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Add a runtime-only recording root without constructing hardware."""

        root, dataset_count = self._validate_data_view_import_path(payload)
        retired_viewers: list[DataViewSession] = []
        with self._lock:
            self._check_data_view_catalog_mutation_locked()
            if root in self._recorded_data_roots:
                self._data_view_root_dataset_counts[root] = dataset_count
                summary = next(
                    item
                    for item in self._data_view_root_summaries_locked()
                    if item["root_id"] == _data_view_root_id(root)
                )
                return {
                    "added": False,
                    "root": {**summary, "dataset_count": dataset_count},
                    "data_view": self._data_view_status_locked(),
                }
            if any(
                root.is_relative_to(existing) or existing.is_relative_to(root)
                for existing in self._recorded_data_roots
            ):
                raise ApiError(
                    "The submitted recording directory overlaps an already configured directory",
                    HTTPStatus.CONFLICT,
                )
            if len(self._data_view_web_roots) >= DATA_VIEW_MAX_IMPORTED_ROOTS:
                raise ApiError(
                    f"At most {DATA_VIEW_MAX_IMPORTED_ROOTS} Web-imported recording directories are allowed",
                    HTTPStatus.CONFLICT,
                )
            self._data_view_root_dataset_counts[root] = dataset_count
            retired_viewers = self._replace_data_view_roots_locked((*self._data_view_web_roots, root))
            summary = next(
                item for item in self._data_view_root_summaries_locked() if item["root_id"] == _data_view_root_id(root)
            )
            result = {
                "added": True,
                "root": {**summary, "dataset_count": dataset_count},
                "data_view": self._data_view_status_locked(),
            }
        self._close_replaced_data_views(retired_viewers)
        self.log("Imported a runtime-only DATA_VIEW recording directory")
        return result

    def data_view_remove_root(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Remove one Web-imported root; startup roots remain immutable."""

        root_id = payload.get("root_id")
        if not isinstance(root_id, str) or not root_id:
            raise ApiError("root_id is required")
        retired_viewers: list[DataViewSession] = []
        with self._lock:
            self._check_data_view_catalog_mutation_locked()
            root = next((item for item in self._recorded_data_roots if _data_view_root_id(item) == root_id), None)
            if root is None:
                raise ApiError("Unknown recording directory", HTTPStatus.NOT_FOUND)
            if root not in self._data_view_web_roots:
                raise ApiError("Recording directories configured at startup cannot be removed", HTTPStatus.CONFLICT)
            remaining_web_roots = tuple(item for item in self._data_view_web_roots if item != root)
            remaining_roots = (*self._startup_recorded_data_roots, *remaining_web_roots)
            if self._connected and not remaining_roots:
                raise ApiError(
                    "Disconnect DATA_VIEW before removing its last recording directory",
                    HTTPStatus.CONFLICT,
                )
            self._data_view_root_dataset_counts.pop(root, None)
            retired_viewers = self._replace_data_view_roots_locked(remaining_web_roots)
            result = {
                "removed": True,
                "root_id": root_id,
                "data_view": self._data_view_status_locked(),
            }
        self._close_replaced_data_views(retired_viewers)
        self.log("Removed a runtime-only DATA_VIEW recording directory")
        return result

    def _data_view_status_locked(self) -> dict[str, Any]:
        manager = self._data_view_open_loop_jobs
        jobs = manager.list_status() if manager is not None else []
        active_states = {OpenLoopJobState.QUEUED.value, OpenLoopJobState.RUNNING.value}
        return {
            # ``ready`` means the iframe can browse recorded files now;
            # ``connected`` means the optional DATA_VIEW worker/lease has
            # been activated by Connect or an explicit open-loop submission.
            "ready": self._runtime_mode is RuntimeMode.DATA_VIEW and self._recorded_data_view is not None,
            "connected": self._data_view_ready_locked(),
            "session_id": self._data_view_session_id,
            "generation_id": self._data_view_generation_id,
            "dataset_roots_configured": len(self._recorded_data_roots),
            "dataset_roots": self._data_view_root_summaries_locked(),
            "web_import_root_limit": DATA_VIEW_MAX_IMPORTED_ROOTS,
            "root_persistence": "runtime_only",
            "open_loop_worker_ready": manager is not None,
            "open_loop_active": any(job["state"] in active_states for job in jobs),
            "open_loop_jobs": jobs,
        }

    def data_view_status(self) -> dict[str, Any]:
        with self._lock:
            return self._data_view_status_locked()

    def data_view_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._runtime_mode is not RuntimeMode.DATA_VIEW:
                raise ApiError("Data view APIs require DATA_VIEW mode", HTTPStatus.CONFLICT)
            # An empty catalog is a normal first-run DATA_VIEW state.  It lets
            # the iframe present its path-import UI without treating absence
            # of a startup CLI flag as an API failure.
            if self._recorded_data_view is None:
                return []
        with self._data_view_viewer_lease() as viewer:
            return viewer.datasets()

    def data_view_episodes(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.episodes(dataset_id)

    def data_view_episode_metadata(self, dataset_id: str, episode_index: int) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.episode_metadata(dataset_id, episode_index)

    def data_view_sample(self, dataset_id: str, episode_index: int, sample_index: int) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.sample(dataset_id, episode_index, sample_index)

    def data_view_sample_at_timestamp(self, dataset_id: str, episode_index: int, timestamp: float) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.sample_at_timestamp(dataset_id, episode_index, timestamp)

    def data_view_camera_frame(
        self, dataset_id: str, episode_index: int, sample_index: int, role: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.camera_frame(dataset_id, episode_index, sample_index, role)

    def data_view_curves(
        self,
        dataset_id: str,
        episode_index: int,
        *,
        series: tuple[str, ...],
        max_points: int,
    ) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.curves(dataset_id, episode_index, series=series, max_points=max_points)

    def data_view_runtime_events(self, dataset_id: str, episode_index: int, *, limit: int) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.runtime_events(dataset_id, episode_index, limit=limit)

    def data_view_metrics(self, dataset_id: str, episode_index: int) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.metrics(dataset_id, episode_index)

    def data_view_selection(self) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            return viewer.selection()

    def data_view_select(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            with self._lock:
                if self._replay_view_locked:
                    raise ApiError(
                        "Recorded playback cursor is locked while robot replay owns the session",
                        HTTPStatus.CONFLICT,
                    )
            return viewer.select(
                str(payload["dataset_id"]),
                int(payload["episode_index"]),
                int(payload["sample_index"]),
                playing=bool(payload.get("playing", False)),
                playback_rate=float(payload.get("playback_rate", 1.0)),
            )

    def data_view_submit_open_loop(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._data_view_viewer_lease() as viewer:
            with self._lock:
                if self._replay_active_locked():
                    raise ApiError(
                        "Stop robot trajectory replay before submitting open-loop evaluation",
                        HTTPStatus.CONFLICT,
                    )
                # The iframe intentionally works before the normal Web Connect
                # action.  Activating this virtual session still creates no robot,
                # camera, or policy client; it only starts the bounded job worker.
                self._activate_data_view_locked()
                assert self._data_view_open_loop_jobs is not None
                jobs = self._data_view_open_loop_jobs
                generation_id = self._data_view_generation_id
                session_id = self._data_view_session_id
            dataset_id = str(payload["dataset_id"])
            episode_index = int(payload["episode_index"])
            source = viewer.replay_source(dataset_id)
            raw_roles = payload.get("camera_roles", ())
            if not isinstance(raw_roles, list | tuple):
                raise ValueError("camera_roles must be a list when supplied")
            config = OpenLoopEvaluationConfig(
                dataset=source.get_dataset_metadata().root,
                episode_indices=(episode_index,),
                policy_url=str(payload["policy_url"]),
                policy_label=str(payload["policy_label"]),
                output_dir=self._open_loop_output_root / "pending",
                prompt_override=_optional_text(payload.get("prompt_override")),
                policy_api_key=_optional_text(payload.get("policy_api_key")),
                connection_timeout_s=float(payload.get("connection_timeout", 10.0)),
                metadata_timeout_s=float(payload.get("metadata_timeout", 10.0)),
                target_source=PredictionResultSource(str(payload.get("target_source", "action"))),
                alignment_mode=AlignmentMode(str(payload.get("alignment", "sample_index"))),
                max_timestamp_error_s=float(payload.get("max_timestamp_error", 0.05)),
                selected_camera_roles=tuple(str(role) for role in raw_roles) or None,
                request_mode=EvaluationRequestMode(str(payload.get("mode", "sequential"))),
                allow_frame_index_as_control_step=bool(payload.get("allow_frame_index_as_control_step", False)),
                limit=_optional_positive_int(payload.get("limit"), field="limit"),
            )
            job = jobs.submit(config)
            # The job ID is the durable result identity.  The DATA_VIEW session and
            # generation let the browser reject a stale result after disconnect or
            # a mode switch without changing the manager's stable artifact layout.
            job["data_view_session_id"] = session_id
            job["data_view_generation_id"] = generation_id
            return job

    def data_view_open_loop_jobs(self) -> list[dict[str, Any]]:
        with self._data_view_viewer_lease():
            with self._lock:
                jobs = self._data_view_open_loop_jobs
            return jobs.list_status() if jobs is not None else []

    def data_view_open_loop_job(self, job_id: str) -> dict[str, Any]:
        _, jobs = self._require_data_view_ready()
        return jobs.status(job_id)

    def data_view_stop_open_loop(self, job_id: str) -> dict[str, Any]:
        _, jobs = self._require_data_view_ready()
        return jobs.stop(job_id)

    def _data_view_stop_active_open_loop(self) -> None:
        with self._lock:
            manager = self._data_view_open_loop_jobs
        if manager is None:
            return
        active = {OpenLoopJobState.QUEUED.value, OpenLoopJobState.RUNNING.value}
        for job in manager.list_status():
            if job["state"] in active:
                manager.stop(str(job["job_id"]))

    def _close_data_view_open_loop_jobs(self, *, timeout: float = 10.0) -> None:
        with self._lock:
            manager = self._data_view_open_loop_jobs
        if manager is None:
            return
        try:
            manager.close(timeout=timeout)
        except TimeoutError as exc:
            raise ApiError("Open-loop worker is still stopping; retry Disconnect shortly", HTTPStatus.CONFLICT) from exc
        with self._lock:
            if self._data_view_open_loop_jobs is manager:
                self._data_view_open_loop_jobs = None
                self._data_view_session_id = None
                self._data_view_generation_id += 1

    def data_view_open_loop_report(self, job_id: str, episode_index: int, *, include_curves: bool) -> dict[str, Any]:
        _, jobs = self._require_data_view_ready()
        report_path = jobs.report_path(job_id, episode_index)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if include_curves:
            report["curves"] = self.data_view_open_loop_curves(job_id, episode_index)
        return report

    def data_view_open_loop_curves(self, job_id: str, episode_index: int) -> list[dict[str, Any]]:
        _, jobs = self._require_data_view_ready()
        prediction_path = jobs.prediction_path(job_id, episode_index)
        with np.load(prediction_path, allow_pickle=False) as archive:
            predicted = np.asarray(archive["predicted_chunks"], dtype=np.float32)
            target = np.asarray(archive["targets"], dtype=np.float32)
            valid = np.asarray(archive["valid_mask"], dtype=np.bool_)
        if predicted.ndim != 3 or target.shape != predicted.shape or valid.shape != predicted.shape[:2]:
            raise DataViewError("open-loop prediction artifact has an invalid shape")
        action_names = self._data_view_open_loop_action_names(
            jobs,
            job_id,
            episode_index,
            expected_dimensions=predicted.shape[2],
        )
        curves: list[dict[str, Any]] = []
        for dimension in range(predicted.shape[2]):
            points = valid[:, 0]
            predicted_values = np.where(points, predicted[:, 0, dimension], np.nan)
            target_values = np.where(points, target[:, 0, dimension], np.nan)
            field_name = action_names[dimension]
            # Pair IDs intentionally share the same suffix so the frontend
            # assigns prediction and recorded target for one ActionSpec field
            # the same color while retaining a stable machine-readable ID.
            curve_suffix = f"{dimension}.{field_name}"
            curves.append(
                {
                    "id": f"prediction.{curve_suffix}",
                    "label": f"prediction {field_name}",
                    "field_name": field_name,
                    "dimension": dimension,
                    "points": downsample_series(predicted_values, max_points=600),
                    "kind": "prediction",
                }
            )
            curves.append(
                {
                    "id": f"target.{curve_suffix}",
                    "label": f"target {field_name}",
                    "field_name": field_name,
                    "dimension": dimension,
                    "points": downsample_series(target_values, max_points=600),
                    "kind": "target",
                }
            )
        return curves

    @staticmethod
    def _data_view_open_loop_action_names(
        jobs: OpenLoopEvaluationJobManager,
        job_id: str,
        episode_index: int,
        *,
        expected_dimensions: int,
    ) -> tuple[str, ...]:
        """Read the persisted ActionSpec rather than assuming a robot layout."""
        report_path = jobs.report_path(job_id, episode_index)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        input_payload = report.get("input", {}) if isinstance(report, Mapping) else {}
        action_spec = input_payload.get("action_spec", {}) if isinstance(input_payload, Mapping) else {}
        raw_names = action_spec.get("action_names") if isinstance(action_spec, Mapping) else None
        if not isinstance(raw_names, list):
            raw_fields = action_spec.get("action_fields", ()) if isinstance(action_spec, Mapping) else ()
            raw_names = [item.get("name") if isinstance(item, Mapping) else None for item in raw_fields]
        names = tuple(
            (str(raw_names[index]).strip() if index < len(raw_names) and raw_names[index] is not None else "")
            or f"dim_{index}"
            for index in range(expected_dimensions)
        )
        # A malformed legacy report must still be viewable.  Fallback labels
        # are only used where the persisted ActionSpec omitted a field name.
        return names

    def replay_plan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Queue a fully offline replay preflight; this never constructs hardware."""
        dataset_id = str(payload.get("dataset_id", ""))
        episode_index = int(payload.get("episode_index", -1))
        start_sample = payload.get("start_sample")
        end_sample = payload.get("end_sample")
        try:
            mode = ReplayMode(str(payload.get("mode", ReplayMode.COMMAND_REPLAY.value)))
            timing_mode = ReplayTimingMode(str(payload.get("timing_mode", ReplayTimingMode.RECORDED_TIMESTAMPS.value)))
        except ValueError as exc:
            raise ApiError(f"invalid replay mode/timing: {exc}") from exc
        speed_scale = float(payload.get("speed_scale", 0.1))
        fps_value = payload.get("fps")
        fps = float(fps_value) if fps_value is not None else None
        constraints = _replay_constraints_from_payload(payload)
        with self._lock:
            if self._recorded_data_view is None:
                raise ApiError("No recorded-data root is configured for this Web server", HTTPStatus.CONFLICT)
            # DATA_VIEW has a virtual recorded-data connection only.  It must
            # not prevent the existing offline replay preflight from running;
            # normal deployment/camera connections still do.
            data_view_ready = self._data_view_ready_locked()
            if (
                (self._connected and not data_view_ready)
                or self._running
                or self._pose_worker_active_locked()
                or self._replay_active_locked()
                or self._pose_phase not in {"idle", "offline_preflighted", "offline_rejected", "failed", "aborted"}
            ):
                raise ApiError("Disconnect active robot control before generating a replay plan", HTTPStatus.CONFLICT)
            if self._replay_plan_thread is not None and self._replay_plan_thread.is_alive():
                raise ApiError("Replay planning is already in progress", HTTPStatus.CONFLICT)
            viewer = self._recorded_data_view
            viewer_key = self._retain_data_view_viewer_locked(viewer)
            profile = self._profile
            args = copy.deepcopy(self._args)
            safety_profile = profile.safety_profile_for_args(args)
            replay_safety_profile = (
                safety_profile
                if getattr(args, "safety_profile_path", None) is not None
                or safety_profile.hardware_motion_enabled
                or safety_profile.development_override.enabled
                else None
            )
            self._replay_generation_id += 1
            generation_id = self._replay_generation_id
            self._replay_stop_event.clear()
            self._replay_plan = None
            self._replay_report = None
            self._replay_controller = None
            self._replay_error = None
            self._replay_phase = ReplayState.PLANNING
            self._replay_view_locked = False
            worker = threading.Thread(
                target=self._run_replay_plan,
                args=(
                    viewer,
                    profile.robot_name,
                    profile.action_spec_for_args(args),
                    dataset_id,
                    episode_index,
                    int(start_sample) if start_sample is not None else None,
                    int(end_sample) if end_sample is not None else None,
                    mode,
                    timing_mode,
                    fps,
                    speed_scale,
                    constraints,
                    generation_id,
                    self._resource_owner_id,
                    replay_safety_profile,
                    viewer_key,
                ),
                name=f"{profile.robot_name}-replay-plan-g{generation_id}",
                daemon=False,
            )
            self._replay_plan_thread = worker
            worker.start()
        return self.replay_status()

    def _run_replay_plan(
        self,
        viewer: DataViewSession,
        robot_name: str,
        action_spec: Any,
        dataset_id: str,
        episode_index: int,
        start_sample: int | None,
        end_sample: int | None,
        mode: ReplayMode,
        timing_mode: ReplayTimingMode,
        fps: float | None,
        speed_scale: float,
        constraints: ReplayConstraints,
        generation_id: int,
        resource_owner_id: str,
        safety_profile: Any,
        viewer_key: int,
    ) -> None:
        try:
            try:
                result = ReplayPlanner(viewer.replay_source(dataset_id)).plan(
                    robot_name=robot_name,
                    target_action_spec=action_spec,
                    dataset_id=dataset_id,
                    episode_index=episode_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    mode=mode,
                    timing_mode=timing_mode,
                    fps=fps,
                    speed_scale=speed_scale,
                    constraints=constraints,
                    generation_id=generation_id,
                    resource_owner_id=resource_owner_id,
                    safety_profile=safety_profile,
                )
            except BaseException as exc:
                with self._lock:
                    if generation_id == self._replay_generation_id:
                        self._replay_phase = ReplayState.ERROR
                        self._replay_error = f"{type(exc).__name__}: {exc}"
                return
            with self._lock:
                if generation_id != self._replay_generation_id or self._replay_stop_event.is_set():
                    return
                self._replay_report = result.report
                self._replay_plan = result.plan
                self._replay_phase = ReplayState.VALIDATED if result.report.valid else ReplayState.ERROR
                self._replay_error = (
                    None if result.report.valid else "; ".join(item.message for item in result.report.errors)
                )
        finally:
            self._release_data_view_viewer_lease(viewer_key)

    def replay_connect(self) -> dict[str, Any]:
        """Create only the selected Robot, then let ReplayController move to start."""
        with self._lock:
            plan = self._replay_plan
            if self._replay_phase is not ReplayState.VALIDATED or plan is None:
                raise ApiError("A valid offline replay plan is required", HTTPStatus.CONFLICT)
            data_view_ready = self._data_view_ready_locked()
            if (
                (self._connected and not data_view_ready)
                or self._running
                or self._pose_worker_active_locked()
                or self._replay_active_locked()
            ):
                raise ApiError("Another robot-control lifecycle is active", HTTPStatus.CONFLICT)
            if data_view_ready and self._data_view_status_locked()["open_loop_active"]:
                raise ApiError("Stop open-loop evaluation before connecting the robot for replay", HTTPStatus.CONFLICT)
            try:
                lease = self._resource_manager.acquire(
                    self._resource_owner_id,
                    (
                        ResourceRequest(ResourceType.ROBOT_CONTROL, self._profile.robot_name),
                        ResourceRequest(ResourceType.RECORDED_DATA, plan.dataset_id),
                    ),
                )
            except ResourceLeaseConflict as exc:
                raise ApiError(str(exc), HTTPStatus.CONFLICT) from exc
            args = copy.deepcopy(self._args)
            self._replay_generation_id += 1
            generation_id = self._replay_generation_id
            try:
                plan = plan.with_integrity_context(
                    generation_id=generation_id,
                    resource_owner_id=lease.owner_id,
                    resource_lease_id=lease.lease_id,
                )
                plan.require_integrity(check_expiration=True)
            except BaseException:
                lease.release()
                raise
            self._replay_stop_event.clear()
            self._replay_lease = lease
            self._replay_plan = plan
            self._replay_phase = ReplayState.CONNECTING
            self._replay_error = None
            self._replay_view_locked = True
            worker = threading.Thread(
                target=self._run_replay_connect,
                args=(plan, args, lease, generation_id),
                name=f"{self._profile.robot_name}-replay-connect-{plan.plan_id[:8]}",
                daemon=False,
            )
            self._replay_connect_thread = worker
            worker.start()
        return self.replay_status()

    def _run_replay_connect(self, plan: ReplayPlan, args: Any, lease: ResourceLease, generation_id: int) -> None:
        robot: Robot | None = None
        recorder: ReplayRecordWriter | None = None
        try:
            plan.require_integrity(check_expiration=True)
            self._require_replay_source_integrity(plan)
            args.reset_on_start = False
            if hasattr(args, "enable_on_start"):
                args.enable_on_start = False
            if hasattr(args, "speed_percent"):
                args.speed_percent = min(int(args.speed_percent), 10)
            robot = self._robot_factory(self._profile.robot_name, args)

            def lease_valid() -> bool:
                with self._lock:
                    return (
                        generation_id == self._replay_generation_id
                        and self._replay_lease is lease
                        and not lease.released
                        and not self._replay_stop_event.is_set()
                    )

            if self._replay_record_root is not None:
                recorder = ReplayRecordWriter(ReplayRecordingConfig(self._replay_record_root), plan)
                recorder.start()
            controller = RobotReplayController(
                robot,
                plan,
                lease_valid=lease_valid,
                record_callback=recorder.emit if recorder is not None else None,
                thread_name=f"{self._profile.robot_name}-replay",
            )
            with self._lock:
                if not lease_valid():
                    raise RuntimeError("replay connection was superseded")
                self._replay_robot = robot
                self._replay_controller = controller
                self._replay_recorder = recorder
            controller.prepare()
            watcher = threading.Thread(
                target=self._watch_replay_prepare,
                args=(controller, robot, lease, generation_id),
                name=f"{self._profile.robot_name}-replay-prepare-watch-{plan.plan_id[:8]}",
                daemon=False,
            )
            with self._lock:
                self._replay_watch_thread = watcher
            watcher.start()
        except BaseException as exc:
            if recorder is not None:
                recorder.stop(result="connection_error")
            if robot is not None:
                try:
                    robot.close()
                except BaseException:
                    pass
            lease.release()
            with self._lock:
                if generation_id == self._replay_generation_id:
                    self._replay_robot = None
                    self._replay_lease = None
                    self._replay_recorder = None
                    self._replay_phase = ReplayState.ERROR
                    self._replay_error = f"{type(exc).__name__}: {exc}"
                    self._replay_view_locked = False

    def _watch_replay_prepare(
        self, controller: RobotReplayController, robot: Robot, lease: ResourceLease, generation_id: int
    ) -> None:
        controller.join()
        cursor = controller.cursor()
        if cursor.state is ReplayState.ARMED:
            with self._lock:
                if generation_id == self._replay_generation_id and self._replay_controller is controller:
                    self._replay_phase = ReplayState.ARMED
            return
        self._release_replay_resources(controller, robot, lease, generation_id)

    def replay_start(self, plan_hash: str) -> dict[str, Any]:
        with self._lock:
            controller = self._replay_controller
            robot = self._replay_robot
            lease = self._replay_lease
            generation_id = self._replay_generation_id
            if (
                controller is None
                or robot is None
                or lease is None
                or controller.cursor().state is not ReplayState.ARMED
            ):
                raise ApiError("Replay must reach ARMED before it can start", HTTPStatus.CONFLICT)
            try:
                controller.plan.require_integrity(check_expiration=True)
                controller.confirm_and_start(plan_hash)
            except BaseException as exc:
                raise ApiError(f"{type(exc).__name__}: {exc}", HTTPStatus.CONFLICT) from exc
            self._replay_phase = ReplayState.RUNNING
            watcher = threading.Thread(
                target=self._watch_replay_run,
                args=(controller, robot, lease, generation_id),
                name=f"{self._profile.robot_name}-replay-run-watch-{controller.plan.plan_id[:8]}",
                daemon=False,
            )
            self._replay_watch_thread = watcher
            watcher.start()
        return self.replay_status()

    def replay_pause(self) -> dict[str, Any]:
        with self._lock:
            controller = self._replay_controller
        if controller is None or not controller.pause():
            raise ApiError("Replay is not running", HTTPStatus.CONFLICT)
        return self.replay_status()

    def replay_resume(self) -> dict[str, Any]:
        with self._lock:
            controller = self._replay_controller
        if controller is None or not controller.resume():
            raise ApiError("Replay resume was rejected; move to the recorded pause state first", HTTPStatus.CONFLICT)
        return self.replay_status()

    def _require_replay_source_integrity(self, plan: ReplayPlan) -> None:
        viewer = self._recorded_data_view
        if viewer is None:
            return
        source = viewer.replay_source(plan.dataset_id)
        info = source.get_dataset_metadata().info
        samples = []
        targets = []
        for sample_index in range(plan.start_sample, plan.end_sample + 1):
            sample = source.get_sample(plan.episode_index, sample_index)
            target = sample.action if plan.mode is ReplayMode.COMMAND_REPLAY else sample.state
            samples.append(
                (
                    sample_index,
                    sample.frame_index,
                    float(sample.timestamp),
                    np.asarray(target, dtype=np.float32).copy(),
                    np.asarray(sample.state, dtype=np.float32).copy(),
                )
            )
            targets.append(np.asarray(target, dtype=np.float32).copy())
        current_hash = build_replay_source_hash(
            plan.dataset_id,
            info,
            plan.episode_index,
            plan.start_sample,
            plan.end_sample,
            samples,
            np.vstack(targets) if targets else np.empty((0, 0), dtype=np.float32),
        )
        if current_hash != plan.dataset_hash:
            raise ReplayPlanStaleError("source dataset changed after replay plan generation")

    def replay_stop(self, *, emergency: bool = False) -> dict[str, Any]:
        with self._lock:
            controller = self._replay_controller
            robot = self._replay_robot
            lease = self._replay_lease
            generation_id = self._replay_generation_id
            armed_without_run_worker = controller is not None and controller.cursor().state is ReplayState.ARMED
        if controller is None:
            return self.replay_status()
        controller.stop(emergency=emergency, wait=False)
        if armed_without_run_worker and robot is not None and lease is not None:
            # The prepare watcher intentionally returns while ARMED so it does
            # not wait for operator confirmation forever.  Re-install a
            # joinable watcher for the cancellation path; ownership checks in
            # _release_replay_resources make this safe if the prepare watcher
            # wins a tight stop/prepare race.
            watcher = threading.Thread(
                target=self._watch_replay_run,
                args=(controller, robot, lease, generation_id),
                name=f"{self._profile.robot_name}-replay-armed-stop-watch-{controller.plan.plan_id[:8]}",
                daemon=False,
            )
            with self._lock:
                if (
                    generation_id == self._replay_generation_id
                    and self._replay_controller is controller
                    and self._replay_robot is robot
                    and self._replay_lease is lease
                ):
                    self._replay_watch_thread = watcher
                    watcher.start()
        return self.replay_status()

    def _watch_replay_run(
        self, controller: RobotReplayController, robot: Robot, lease: ResourceLease, generation_id: int
    ) -> None:
        controller.join()
        self._release_replay_resources(controller, robot, lease, generation_id)

    def _release_replay_resources(
        self, controller: RobotReplayController, robot: Robot, lease: ResourceLease, generation_id: int
    ) -> None:
        with self._lock:
            if (
                generation_id != self._replay_generation_id
                or self._replay_controller is not controller
                or self._replay_robot is not robot
                or self._replay_lease is not lease
            ):
                return
            recorder = self._replay_recorder
            # Claim these ownership slots before close/release.  A prepare
            # watcher and an ARMED-stop watcher can otherwise race to close
            # the same real robot after a cancellation at the state boundary.
            self._replay_recorder = None
            self._replay_robot = None
            self._replay_lease = None
        if recorder is not None:
            recorder.stop(result=controller.cursor().state.value)
        try:
            robot.close()
        finally:
            lease.release()
        with self._lock:
            if generation_id == self._replay_generation_id and self._replay_controller is controller:
                self._replay_phase = controller.cursor().state
                if recorder is not None and recorder.error is not None:
                    self._replay_phase = ReplayState.ERROR
                    self._replay_error = f"ReplayRecordingError: {recorder.error}"
                else:
                    self._replay_error = controller.cursor().message if controller.error() is not None else None
                self._replay_view_locked = False

    def replay_status(self) -> dict[str, Any]:
        with self._lock:
            controller = self._replay_controller
            cursor = controller.cursor() if controller is not None else None
            state = (
                ReplayState.ERROR
                if self._replay_phase is ReplayState.ERROR and self._replay_error is not None
                else cursor.state
                if cursor is not None
                else self._replay_phase
            )
            return {
                "state": state.value,
                "error": self._replay_error,
                "view_cursor_locked": self._replay_view_locked,
                "plan": _replay_plan_json(self._replay_plan),
                "safety_report": _replay_report_json(self._replay_report),
                "cursor": _replay_cursor_json(cursor),
                "progress": _replay_progress_json(cursor),
            }

    def _replay_active_locked(self) -> bool:
        cursor = self._replay_controller.cursor() if self._replay_controller is not None else None
        if cursor is not None and cursor.state in {
            ReplayState.CONNECTING,
            ReplayState.MOVING_TO_START,
            ReplayState.ARMED,
            ReplayState.RUNNING,
            ReplayState.PAUSED,
            ReplayState.STOPPING,
        }:
            return True
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._replay_plan_thread, self._replay_connect_thread, self._replay_watch_thread)
        )

    def pose_select(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Select and preflight a recorded state without constructing hardware."""
        dataset_id = str(payload.get("dataset_id", ""))
        episode_index = int(payload.get("episode_index", -1))
        sample_index = int(payload.get("sample_index", -1))
        with self._lock:
            if self._recorded_data_view is None:
                raise ApiError("No recorded-data root is configured for this Web server", HTTPStatus.CONFLICT)
            if (
                self._connected
                or self._running
                or self._pose_robot is not None
                or self._replay_active_locked()
                or self._pose_phase not in {"idle", "offline_preflighted", "offline_rejected", "failed", "aborted"}
            ):
                raise ApiError(
                    "Disconnect deployment and finish the current pose session before selecting a sample",
                    HTTPStatus.CONFLICT,
                )
            viewer = self._recorded_data_view
            viewer_key = self._retain_data_view_viewer_locked(viewer)
            profile = self._profile
            args = copy.deepcopy(self._args)
        try:
            target = viewer.pose_target(dataset_id, episode_index, sample_index)
        finally:
            self._release_data_view_viewer_lease(viewer_key)
        validation = (
            MoveToStateValidator(
                profile.robot_name,
                profile.action_spec_for_args(args),
                mapping_config=self._pose_mapping_config,
            )
            .validate(target)
            .report
        )
        with self._lock:
            self._pose_target = target
            self._pose_validation = validation
            self._pose_live_validation = None
            self._pose_plan = None
            self._pose_phase = "offline_preflighted" if validation.valid else "offline_rejected"
            self._pose_error = None if validation.valid else "; ".join(issue.message for issue in validation.issues)
        return self.pose_status()

    def pose_connect(self) -> dict[str, Any]:
        """Queue a robot-only pose connection.  It never creates policy/cameras."""
        with self._lock:
            if self._pose_target is None or self._pose_validation is None:
                raise ApiError("Select and preflight a recorded sample first", HTTPStatus.CONFLICT)
            if not self._pose_validation.valid:
                raise ApiError(
                    "Offline pose preflight failed; an explicit mapping/configuration is required",
                    HTTPStatus.CONFLICT,
                )
            if self._connected or self._running or self._pose_worker_active_locked():
                raise ApiError("Normal deployment or another pose connection is active", HTTPStatus.CONFLICT)
            if self._replay_active_locked():
                raise ApiError("Trajectory replay owns robot control", HTTPStatus.CONFLICT)
            target = self._pose_target
            args = copy.deepcopy(self._args)
            try:
                lease = self._resource_manager.acquire(
                    self._resource_owner_id,
                    (
                        ResourceRequest(ResourceType.ROBOT_CONTROL, self._profile.robot_name),
                        ResourceRequest(ResourceType.RECORDED_DATA, target.dataset_id),
                    ),
                )
            except ResourceLeaseConflict as exc:
                raise ApiError(str(exc), HTTPStatus.CONFLICT) from exc
            self._pose_lease = lease
            self._pose_phase = "connecting_robot"
            self._pose_error = None
            self._pose_generation_id += 1
            generation_id = self._pose_generation_id
            self._pose_stop_event.clear()
            self._pose_connect_thread = threading.Thread(
                target=self._run_pose_connect,
                args=(target, args, lease, generation_id),
                name=f"{self._profile.robot_name}-pose-connect-{target.target_id[:8]}",
                daemon=False,
            )
            self._pose_connect_thread.start()
        return self.pose_status()

    def _run_pose_connect(self, target: Any, args: Any, lease: ResourceLease, generation_id: int) -> None:
        robot: Robot | None = None
        try:
            args.reset_on_start = False
            if hasattr(args, "enable_on_start"):
                args.enable_on_start = False
            if hasattr(args, "speed_percent"):
                args.speed_percent = min(int(args.speed_percent), 10)
            robot = self._robot_factory(self._profile.robot_name, args)
            self._require_pose_active(generation_id)
            if not isinstance(robot, PoseControlCapability):
                raise RuntimeError(f"{self._profile.robot_name} does not implement PoseControlCapability")
            validation = MoveToStateValidator(
                self._profile.robot_name,
                robot.action_spec,
                mapping_config=self._pose_mapping_config,
            ).validate(target)
            validation.report.require_valid()
            capability_report = robot.validate_pose_target(target)
            capability_report.require_valid()
            plan = MoveToRecordedStatePlan.build(
                target=target,
                current_state=robot.get_current_pose_state(),
                target_state=validation.values,
                gripper_indices=validation.gripper_indices,
                mapped_joint_names=validation.field_names,
                conversions=validation.mappings,
                constraints=PoseMotionConstraints(),
                safety_warnings=("vendor command speed capped at 10 percent",),
                mapping_fingerprint=validation.report.mapping_fingerprint,
                session_id=f"pose-{target.target_id[:16]}",
                generation_id=generation_id,
                resource_owner_id=lease.owner_id,
                resource_lease_id=lease.lease_id,
                safety_profile_hash=capability_report.safety_profile_hash,
                safety_policy=capability_report.safety_policy,
            )
            plan = robot.plan_move_to_state(plan)
            plan.require_integrity(check_expiration=True)
            self._require_pose_active(generation_id)
        except BaseException as exc:
            if robot is not None:
                try:
                    robot.close()
                except BaseException:
                    pass
            lease.release()
            with self._lock:
                if self._pose_lease is lease:
                    self._pose_lease = None
                if generation_id == self._pose_generation_id:
                    self._pose_phase = "aborted" if self._pose_stop_event.is_set() else "failed"
                    self._pose_error = f"{type(exc).__name__}: {exc}"
            self.log(f"Recorded-state pose connection ended: {type(exc).__name__}: {exc}")
            return
        with self._lock:
            if self._pose_lease is not lease or lease.released or not self._pose_active_locked(generation_id):
                should_close = True
            else:
                should_close = False
                self._pose_robot = robot
                self._pose_plan = plan
                self._pose_live_validation = capability_report
                self._pose_phase = "awaiting_move_confirmation"
                self._pose_error = None
        if should_close:
            robot.close()

    def pose_execute(self, plan_hash: str) -> dict[str, Any]:
        with self._lock:
            plan = self._pose_plan
            robot = self._pose_robot
            if self._pose_phase != "awaiting_move_confirmation" or plan is None or robot is None:
                raise ApiError("A connected, revalidated pose plan is required", HTTPStatus.CONFLICT)
            controller = PoseMoveController(robot, thread_name=f"{self._profile.robot_name}-pose")
            self._pose_controller = controller
            self._pose_phase = "moving"
            self._pose_progress = None
            try:
                plan.require_integrity(check_expiration=True)
                if not secrets.compare_digest(plan.plan_hash, str(plan_hash)):
                    raise ApiError("The confirmation plan hash does not match the current plan", HTTPStatus.CONFLICT)
                if not isinstance(robot, PoseControlCapability):
                    raise ApiError("Robot no longer supports recorded-state pose control", HTTPStatus.CONFLICT)
                controller.start(plan, on_progress=self._record_pose_progress)
            except BaseException as exc:
                self._pose_controller = None
                self._pose_phase = "failed"
                self._pose_error = f"{type(exc).__name__}: {exc}"
                raise ApiError(self._pose_error, HTTPStatus.CONFLICT) from exc
            watcher = threading.Thread(
                target=self._watch_pose_move,
                args=(controller, plan),
                name=f"{self._profile.robot_name}-pose-watch-{plan.plan_id[:8]}",
                daemon=False,
            )
            self._pose_watch_thread = watcher
            watcher.start()
        return self.pose_status()

    def _record_pose_progress(self, progress: PoseMoveProgress) -> None:
        with self._lock:
            if self._pose_plan is not None and progress.plan_id == self._pose_plan.plan_id:
                self._pose_progress = progress

    def _watch_pose_move(self, controller: PoseMoveController, plan: MoveToRecordedStatePlan) -> None:
        controller.join(raise_on_error=False)
        error = controller.error()
        result = controller.result()
        with self._lock:
            if self._pose_controller is not controller or self._pose_plan is not plan:
                return
            if error is not None:
                self._pose_phase = "failed"
                self._pose_error = f"{type(error).__name__}: {error}"
            elif result is not None and result.status in {"reached", "reached_with_warning"}:
                self._pose_phase = result.status
                self._pose_error = result.message
            else:
                self._pose_phase = "failed"
                self._pose_error = "pose controller ended without a verified result"

    def pose_stop(self) -> dict[str, Any]:
        with self._lock:
            controller = self._pose_controller
            if controller is None or self._pose_phase != "moving":
                return self.pose_status()
            self._pose_phase = "stopping"
        controller.stop(wait=False)
        return self.pose_status()

    def pose_prepare_deployment(self, plan_hash: str) -> dict[str, Any]:
        """Handoff a verified pose connection to cameras/policy without reset."""
        with self._lock:
            plan = self._pose_plan
            robot = self._pose_robot
            lease = self._pose_lease
            if (
                self._pose_phase not in {"reached", "reached_with_warning"}
                or plan is None
                or robot is None
                or lease is None
            ):
                raise ApiError(
                    "A reached recorded-state pose is required before deployment handoff",
                    HTTPStatus.CONFLICT,
                )
            plan.require_integrity(check_expiration=True)
            if not secrets.compare_digest(plan.plan_hash, str(plan_hash)):
                raise ApiError("The handoff plan hash does not match the reached plan", HTTPStatus.CONFLICT)
            if self._pose_handoff_thread is not None and self._pose_handoff_thread.is_alive():
                raise ApiError("Recorded-state deployment handoff is already running", HTTPStatus.CONFLICT)
            self._pose_phase = "handoff_connecting"
            self._pose_error = None
            generation_id = self._pose_generation_id
            args = copy.deepcopy(self._args)
            self._pose_handoff_thread = threading.Thread(
                target=self._run_pose_handoff,
                args=(plan, robot, lease, args, generation_id),
                name=f"{self._profile.robot_name}-pose-handoff-{plan.plan_id[:8]}",
                daemon=False,
            )
            self._pose_handoff_thread.start()
        return self.pose_status()

    def _run_pose_handoff(
        self,
        plan: MoveToRecordedStatePlan,
        robot: Robot,
        lease: ResourceLease,
        args: Any,
        generation_id: int,
    ) -> None:
        client: PolicyClient | None = None
        cameras: dict[str, infer_piper.Camera] = {}
        controller: RuntimeController | None = None
        replacement: ResourceLease | None = None
        try:
            plan.require_integrity(check_expiration=True)
            requested = (
                *self._resource_requests(RuntimeMode.DEPLOYMENT, args),
                ResourceRequest(ResourceType.RECORDED_DATA, plan.target.dataset_id),
            )
            replacement = lease.replace(requested)
            self._require_pose_active(generation_id)
            client = self._create_policy_client(args)
            self._require_pose_active(generation_id)
            cameras = self._camera_factory(args)
            self._require_pose_active(generation_id)
            adapter = self._make_inference_adapter(robot, args)
            controller = RuntimeController(
                robot,
                adapter,
                client,
                self._loop_config(args),
                hooks=self._loop_hooks,
                on_step=self._loop_hooks.on_step,
                thread_name=f"{self._profile.robot_name}-web-run",
                print_infer_only_chunks=False,
                event_sink=CompositeRuntimeEventSink(InMemoryRuntimeEventSink(), self._evaluation_service),
            )
            with self._lock:
                if self._pose_robot is not robot or self._pose_lease is not lease:
                    raise PolicyStartupCancelled("pose handoff was superseded")
                self._require_pose_active(generation_id)
                self._resource_lease = replacement
                self._pose_lease = None
                self._pose_robot = None
                self._pose_controller = None
                self._controller = controller
                self._cameras = cameras
                self._connected = True
                self._policy_connected = True
                self._policy_state = "WARMING_UP"
                self._phase = "pose_warming_up"
                self._ensure_camera_thread_locked(args, self._camera_stream_fps)
                startup_config = self._policy_startup_config_locked()
            coordinator = PolicyStartupCoordinator(
                controller.policy_client,
                adapter,
                self._loop_config(args),
                startup_config,
                hooks=self._loop_hooks,
                stop_requested=lambda: self._pose_cancelled(generation_id),
            )
            prepared = coordinator.prepare()
            with self._lock:
                if self._controller is not controller:
                    raise PolicyStartupCancelled("pose handoff was superseded")
                self._require_pose_active(generation_id)
                self._pose_prepared = prepared
                self._pose_phase = "awaiting_deployment_confirmation"
                self._phase = "pose_ready_to_deploy"
                self._policy_state = "READY"
        except BaseException as exc:
            if cameras:
                close_profile_cameras(cameras)
            if controller is not None:
                controller.close()
            else:
                if client is not None:
                    client.close()
                if robot is not None:
                    robot.close()
            if replacement is not None:
                replacement.release()
            elif not lease.released:
                lease.release()
            with self._lock:
                self._controller = None
                self._cameras = {}
                self._resource_lease = None
                self._pose_lease = None
                self._pose_robot = None
                self._connected = False
                self._policy_connected = False
                self._pose_phase = "aborted" if self._pose_cancelled(generation_id) else "failed"
                self._pose_error = f"{type(exc).__name__}: {exc}"
                self._phase = "error"
                self._policy_state = "ERROR"
            self.log(f"Recorded-state deployment handoff failed: {type(exc).__name__}: {exc}")

    def pose_start_deployment(self, plan_hash: str) -> dict[str, Any]:
        """After a second confirmation, request a new live first chunk and start."""
        with self._lock:
            plan = self._pose_plan
            controller = self._controller
            if self._pose_phase != "awaiting_deployment_confirmation" or plan is None or controller is None:
                raise ApiError(
                    "Recorded-state policy warmup must finish before deployment can start",
                    HTTPStatus.CONFLICT,
                )
            plan.require_integrity(check_expiration=True)
            if not secrets.compare_digest(plan.plan_hash, str(plan_hash)):
                raise ApiError("The deployment plan hash does not match the reached plan", HTTPStatus.CONFLICT)
            current = controller.robot.read_state()
            tracking_error = float(np.max(np.abs(current.values - plan.target_state)))
            if tracking_error > plan.constraints.tracking_tolerance:
                raise ApiError(
                    "Robot pose changed after handoff; generate and revalidate a new move plan",
                    HTTPStatus.CONFLICT,
                )
            if self._pose_deploy_thread is not None and self._pose_deploy_thread.is_alive():
                raise ApiError("Recorded-state deployment start is already running", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            baseline_sequences = {name: frame.sequence for name, frame in self._frames.items()}
            self._pose_phase = "prefetching_fresh_live_chunk"
            generation_id = self._pose_generation_id
            self._pose_deploy_thread = threading.Thread(
                target=self._run_pose_deployment_start,
                args=(plan, controller, args, baseline_sequences, tracking_error, generation_id),
                name=f"{self._profile.robot_name}-pose-deploy-{plan.plan_id[:8]}",
                daemon=False,
            )
            self._pose_deploy_thread.start()
        return self.pose_status()

    def _run_pose_deployment_start(
        self,
        plan: MoveToRecordedStatePlan,
        controller: RuntimeController,
        args: Any,
        baseline_sequences: Mapping[str, int],
        tracking_error: float,
        generation_id: int,
    ) -> None:
        try:
            self._wait_for_fresh_pose_frames(args, baseline_sequences, generation_id)
            adapter = self._make_inference_adapter(controller.robot, args)
            startup_config = dataclasses.replace(self._policy_startup_config_locked(), warmup_enabled=False)
            prepared = PolicyStartupCoordinator(
                controller.policy_client,
                adapter,
                self._loop_config(args),
                startup_config,
                hooks=self._loop_hooks,
                stop_requested=lambda: self._pose_cancelled(generation_id),
            ).prepare()
            with self._lock:
                if self._controller is not controller or self._pose_plan is not plan:
                    return
                self._require_pose_active(generation_id)
                self._recorded_start_context = {
                    "source_dataset": plan.target.dataset_id,
                    "source_episode_index": plan.target.episode_index,
                    "source_sample_index": plan.target.sample_index,
                    "started_from_recorded_state": True,
                    "move_plan_id": plan.plan_id,
                    "target_tracking_error": tracking_error,
                }
                controller.configure_event_identity(session_id=plan.session_id, episode_id=None)
                controller.configure(
                    adapter,
                    self._loop_config(args),
                    hooks=self._loop_hooks,
                    on_step=self._loop_hooks.on_step,
                    initial_chunk=prepared.initial_chunk,
                    initial_provenance=prepared.initial_provenance,
                )
                self._running = True
                self._phase = "running"
                self._policy_state = "RUNNING"
                self._pose_phase = "deploying"
            controller.start()
        except BaseException as exc:
            with self._lock:
                if self._controller is controller:
                    self._running = False
                    self._phase = "pose_ready_to_deploy"
                    self._policy_state = "READY"
                    self._pose_phase = "awaiting_deployment_confirmation"
                    self._pose_error = f"{type(exc).__name__}: {exc}"

    def _wait_for_fresh_pose_frames(
        self, args: Any, baseline_sequences: Mapping[str, int], generation_id: int
    ) -> None:
        deadline = time.monotonic() + args.camera_timeout
        roles = self._profile.camera_roles_for_args(args)
        with self._frame_condition:
            while any(self._frames[name].sequence <= baseline_sequences.get(name, -1) for name in roles):
                self._require_pose_active(generation_id)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No fresh real camera frames arrived after recorded-state handoff")
                self._frame_condition.wait(min(0.05, remaining))

    def pose_status(self) -> dict[str, Any]:
        with self._lock:
            target = self._pose_target
            plan = self._pose_plan
            progress = self._pose_progress
            return {
                "phase": self._pose_phase,
                "error": self._pose_error,
                "target": {
                    "dataset_id": target.dataset_id,
                    "episode_index": target.episode_index,
                    "sample_index": target.sample_index,
                    "robot_name": target.robot_name,
                    "state_schema": list(target.state_schema),
                    "state_values": target.state_values.tolist(),
                    "joint_unit": target.joint_unit,
                    "target_id": target.target_id,
                }
                if target is not None
                else None,
                "offline_validation": _pose_validation_json(self._pose_validation),
                "live_validation": _pose_validation_json(self._pose_live_validation),
                "plan": _pose_plan_json(plan),
                "progress": {
                    "waypoint_index": progress.waypoint_index,
                    "waypoint_count": progress.waypoint_count,
                    "tracking_error": progress.tracking_error,
                    "timestamp_monotonic_ns": progress.monotonic_timestamp_ns,
                }
                if progress is not None
                else None,
            }

    def _pose_connect_active_locked(self) -> bool:
        return self._pose_connect_thread is not None and self._pose_connect_thread.is_alive()

    def _pose_worker_active_locked(self) -> bool:
        return any(
            thread is not None and thread.is_alive()
            for thread in (
                self._pose_connect_thread,
                self._pose_watch_thread,
                self._pose_handoff_thread,
                self._pose_deploy_thread,
            )
        )

    def _pose_active_locked(self, generation_id: int) -> bool:
        return generation_id == self._pose_generation_id and not self._pose_stop_event.is_set()

    def _pose_cancelled(self, generation_id: int) -> bool:
        with self._lock:
            return not self._pose_active_locked(generation_id)

    def _require_pose_active(self, generation_id: int) -> None:
        if self._pose_cancelled(generation_id):
            raise PolicyStartupCancelled("recorded-state pose session was cancelled")

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._pending_deployment_cleanup is not None:
                raise ApiError(
                    "Previous deployment cleanup is incomplete; retry Disconnect before starting",
                    HTTPStatus.CONFLICT,
                )
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns robot control; use the evaluation API", HTTPStatus.CONFLICT)
            if self._replay_active_locked():
                raise ApiError(
                    "Trajectory replay owns robot control; stop it before starting deployment", HTTPStatus.CONFLICT
                )
            mode = self._runtime_mode
            if mode is RuntimeMode.CAMERA_PREVIEW:
                if self._connected:
                    return self.status()
                args = copy.deepcopy(self._args)
            elif mode is RuntimeMode.OFFLINE_REPLAY:
                if not self._connected:
                    self._connect_offline_replay_locked()
                return self.status()
            elif mode is RuntimeMode.DATA_VIEW:
                if not self._connected:
                    self._connect_data_view_locked()
                return self.status()
            elif self._running or self._stop_requested or self._startup_active_locked():
                raise ApiError("Runtime is already running or stopping", HTTPStatus.CONFLICT)
            elif self._connection_active_locked():
                self._start_after_connect = True
                return self.status()
            else:
                args = copy.deepcopy(self._args)

        if mode is RuntimeMode.CAMERA_PREVIEW:
            self._connect_camera_preview(args)
            return self.status()

        if not self._connected:
            with self._lock:
                self._begin_deployment_connect_locked(args, start_after_connect=True)
            return self.status()

        with self._lock:
            self._refresh_controller_state_locked()
            controller = self._controller
            if controller is None:
                raise ApiError("Runtime is not fully connected", HTTPStatus.CONFLICT)
            self._stop_requested = False
            self._phase = "warming_up"
            self._policy_state = "WARMING_UP"
            self._last_error = None
            self._ensure_camera_thread_locked(args, self._camera_stream_fps)
            self._start_policy_worker_locked(controller, args)
            self.log("Started policy warmup")
            return self.status()

    def connect(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._pending_deployment_cleanup is not None:
                raise ApiError(
                    "Previous deployment cleanup is incomplete; retry Disconnect before connecting",
                    HTTPStatus.CONFLICT,
                )
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns this deployment; it must finish before reconnecting",
                    HTTPStatus.CONFLICT,
                )
            if self._replay_active_locked():
                raise ApiError(
                    "Trajectory replay owns robot control; stop it before connecting deployment", HTTPStatus.CONFLICT
                )
            if self._running or self._stop_requested or self._startup_active_locked():
                raise ApiError("Runtime is running or stopping", HTTPStatus.CONFLICT)
            if self._connected:
                return self.status()
            if self._connection_active_locked():
                return self.status()
            args = copy.deepcopy(self._args)
            mode = self._runtime_mode

        if mode is RuntimeMode.CAMERA_PREVIEW:
            self._connect_camera_preview(args)
        elif mode is RuntimeMode.OFFLINE_REPLAY:
            with self._lock:
                self._connect_offline_replay_locked()
        elif mode is RuntimeMode.DATA_VIEW:
            with self._lock:
                self._connect_data_view_locked()
        else:
            with self._lock:
                self._begin_deployment_connect_locked(args, start_after_connect=False)
        return self.status()

    def _resource_requests(self, mode: RuntimeMode, args: Any) -> tuple[ResourceRequest, ...]:
        robot_scope = self._profile.robot_name
        camera_scope = f"{robot_scope}:{','.join(self._profile.camera_roles_for_args(args))}"
        if mode is RuntimeMode.DEPLOYMENT:
            return (
                ResourceRequest(ResourceType.ROBOT_CONTROL, robot_scope),
                ResourceRequest(ResourceType.CAMERAS, camera_scope),
                ResourceRequest(ResourceType.POLICY_CLIENT, str(args.server_url)),
            )
        if mode is RuntimeMode.CAMERA_PREVIEW:
            return (ResourceRequest(ResourceType.CAMERAS, camera_scope),)
        if mode is RuntimeMode.DATA_VIEW:
            return (ResourceRequest(ResourceType.RECORDED_DATA, f"{robot_scope}:data-view"),)
        return (ResourceRequest(ResourceType.RECORDED_DATA, robot_scope),)

    def _acquire_resources_locked(self, mode: RuntimeMode, args: Any) -> None:
        if self._resource_lease is not None and not self._resource_lease.released:
            return
        try:
            self._resource_lease = self._resource_manager.acquire(
                self._resource_owner_id,
                self._resource_requests(mode, args),
            )
        except ResourceLeaseConflict as exc:
            raise ApiError(str(exc), HTTPStatus.CONFLICT) from exc

    def _release_resources_locked(self) -> None:
        if self._resource_lease is not None:
            self._resource_lease.release()
            self._resource_lease = None

    def _begin_deployment_connect_locked(self, args: Any, *, start_after_connect: bool) -> None:
        if self._pending_deployment_cleanup is not None:
            raise ApiError(
                "Previous deployment cleanup is incomplete; retry Disconnect before connecting",
                HTTPStatus.CONFLICT,
            )
        if self._connection_active_locked():
            self._start_after_connect = self._start_after_connect or start_after_connect
            return
        self._acquire_resources_locked(RuntimeMode.DEPLOYMENT, args)
        self._connect_generation_id += 1
        generation_id = self._connect_generation_id
        self._connect_stop_event = threading.Event()
        self._start_after_connect = start_after_connect
        self._phase = "connecting"
        self._policy_state = "CONNECTING"
        self._last_error = None
        self._connect_thread = threading.Thread(
            target=self._run_deployment_connect,
            args=(generation_id, self._connect_stop_event, copy.deepcopy(args)),
            name=f"{self._profile.robot_name}-web-connect-g{generation_id}",
            daemon=False,
        )
        self._connect_thread.start()

    def _run_deployment_connect(
        self,
        generation_id: int,
        stop_event: threading.Event,
        args: Any,
    ) -> None:
        client: PolicyClient | None = None
        robot: Robot | None = None
        cameras: dict[str, infer_piper.Camera] = {}
        controller: RuntimeController | None = None
        try:
            if self._profile.initialize_cameras_before_robot:
                self.log(f"Connecting policy server {args.server_url}")
                client = self._create_policy_client(args)
                if stop_event.is_set():
                    raise PolicyStartupCancelled("Deployment connection was cancelled")
                self.log("Connected policy server")
                self.log(f"Connecting {self._profile.robot_name} cameras")
                cameras = self._camera_factory(args)
                if stop_event.is_set():
                    raise PolicyStartupCancelled("Deployment connection was cancelled")

            self.log(f"Connecting {self._profile.robot_name} robot")
            robot = self._robot_factory(self._profile.robot_name, args)
            if stop_event.is_set():
                raise PolicyStartupCancelled("Deployment connection was cancelled")
            if not args.infer_only:
                robot.reset()
            if stop_event.is_set():
                raise PolicyStartupCancelled("Deployment connection was cancelled")

            if not self._profile.initialize_cameras_before_robot:
                self.log(f"Connecting policy server {args.server_url}")
                client = self._create_policy_client(args)
                if stop_event.is_set():
                    raise PolicyStartupCancelled("Deployment connection was cancelled")
                self.log("Connected policy server")
                self.log(f"Connecting {self._profile.robot_name} cameras")
                cameras = self._camera_factory(args)
                if stop_event.is_set():
                    raise PolicyStartupCancelled("Deployment connection was cancelled")

            assert client is not None
            adapter = self._make_inference_adapter(robot, args)
            controller = RuntimeController(
                robot,
                adapter,
                client,
                self._loop_config(args),
                hooks=self._loop_hooks,
                on_step=self._loop_hooks.on_step,
                thread_name=f"{self._profile.robot_name}-web-run",
                print_infer_only_chunks=False,
                event_sink=CompositeRuntimeEventSink(InMemoryRuntimeEventSink(), self._evaluation_service),
            )
        except BaseException as exc:
            resources = _DeploymentResources(
                cameras=cameras,
                controller=controller,
                robot=robot if controller is None else None,
                client=client if controller is None else None,
            )
            cleanup_error: BaseException | None = None
            try:
                resources.close()
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
            with self._lock:
                if not resources.complete:
                    self._pending_deployment_cleanup = resources
                if generation_id != self._connect_generation_id:
                    return
                self._connected = False
                self._policy_connected = False
                if not resources.complete:
                    self._policy_state = "ERROR"
                    self._phase = "cleanup_failed"
                    detail = f"{type(cleanup_error).__name__}: {cleanup_error}"
                    self._last_error = f"Connect failed and cleanup is incomplete: {detail}"
                    self._stop_requested = False
                    self._start_after_connect = False
                elif isinstance(exc, PolicyStartupCancelled) and cleanup_error is None:
                    self._policy_state = "DISCONNECTED"
                    self._phase = "idle"
                    self._last_error = None
                    self._stop_requested = False
                    self._start_after_connect = False
                else:
                    self._policy_state = "ERROR"
                    self._phase = "error"
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    if cleanup_error is not None:
                        self._last_error += f"; cleanup reported {type(cleanup_error).__name__}: {cleanup_error}"
                if resources.complete:
                    self._release_resources_locked()
            self.log(f"Connect failed: {type(exc).__name__}: {exc}")
            if cleanup_error is not None:
                self.log(f"Connect cleanup reported: {type(cleanup_error).__name__}: {cleanup_error}")
            return

        with self._lock:
            if generation_id != self._connect_generation_id or stop_event.is_set():
                should_close = True
            else:
                should_close = False
                start_after_connect = self._start_after_connect
                self._controller = controller
                self._cameras = cameras
                self._server_metadata = getattr(client, "metadata", {})
                self._policy_metrics = PolicyMetrics(
                    connect_latency_ms=getattr(client, "connect_latency_ms", None),
                    metadata_latency_ms=getattr(client, "metadata_latency_ms", None),
                )
                self._connected = True
                self._policy_connected = True
                self._policy_state = "CONNECTED"
                self._phase = "stopped"
                self._ensure_camera_thread_locked(args, self._camera_stream_fps)
                if start_after_connect:
                    self._stop_requested = False
                    self._phase = "warming_up"
                    self._policy_state = "WARMING_UP"
                    self._start_policy_worker_locked(controller, args)
        if should_close:
            resources = _DeploymentResources(cameras=cameras, controller=controller)
            try:
                resources.close()
            except BaseException as exc:
                with self._lock:
                    if not resources.complete:
                        self._pending_deployment_cleanup = resources
                        self._phase = "cleanup_failed"
                        self._last_error = f"Deployment cancellation cleanup failed: {type(exc).__name__}: {exc}"
                self.log(f"Deployment cancellation cleanup reported: {type(exc).__name__}: {exc}")
            return
        self.log(f"Connected {self._profile.robot_name} runtime")

    def _connect_camera_preview(self, args: Any) -> None:
        with self._lock:
            self._acquire_resources_locked(RuntimeMode.CAMERA_PREVIEW, args)
            self._phase = "connecting_cameras"
            self._last_error = None
            self._policy_state = "DISCONNECTED"
        cameras: dict[str, infer_piper.Camera] = {}
        try:
            cameras = self._camera_factory(args)
        except Exception as exc:
            if cameras:
                close_profile_cameras(cameras)
            with self._lock:
                self._phase = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._release_resources_locked()
            raise ApiError(str(exc), HTTPStatus.BAD_GATEWAY) from exc
        with self._lock:
            self._cameras = cameras
            self._connected = True
            self._policy_connected = False
            self._phase = "previewing"
            self._ensure_camera_thread_locked(args, self._camera_stream_fps)
        self.log("Started camera-only preview")

    def _connect_offline_replay_locked(self) -> None:
        self._acquire_resources_locked(RuntimeMode.OFFLINE_REPLAY, self._args)
        self._connected = True
        self._policy_connected = False
        self._policy_state = "DISCONNECTED"
        self._phase = "offline_replay"
        self._last_error = None
        self.log("Offline replay mode is ready; playback is scheduled for stage 7")

    def _make_data_view_policy_client(
        self, server_url: str, api_key: str | None, timeout: float, metadata_timeout: float
    ) -> _LeasedPolicyClient:
        """Create a policy client for one open-loop job without claiming a robot."""
        owner_id = f"{self._resource_owner_id}-open-loop-{secrets.token_hex(8)}"
        try:
            lease = self._resource_manager.acquire(
                owner_id,
                (ResourceRequest(ResourceType.POLICY_CLIENT, str(server_url)),),
            )
        except ResourceLeaseConflict as exc:
            raise RuntimeError(f"open-loop policy resource is unavailable: {exc}") from exc
        try:
            if self._policy_client_factory is not None:
                client = self._policy_client_factory(server_url, api_key, timeout)
            else:
                client = PolicyClient(server_url, api_key, timeout=timeout, metadata_timeout=metadata_timeout)
        except BaseException:
            lease.release()
            raise
        return _LeasedPolicyClient(client, lease)

    def _connect_data_view_locked(self) -> None:
        if self._recorded_data_view is None:
            raise ApiError(
                "Import a recording directory in DATA_VIEW before connecting",
                HTTPStatus.CONFLICT,
            )
        self._activate_data_view_locked()

    def _activate_data_view_locked(self) -> None:
        """Activate the optional DATA_VIEW worker/lease without hardware."""
        if self._recorded_data_view is None:
            raise ApiError(
                "Import a recording directory in DATA_VIEW before connecting",
                HTTPStatus.CONFLICT,
            )
        if self._data_view_ready_locked():
            return
        self._acquire_resources_locked(RuntimeMode.DATA_VIEW, self._args)
        if self._data_view_open_loop_jobs is None:
            try:
                self._data_view_open_loop_jobs = OpenLoopEvaluationJobManager(
                    self._open_loop_output_root,
                    policy_factory=self._make_data_view_policy_client,
                )
            except BaseException:
                self._release_resources_locked()
                raise
            self._data_view_generation_id += 1
            self._data_view_session_id = f"data-view-{secrets.token_hex(8)}"
        self._connected = True
        self._policy_connected = False
        self._policy_state = "DISCONNECTED"
        self._phase = "data_view"
        self._last_error = None
        self.log("DATA_VIEW is ready; no robot, camera, or policy client was created")

    def _start_policy_worker_locked(self, controller: RuntimeController, args: Any) -> None:
        self._startup_generation_id += 1
        generation_id = self._startup_generation_id
        stop_event = threading.Event()
        self._startup_stop_event = stop_event
        self._startup_thread = threading.Thread(
            target=self._run_policy_startup,
            args=(generation_id, stop_event, controller, copy.deepcopy(args)),
            name=f"{self._profile.robot_name}-web-policy-startup-g{generation_id}",
            daemon=False,
        )
        self._startup_thread.start()

    def _run_policy_startup(
        self,
        generation_id: int,
        stop_event: threading.Event,
        controller: RuntimeController,
        args: Any,
    ) -> None:
        try:
            controller.configure_robot(args)
            adapter = self._make_inference_adapter(controller.robot, args)
            loop_config = self._loop_config(args)
            self._loop_hooks.reset()
            with self._lock:
                startup_config = self._policy_startup_config_locked()
            startup_event_hooks = RuntimeEventHooks(
                controller.event_sink,
                RuntimeEventIdentity(
                    runtime_id=controller.runtime_id,
                    generation_id=controller.status().generation_id + 1,
                    session_id=f"{self._profile.robot_name}-web-connect-{generation_id}",
                ),
            )
            coordinator = PolicyStartupCoordinator(
                controller.policy_client,
                adapter,
                loop_config,
                startup_config,
                hooks=CompositeInferenceHooks(self._loop_hooks, startup_event_hooks),
                stop_requested=stop_event.is_set,
                on_phase=lambda phase: self._set_policy_startup_phase(generation_id, phase),
            )
            prepared = coordinator.prepare()
            with self._lock:
                if (
                    generation_id != self._startup_generation_id
                    or stop_event.is_set()
                    or self._controller is not controller
                ):
                    return
                self._policy_metrics.cold_inference_latency_ms = prepared.metrics.cold_inference_latency_ms
                self._policy_metrics.warmup_latency_ms = prepared.metrics.warmup_latency_ms
                self._policy_metrics.first_live_inference_latency_ms = prepared.metrics.first_live_inference_latency_ms
                controller.configure_event_identity(session_id=None, episode_id=None)
                controller.configure(
                    adapter,
                    loop_config,
                    hooks=self._loop_hooks,
                    on_step=self._loop_hooks.on_step,
                    initial_chunk=prepared.initial_chunk,
                    initial_provenance=prepared.initial_provenance,
                )
                self._running = True
                self._phase = "running"
                self._policy_state = "RUNNING"
            controller.start()
            self.log("Policy warmup complete; started inference loop")
        except PolicyStartupCancelled:
            with self._lock:
                if generation_id == self._startup_generation_id and self._phase != "idle":
                    self._running = False
                    self._stop_requested = False
                    self._phase = "stopped"
                    if self._policy_state != "DISCONNECTED":
                        self._policy_state = "CONNECTED"
            self.log("Policy startup stopped")
        except BaseException as exc:
            with self._lock:
                if generation_id == self._startup_generation_id:
                    self._running = False
                    self._stop_requested = False
                    self._policy_state = "WARMUP_FAILED"
                    self._phase = "warmup_failed"
                    self._last_error = f"{type(exc).__name__}: {exc}"
            self.log(f"Policy startup failed: {type(exc).__name__}: {exc}")

    def _set_policy_startup_phase(self, generation_id: int, phase: str) -> None:
        with self._lock:
            if generation_id != self._startup_generation_id or self._startup_stop_event.is_set():
                return
            self._policy_state = phase
            self._phase = phase.lower()

    def _ensure_camera_thread_locked(self, args: Any, camera_stream_fps: float) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(copy.deepcopy(args), camera_stream_fps),
            daemon=False,
            name=f"{self._profile.robot_name}-web-camera",
        )
        self._camera_thread.start()

    def stop(self, *, wait: bool = False) -> dict[str, Any]:
        stop_data_view_jobs = False
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns robot control; use stop-episode or abort", HTTPStatus.CONFLICT)
            if self._runtime_mode is RuntimeMode.CAMERA_PREVIEW:
                if self._phase == "previewing":
                    self._phase = "stopping"
                    self._camera_stop_event.set()
                camera_thread = self._camera_thread
                controller = None
                startup_thread = None
                startup_active = False
                connection_thread = None
                connection_active = False
            elif self._runtime_mode is RuntimeMode.OFFLINE_REPLAY:
                return self.status()
            elif self._runtime_mode is RuntimeMode.DATA_VIEW:
                # Stop only explicitly submitted policy work.  Browsing stays
                # ready and does not close its read-only data session.
                stop_data_view_jobs = True
                camera_thread = None
                controller = None
                startup_thread = None
                startup_active = False
                connection_thread = None
                connection_active = False
            else:
                startup_thread = self._startup_thread
                startup_active = self._startup_active_locked()
                connection_thread = self._connect_thread
                connection_active = self._connection_active_locked()
                if not self._running and not self._stop_requested and not startup_active and not connection_active:
                    return self.status()
                self._stop_requested = True
                if connection_active:
                    self._phase = "stopping"
                    self._connect_stop_event.set()
                if startup_active:
                    self._phase = "stopping"
                    self._startup_stop_event.set()
                    self._policy_connected = False
                    self._policy_state = "DISCONNECTED"
                controller = self._controller
                camera_thread = None
        self.log("Stop requested")
        if stop_data_view_jobs:
            self._data_view_stop_active_open_loop()
            return self.status()
        if startup_active and controller is not None:
            close_client = getattr(controller.policy_client, "close", None)
            if callable(close_client):
                close_client()
        if startup_thread is not None and wait and startup_thread is not threading.current_thread():
            startup_thread.join(timeout=5.0)
        if connection_thread is not None and wait and connection_thread is not threading.current_thread():
            connection_thread.join(timeout=5.0)
        if controller is not None:
            controller.stop(wait=wait, timeout=5.0 if wait else None)
        if camera_thread is not None and wait and camera_thread is not threading.current_thread():
            camera_thread.join(timeout=max(1.0, self._args.camera_timeout + 1.0))
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns this deployment; abort it before disconnecting", HTTPStatus.CONFLICT)
            # Cancel replay before touching normal or pose resources.  The
            # replay lease predicate observes this generation immediately.
            self._replay_generation_id += 1
            self._replay_stop_event.set()
            replay_controller = self._replay_controller
        if replay_controller is not None:
            replay_controller.stop(emergency=True, wait=True, timeout=2.0)
        with self._lock:
            replay_threads = tuple(
                thread
                for thread in (self._replay_plan_thread, self._replay_connect_thread, self._replay_watch_thread)
                if thread is not None and thread is not threading.current_thread()
            )
        replay_deadline = time.monotonic() + 5.0
        for thread in replay_threads:
            thread.join(max(0.0, replay_deadline - time.monotonic()))
        if any(thread.is_alive() for thread in replay_threads):
            raise ApiError("Replay worker is still stopping; retry disconnect shortly", HTTPStatus.CONFLICT)
        with self._lock:
            replay_robot = self._replay_robot
            replay_lease = self._replay_lease
            replay_recorder = self._replay_recorder
            self._replay_robot = None
            self._replay_lease = None
            self._replay_recorder = None
            self._replay_view_locked = False
            if self._replay_phase not in {ReplayState.COMPLETED, ReplayState.ABORTED, ReplayState.ERROR}:
                self._replay_phase = ReplayState.ABORTED
            # Every pose worker observes this generation.  Join before normal
            # deployment cleanup so an old handoff cannot resurrect resources.
            self._pose_generation_id += 1
            self._pose_stop_event.set()
            pose_controller = self._pose_controller
            pose_moving = self._pose_phase in {"moving", "stopping"}
        if replay_robot is not None:
            replay_robot.close()
        if replay_lease is not None:
            replay_lease.release()
        if replay_recorder is not None:
            replay_recorder.stop(result="disconnected")
        # DATA_VIEW has a non-daemon open-loop worker.  Close it before
        # releasing the read-only resource lease so a stale worker cannot
        # publish a result into a later DATA_VIEW session.
        self._close_data_view_open_loop_jobs()
        if pose_controller is not None and pose_moving:
            pose_controller.stop(wait=True, timeout=2.0)
        with self._lock:
            pose_threads = tuple(
                thread
                for thread in (
                    self._pose_connect_thread,
                    self._pose_watch_thread,
                    self._pose_handoff_thread,
                    self._pose_deploy_thread,
                )
                if thread is not None and thread is not threading.current_thread()
            )
        deadline = time.monotonic() + 5.0
        for thread in pose_threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in pose_threads):
            raise ApiError("Recorded-state worker is still stopping; retry disconnect shortly", HTTPStatus.CONFLICT)

        # A pose-only connection has no RuntimeController yet, so normal
        # deployment cleanup would otherwise leave its Robot/lease behind.
        with self._lock:
            pose_robot = self._pose_robot
            pose_lease = self._pose_lease
            self._pose_controller = None
            self._pose_robot = None
            self._pose_lease = None
            self._pose_plan = None
            self._pose_live_validation = None
            self._pose_progress = None
            self._pose_prepared = None
            self._pose_phase = "idle"
            self._pose_error = None
        if pose_robot is not None:
            pose_robot.close()
        if pose_lease is not None:
            pose_lease.release()
        self.stop(wait=True)
        with self._lock:
            if self._running or self._startup_active_locked() or self._connection_active_locked():
                raise ApiError("Runtime is still stopping; retry disconnect shortly", HTTPStatus.CONFLICT)
            self._camera_stop_event.set()
            camera_thread = self._camera_thread

        if camera_thread is not None and camera_thread is not threading.current_thread():
            camera_thread.join(timeout=max(1.0, self._args.camera_timeout + 1.0))
            if camera_thread.is_alive():
                raise ApiError("Camera preview is still stopping; retry disconnect shortly", HTTPStatus.CONFLICT)

        with self._lock:
            resources = self._pending_deployment_cleanup
            if resources is None:
                resources = _DeploymentResources(
                    controller=self._controller,
                    cameras=self._cameras,
                )
                self._pending_deployment_cleanup = resources

        cleanup_error: BaseException | None = None
        try:
            resources.close()
        except BaseException as exc:
            cleanup_error = exc

        if not resources.complete:
            with self._lock:
                self._phase = "cleanup_failed"
                self._policy_state = "ERROR"
                self._last_error = f"Deployment cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
            raise ApiError(
                "Deployment cleanup is incomplete; retry Disconnect after the worker stops",
                HTTPStatus.CONFLICT,
            ) from cleanup_error

        with self._lock:
            if self._pending_deployment_cleanup is resources:
                self._pending_deployment_cleanup = None
            self._controller = None
            self._cameras = {}
            self._camera_thread = None
            self._connect_thread = None
            self._startup_thread = None
            self._connected = False
            self._policy_connected = False
            self._running = False
            self._stop_requested = False
            self._phase = "idle"
            self._policy_state = "DISCONNECTED"
            self._last_error = None
            self._server_metadata = None
            self._policy_metrics = PolicyMetrics()
            self._loop_hooks.reset()
            self._reset_placeholder_frames_locked(self._args.resize_size)
            self._release_resources_locked()
        if cleanup_error is not None:
            self.log(f"Deployment resources closed after reporting: {type(cleanup_error).__name__}: {cleanup_error}")
        self.log("Disconnected runtime")
        return self.status()

    def reset_arms(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns robot control; reset is an operator step in the evaluation flow",
                    HTTPStatus.CONFLICT,
                )
            if self._runtime_mode is not RuntimeMode.DEPLOYMENT:
                raise ApiError("Robot reset is only available in deployment mode", HTTPStatus.CONFLICT)
            if self._startup_active_locked():
                raise ApiError("Cannot reset while policy startup is in progress", HTTPStatus.CONFLICT)
            if self._running and self._policy_connected:
                raise ApiError("Reset is only allowed while stopped or when the policy server is disconnected")
            controller = self._controller
            args = copy.deepcopy(self._args)

        if self._running:
            self.stop(wait=True)
        if controller is None:
            raise ApiError("Arms are not connected", HTTPStatus.CONFLICT)

        if not self._profile.capabilities.supports_reset:
            raise ApiError(f"{self._profile.robot_name} does not support reset", HTTPStatus.CONFLICT)
        args = self._profile.configure_reset(args)
        controller.configure_robot(args)
        try:
            controller.reset_robot()
        except ControllerAlreadyRunningError as exc:
            raise ApiError(str(exc), HTTPStatus.CONFLICT) from exc
        self.log(f"Reset {self._profile.robot_name} through the robot boundary")
        return self.status()

    def ping_policy(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._runtime_mode is not RuntimeMode.DEPLOYMENT:
                return {
                    "robot": self._profile.robot_name,
                    "ok": False,
                    "connected": False,
                    "error": "Policy is unavailable outside deployment mode",
                }
            if self._running:
                return {
                    "ok": self._policy_connected,
                    "connected": self._policy_connected,
                    "metadata": _json_safe(self._server_metadata or {}),
                }
            args = copy.deepcopy(self._args)

        client: PolicyClient | None = None
        started = time.monotonic()
        try:
            client = self._create_policy_client(args)
            latency_ms = (time.monotonic() - started) * 1000.0
            return {
                "robot": self._profile.robot_name,
                "ok": True,
                "connected": True,
                "latency_ms": round(latency_ms, 2),
                "metadata": _json_safe(client.metadata),
            }
        except Exception as exc:
            return {"ok": False, "connected": False, "error": str(exc)}
        finally:
            if client is not None:
                client.close()

    def _camera_loop(self, args: Any, camera_stream_fps: float) -> None:
        period = 1.0 / camera_stream_fps
        while not self._camera_stop_event.is_set():
            loop_t0 = time.monotonic()
            with self._lock:
                cameras = dict(self._cameras)

            for name, camera in cameras.items():
                if self._camera_stop_event.is_set():
                    break
                try:
                    frame = camera.read_frame(timeout=args.camera_timeout)
                    image = preprocess_image(frame.image, args.resize_size)
                    jpeg = _encode_jpeg_rgb(image)
                    with self._frame_condition:
                        old = self._frames.get(name, FrameSnapshot())
                        self._frames[name] = FrameSnapshot(
                            image=image,
                            jpeg=jpeg,
                            sequence=old.sequence + 1,
                            updated_at=frame.timestamp_monotonic,
                            error=None,
                            frame_id=frame.frame_id,
                            timestamp_monotonic_ns=frame.timestamp_monotonic_ns,
                            source_sequence=frame.source_sequence,
                            capture_latency_ns=frame.capture_latency_ns,
                        )
                        self._frame_condition.notify_all()
                except Exception as exc:
                    with self._frame_condition:
                        old = self._frames.get(name, FrameSnapshot())
                        self._frames[name] = dataclasses.replace(old, error=str(exc), updated_at=time.monotonic())
                        self._frame_condition.notify_all()

            elapsed = time.monotonic() - loop_t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                self._camera_stop_event.wait(sleep_s)

    def _latest_images_for_inference(
        self, args: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, Any] | None, dict[str, FrameSnapshot]]:
        deadline = time.monotonic() + args.camera_timeout
        roles = self._profile.camera_roles_for_args(args)
        with self._frame_condition:
            while True:
                missing = [
                    name
                    for name in roles
                    if self._frames.get(name) is None or self._frames[name].image is None or self._frames[name].error
                ]
                if not missing:
                    images = {name: np.asarray(self._frames[name].image).copy() for name in roles}
                    camera_params = (
                        {name: camera.camera_info() for name, camera in self._cameras.items()}
                        if self._profile.include_camera_params
                        else None
                    )
                    return images, camera_params, {name: self._frames[name] for name in roles}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    errors = {
                        name: self._frames[name].error
                        for name in missing
                        if self._frames.get(name) is not None and self._frames[name].error
                    }
                    raise RuntimeError(f"Camera frames unavailable: missing={missing}, errors={errors}")
                self._frame_condition.wait(min(0.05, remaining))

    def wait_frame(self, camera_name: str, last_sequence: int, *, timeout: float = 5.0) -> FrameSnapshot:
        if camera_name not in self._profile.camera_roles_for_args(self._args):
            raise ApiError(f"Unknown camera {camera_name}", HTTPStatus.NOT_FOUND)
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while True:
                frame = self._frames[camera_name]
                if frame.sequence != last_sequence and frame.jpeg:
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return frame
                self._frame_condition.wait(min(0.2, remaining))

    def status(self) -> dict[str, Any]:
        evaluation = self._evaluation_service.current()
        with self._lock:
            self._refresh_controller_state_locked()
            loop = self._loop_hooks.snapshot()
            now = time.monotonic()
            controller_status = self._controller.status() if self._controller is not None else None
            started_at_ns = loop.started_at_monotonic_ns
            if started_at_ns is None and controller_status is not None:
                started_at_ns = controller_status.started_at_monotonic_ns
            frames = {
                name: {
                    "sequence": frame.sequence,
                    "frame_id": frame.frame_id,
                    "timestamp_monotonic_ns": frame.timestamp_monotonic_ns or None,
                    "source_sequence": frame.source_sequence,
                    "capture_latency_ns": frame.capture_latency_ns,
                    "age_ms": round((now - frame.updated_at) * 1000.0, 1) if frame.updated_at else None,
                    "error": frame.error,
                }
                for name, frame in self._frames.items()
            }
            startup_active = self._startup_active_locked()
            connection_active = self._connection_active_locked()
            replay_active = self._replay_active_locked()
            cleanup_pending = self._pending_deployment_cleanup is not None
            data_view = self._data_view_status_locked()
            can_reset = (
                self._runtime_mode is RuntimeMode.DEPLOYMENT
                and self._controller is not None
                and not cleanup_pending
                and not startup_active
                and (not self._running or not self._policy_connected)
            )
            return {
                "robot": self._profile.robot_name,
                "camera_roles": list(self._profile.camera_roles_for_args(self._args)),
                "action_spec": dataclasses.asdict(self._profile.action_spec_for_args(self._args)),
                "safety_profile": _replay_json_safe(self._profile.safety_profile_for_args(self._args).to_dict()),
                "capabilities": dataclasses.asdict(self._profile.capabilities),
                "runtime_mode": self._runtime_mode.value,
                "phase": self._phase,
                "policy_state": self._policy_state,
                "connected": self._connected,
                "policy_connected": self._policy_connected,
                "running": self._running,
                "stop_requested": self._stop_requested,
                "cleanup_pending": cleanup_pending,
                "can_edit_config": self._can_edit_config_locked(),
                "can_edit_connection_config": self._can_edit_connection_config_locked(),
                "can_connect": not self._connected
                and not self._running
                and not self._stop_requested
                and not connection_active
                and not startup_active
                and not replay_active
                and not cleanup_pending
                and self._phase in {"idle", "error", "warmup_failed"},
                "can_start": (
                    self._runtime_mode is RuntimeMode.CAMERA_PREVIEW
                    and not self._connected
                    and not cleanup_pending
                    and self._phase in {"idle", "error"}
                )
                or (
                    self._runtime_mode is RuntimeMode.DATA_VIEW
                    and not self._connected
                    and not cleanup_pending
                    and self._phase in {"idle", "error"}
                )
                or (
                    self._runtime_mode is RuntimeMode.DEPLOYMENT
                    and not self._running
                    and not self._stop_requested
                    and not connection_active
                    and not startup_active
                    and not cleanup_pending
                    and self._phase in {"idle", "stopped", "warmup_failed", "error"}
                ),
                "can_stop": self._running
                or self._stop_requested
                or connection_active
                or startup_active
                or replay_active
                or (self._runtime_mode is RuntimeMode.CAMERA_PREVIEW and self._phase == "previewing")
                or data_view["open_loop_active"],
                "can_disconnect": self._connected
                or self._running
                or connection_active
                or startup_active
                or replay_active
                or cleanup_pending
                or self._phase in {"connecting", "connecting_cameras", "error", "stopped", "warmup_failed", "stopping"},
                "can_reset": can_reset,
                "step": loop.step,
                "max_steps": self._args.max_steps,
                "prompt": self._args.prompt,
                "infer_only": self._args.infer_only,
                "server_url": self._args.server_url,
                "server_metadata": _json_safe(self._server_metadata or {}),
                "last_error": self._last_error,
                "metrics": {
                    "infer_latency_ms": round(loop.infer_latency_ms, 2) if loop.infer_latency_ms is not None else None,
                    "infer_hz": round(loop.infer_hz, 2) if loop.infer_hz is not None else None,
                    "loop_ms": round(loop.loop_ms, 2) if loop.loop_ms is not None else None,
                    "control_hz": round(loop.control_hz, 2) if loop.control_hz is not None else None,
                    "action_queue_len": loop.action_queue_len,
                    "connect_latency_ms": round(self._policy_metrics.connect_latency_ms, 2)
                    if self._policy_metrics.connect_latency_ms is not None
                    else None,
                    "metadata_latency_ms": round(self._policy_metrics.metadata_latency_ms, 2)
                    if self._policy_metrics.metadata_latency_ms is not None
                    else None,
                    "cold_inference_latency_ms": round(self._policy_metrics.cold_inference_latency_ms, 2)
                    if self._policy_metrics.cold_inference_latency_ms is not None
                    else None,
                    "warmup_latency_ms": round(self._policy_metrics.warmup_latency_ms, 2)
                    if self._policy_metrics.warmup_latency_ms is not None
                    else None,
                    "first_live_inference_latency_ms": round(self._policy_metrics.first_live_inference_latency_ms, 2)
                    if self._policy_metrics.first_live_inference_latency_ms is not None
                    else None,
                    "steady_inference_latency_ms": round(self._policy_metrics.steady_inference_latency_ms, 2)
                    if self._policy_metrics.steady_inference_latency_ms is not None
                    else None,
                    "uptime_s": round((time.monotonic_ns() - started_at_ns) / 1e9, 1)
                    if started_at_ns is not None
                    else None,
                },
                "frames": frames,
                "evaluation": evaluation,
                "pose": self.pose_status(),
                "replay": self.replay_status(),
                "data_view": data_view,
                "logs": list(self._logs),
                "resource_leases": self._resource_manager.snapshot(),
            }


class PiperWebRuntime(RobotWebRuntime):
    """Backward-compatible Piper constructor for the profile-based runtime."""

    def __init__(self, initial_args: infer_piper.Args | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("profile", PIPER_WEB_PROFILE)
        super().__init__(initial_args, **kwargs)


_RM_STREAM_TO_POLICY = {
    "cam_head": "head_color",
    "cam_left_wrist": "left_color",
    "cam_right_wrist": "right_color",
}


class _LegacyRm2WebRuntime:
    """RM2 implementation of the same Web lifecycle exposed by PiperWebRuntime."""

    def __init__(
        self,
        initial_args: infer_rm2.Args | None = None,
        *,
        policy_timeout: float = 3.0,
        robot_factory: Callable[[str, Any], Robot] = create_robot,
        policy_client_factory: Callable[[str, str | None, float], PolicyClient] | None = None,
        camera_factory: Callable[[infer_rm2.Args], dict[str, infer_rm2.Camera]] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._args = copy.deepcopy(initial_args) if initial_args is not None else infer_rm2.Args()
        self._policy_timeout = policy_timeout
        self._camera_stream_fps = 10.0
        self._robot_factory = robot_factory
        self._policy_client_factory = policy_client_factory or (
            lambda server_url, api_key, timeout: PolicyClient(server_url, api_key, timeout=timeout)
        )
        self._camera_factory = camera_factory or infer_rm2.make_cameras
        self._evaluation_owner: str | None = None
        self._evaluation_service = EvaluationService(self)
        self._controller: RuntimeController | None = None
        self._loop_hooks = WebLoopHooks(error_callback=self._record_loop_error)
        self._cameras: dict[str, infer_rm2.Camera] = {}
        self._frames = {
            name: FrameSnapshot(
                _black_frame(self._args.resize_size),
                _encode_jpeg_rgb(_black_frame(self._args.resize_size)),
            )
            for name in CAMERA_NAMES
        }
        self._camera_thread: threading.Thread | None = None
        self._camera_stop_event = threading.Event()
        self._connected = False
        self._running = False
        self._phase = "idle"
        self._last_error: str | None = None
        self._logs: deque[str] = deque(maxlen=200)

    @property
    def evaluation_service(self) -> EvaluationService:
        return self._evaluation_service

    def acquire_evaluation_control(self, evaluation_id: str) -> EvaluationRuntimeLease:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise EvaluationConflict("Another evaluation already owns this deployment controller")
            if not self._connected or self._controller is None:
                raise EvaluationConflict("Connect the deployment runtime before creating an evaluation")
            if self._running:
                raise EvaluationConflict("Stop normal deployment control before creating an evaluation")
            if self._args.infer_only:
                raise EvaluationConflict(
                    "Evaluation requires a motion-capable (non-infer-only) deployment configuration"
                )

            controller = self._controller
            args = copy.deepcopy(self._args)
            runtime_snapshot = self.get_config()
            runtime_snapshot["policy_metadata"] = {}
            runtime_snapshot["git_commit"] = _git_commit()
            action_spec_snapshot = dataclasses.asdict(controller.robot.action_spec)
            startup_config = PolicyStartupConfig(
                warmup_timeout_s=self._policy_timeout,
                inference_timeout_s=self._policy_timeout,
            )
            self._evaluation_owner = evaluation_id

        def make_args(prompt: str) -> infer_rm2.Args:
            evaluation_args = copy.deepcopy(args)
            evaluation_args.prompt = prompt
            return evaluation_args

        def release() -> None:
            with self._lock:
                if self._evaluation_owner == evaluation_id:
                    self._evaluation_owner = None

        return EvaluationRuntimeLease(
            controller=controller,
            runtime_config_snapshot=runtime_snapshot,
            action_spec_snapshot=action_spec_snapshot,
            robot_name="rm2",
            make_adapter=lambda prompt: self._make_inference_adapter(controller.robot, make_args(prompt)),
            make_loop_config=lambda prompt: InferenceLoopConfig.from_args(make_args(prompt)),
            make_startup_config=lambda: startup_config,
            release=release,
        )

    def log(self, message: str) -> None:
        with self._lock:
            self._logs.append(f"{time.strftime('%H:%M:%S')} {message}")
        logging.info(message)

    def _record_loop_error(self, error: BaseException) -> None:
        with self._lock:
            self._phase = "error"
            self._last_error = str(error)
            self._logs.append(f"{time.strftime('%H:%M:%S')} RM2 inference failed: {error}")

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            args = copy.deepcopy(self._args)
        return {
            "robot": "rm2",
            "runtime_mode": RuntimeMode.DEPLOYMENT.value,
            "server_url": args.server_url,
            "api_key": args.api_key or "",
            "prompt": args.prompt,
            "fps": args.fps,
            "replan_steps": args.replan_steps,
            "max_steps": args.max_steps,
            "dry_run": args.dry_run,
            "infer_only": args.infer_only,
            "use_rtc": args.use_rtc,
            "reset_on_start": args.reset_on_start,
            "camera_backend": args.camera_backend,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "camera_fps": args.camera_fps,
            "camera_timeout": args.camera_timeout,
            "resize_size": args.resize_size,
            "rm_config": args.rm_config,
            "rm_sdk_lib": args.rm_sdk_lib,
            "left_ip": args.left_ip or "",
            "right_ip": args.right_ip or "",
            "arm_port": args.arm_port,
            "joint_dof": args.joint_dof,
            "policy_joint_unit": args.policy_joint_unit,
            "cam_left_topic": args.cam_left_topic,
            "cam_right_topic": args.cam_right_topic,
            "cam_head_topic": args.cam_head_topic,
        }

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns this deployment; abort or complete it before changing configuration",
                    HTTPStatus.CONFLICT,
                )
            if self._connected or self._running:
                raise ApiError("Disconnect RM2 before changing parameters", HTTPStatus.CONFLICT)
            if (
                "runtime_mode" in payload
                and _runtime_mode(payload["runtime_mode"]) is not RuntimeMode.DEPLOYMENT
            ):
                raise ApiError(
                    "CAMERA_PREVIEW and OFFLINE_REPLAY are currently available for Piper Web only",
                    HTTPStatus.CONFLICT,
                )
            args = copy.deepcopy(self._args)
            for field in (
                "server_url",
                "api_key",
                "prompt",
                "left_ip",
                "right_ip",
                "cam_left_topic",
                "cam_right_topic",
                "cam_head_topic",
            ):
                if field in payload:
                    value = str(payload[field])
                    setattr(args, field, value if field != "api_key" or value else None)
            for field in ("rm_config", "rm_sdk_lib"):
                if field in payload:
                    value = str(payload[field]).strip()
                    setattr(args, field, pathlib.Path(value).expanduser() if value else None)
            for field in ("fps", "camera_timeout"):
                if field in payload:
                    setattr(args, field, float(payload[field]))
            for field in ("replan_steps", "camera_width", "camera_height", "camera_fps", "resize_size", "joint_dof"):
                if field in payload:
                    setattr(args, field, int(payload[field]))
            if "max_steps" in payload:
                args.max_steps = _coerce_optional_int(payload["max_steps"])
            if "arm_port" in payload:
                args.arm_port = int(payload["arm_port"]) if payload["arm_port"] not in (None, "") else None
            if "camera_backend" in payload:
                args.camera_backend = str(payload["camera_backend"])
            if "policy_joint_unit" in payload:
                args.policy_joint_unit = str(payload["policy_joint_unit"])
            for field in ("dry_run", "infer_only", "use_rtc", "reset_on_start"):
                if field in payload:
                    setattr(args, field, _coerce_bool(payload[field]))
            infer_rm2.validate_args(args)
            self._args = args
            return self.get_config()

    def _make_inference_adapter(self, robot: Robot, args: infer_rm2.Args) -> WebInferenceAdapter:
        source = CachedFrameObservationSource(
            robot=robot,
            read_images=self._latest_policy_images,
            image_masks={
                policy_name: np.bool_(args.camera_backend != "black")
                for policy_name in _RM_STREAM_TO_POLICY.values()
            },
            prompt=args.prompt,
        )

        def decode(response: dict[str, Any], replan_steps: int) -> np.ndarray:
            if replan_steps != args.replan_steps:
                raise ValueError("RM2 replan_steps must match the Web runtime configuration")
            return infer_rm2.response_to_action_chunk(response, args)

        def profile(stage: str, elapsed_s: float) -> None:
            if args.profile_timing:
                logging.info("rm2 web profile %s=%.3fs", stage, elapsed_s)

        return WebInferenceAdapter(
            name="rm2",
            robot=robot,
            observation_source=source,
            decode_chunk=decode,
            stabilize=lambda action, previous: infer_rm2.stabilize_action(action, previous, args),
            metadata_keys=("camera_params",),
            profile_callback=profile,
        )

    def _refresh_controller_state_locked(self) -> None:
        if self._controller is None:
            return
        controller_status = self._controller.status()
        self._running = controller_status.running
        if controller_status.error is not None:
            self._phase = "error"
            self._last_error = str(controller_status.error)
        elif not controller_status.running and self._phase == "running":
            self._phase = "stopped"

    def connect(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._connected:
                return self.status()
            args = copy.deepcopy(self._args)
            self._phase = "connecting"
        client: PolicyClient | None = None
        robot: Robot | None = None
        cameras: dict[str, infer_rm2.Camera] = {}
        try:
            client = self._policy_client_factory(args.server_url, args.api_key, self._policy_timeout)
            cameras = self._camera_factory(args)
            robot = self._robot_factory("rm2", args)
            robot.reset()
            adapter = self._make_inference_adapter(robot, args)
            controller = RuntimeController(
                robot,
                adapter,
                client,
                InferenceLoopConfig.from_args(args),
                hooks=self._loop_hooks,
                on_step=self._loop_hooks.on_step,
                thread_name="rm2-web-run",
                print_infer_only_chunks=False,
                event_sink=CompositeRuntimeEventSink(InMemoryRuntimeEventSink(), self._evaluation_service),
            )
        except Exception as exc:
            if cameras:
                infer_rm2.close_cameras(cameras)
            if robot is not None:
                robot.close()
            if client is not None:
                client.close()
            with self._lock:
                self._phase, self._last_error = "error", str(exc)
            raise ApiError(str(exc), HTTPStatus.BAD_GATEWAY) from exc
        with self._lock:
            self._controller, self._cameras = controller, cameras
            self._connected, self._phase, self._last_error = True, "stopped", None
            self._ensure_camera_thread_locked(args)
        self.log("Connected RM2 runtime")
        return self.status()

    def start(self) -> dict[str, Any]:
        if not self._connected:
            self.connect()
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns robot control; use the evaluation API", HTTPStatus.CONFLICT)
            controller = self._controller
            if self._running or controller is None:
                raise ApiError("RM2 runtime is not ready", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            self._loop_hooks.reset()
            controller.configure_event_identity(session_id=None, episode_id=None)
            controller.configure(
                self._make_inference_adapter(controller.robot, args),
                InferenceLoopConfig.from_args(args),
                hooks=self._loop_hooks,
                on_step=self._loop_hooks.on_step,
            )
            self._running, self._phase, self._last_error = True, "running", None
            try:
                controller.start()
            except ControllerAlreadyRunningError as exc:
                self._refresh_controller_state_locked()
                raise ApiError(str(exc), HTTPStatus.CONFLICT) from exc
        return self.status()

    def stop(self, *, wait: bool = False) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns robot control; use stop-episode or abort", HTTPStatus.CONFLICT)
            controller = self._controller
        if controller is not None:
            controller.stop(wait=wait, timeout=5.0 if wait else None)
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns this deployment; abort it before disconnecting", HTTPStatus.CONFLICT)
        self.stop(wait=True)
        with self._lock:
            self._refresh_controller_state_locked()
            if self._running:
                raise ApiError("RM2 runtime is still stopping", HTTPStatus.CONFLICT)
        self._camera_stop_event.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
            if self._camera_thread.is_alive():
                raise ApiError("RM2 camera preview is still stopping", HTTPStatus.CONFLICT)
        with self._lock:
            controller, cameras = self._controller, self._cameras
            self._controller, self._cameras = None, {}
            self._camera_thread = None
            self._connected, self._running, self._phase = False, False, "idle"
            self._loop_hooks.reset()
        infer_rm2.close_cameras(cameras)
        if controller is not None:
            controller.close()
        return self.status()

    def reset_arms(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns robot control; reset is an operator step in the evaluation flow",
                    HTTPStatus.CONFLICT,
                )
            controller = self._controller
        if controller is None:
            raise ApiError("RM2 is not connected", HTTPStatus.CONFLICT)
        try:
            controller.reset_robot()
        except ControllerAlreadyRunningError as exc:
            raise ApiError(str(exc), HTTPStatus.CONFLICT) from exc
        return self.status()

    def ping_policy(self) -> dict[str, Any]:
        client: PolicyClient | None = None
        try:
            client = self._policy_client_factory(self._args.server_url, self._args.api_key, self._policy_timeout)
            return {"ok": True, "connected": True, "metadata": _json_safe(client.metadata)}
        except Exception as exc:
            return {"ok": False, "connected": False, "error": str(exc)}
        finally:
            if client is not None:
                client.close()

    def _ensure_camera_thread_locked(self, args: infer_rm2.Args) -> None:
        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(args,),
            daemon=False,
            name="rm2-web-camera",
        )
        self._camera_thread.start()

    def _camera_loop(self, args: infer_rm2.Args) -> None:
        while not self._camera_stop_event.is_set():
            for stream_name, policy_name in _RM_STREAM_TO_POLICY.items():
                camera = self._cameras.get(policy_name)
                if camera is None:
                    continue
                try:
                    frame = camera.read_frame(timeout=args.camera_timeout)
                    image = preprocess_image(frame.image, args.resize_size)
                    with self._frame_condition:
                        old = self._frames[stream_name]
                        self._frames[stream_name] = FrameSnapshot(
                            image,
                            _encode_jpeg_rgb(image),
                            old.sequence + 1,
                            frame.timestamp_monotonic,
                            None,
                            frame.frame_id,
                            frame.timestamp_monotonic_ns,
                            frame.source_sequence,
                            frame.capture_latency_ns,
                        )
                        self._frame_condition.notify_all()
                except Exception as exc:
                    with self._frame_condition:
                        self._frames[stream_name] = dataclasses.replace(self._frames[stream_name], error=str(exc))
            self._camera_stop_event.wait(1.0 / self._camera_stream_fps)

    def _latest_policy_images(self) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, FrameSnapshot]]:
        deadline = time.monotonic() + self._args.camera_timeout
        with self._frame_condition:
            while any(self._frames[name].image is None or self._frames[name].error for name in CAMERA_NAMES):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("RM2 camera frames are unavailable")
                self._frame_condition.wait(min(0.05, remaining))
            images = {policy: self._frames[stream].image.copy() for stream, policy in _RM_STREAM_TO_POLICY.items()}
            frame_metadata = {policy: self._frames[stream] for stream, policy in _RM_STREAM_TO_POLICY.items()}
        return images, {name: camera.camera_info() for name, camera in self._cameras.items()}, frame_metadata

    def wait_frame(self, camera_name: str, last_sequence: int, *, timeout: float = 5.0) -> FrameSnapshot:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while self._frames[camera_name].sequence == last_sequence and time.monotonic() < deadline:
                self._frame_condition.wait(0.2)
            return self._frames[camera_name]

    def status(self) -> dict[str, Any]:
        evaluation = self._evaluation_service.current()
        with self._lock:
            self._refresh_controller_state_locked()
            loop = self._loop_hooks.snapshot()
            return {
                "robot": "rm2", "runtime_mode": RuntimeMode.DEPLOYMENT.value,
                "phase": self._phase, "connected": self._connected, "policy_connected": self._connected,
                "running": self._running, "can_edit_config": not self._connected and not self._running,
                "can_edit_connection_config": not self._connected, "can_connect": not self._connected,
                "can_start": self._connected and not self._running, "can_stop": self._running,
                "can_disconnect": self._connected, "can_reset": self._controller is not None and not self._running,
                "step": loop.step, "max_steps": self._args.max_steps, "prompt": self._args.prompt,
                "server_url": self._args.server_url, "server_metadata": {}, "last_error": self._last_error,
                "metrics": {},
                "frames": {
                    name: {
                        "sequence": frame.sequence,
                        "frame_id": frame.frame_id,
                        "timestamp_monotonic_ns": frame.timestamp_monotonic_ns or None,
                        "source_sequence": frame.source_sequence,
                        "capture_latency_ns": frame.capture_latency_ns,
                        "age_ms": None,
                        "error": frame.error,
                    }
                    for name, frame in self._frames.items()
                },
                "evaluation": evaluation,
                "logs": list(self._logs),
            }


class Rm2WebRuntime(RobotWebRuntime):
    """Backward-compatible RM2 constructor using the shared Web lifecycle."""

    def __init__(self, initial_args: infer_rm2.Args | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("profile", RM2_WEB_PROFILE)
        super().__init__(initial_args, **kwargs)


class PiperWebHandler(BaseHTTPRequestHandler):
    server: PiperWebServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path in {"/", "/replay"}:
                self._send_file(self.server.static_dir / "index.html")
            elif path == "/data-view":
                self._send_file(self.server.static_dir / "data_view.html")
            elif path.startswith("/static/"):
                self._send_file(self.server.static_dir / path.removeprefix("/static/"))
            elif path.startswith("/api/data-view/"):
                self._handle_data_view_get(path, urllib.parse.parse_qs(parsed.query))
            elif path == "/api/status":
                self._send_json(self.server.runtime.status())
            elif path == "/api/config":
                self._send_json(self.server.runtime.get_config())
            elif path == "/api/pose/status":
                self._send_json({"ok": True, "pose": self.server.runtime.pose_status()})
            elif path == "/api/evaluations/current":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.current()})
            elif path == "/api/baselines":
                self._send_json(
                    {
                        "ok": True,
                        "baselines": [item.to_dict() for item in self.server.runtime.baseline_service.list()],
                        "writer": self.server.runtime.baseline_status(),
                    }
                )
            elif path.startswith("/api/baseline-jobs/"):
                job_id = urllib.parse.unquote(path.removeprefix("/api/baseline-jobs/"))
                self._send_json({"ok": True, "job": self.server.runtime.baseline_job_status(job_id)})
            elif path.startswith("/api/baselines/"):
                baseline_id = urllib.parse.unquote(path.removeprefix("/api/baselines/"))
                baseline = self.server.runtime.baseline_service.get(baseline_id)
                self._send_json({"ok": True, "baseline": baseline.to_dict()})
            elif path == "/api/replay/status":
                self._send_json({"ok": True, "replay": self.server.runtime.replay_status()})
            elif path.startswith("/snapshot/") and path.endswith(".jpg"):
                camera_name = path.removeprefix("/snapshot/").removesuffix(".jpg")
                frame = self.server.runtime.wait_frame(camera_name, -1, timeout=0.1)
                self._send_bytes(frame.jpeg, "image/jpeg")
            elif path.startswith("/stream/") and path.endswith(".mjpg"):
                camera_name = path.removeprefix("/stream/").removesuffix(".mjpg")
                self._stream_camera(camera_name)
            else:
                self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        except ApiError as exc:
            self._send_json({"ok": False, "error": str(exc)}, exc.status)
        except DataViewError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            pass
        except Exception as exc:
            logging.exception("GET failed")
            self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if not self.server.authorize(self.headers.get("X-Motrix-Key")):
                raise ApiError("Invalid access key", HTTPStatus.UNAUTHORIZED)
            if path == "/api/data-view/datasets":
                imported = self.server.runtime_call("data_view_import_root", self._read_json())
                self._send_json({"ok": True, **imported}, HTTPStatus.CREATED)
            elif path == "/api/data-view/datasets/remove":
                removed = self.server.runtime_call("data_view_remove_root", self._read_json())
                self._send_json({"ok": True, **removed})
            elif path == "/api/data-view/selection":
                selection = self.server.runtime_call("data_view_select", self._read_json())
                self._send_json({"ok": True, **selection})
            elif path == "/api/data-view/open-loop-evaluations":
                job = self.server.runtime_call("data_view_submit_open_loop", self._read_json())
                self._send_json({"ok": True, "job": job}, HTTPStatus.ACCEPTED)
            elif self._is_data_view_open_loop_stop(path):
                job_id = path.split("/")[4]
                job = self.server.runtime_call("data_view_stop_open_loop", job_id)
                self._send_json({"ok": True, "job": job})
            elif path == "/api/baselines":
                job = self.server.runtime_call("create_baseline", self._read_json())
                self._send_json({"ok": True, "job": job}, HTTPStatus.ACCEPTED)
            elif path == "/api/baselines/from-evaluation":
                job = self.server.runtime_call("create_baseline_from_evaluation", self._read_json())
                self._send_json({"ok": True, "job": job}, HTTPStatus.ACCEPTED)
            elif path == "/api/baselines/compare":
                payload = self._read_json()
                baseline_ids = payload.get("baseline_ids")
                if not isinstance(baseline_ids, list) or not all(isinstance(item, str) for item in baseline_ids):
                    raise ApiError("baseline_ids must be a list of IDs")
                comparison = self.server.runtime_call("compare_baselines", tuple(baseline_ids))
                self._send_json({"ok": True, "comparison": comparison})
            elif path.startswith("/api/baselines/"):
                suffix = path.removeprefix("/api/baselines/")
                baseline_id, separator, operation = suffix.partition("/")
                if not separator or not baseline_id:
                    raise ApiError("Baseline operation is required", HTTPStatus.NOT_FOUND)
                baseline_id = urllib.parse.unquote(baseline_id)
                payload = self._read_json()
                if operation == "clone":
                    job = self.server.runtime_call("clone_baseline", baseline_id, payload)
                    self._send_json({"ok": True, "job": job}, HTTPStatus.ACCEPTED)
                elif operation == "diff":
                    other = str(payload.get("other_baseline_id", ""))
                    if not other:
                        raise ApiError("other_baseline_id is required")
                    diff = self.server.runtime_call("baseline_diff", baseline_id, other)
                    self._send_json({"ok": True, "diff": diff})
                elif operation == "run":
                    evaluation = self.server.runtime_call("run_baseline", baseline_id)
                    self._send_json({"ok": True, "evaluation": evaluation})
                elif operation == "attach-open-loop":
                    job = self.server.runtime_call("attach_open_loop_baseline", baseline_id, payload)
                    self._send_json(
                        {"ok": True, "job": job},
                        HTTPStatus.ACCEPTED,
                    )
                else:
                    raise ApiError("unknown Baseline operation", HTTPStatus.NOT_FOUND)
            elif path == "/api/evaluations":
                payload = self._read_json()
                evaluation = self.server.runtime_operation(lambda runtime: runtime.evaluation_service.create(payload))
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/warmup":
                evaluation = self.server.runtime_operation(lambda runtime: runtime.evaluation_service.warmup())
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/reset-ready":
                evaluation = self.server.runtime_operation(lambda runtime: runtime.evaluation_service.reset_ready())
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/start-episode":
                evaluation = self.server.runtime_operation(lambda runtime: runtime.evaluation_service.start_episode())
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/stop-episode":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.stop_episode()})
            elif path == "/api/evaluations/current/label":
                payload = self._read_json()
                evaluation = self.server.runtime_operation(lambda runtime: runtime.evaluation_service.label(payload))
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/abort":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.abort()})
            elif path == "/api/evaluations/current/complete":
                evaluation = self.server.runtime_operation(lambda runtime: runtime.evaluation_service.complete())
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/config":
                config = self.server.runtime_call("update_config", self._read_json())
                self._send_json({"ok": True, "config": config})
            elif path == "/api/replay/plan":
                replay = self.server.runtime_call("replay_plan", self._read_json())
                self._send_json({"ok": True, "replay": replay})
            elif path == "/api/replay/connect":
                self._send_json({"ok": True, "replay": self.server.runtime_call("replay_connect")})
            elif path == "/api/replay/start":
                payload = self._read_json()
                replay = self.server.runtime_call("replay_start", str(payload.get("plan_hash", "")))
                self._send_json({"ok": True, "replay": replay})
            elif path == "/api/replay/pause":
                self._send_json({"ok": True, "replay": self.server.runtime_call("replay_pause")})
            elif path == "/api/replay/resume":
                self._send_json({"ok": True, "replay": self.server.runtime_call("replay_resume")})
            elif path == "/api/replay/stop":
                self._send_json({"ok": True, "replay": self.server.runtime.replay_stop()})
            elif path == "/api/replay/emergency-stop":
                self._send_json({"ok": True, "replay": self.server.runtime.replay_stop(emergency=True)})
            elif path == "/api/pose/select":
                pose = self.server.runtime_call("pose_select", self._read_json())
                self._send_json({"ok": True, "pose": pose})
            elif path == "/api/pose/connect":
                self._send_json({"ok": True, "pose": self.server.runtime_call("pose_connect")})
            elif path == "/api/pose/execute":
                payload = self._read_json()
                pose = self.server.runtime_call("pose_execute", str(payload.get("plan_hash", "")))
                self._send_json({"ok": True, "pose": pose})
            elif path == "/api/pose/stop":
                self._send_json({"ok": True, "pose": self.server.runtime.pose_stop()})
            elif path == "/api/pose/prepare-deployment":
                payload = self._read_json()
                pose = self.server.runtime_call("pose_prepare_deployment", str(payload.get("plan_hash", "")))
                self._send_json({"ok": True, "pose": pose})
            elif path == "/api/pose/start-deployment":
                payload = self._read_json()
                pose = self.server.runtime_call("pose_start_deployment", str(payload.get("plan_hash", "")))
                self._send_json({"ok": True, "pose": pose})
            elif path == "/api/connect":
                self._send_json({"ok": True, "status": self.server.runtime_call("connect")})
            elif path == "/api/start":
                self._send_json({"ok": True, "status": self.server.runtime_call("start")})
            elif path == "/api/stop":
                self._send_json({"ok": True, "status": self.server.runtime.stop(wait=False)})
            elif path == "/api/disconnect":
                self._send_json({"ok": True, "status": self.server.runtime_call("disconnect")})
            elif path == "/api/reset":
                self._send_json({"ok": True, "status": self.server.runtime_call("reset_arms")})
            elif path == "/api/ping_policy":
                self._send_json(self.server.runtime_call("ping_policy"))
            elif path == "/api/robot":
                payload = self._read_json()
                self._send_json({"ok": True, "config": self.server.select_robot(str(payload.get("robot", "")))})
            else:
                self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        except EvaluationConflict as exc:
            self._send_json(
                {"ok": False, "error": str(exc), "legal_operations": list(exc.legal_operations)},
                HTTPStatus.CONFLICT,
            )
        except BaselineConfigurationConflict as exc:
            self._send_json({"ok": False, "error": str(exc), "diff": exc.diff.to_dict()}, HTTPStatus.CONFLICT)
        except ApiError as exc:
            self._send_json({"ok": False, "error": str(exc)}, exc.status)
        except DataViewError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logging.exception("POST failed")
            self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    @staticmethod
    def _query_text(query: Mapping[str, list[str]], name: str, default: str | None = None) -> str:
        values = query.get(name)
        if not values:
            if default is None:
                raise DataViewError(f"missing query parameter: {name}")
            return default
        return values[0]

    def _query_int(self, query: Mapping[str, list[str]], name: str) -> int:
        return int(self._query_text(query, name))

    def _query_float(self, query: Mapping[str, list[str]], name: str) -> float:
        return float(self._query_text(query, name))

    @staticmethod
    def _is_data_view_open_loop_stop(path: str) -> bool:
        parts = path.split("/")
        return (
            len(parts) == 6
            and parts[:4] == ["", "api", "data-view", "open-loop-evaluations"]
            and bool(parts[4])
            and parts[5] == "stop"
        )

    def _handle_data_view_get(self, path: str, query: Mapping[str, list[str]]) -> None:
        """Serve the standalone mp-data-view API from the shared Web runtime."""
        if path == "/api/data-view/status":
            self._send_json({"ok": True, "data_view": self.server.runtime_call("data_view_status")})
            return
        if path == "/api/data-view/datasets":
            self._send_json({"ok": True, "datasets": self.server.runtime_call("data_view_datasets")})
            return
        if path == "/api/data-view/selection":
            self._send_json({"ok": True, **self.server.runtime_call("data_view_selection")})
            return
        if path == "/api/data-view/open-loop-evaluations":
            self._send_json({"ok": True, "jobs": self.server.runtime_call("data_view_open_loop_jobs")})
            return

        open_loop_prefix = "/api/data-view/open-loop-evaluations/"
        if path.startswith(open_loop_prefix):
            parts = path.split("/")
            if len(parts) == 5 and parts[4]:
                self._send_json({"ok": True, "job": self.server.runtime_call("data_view_open_loop_job", parts[4])})
                return
            if len(parts) == 7 and parts[4] and parts[5] == "reports":
                report = self.server.runtime_call(
                    "data_view_open_loop_report",
                    parts[4],
                    int(parts[6]),
                    include_curves=self._query_text(query, "curves", "0") == "1",
                )
                self._send_json({"ok": True, "report": report})
                return
            raise ApiError("not found", HTTPStatus.NOT_FOUND)

        parts = path.split("/")
        # /api/data-view/datasets/{id}/episodes[/index/{operation}]
        if len(parts) < 6 or parts[3] != "datasets" or parts[5] != "episodes":
            raise ApiError("not found", HTTPStatus.NOT_FOUND)
        dataset_id = urllib.parse.unquote(parts[4])
        if len(parts) == 6:
            self._send_json({"ok": True, "episodes": self.server.runtime_call("data_view_episodes", dataset_id)})
            return
        if len(parts) < 8:
            raise ApiError("not found", HTTPStatus.NOT_FOUND)
        episode_index = int(parts[6])
        operation = parts[7]
        if operation == "metadata":
            self._send_json(
                {"ok": True, **self.server.runtime_call("data_view_episode_metadata", dataset_id, episode_index)}
            )
        elif operation == "sample":
            self._send_json(
                {
                    "ok": True,
                    **self.server.runtime_call(
                        "data_view_sample", dataset_id, episode_index, self._query_int(query, "sample_index")
                    ),
                }
            )
        elif operation == "sample-at":
            self._send_json(
                {
                    "ok": True,
                    **self.server.runtime_call(
                        "data_view_sample_at_timestamp",
                        dataset_id,
                        episode_index,
                        self._query_float(query, "timestamp"),
                    ),
                }
            )
        elif operation == "frame":
            frame, metadata = self.server.runtime_call(
                "data_view_camera_frame",
                dataset_id,
                episode_index,
                self._query_int(query, "sample_index"),
                self._query_text(query, "role"),
            )
            self._send_data_view_jpeg(frame, metadata)
        elif operation == "curves":
            requested = tuple(item for item in self._query_text(query, "series", "action").split(",") if item)
            max_points = int(self._query_text(query, "max_points", "600"))
            self._send_json(
                {
                    "ok": True,
                    **self.server.runtime_call(
                        "data_view_curves",
                        dataset_id,
                        episode_index,
                        series=requested,
                        max_points=max_points,
                    ),
                }
            )
        elif operation == "events":
            limit = int(self._query_text(query, "limit", "2000"))
            self._send_json(
                {
                    "ok": True,
                    **self.server.runtime_call("data_view_runtime_events", dataset_id, episode_index, limit=limit),
                }
            )
        elif operation == "metrics":
            self._send_json({"ok": True, **self.server.runtime_call("data_view_metrics", dataset_id, episode_index)})
        else:
            raise ApiError("not found", HTTPStatus.NOT_FOUND)

    def _send_data_view_jpeg(self, frame: np.ndarray, metadata: Mapping[str, Any]) -> None:
        data = _encode_jpeg_rgb(np.asarray(frame, dtype=np.uint8), quality=90)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Index", str(metadata["frame_index"]))
        self.send_header("X-Frame-Id", str(metadata["frame_id"]))
        self.send_header("X-Rendered-Frame-Index", str(metadata.get("rendered_frame_index", metadata["frame_index"])))
        self.send_header("X-Frame-Reused", "true" if metadata.get("frame_reused") else "false")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: pathlib.Path) -> None:
        root = self.server.static_dir.resolve()
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise ApiError("not found", HTTPStatus.NOT_FOUND)
        if not resolved.is_file():
            raise ApiError("not found", HTTPStatus.NOT_FOUND)
        content_type = "text/html; charset=utf-8"
        if resolved.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif resolved.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self._send_bytes(resolved.read_bytes(), content_type)

    def _stream_camera(self, camera_name: str) -> None:
        if camera_name not in self.server.runtime.get_config().get("camera_roles", []):
            raise ApiError(f"Unknown camera {camera_name}", HTTPStatus.NOT_FOUND)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_sequence = -1
        while True:
            frame = self.server.runtime.wait_frame(camera_name, last_sequence)
            last_sequence = frame.sequence
            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame.jpeg)}\r\n\r\n".encode("ascii")
                + frame.jpeg
                + b"\r\n"
            )
            self.wfile.write(chunk)
            self.wfile.flush()


class PiperWebServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[PiperWebHandler],
        runtime: Any,
        *,
        access_key: str | None = None,
    ) -> None:
        super().__init__(server_address, handler)
        self.runtime = runtime
        # do_POST holds this lock and /api/robot re-enters it through
        # select_robot(), hence an RLock is intentional.
        self._runtime_switch_lock = threading.RLock()
        self.access_key = access_key
        packaged_static = pathlib.Path(__file__).resolve().parent / "static"
        source_static = pathlib.Path(__file__).resolve().parents[3] / "static"
        self.static_dir = packaged_static if packaged_static.is_dir() else source_static

    def authorize(self, provided_key: str | None) -> bool:
        return self.access_key is None or (
            provided_key is not None and secrets.compare_digest(provided_key, self.access_key)
        )

    def runtime_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a state-changing runtime method without racing robot switch.

        Request parsing and response I/O intentionally stay outside this
        short critical section. Stop and emergency-stop handlers call the
        runtime directly so safety operations are never queued behind an
        unrelated request holding the switch lock. Disconnect is serialized
        because it deletes SDK handles and must not race Reset.
        """

        with self._runtime_switch_lock:
            method = getattr(self.runtime, method_name)
            return method(*args, **kwargs)

    def runtime_operation(self, operation: Callable[[Any], Any]) -> Any:
        with self._runtime_switch_lock:
            return operation(self.runtime)

    def select_robot(self, name: str) -> dict[str, Any]:
        with self._runtime_switch_lock:
            if name not in {"piper", "rm2"}:
                raise ApiError("robot must be piper or rm2")
            status = self.runtime.status()
            if status.get("cleanup_pending"):
                raise ApiError(
                    "Retry Disconnect until deployment cleanup completes before changing robot",
                    HTTPStatus.CONFLICT,
                )
            if (
                status["connected"]
                or status["running"]
                or status.get("stop_requested")
                or status.get("can_stop")
            ):
                raise ApiError(
                    "Stop and Disconnect the active runtime before changing robot",
                    HTTPStatus.CONFLICT,
                )
            if self.runtime.pose_status()["phase"] not in {
                "idle",
                "offline_preflighted",
                "offline_rejected",
                "failed",
                "aborted",
            }:
                raise ApiError("Finish the recorded-state pose session before changing robot", HTTPStatus.CONFLICT)
            self.runtime.shutdown_baselines()
            profile = get_web_profile(name)
            self.runtime = RobotWebRuntime(
                profile=profile,
                policy_timeout=self.runtime._policy_timeout,
                resource_manager=self.runtime.resource_manager,
                recorded_data_roots=self.runtime._startup_recorded_data_roots,
                data_view_web_roots=self.runtime._data_view_web_roots,
                pose_mapping_config=self.runtime._pose_mapping_config,
                replay_record_root=self.runtime._replay_record_root,
                baseline_root=self.runtime._baseline_root,
                open_loop_output_root=self.runtime._open_loop_output_root,
            )
            return self.runtime.get_config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot web control panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--camera-profile",
        choices=("hardware", "black"),
        default="hardware",
        help="Use hardware camera defaults, or black placeholders for deployment and policy-connect checks.",
    )
    parser.add_argument("--server-url", help="Initial policy websocket URL. It can still be changed in Settings.")
    parser.add_argument("--api-key", help="Initial policy API key. It can still be changed in Settings.")
    parser.add_argument("--prompt", help="Initial policy prompt.")
    parser.add_argument("--left-can", help="Initial left Piper CAN interface.")
    parser.add_argument("--right-can", help="Initial right Piper CAN interface.")
    for name in CAMERA_NAMES:
        option = name.removeprefix("cam_").replace("_", "-")
        parser.add_argument(f"--cam-{option}-backend", choices=CAMERA_BACKENDS)
        parser.add_argument(f"--cam-{option}")
    parser.add_argument("--dry-run", action="store_true", help="Do not send arm action commands.")
    parser.add_argument("--no-enable-on-start", action="store_false", dest="enable_on_start", default=None)
    parser.add_argument("--no-reset-on-start", action="store_false", dest="reset_on_start", default=None)
    parser.add_argument("--policy-timeout", type=float, default=3.0)
    parser.add_argument("--robot", choices=("piper", "rm2"), default="piper")
    parser.add_argument("--access-key", default=None, help="Require X-Motrix-Key for Web control requests.")
    parser.add_argument(
        "--recorded-data-root",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Allowed local recording root for safe recorded-state selection; may be repeated.",
    )
    parser.add_argument(
        "--pose-mapping-config",
        type=pathlib.Path,
        help="Versioned explicit JSON mapping for a non-identical recorded-state schema.",
    )
    parser.add_argument(
        "--replay-record-root",
        type=pathlib.Path,
        default=pathlib.Path("recordings/replay"),
        help="Directory for atomic, explicit real-robot replay records.",
    )
    parser.add_argument(
        "--baseline-root",
        type=pathlib.Path,
        default=pathlib.Path("recordings/baselines"),
        help="Directory for immutable Baseline JSON documents and compact run references.",
    )
    parser.add_argument(
        "--open-loop-output-root",
        type=pathlib.Path,
        default=pathlib.Path("open_loop_results"),
        help="Directory for isolated DATA_VIEW teacher-forced evaluation artifacts.",
    )
    cli_args = parser.parse_args()

    if cli_args.policy_timeout <= 0:
        parser.error("--policy-timeout must be positive")

    try:
        pose_mapping_config = (
            load_pose_mapping_config(cli_args.pose_mapping_config)
            if cli_args.pose_mapping_config is not None
            else None
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        parser.error(f"invalid --pose-mapping-config: {exc}")

    profile = get_web_profile(cli_args.robot)
    runtime_args = (
        _default_args(camera_profile=cli_args.camera_profile)
        if cli_args.robot == "piper"
        else clone_default_args(profile)
    )
    for field in ("server_url", "api_key", "prompt", "enable_on_start", "reset_on_start"):
        value = getattr(cli_args, field)
        if value is not None and hasattr(runtime_args, field):
            setattr(runtime_args, field, value)
    if cli_args.robot == "piper":
        for field in ("left_can", "right_can"):
            value = getattr(cli_args, field)
            if value is not None:
                setattr(runtime_args, field, value)
        for name in CAMERA_NAMES:
            backend = getattr(cli_args, f"{name}_backend") or getattr(runtime_args, f"{name}_backend")
            setattr(runtime_args, f"{name}_backend", backend)
            selector = getattr(cli_args, name)
            if selector is not None:
                setattr(runtime_args, name, selector)
    runtime_args.dry_run = cli_args.dry_run

    logging.basicConfig(
        level=getattr(logging, cli_args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    runtime: Any = RobotWebRuntime(
        runtime_args,
        profile=profile,
        policy_timeout=cli_args.policy_timeout,
        recorded_data_roots=tuple(cli_args.recorded_data_root),
        pose_mapping_config=pose_mapping_config,
        replay_record_root=cli_args.replay_record_root,
        baseline_root=cli_args.baseline_root,
        open_loop_output_root=cli_args.open_loop_output_root,
    )
    access_key = cli_args.access_key or os.environ.get("MOTRIX_WEB_ACCESS_KEY")
    server = PiperWebServer((cli_args.host, cli_args.port), PiperWebHandler, runtime, access_key=access_key)
    logging.info("Robot web UI (%s): http://%s:%s", profile.robot_name, cli_args.host, cli_args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping server")
    finally:
        try:
            runtime.evaluation_service.shutdown()
        except Exception:
            logging.exception("Failed to stop evaluation workers during shutdown")
        try:
            runtime.shutdown_baselines()
        except Exception:
            logging.exception("Failed to stop Baseline writer during shutdown")
        try:
            runtime.disconnect()
        except Exception:
            logging.exception("Failed to disconnect runtime during shutdown")
        server.server_close()


if __name__ == "__main__":
    main()
