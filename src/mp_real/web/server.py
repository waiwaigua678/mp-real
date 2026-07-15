from __future__ import annotations

# ruff: noqa: I001, N802

import argparse
from collections import deque
from collections.abc import Callable
import copy
import dataclasses
import enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import logging
import os
import pathlib
import secrets
import threading
import time
from typing import Any
import urllib.parse

import numpy as np
import websockets.sync.client

try:
    import cv2
except ImportError:
    cv2 = None

from mp_real.common.image import preprocess_image
from mp_real.common.runtime import rtc_replan_stride
from mp_real.evaluation.service import EvaluationConflict, EvaluationRuntimeLease, EvaluationService
from mp_real.policy_client import msgpack_numpy
from mp_real.robots.base import Robot
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
RESET_LEFT_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RESET_RIGHT_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RESET_LEFT_GRIPPER = 1.0
RESET_RIGHT_GRIPPER = 1.0

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


def _runtime_mode(value: Any) -> RuntimeMode:
    try:
        return RuntimeMode(str(value))
    except ValueError as exc:
        available = ", ".join(mode.value for mode in RuntimeMode)
        raise ApiError(f"runtime_mode must be one of: {available}") from exc


class PolicyClient:
    def __init__(
        self,
        server_url: str,
        api_key: str | None,
        *,
        timeout: float,
        metadata_timeout: float | None = None,
    ) -> None:
        self.uri = self._normalize_uri(server_url)
        self.timeout = timeout
        self.connect_latency_ms: float | None = None
        self.metadata_latency_ms: float | None = None
        self._packer = msgpack_numpy.Packer()
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        connect_kwargs = {
            "compression": None,
            "max_size": None,
            "additional_headers": headers,
        }
        connect_started_ns = time.monotonic_ns()
        try:
            self._ws = websockets.sync.client.connect(self.uri, open_timeout=timeout, **connect_kwargs)
        except TypeError:
            self._ws = websockets.sync.client.connect(self.uri, **connect_kwargs)
        self.connect_latency_ms = (time.monotonic_ns() - connect_started_ns) / 1e6

        metadata_started_ns = time.monotonic_ns()
        previous_timeout = self.timeout
        self.timeout = metadata_timeout if metadata_timeout is not None else timeout
        try:
            self.metadata = msgpack_numpy.unpackb(self._recv())
        finally:
            self.timeout = previous_timeout
        self.metadata_latency_ms = (time.monotonic_ns() - metadata_started_ns) / 1e6

    @staticmethod
    def _normalize_uri(server_url: str) -> str:
        server_url = server_url.strip()
        if not server_url:
            raise ValueError("server_url cannot be empty")
        parsed = urllib.parse.urlparse(server_url)
        if parsed.scheme:
            return server_url
        return f"ws://{server_url}"

    def _recv(self) -> bytes | str:
        try:
            return self._ws.recv(timeout=self.timeout)
        except TypeError:
            return self._ws.recv()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def set_timeout(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("Policy timeout must be positive")
        self.timeout = timeout_s

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            logging.exception("Failed to close policy websocket")


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
        self._evaluation_service = EvaluationService(self)

        self._controller: RuntimeController | None = None
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
            for field in ("fps", "camera_timeout", "action_smoothing", "gripper_smoothing"):
                if field in payload:
                    value = float(payload[field])
                    if field in {"fps", "camera_timeout"} and value <= 0:
                        raise ApiError(f"{field} must be positive")
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
            ):
                if field in payload:
                    value = int(payload[field])
                    if value < 0 or (field not in {"rtc_replan_stride", "rtc_prefetch_steps"} and value == 0):
                        raise ApiError(f"{field} must be positive")
                    setattr(args, field, value)
            if "max_steps" in payload:
                args.max_steps = _coerce_optional_int(payload["max_steps"])
            if "camera_backend" in payload:
                args.camera_backend = str(payload["camera_backend"])
            if "policy_joint_unit" in payload:
                args.policy_joint_unit = str(payload["policy_joint_unit"])
            if "arm_port" in payload:
                args.arm_port = int(payload["arm_port"]) if payload["arm_port"] not in (None, "") else None
            for field in ("dry_run", "infer_only", "use_rtc", "reset_on_start", "hold_last_action", "log_timing"):
                if field in payload:
                    setattr(args, field, _coerce_bool(payload[field]))
            for field in (
                "policy_connect_timeout_s",
                "policy_metadata_timeout_s",
                "policy_warmup_timeout_s",
                "policy_inference_timeout_s",
                "policy_timeout",
                "camera_stream_fps",
            ):
                if field in payload:
                    value = float(payload[field])
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
                self._policy_warmup_requests = int(payload["policy_warmup_requests"])
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

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_controller_state_locked()
            if self._evaluation_owner is not None:
                raise ApiError("Evaluation owns robot control; use the evaluation API", HTTPStatus.CONFLICT)
            mode = self._runtime_mode
            if mode is RuntimeMode.CAMERA_PREVIEW:
                if self._connected:
                    return self.status()
                args = copy.deepcopy(self._args)
            elif mode is RuntimeMode.OFFLINE_REPLAY:
                if not self._connected:
                    self._connect_offline_replay_locked()
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
            if self._evaluation_owner is not None:
                raise ApiError(
                    "Evaluation owns this deployment; it must finish before reconnecting",
                    HTTPStatus.CONFLICT,
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
        self.log(f"Connecting {self._profile.robot_name} robot")

        client: PolicyClient | None = None
        robot: Robot | None = None
        cameras: dict[str, infer_piper.Camera] = {}
        controller: RuntimeController | None = None
        try:
            robot = self._robot_factory(self._profile.robot_name, args)
            if stop_event.is_set():
                raise PolicyStartupCancelled("Deployment connection was cancelled")
            if not args.infer_only:
                robot.reset()
            if stop_event.is_set():
                raise PolicyStartupCancelled("Deployment connection was cancelled")

            self.log(f"Connecting policy server {args.server_url}")
            client = self._create_policy_client(args)
            if stop_event.is_set():
                raise PolicyStartupCancelled("Deployment connection was cancelled")
            self.log("Connected policy server")
            cameras = self._camera_factory(args)
            if stop_event.is_set():
                raise PolicyStartupCancelled("Deployment connection was cancelled")
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
            if cameras:
                close_profile_cameras(cameras)
            if controller is not None:
                controller.close()
            else:
                if robot is not None:
                    robot.close()
                if client is not None:
                    client.close()
            with self._lock:
                if generation_id != self._connect_generation_id:
                    return
                self._connected = False
                self._policy_connected = False
                if isinstance(exc, PolicyStartupCancelled):
                    self._policy_state = "DISCONNECTED"
                    self._phase = "idle"
                    self._last_error = None
                    self._stop_requested = False
                    self._start_after_connect = False
                else:
                    self._policy_state = "ERROR"
                    self._phase = "error"
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self._release_resources_locked()
            self.log(f"Connect failed: {type(exc).__name__}: {exc}")
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
            if cameras:
                close_profile_cameras(cameras)
            controller.close()
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
            controller = self._controller
            cameras = self._cameras
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

        if cameras:
            close_profile_cameras(cameras)
        if controller is not None:
            controller.close()
        with self._lock:
            self._release_resources_locked()
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

    def _latest_images_for_inference(self, args: Any) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
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
                    return images, camera_params
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
            can_reset = (
                self._runtime_mode is RuntimeMode.DEPLOYMENT
                and self._controller is not None
                and not startup_active
                and (not self._running or not self._policy_connected)
            )
            return {
                "robot": self._profile.robot_name,
                "camera_roles": list(self._profile.camera_roles_for_args(self._args)),
                "action_spec": dataclasses.asdict(self._profile.action_spec_for_args(self._args)),
                "capabilities": dataclasses.asdict(self._profile.capabilities),
                "runtime_mode": self._runtime_mode.value,
                "phase": self._phase,
                "policy_state": self._policy_state,
                "connected": self._connected,
                "policy_connected": self._policy_connected,
                "running": self._running,
                "stop_requested": self._stop_requested,
                "can_edit_config": self._can_edit_config_locked(),
                "can_edit_connection_config": self._can_edit_connection_config_locked(),
                "can_connect": not self._connected
                and not self._running
                and not self._stop_requested
                and not connection_active
                and not startup_active
                and self._phase in {"idle", "error", "warmup_failed"},
                "can_start": (
                    self._runtime_mode is RuntimeMode.CAMERA_PREVIEW
                    and not self._connected
                    and self._phase in {"idle", "error"}
                )
                or (
                    self._runtime_mode is RuntimeMode.DEPLOYMENT
                    and not self._running
                    and not self._stop_requested
                    and not connection_active
                    and not startup_active
                    and self._phase in {"idle", "stopped", "warmup_failed", "error"}
                ),
                "can_stop": self._running
                or self._stop_requested
                or connection_active
                or startup_active
                or (self._runtime_mode is RuntimeMode.CAMERA_PREVIEW and self._phase == "previewing"),
                "can_disconnect": self._connected
                or self._running
                or connection_active
                or startup_active
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

    def _latest_policy_images(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        deadline = time.monotonic() + self._args.camera_timeout
        with self._frame_condition:
            while any(self._frames[name].image is None or self._frames[name].error for name in CAMERA_NAMES):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("RM2 camera frames are unavailable")
                self._frame_condition.wait(min(0.05, remaining))
            images = {policy: self._frames[stream].image.copy() for stream, policy in _RM_STREAM_TO_POLICY.items()}
        return images, {name: camera.camera_info() for name, camera in self._cameras.items()}

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
            elif path.startswith("/static/"):
                self._send_file(self.server.static_dir / path.removeprefix("/static/"))
            elif path == "/api/status":
                self._send_json(self.server.runtime.status())
            elif path == "/api/config":
                self._send_json(self.server.runtime.get_config())
            elif path == "/api/evaluations/current":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.current()})
            elif path == "/api/replay/status":
                self._send_json(
                    {
                        "ok": True,
                        "available": False,
                        "message": "Recorded-session playback will be implemented in stage 7.",
                    }
                )
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
        except BrokenPipeError:
            pass
        except Exception as exc:
            logging.exception("GET failed")
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if not self.server.authorize(self.headers.get("X-Motrix-Key")):
                raise ApiError("Invalid access key", HTTPStatus.UNAUTHORIZED)
            if path == "/api/evaluations":
                evaluation = self.server.runtime.evaluation_service.create(self._read_json())
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/warmup":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.warmup()})
            elif path == "/api/evaluations/current/reset-ready":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.reset_ready()})
            elif path == "/api/evaluations/current/start-episode":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.start_episode()})
            elif path == "/api/evaluations/current/stop-episode":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.stop_episode()})
            elif path == "/api/evaluations/current/label":
                evaluation = self.server.runtime.evaluation_service.label(self._read_json())
                self._send_json({"ok": True, "evaluation": evaluation})
            elif path == "/api/evaluations/current/abort":
                self._send_json({"ok": True, "evaluation": self.server.runtime.evaluation_service.abort()})
            elif path == "/api/config":
                self._send_json({"ok": True, "config": self.server.runtime.update_config(self._read_json())})
            elif path == "/api/connect":
                self._send_json({"ok": True, "status": self.server.runtime.connect()})
            elif path == "/api/start":
                self._send_json({"ok": True, "status": self.server.runtime.start()})
            elif path == "/api/stop":
                self._send_json({"ok": True, "status": self.server.runtime.stop(wait=False)})
            elif path == "/api/disconnect":
                self._send_json({"ok": True, "status": self.server.runtime.disconnect()})
            elif path == "/api/reset":
                self._send_json({"ok": True, "status": self.server.runtime.reset_arms()})
            elif path == "/api/ping_policy":
                self._send_json(self.server.runtime.ping_policy())
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
        except ApiError as exc:
            self._send_json({"ok": False, "error": str(exc)}, exc.status)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logging.exception("POST failed")
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

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
        self.access_key = access_key
        packaged_static = pathlib.Path(__file__).resolve().parent / "static"
        source_static = pathlib.Path(__file__).resolve().parents[3] / "static"
        self.static_dir = packaged_static if packaged_static.is_dir() else source_static

    def authorize(self, provided_key: str | None) -> bool:
        return self.access_key is None or (
            provided_key is not None and secrets.compare_digest(provided_key, self.access_key)
        )

    def select_robot(self, name: str) -> dict[str, Any]:
        if name not in {"piper", "rm2"}:
            raise ApiError("robot must be piper or rm2")
        if self.runtime.status()["connected"] or self.runtime.status()["running"]:
            raise ApiError("Disconnect before changing robot", HTTPStatus.CONFLICT)
        profile = get_web_profile(name)
        self.runtime = RobotWebRuntime(profile=profile, resource_manager=self.runtime.resource_manager)
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
    cli_args = parser.parse_args()

    if cli_args.policy_timeout <= 0:
        parser.error("--policy-timeout must be positive")

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
    runtime: Any = RobotWebRuntime(runtime_args, profile=profile, policy_timeout=cli_args.policy_timeout)
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
            runtime.disconnect()
        except Exception:
            logging.exception("Failed to disconnect runtime during shutdown")
        server.server_close()


if __name__ == "__main__":
    main()
