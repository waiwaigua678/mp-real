from __future__ import annotations

# ruff: noqa: I001, N802

import argparse
from collections import deque
import copy
import dataclasses
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import logging
import os
import pathlib
import queue
import secrets
import threading
import time
import traceback
from typing import Any
import urllib.parse

import numpy as np
import websockets.sync.client

try:
    import cv2
except ImportError:
    cv2 = None

from mp_real.common.image import preprocess_image
from mp_real.common.runtime import RealTimeChunkingBuffer, rtc_replan_stride, select_rtc_cursor
from mp_real.policy_client import msgpack_numpy
from mp_real.robots.piper import infer as infer_piper
from mp_real.robots.rm2 import infer as infer_rm2
from mp_real.robots.registry import create_robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import run_policy_loop


CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_BACKENDS = ("realsense", "v4l2", "black")
ARM_COMMAND_MODES = ("move_j", "move_js", "auto")
RESET_LEFT_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RESET_RIGHT_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
RESET_LEFT_GRIPPER = 1.0
RESET_RIGHT_GRIPPER = 1.0

CONNECTION_CONFIG_FIELDS = {
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


class PolicyClient:
    def __init__(self, server_url: str, api_key: str | None, *, timeout: float) -> None:
        self.uri = self._normalize_uri(server_url)
        self.timeout = timeout
        self._packer = msgpack_numpy.Packer()
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        connect_kwargs = {
            "compression": None,
            "max_size": None,
            "additional_headers": headers,
        }
        try:
            self._ws = websockets.sync.client.connect(self.uri, open_timeout=timeout, **connect_kwargs)
        except TypeError:
            self._ws = websockets.sync.client.connect(self.uri, **connect_kwargs)

        self.metadata = msgpack_numpy.unpackb(self._recv())

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


def _config_to_dict(args: infer_piper.Args, *, camera_stream_fps: float, policy_timeout: float) -> dict[str, Any]:
    return {
        "robot": "piper",
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


class PiperWebRuntime:
    def __init__(self, initial_args: infer_piper.Args | None = None, *, policy_timeout: float = 3.0) -> None:
        self._lock = threading.RLock()
        self._robot_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._args = copy.deepcopy(initial_args) if initial_args is not None else _default_args()
        self._camera_stream_fps = 10.0
        self._policy_timeout = policy_timeout

        self._client: PolicyClient | None = None
        self._left: infer_piper.PiperArm | None = None
        self._right: infer_piper.PiperArm | None = None
        self._cameras: dict[str, infer_piper.Camera] = {}
        self._frames: dict[str, FrameSnapshot] = {
            name: FrameSnapshot(
                image=_black_frame(self._args.resize_size), jpeg=_encode_jpeg_rgb(_black_frame(self._args.resize_size))
            )
            for name in CAMERA_NAMES
        }

        self._run_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._camera_stop_event = threading.Event()
        self._logs: deque[str] = deque(maxlen=200)

        self._connected = False
        self._policy_connected = False
        self._running = False
        self._stop_requested = False
        self._phase = "idle"
        self._last_error: str | None = None
        self._server_metadata: dict[str, Any] | None = None

        self._step = 0
        self._action_queue_len = 0
        self._last_infer_latency_ms: float | None = None
        self._last_loop_ms: float | None = None
        self._last_control_hz: float | None = None
        self._last_infer_hz: float | None = None
        self._started_at: float | None = None

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self._lock:
            self._logs.append(line)
        logging.info(message)

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            return _config_to_dict(
                copy.deepcopy(self._args),
                camera_stream_fps=self._camera_stream_fps,
                policy_timeout=self._policy_timeout,
            )

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._can_edit_config_locked():
                raise ApiError("Parameters can only be changed while the robot is not running")
            args = copy.deepcopy(self._args)
            camera_stream_fps = self._camera_stream_fps
            policy_timeout = self._policy_timeout
            protected_before = _config_to_dict(
                args,
                camera_stream_fps=camera_stream_fps,
                policy_timeout=policy_timeout,
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

            protected_after = _config_to_dict(
                args,
                camera_stream_fps=camera_stream_fps,
                policy_timeout=policy_timeout,
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
            if not self._connected:
                self._reset_placeholder_frames_locked(args.resize_size)
            self.log("Updated runtime parameters")
            return self.get_config()

    def _reset_placeholder_frames_locked(self, resize_size: int) -> None:
        placeholder = _black_frame(resize_size)
        jpeg = _encode_jpeg_rgb(placeholder)
        now = time.monotonic()
        self._frames = {
            name: FrameSnapshot(image=placeholder.copy(), jpeg=jpeg, sequence=0, updated_at=now)
            for name in CAMERA_NAMES
        }
        self._frame_condition.notify_all()

    def _can_edit_config_locked(self) -> bool:
        return not self._running and not self._stop_requested and self._phase not in {"connecting"}

    def _can_edit_connection_config_locked(self) -> bool:
        return self._can_edit_config_locked() and not self._connected

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running or self._stop_requested:
                raise ApiError("Runtime is already running or stopping", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            camera_stream_fps = self._camera_stream_fps
            policy_timeout = self._policy_timeout

        if not self._connected:
            self._connect(args, policy_timeout=policy_timeout)

        with self._lock:
            if self._running:
                raise ApiError("Runtime is already running", HTTPStatus.CONFLICT)
            if self._client is None or self._left is None or self._right is None:
                raise ApiError("Runtime is not fully connected", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            self._apply_arm_runtime_settings(args)
            self._stop_event.clear()
            self._stop_requested = False
            self._running = True
            self._phase = "running"
            self._last_error = None
            self._step = 0
            self._action_queue_len = 0
            self._last_infer_latency_ms = None
            self._last_loop_ms = None
            self._last_control_hz = None
            self._last_infer_hz = None
            self._started_at = time.monotonic()
            self._ensure_camera_thread_locked(args, camera_stream_fps)
            self._run_thread = threading.Thread(target=self._run_loop, args=(args,), daemon=True, name="piper-web-run")
            self._run_thread.start()
            self.log("Started inference loop")
            return self.status()

    def connect(self) -> dict[str, Any]:
        with self._lock:
            if self._running or self._stop_requested:
                raise ApiError("Runtime is running or stopping", HTTPStatus.CONFLICT)
            if self._connected:
                return self.status()
            args = copy.deepcopy(self._args)
            policy_timeout = self._policy_timeout

        self._connect(args, policy_timeout=policy_timeout)
        return self.status()

    def _connect(self, args: infer_piper.Args, *, policy_timeout: float) -> None:
        with self._lock:
            self._phase = "connecting"
            self._last_error = None
        self.log("Connecting Piper arms")

        client: PolicyClient | None = None
        left: infer_piper.PiperArm | None = None
        right: infer_piper.PiperArm | None = None
        cameras: dict[str, infer_piper.Camera] = {}
        try:
            robot = create_robot("piper", args)
            if not isinstance(robot, infer_piper.PiperRobot):
                raise RuntimeError("Piper registry factory returned an incompatible robot")
            left = robot.left
            right = robot.right
            if not args.infer_only:
                with self._robot_lock:
                    infer_piper.maybe_reset_arms(left, right, args)

            self.log(f"Connecting policy server {args.server_url}")
            client = PolicyClient(args.server_url, args.api_key, timeout=policy_timeout)
            self.log("Connected policy server")
            cameras = infer_piper.make_cameras(args)
        except Exception as exc:
            if cameras:
                infer_piper.close_cameras(cameras)
            infer_piper.close_arm(left)
            infer_piper.close_arm(right)
            if client is not None:
                client.close()
            with self._lock:
                self._connected = False
                self._policy_connected = False
                self._phase = "error"
                self._last_error = str(exc)
            self.log(f"Connect failed: {exc}")
            raise ApiError(str(exc), HTTPStatus.BAD_GATEWAY) from exc

        with self._lock:
            self._client = client
            self._left = left
            self._right = right
            self._cameras = cameras
            self._server_metadata = client.metadata
            self._connected = True
            self._policy_connected = True
            self._phase = "stopped"
            self._ensure_camera_thread_locked(args, self._camera_stream_fps)

    def _ensure_camera_thread_locked(self, args: infer_piper.Args, camera_stream_fps: float) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(copy.deepcopy(args), camera_stream_fps),
            daemon=True,
            name="piper-web-camera",
        )
        self._camera_thread.start()

    def stop(self, *, wait: bool = False) -> dict[str, Any]:
        with self._lock:
            if not self._running and not self._stop_requested:
                return self.status()
            self._stop_requested = True
            self._stop_event.set()
            thread = self._run_thread
        self.log("Stop requested")
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.stop(wait=True)
        with self._lock:
            if self._running:
                raise ApiError("Runtime is still stopping; retry disconnect shortly", HTTPStatus.CONFLICT)
            self._camera_stop_event.set()
            camera_thread = self._camera_thread

        if camera_thread is not None and camera_thread is not threading.current_thread():
            camera_thread.join(timeout=max(1.0, self._args.camera_timeout + 1.0))

        with self._lock:
            client = self._client
            left = self._left
            right = self._right
            cameras = self._cameras
            self._client = None
            self._left = None
            self._right = None
            self._cameras = {}
            self._camera_thread = None
            self._run_thread = None
            self._connected = False
            self._policy_connected = False
            self._running = False
            self._stop_requested = False
            self._phase = "idle"
            self._last_error = None
            self._server_metadata = None
            self._action_queue_len = 0
            self._reset_placeholder_frames_locked(self._args.resize_size)

        if cameras:
            infer_piper.close_cameras(cameras)
        infer_piper.close_arm(left)
        infer_piper.close_arm(right)
        if client is not None:
            client.close()
        self.log("Disconnected runtime")
        return self.status()

    def reset_arms(self) -> dict[str, Any]:
        with self._lock:
            if self._running and self._policy_connected:
                raise ApiError("Reset is only allowed while stopped or when the policy server is disconnected")
            left = self._left
            right = self._right
            args = copy.deepcopy(self._args)

        if self._running:
            self.stop(wait=True)
        if left is None or right is None:
            raise ApiError("Arms are not connected", HTTPStatus.CONFLICT)

        args.reset_on_start = True
        args.init_left_joints = RESET_LEFT_JOINTS
        args.init_right_joints = RESET_RIGHT_JOINTS
        args.init_left_gripper = RESET_LEFT_GRIPPER
        args.init_right_gripper = RESET_RIGHT_GRIPPER
        with self._robot_lock:
            infer_piper.maybe_reset_arms(left, right, args)
        self.log("Reset arms to zero joints and open grippers")
        return self.status()

    def ping_policy(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                return {
                    "ok": self._policy_connected,
                    "connected": self._policy_connected,
                    "metadata": _json_safe(self._server_metadata or {}),
                }
            args = copy.deepcopy(self._args)
            timeout = self._policy_timeout

        client: PolicyClient | None = None
        started = time.monotonic()
        try:
            client = PolicyClient(args.server_url, args.api_key, timeout=timeout)
            latency_ms = (time.monotonic() - started) * 1000.0
            return {
                "robot": "piper",
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

    def _camera_loop(self, args: infer_piper.Args, camera_stream_fps: float) -> None:
        period = 1.0 / camera_stream_fps
        while not self._camera_stop_event.is_set():
            loop_t0 = time.monotonic()
            with self._lock:
                cameras = dict(self._cameras)

            for name, camera in cameras.items():
                if self._camera_stop_event.is_set():
                    break
                try:
                    raw = camera.read(timeout=args.camera_timeout)
                    image = preprocess_image(raw, args.resize_size)
                    jpeg = _encode_jpeg_rgb(image)
                    with self._frame_condition:
                        old = self._frames.get(name, FrameSnapshot())
                        self._frames[name] = FrameSnapshot(
                            image=image,
                            jpeg=jpeg,
                            sequence=old.sequence + 1,
                            updated_at=time.monotonic(),
                            error=None,
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

    def _latest_images_for_inference(self, args: infer_piper.Args) -> dict[str, np.ndarray]:
        deadline = time.monotonic() + args.camera_timeout
        with self._frame_condition:
            while True:
                missing = [
                    name
                    for name in CAMERA_NAMES
                    if self._frames.get(name) is None or self._frames[name].image is None or self._frames[name].error
                ]
                if not missing:
                    return {name: np.asarray(self._frames[name].image).copy() for name in CAMERA_NAMES}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    errors = {
                        name: self._frames[name].error
                        for name in missing
                        if self._frames.get(name) is not None and self._frames[name].error
                    }
                    raise RuntimeError(f"Camera frames unavailable: missing={missing}, errors={errors}")
                self._frame_condition.wait(min(0.05, remaining))

    def _prepare_observation(self, args: infer_piper.Args) -> dict[str, Any]:
        with self._lock:
            left = self._left
            right = self._right
        if left is None or right is None:
            raise RuntimeError("Arms are not connected")
        return {
            "images": self._latest_images_for_inference(args),
            "image_masks": _camera_masks(args),
            "state": infer_piper.read_state(left, right, args, robot_lock=self._robot_lock),
            "prompt": args.prompt,
        }

    def _infer_action_chunk(self, args: infer_piper.Args) -> np.ndarray:
        with self._lock:
            client = self._client
        if client is None:
            raise RuntimeError("Policy client is not connected")

        obs = self._prepare_observation(args)
        infer_t0 = time.monotonic()
        response = client.infer(obs)
        infer_latency = time.monotonic() - infer_t0
        action_chunk = infer_piper.response_to_action_chunk(response, args)

        with self._lock:
            self._last_infer_latency_ms = infer_latency * 1000.0
            self._last_infer_hz = 1.0 / infer_latency if infer_latency > 0 else None
        return action_chunk

    def _update_loop_metrics(self, loop_t0: float, *, action_queue_len: int = 0) -> None:
        elapsed = time.monotonic() - loop_t0
        with self._lock:
            self._last_loop_ms = elapsed * 1000.0
            self._last_control_hz = 1.0 / elapsed if elapsed > 0 else None
            self._action_queue_len = action_queue_len

    def _get_arms(self) -> tuple[infer_piper.PiperArm, infer_piper.PiperArm]:
        with self._lock:
            left = self._left
            right = self._right
        if left is None or right is None:
            raise RuntimeError("Arms are not connected")
        return left, right

    def _apply_arm_runtime_settings(self, args: infer_piper.Args) -> None:
        left, right = self._get_arms()
        with self._robot_lock:
            left.arm.set_speed_percent(args.speed_percent)
            right.arm.set_speed_percent(args.speed_percent)

    def _run_infer_only_loop(self, args: infer_piper.Args) -> None:
        dt = 1.0 / args.fps
        max_chunks = args.max_steps if args.max_steps is not None else args.infer_only_chunks
        while not self._stop_event.is_set() and self._step < max_chunks:
            loop_t0 = time.monotonic()
            action_chunk = self._infer_action_chunk(args)
            with self._lock:
                self._step += 1
            self._update_loop_metrics(loop_t0, action_queue_len=len(action_chunk))
            self._stop_event.wait(max(0.0, dt - (time.monotonic() - loop_t0)))

    def _run_sync_control_loop(self, args: infer_piper.Args) -> None:
        action_plan: deque[np.ndarray] = deque()
        dt = 1.0 / args.fps
        left, right = self._get_arms()
        last_action: np.ndarray | None = infer_piper.read_state(left, right, args, robot_lock=self._robot_lock)

        while not self._stop_event.is_set() and (args.max_steps is None or self._step < args.max_steps):
            loop_t0 = time.monotonic()
            if not action_plan:
                action_plan.extend(self._infer_action_chunk(args))

            action = infer_piper.stabilize_action(action_plan.popleft(), last_action, args)
            last_action = infer_piper.execute_action_transition(
                last_action,
                action,
                left,
                right,
                args,
                robot_lock=self._robot_lock,
            )

            with self._lock:
                self._step += 1
            self._update_loop_metrics(loop_t0, action_queue_len=len(action_plan))
            self._stop_event.wait(max(0.0, dt - (time.monotonic() - loop_t0)))

    def _rtc_action_producer_loop(
        self,
        args: infer_piper.Args,
        rtc: RealTimeChunkingBuffer,
        producer_stop: threading.Event,
        errors: queue.Queue[BaseException],
    ) -> None:
        while not producer_stop.is_set() and not self._stop_event.is_set():
            cursor = select_rtc_cursor(rtc, args)
            if cursor is None:
                producer_stop.wait(min(0.005, 0.25 / args.fps))
                continue

            generation = rtc.get_generation()
            try:
                action_chunk = self._infer_action_chunk(args)
                rtc.enqueue(action_chunk, cursor, generation)
            except Exception as exc:
                errors.put(exc)
                producer_stop.set()
                self._stop_event.set()
                return

    def _raise_producer_error(self, errors: queue.Queue[BaseException]) -> None:
        try:
            exc = errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("RTC action producer failed") from exc

    def _run_rtc_control_loop(self, args: infer_piper.Args) -> None:
        left, right = self._get_arms()
        rtc = RealTimeChunkingBuffer(exp_weight=args.rtc_exp_weight)
        rtc.clear()

        producer_stop = threading.Event()
        errors: queue.Queue[BaseException] = queue.Queue()
        producer = threading.Thread(
            target=self._rtc_action_producer_loop,
            args=(args, rtc, producer_stop, errors),
            daemon=True,
            name="piper-web-rtc-producer",
        )
        producer.start()

        dt = 1.0 / args.fps
        last_action: np.ndarray | None = infer_piper.read_state(left, right, args, robot_lock=self._robot_lock)
        try:
            while not self._stop_event.is_set() and (args.max_steps is None or self._step < args.max_steps):
                loop_t0 = time.monotonic()
                self._raise_producer_error(errors)

                rtc.set_control_time(self._step)
                action = rtc.get_action(self._step)
                if action is None:
                    if last_action is not None and args.hold_last_action:
                        last_action = infer_piper.execute_action_transition(
                            last_action,
                            last_action,
                            left,
                            right,
                            args,
                            robot_lock=self._robot_lock,
                        )
                    self._update_loop_metrics(loop_t0)
                    self._stop_event.wait(max(0.0, dt - (time.monotonic() - loop_t0)))
                    continue

                action = infer_piper.stabilize_action(action, last_action, args)
                last_action = infer_piper.execute_action_transition(
                    last_action,
                    action,
                    left,
                    right,
                    args,
                    robot_lock=self._robot_lock,
                )

                with self._lock:
                    self._step += 1
                self._update_loop_metrics(loop_t0)
                self._stop_event.wait(max(0.0, dt - (time.monotonic() - loop_t0)))
        finally:
            producer_stop.set()
            producer.join(timeout=2.0)
            self._raise_producer_error(errors)

    def _run_loop(self, args: infer_piper.Args) -> None:
        error: str | None = None

        try:
            if args.infer_only:
                self._run_infer_only_loop(args)
            elif args.use_rtc:
                self._run_rtc_control_loop(args)
            else:
                self._run_sync_control_loop(args)
        except Exception as exc:
            error = str(exc)
            with self._lock:
                self._policy_connected = False
                self._last_error = error
            self.log(f"Inference loop failed: {exc}")
            logging.debug("Inference loop traceback:\n%s", traceback.format_exc())
        finally:
            with self._lock:
                self._running = False
                self._stop_requested = False
                self._action_queue_len = 0
                if error:
                    self._phase = "error"
                else:
                    self._phase = "stopped"
            self.log("Inference loop stopped")

    def wait_frame(self, camera_name: str, last_sequence: int, *, timeout: float = 5.0) -> FrameSnapshot:
        if camera_name not in CAMERA_NAMES:
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
        with self._lock:
            now = time.monotonic()
            frames = {
                name: {
                    "sequence": frame.sequence,
                    "age_ms": round((now - frame.updated_at) * 1000.0, 1) if frame.updated_at else None,
                    "error": frame.error,
                }
                for name, frame in self._frames.items()
            }
            can_reset = (
                self._left is not None and self._right is not None and (not self._running or not self._policy_connected)
            )
            return {
                "robot": "piper",
                "phase": self._phase,
                "connected": self._connected,
                "policy_connected": self._policy_connected,
                "running": self._running,
                "stop_requested": self._stop_requested,
                "can_edit_config": self._can_edit_config_locked(),
                "can_edit_connection_config": self._can_edit_connection_config_locked(),
                "can_connect": not self._connected
                and not self._running
                and not self._stop_requested
                and self._phase in {"idle", "error"},
                "can_start": not self._running
                and not self._stop_requested
                and self._phase in {"idle", "stopped", "error"},
                "can_stop": self._running or self._stop_requested,
                "can_disconnect": self._connected or self._running or self._phase in {"connecting", "error", "stopped"},
                "can_reset": can_reset,
                "step": self._step,
                "max_steps": self._args.max_steps,
                "prompt": self._args.prompt,
                "infer_only": self._args.infer_only,
                "server_url": self._args.server_url,
                "server_metadata": _json_safe(self._server_metadata or {}),
                "last_error": self._last_error,
                "metrics": {
                    "infer_latency_ms": round(self._last_infer_latency_ms, 2)
                    if self._last_infer_latency_ms is not None
                    else None,
                    "infer_hz": round(self._last_infer_hz, 2) if self._last_infer_hz is not None else None,
                    "loop_ms": round(self._last_loop_ms, 2) if self._last_loop_ms is not None else None,
                    "control_hz": round(self._last_control_hz, 2) if self._last_control_hz is not None else None,
                    "action_queue_len": self._action_queue_len,
                    "uptime_s": round(now - self._started_at, 1) if self._started_at else None,
                },
                "frames": frames,
                "logs": list(self._logs),
            }


_RM_STREAM_TO_POLICY = {
    "cam_head": "head_color",
    "cam_left_wrist": "left_color",
    "cam_right_wrist": "right_color",
}


class _Rm2WebAdapter:
    name = "rm2"

    def __init__(self, runtime: "Rm2WebRuntime", args: infer_rm2.Args) -> None:
        self.runtime = runtime
        self.args = args

    def observe(self) -> dict[str, Any]:
        images, camera_params = self.runtime._latest_policy_images()
        return {
            "images": images,
            "image_masks": {name: np.bool_(self.args.camera_backend != "black") for name in images},
            "camera_params": camera_params,
            "state": self.runtime._robot.read_state().values,
            "prompt": self.args.prompt,
        }

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        if replan_steps != self.args.replan_steps:
            raise ValueError("RM2 replan_steps must match the Web runtime configuration")
        return infer_rm2.response_to_action_chunk(response, self.args)

    def initial_action(self) -> np.ndarray:
        return self.runtime._robot.read_state().values

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        return infer_rm2.stabilize_action(action, previous, self.args)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return self.runtime._robot.execute_transition(previous, target)

    def infer_only_metadata(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {"camera_params": observation["camera_params"]}

    def profile(self, stage: str, elapsed_s: float) -> None:
        if self.args.profile_timing:
            logging.info("rm2 web profile %s=%.3fs", stage, elapsed_s)

    def infer_only_interval_s(self) -> float:
        return 0.0


class Rm2WebRuntime:
    """RM2 implementation of the same Web lifecycle exposed by PiperWebRuntime."""

    def __init__(self, initial_args: infer_rm2.Args | None = None, *, policy_timeout: float = 3.0) -> None:
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._args = copy.deepcopy(initial_args) if initial_args is not None else infer_rm2.Args()
        self._policy_timeout = policy_timeout
        self._camera_stream_fps = 10.0
        self._client: PolicyClient | None = None
        self._robot: infer_rm2.Rm2Robot | None = None
        self._cameras: dict[str, infer_rm2.Camera] = {}
        self._frames = {
            name: FrameSnapshot(_black_frame(self._args.resize_size), _encode_jpeg_rgb(_black_frame(self._args.resize_size)))
            for name in CAMERA_NAMES
        }
        self._run_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._camera_stop_event = threading.Event()
        self._connected = False
        self._running = False
        self._phase = "idle"
        self._last_error: str | None = None
        self._step = 0
        self._logs: deque[str] = deque(maxlen=200)

    def log(self, message: str) -> None:
        with self._lock:
            self._logs.append(f"{time.strftime('%H:%M:%S')} {message}")
        logging.info(message)

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            args = copy.deepcopy(self._args)
        return {
            "robot": "rm2",
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
            if self._connected or self._running:
                raise ApiError("Disconnect RM2 before changing parameters", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            for field in ("server_url", "api_key", "prompt", "left_ip", "right_ip", "cam_left_topic", "cam_right_topic", "cam_head_topic"):
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

    def connect(self) -> dict[str, Any]:
        with self._lock:
            if self._connected:
                return self.status()
            args = copy.deepcopy(self._args)
            self._phase = "connecting"
        client: PolicyClient | None = None
        cameras: dict[str, infer_rm2.Camera] = {}
        try:
            client = PolicyClient(args.server_url, args.api_key, timeout=self._policy_timeout)
            cameras = infer_rm2.make_cameras(args)
            robot = create_robot("rm2", args)
            if not isinstance(robot, infer_rm2.Rm2Robot):
                raise RuntimeError("RM2 registry factory returned an incompatible robot")
            robot.reset()
        except Exception as exc:
            if cameras:
                infer_rm2.close_cameras(cameras)
            if client is not None:
                client.close()
            with self._lock:
                self._phase, self._last_error = "error", str(exc)
            raise ApiError(str(exc), HTTPStatus.BAD_GATEWAY) from exc
        with self._lock:
            self._client, self._cameras, self._robot = client, cameras, robot
            self._connected, self._phase, self._last_error = True, "stopped", None
            self._ensure_camera_thread_locked(args)
        self.log("Connected RM2 runtime")
        return self.status()

    def start(self) -> dict[str, Any]:
        if not self._connected:
            self.connect()
        with self._lock:
            if self._running or self._client is None or self._robot is None:
                raise ApiError("RM2 runtime is not ready", HTTPStatus.CONFLICT)
            args = copy.deepcopy(self._args)
            self._stop_event.clear()
            self._running, self._phase, self._step, self._last_error = True, "running", 0, None
            self._run_thread = threading.Thread(target=self._run_loop, args=(args,), daemon=True, name="rm2-web-run")
            self._run_thread.start()
        return self.status()

    def stop(self, *, wait: bool = False) -> dict[str, Any]:
        self._stop_event.set()
        thread = self._run_thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.stop(wait=True)
        self._camera_stop_event.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
        with self._lock:
            client, cameras, robot = self._client, self._cameras, self._robot
            self._client, self._cameras, self._robot = None, {}, None
            self._connected, self._running, self._phase = False, False, "idle"
        infer_rm2.close_cameras(cameras)
        if robot is not None:
            robot.close()
        if client is not None:
            client.close()
        return self.status()

    def reset_arms(self) -> dict[str, Any]:
        if self._robot is None:
            raise ApiError("RM2 is not connected", HTTPStatus.CONFLICT)
        self._robot.reset()
        return self.status()

    def ping_policy(self) -> dict[str, Any]:
        client: PolicyClient | None = None
        try:
            client = PolicyClient(self._args.server_url, self._args.api_key, timeout=self._policy_timeout)
            return {"ok": True, "connected": True, "metadata": _json_safe(client.metadata)}
        except Exception as exc:
            return {"ok": False, "connected": False, "error": str(exc)}
        finally:
            if client is not None:
                client.close()

    def _ensure_camera_thread_locked(self, args: infer_rm2.Args) -> None:
        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(target=self._camera_loop, args=(args,), daemon=True, name="rm2-web-camera")
        self._camera_thread.start()

    def _camera_loop(self, args: infer_rm2.Args) -> None:
        while not self._camera_stop_event.is_set():
            for stream_name, policy_name in _RM_STREAM_TO_POLICY.items():
                camera = self._cameras.get(policy_name)
                if camera is None:
                    continue
                try:
                    image = preprocess_image(camera.read(timeout=args.camera_timeout), args.resize_size)
                    with self._frame_condition:
                        old = self._frames[stream_name]
                        self._frames[stream_name] = FrameSnapshot(image, _encode_jpeg_rgb(image), old.sequence + 1, time.monotonic())
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

    def _run_loop(self, args: infer_rm2.Args) -> None:
        try:
            adapter = _Rm2WebAdapter(self, args)
            run_policy_loop(
                self._client,
                adapter,
                InferenceLoopConfig.from_args(args),
                stop_event=self._stop_event,
                on_step=lambda step, _: setattr(self, "_step", step),
            )
            with self._lock:
                self._phase = "stopped"
        except Exception as exc:
            with self._lock:
                self._phase, self._last_error = "error", str(exc)
            self.log(f"RM2 inference failed: {exc}")
        finally:
            with self._lock:
                self._running = False

    def wait_frame(self, camera_name: str, last_sequence: int, *, timeout: float = 5.0) -> FrameSnapshot:
        deadline = time.monotonic() + timeout
        with self._frame_condition:
            while self._frames[camera_name].sequence == last_sequence and time.monotonic() < deadline:
                self._frame_condition.wait(0.2)
            return self._frames[camera_name]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "robot": "rm2", "phase": self._phase, "connected": self._connected, "policy_connected": self._connected,
                "running": self._running, "can_edit_config": not self._connected and not self._running,
                "can_edit_connection_config": not self._connected, "can_connect": not self._connected,
                "can_start": self._connected and not self._running, "can_stop": self._running,
                "can_disconnect": self._connected, "can_reset": self._robot is not None and not self._running,
                "step": self._step, "max_steps": self._args.max_steps, "prompt": self._args.prompt,
                "server_url": self._args.server_url, "server_metadata": {}, "last_error": self._last_error,
                "metrics": {}, "frames": {name: {"sequence": frame.sequence, "age_ms": None, "error": frame.error} for name, frame in self._frames.items()},
                "logs": list(self._logs),
            }


class PiperWebHandler(BaseHTTPRequestHandler):
    server: PiperWebServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self._send_file(self.server.static_dir / "index.html")
            elif path.startswith("/static/"):
                self._send_file(self.server.static_dir / path.removeprefix("/static/"))
            elif path == "/api/status":
                self._send_json(self.server.runtime.status())
            elif path == "/api/config":
                self._send_json(self.server.runtime.get_config())
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
            if path == "/api/config":
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
        except ApiError as exc:
            self._send_json({"ok": False, "error": str(exc)}, exc.status)
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
        if camera_name not in CAMERA_NAMES:
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
        self, server_address: tuple[str, int], handler: type[PiperWebHandler], runtime: Any, *, access_key: str | None = None
    ) -> None:
        super().__init__(server_address, handler)
        self.runtime = runtime
        self.access_key = access_key
        packaged_static = pathlib.Path(__file__).resolve().parent / "static"
        source_static = pathlib.Path(__file__).resolve().parents[3] / "static"
        self.static_dir = packaged_static if packaged_static.is_dir() else source_static

    def authorize(self, provided_key: str | None) -> bool:
        return self.access_key is None or (provided_key is not None and secrets.compare_digest(provided_key, self.access_key))

    def select_robot(self, name: str) -> dict[str, Any]:
        if name not in {"piper", "rm2"}:
            raise ApiError("robot must be piper or rm2")
        if self.runtime.status()["connected"] or self.runtime.status()["running"]:
            raise ApiError("Disconnect before changing robot", HTTPStatus.CONFLICT)
        self.runtime = Rm2WebRuntime() if name == "rm2" else PiperWebRuntime()
        return self.runtime.get_config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Piper web control panel")
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

    runtime_args = _default_args(camera_profile=cli_args.camera_profile)
    for field in ("server_url", "api_key", "prompt", "left_can", "right_can", "enable_on_start", "reset_on_start"):
        value = getattr(cli_args, field)
        if value is not None:
            setattr(runtime_args, field, value)
    for name in CAMERA_NAMES:
        setattr(runtime_args, f"{name}_backend", getattr(cli_args, f"{name}_backend") or getattr(runtime_args, f"{name}_backend"))
        selector = getattr(cli_args, name)
        if selector is not None:
            setattr(runtime_args, name, selector)
    runtime_args.dry_run = cli_args.dry_run

    logging.basicConfig(
        level=getattr(logging, cli_args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    runtime: Any = (
        PiperWebRuntime(runtime_args, policy_timeout=cli_args.policy_timeout)
        if cli_args.robot == "piper"
        else Rm2WebRuntime(policy_timeout=cli_args.policy_timeout)
    )
    access_key = cli_args.access_key or os.environ.get("MOTRIX_WEB_ACCESS_KEY")
    server = PiperWebServer((cli_args.host, cli_args.port), PiperWebHandler, runtime, access_key=access_key)
    logging.info("Piper web UI: http://%s:%s", cli_args.host, cli_args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping server")
    finally:
        runtime.disconnect()
        server.server_close()


if __name__ == "__main__":
    main()
