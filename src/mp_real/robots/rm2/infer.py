from __future__ import annotations

import configparser
import ctypes
import dataclasses
import logging
import os
import pathlib
import threading
import time
from typing import Any, Literal

import numpy as np
import tyro

def env_path(name: str) -> pathlib.Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return pathlib.Path(value).expanduser()


from mp_real.common.camera import BlackCamera, Camera, ROSImageCamera, close_cameras, init_ros_node
from mp_real.common.camera import make_realsense_cameras
from mp_real.common.runtime import parse_server_url, sleep_until
from mp_real.policy_client import websocket_client_policy
from mp_real.robots.base import Robot
from mp_real.robots.registry import register_robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import run_infer_only as run_generic_infer_only
from mp_real.runtime.inference import run_rtc_loop as run_generic_rtc_loop
from mp_real.runtime.inference import run_sync_loop as run_generic_sync_loop
from mp_real.runtime.models import ActionSpec, ObservationSnapshot, RobotState
from mp_real.runtime.observation import capture_observation

CameraBackend = Literal["ros", "realsense", "black"]
ArmCommandMode = Literal["canfd", "follow", "movej"]
RobotBackend = Literal["rm", "mock"]
JointUnit = Literal["rad", "deg"]


@dataclasses.dataclass
class Args:
    """Run RM dual-arm VLA inference against an OpenPI websocket policy server."""

    server_url: str = "ws://127.0.0.1:8000"
    api_key: str | None = None
    prompt: str = "stack the bowls"

    rm_config: pathlib.Path | None = None
    rm_sdk_lib: pathlib.Path | None = None
    robot_backend: RobotBackend = "rm"
    left_ip: str | None = None
    right_ip: str | None = None
    arm_port: int | None = None
    joint_dof: int = 6
    policy_joint_unit: JointUnit = "rad"

    camera_backend: CameraBackend = "ros"
    cam_left_topic: str = "/camera_d435_0/color/image_raw"
    cam_right_topic: str = "/camera_d435_1/color/image_raw"
    cam_head_topic: str = "/camera_d435_2/color/image_raw"
    cam_left_info_topic: str | None = None
    cam_right_info_topic: str | None = None
    cam_head_info_topic: str | None = None
    cam_left_serial: str = ""
    cam_right_serial: str = ""
    cam_head_serial: str = ""
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_timeout: float = 2.0
    resize_size: int = 224

    fps: float = 10.0
    replan_steps: int = 5
    max_steps: int | None = None
    use_rtc: bool = True
    rtc_replan_stride: int = 0
    rtc_prefetch_steps: int = 0
    rtc_exp_weight: float = 0.0

    dry_run: bool = False
    infer_only: bool = False
    infer_only_chunks: int = 1
    infer_only_output: pathlib.Path | None = None

    reset_on_start: bool = False
    reset_only: bool = False
    init_left_joints: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_right_joints: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_left_gripper: float = 1.0
    init_right_gripper: float = 1.0
    read_gripper_state: bool = False

    arm_command: ArmCommandMode = "canfd"
    speed_percent: int = 30
    movej_block: int = 0
    canfd_follow: bool = True
    canfd_expand: int = 0
    canfd_trajectory_mode: int = 2
    canfd_smooth_radio: int = 20
    max_joint_step_deg: float = 6.0
    max_action_step_deg: float = 2.0
    action_smoothing: float = 0.1
    gripper_smoothing: float = 0.0
    gripper_min: int = 1
    gripper_max: int = 1000
    gripper_timeout: int = 0
    command_gripper: bool = True

    interpolate_actions: bool = True
    command_rate_hz: float = 50.0
    command_gripper_every_step: bool = False
    hold_last_action: bool = True
    log_timing: bool = True
    profile_timing: bool = False


class RmRobotHandle(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int)]


class RmQuat(ctypes.Structure):
    _fields_ = [("w", ctypes.c_float), ("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class RmPosition(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class RmEuler(ctypes.Structure):
    _fields_ = [("rx", ctypes.c_float), ("ry", ctypes.c_float), ("rz", ctypes.c_float)]


class RmPose(ctypes.Structure):
    _fields_ = [("position", RmPosition), ("quaternion", RmQuat), ("euler", RmEuler)]


class RmErr(ctypes.Structure):
    _fields_ = [("err_len", ctypes.c_uint8), ("err", ctypes.c_int * 24)]


class RmCurrentArmState(ctypes.Structure):
    _fields_ = [("pose", RmPose), ("joint", ctypes.c_float * 7), ("err", RmErr)]


class RmGripperState(ctypes.Structure):
    _fields_ = [
        ("enable_state", ctypes.c_int),
        ("status", ctypes.c_int),
        ("error", ctypes.c_int),
        ("mode", ctypes.c_int),
        ("current_force", ctypes.c_int),
        ("temperature", ctypes.c_int),
        ("actpos", ctypes.c_int),
    ]


class RmSdk:
    def __init__(self, lib_path: pathlib.Path) -> None:
        if not lib_path.exists():
            raise FileNotFoundError(f"RM SDK library not found: {lib_path}")
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        rc = self.lib.rm_init(1)
        if rc != 0:
            raise RuntimeError(f"rm_init failed with code {rc}")

    def _bind(self) -> None:
        handle_p = ctypes.POINTER(RmRobotHandle)
        self.lib.rm_init.argtypes = [ctypes.c_int]
        self.lib.rm_init.restype = ctypes.c_int
        self.lib.rm_create_robot_arm.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.rm_create_robot_arm.restype = handle_p
        self.lib.rm_delete_robot_arm.argtypes = [handle_p]
        self.lib.rm_delete_robot_arm.restype = ctypes.c_int
        self.lib.rm_get_current_arm_state.argtypes = [handle_p, ctypes.POINTER(RmCurrentArmState)]
        self.lib.rm_get_current_arm_state.restype = ctypes.c_int
        self.lib.rm_get_joint_degree.argtypes = [handle_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.rm_get_joint_degree.restype = ctypes.c_int
        self.lib.rm_movej.argtypes = [
            handle_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.rm_movej.restype = ctypes.c_int
        self.lib.rm_movej_canfd.argtypes = [
            handle_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.rm_movej_canfd.restype = ctypes.c_int
        self.lib.rm_movej_follow.argtypes = [handle_p, ctypes.POINTER(ctypes.c_float)]
        self.lib.rm_movej_follow.restype = ctypes.c_int
        self.lib.rm_set_gripper_position.argtypes = [handle_p, ctypes.c_int, ctypes.c_bool, ctypes.c_int]
        self.lib.rm_set_gripper_position.restype = ctypes.c_int
        self.lib.rm_get_gripper_state.argtypes = [handle_p, ctypes.POINTER(RmGripperState)]
        self.lib.rm_get_gripper_state.restype = ctypes.c_int
        self.lib.rm_set_arm_stop.argtypes = [handle_p]
        self.lib.rm_set_arm_stop.restype = ctypes.c_int


@dataclasses.dataclass
class RmArm:
    name: str
    sdk: RmSdk
    ip: str
    port: int
    joint_dof: int
    last_gripper: float
    handle: Any | None = None

    def connect(self) -> None:
        self.handle = self.sdk.lib.rm_create_robot_arm(self.ip.encode("utf-8"), self.port)
        if not self.handle:
            raise RuntimeError(f"Failed to connect {self.name} RM arm at {self.ip}:{self.port}")
        handle_id = int(self.handle.contents.id)
        if handle_id <= 0:
            raise RuntimeError(f"{self.name} RM arm returned invalid handle id={handle_id}")
        logging.info("Connected %s RM arm %s:%d handle=%d", self.name, self.ip, self.port, handle_id)

    def close(self) -> None:
        if self.handle is None:
            return
        try:
            self.sdk.lib.rm_delete_robot_arm(self.handle)
        except Exception as exc:
            logging.warning("Failed to close %s arm: %s", self.name, exc)
        self.handle = None

    def read_joints(self) -> np.ndarray:
        self._require_handle()
        joints = (ctypes.c_float * 7)()
        rc = self.sdk.lib.rm_get_joint_degree(self.handle, joints)
        if rc != 0:
            state = RmCurrentArmState()
            rc = self.sdk.lib.rm_get_current_arm_state(self.handle, ctypes.byref(state))
            if rc != 0:
                raise RuntimeError(f"{self.name} rm_get_joint_degree/current_arm_state failed with code {rc}")
            return np.asarray(list(state.joint)[: self.joint_dof], dtype=np.float32)
        return np.asarray(list(joints)[: self.joint_dof], dtype=np.float32)

    def read_gripper(self, args: Args) -> float:
        self._require_handle()
        if not args.read_gripper_state:
            return self.last_gripper
        state = RmGripperState()
        rc = self.sdk.lib.rm_get_gripper_state(self.handle, ctypes.byref(state))
        if rc == 0 and state.actpos > 0:
            denom = max(1, args.gripper_max - args.gripper_min)
            self.last_gripper = float(np.clip((state.actpos - args.gripper_min) / denom, 0.0, 1.0))
        return self.last_gripper

    def command_joints(self, joints: np.ndarray, args: Args) -> None:
        self._require_handle()
        padded = np.zeros(7, dtype=np.float32)
        padded[: self.joint_dof] = np.asarray(joints, dtype=np.float32)[: self.joint_dof]
        arr = (ctypes.c_float * 7)(*padded.tolist())
        t0 = time.monotonic()
        if args.arm_command == "canfd":
            rc = self.sdk.lib.rm_movej_canfd(
                self.handle,
                arr,
                bool(args.canfd_follow),
                int(args.canfd_expand),
                int(args.canfd_trajectory_mode),
                int(args.canfd_smooth_radio),
            )
        elif args.arm_command == "follow":
            rc = self.sdk.lib.rm_movej_follow(self.handle, arr)
        else:
            rc = self.sdk.lib.rm_movej(self.handle, arr, int(args.speed_percent), 0, 0, int(args.movej_block))
        elapsed = time.monotonic() - t0
        if args.profile_timing:
            logging.info("%s command_joints mode=%s rc=%s elapsed=%.3fs", self.name, args.arm_command, rc, elapsed)
        if rc != 0:
            raise RuntimeError(f"{self.name} {args.arm_command} failed with code {rc}")

    def command_gripper(self, value: float, args: Args) -> None:
        self._require_handle()
        value = float(np.clip(value, 0.0, 1.0))
        position = int(round(args.gripper_min + value * (args.gripper_max - args.gripper_min)))
        t0 = time.monotonic()
        rc = self.sdk.lib.rm_set_gripper_position(self.handle, position, False, int(args.gripper_timeout))
        elapsed = time.monotonic() - t0
        if args.profile_timing:
            logging.info("%s command_gripper rc=%s elapsed=%.3fs", self.name, rc, elapsed)
        if rc != 0:
            logging.warning("%s gripper command failed with code %s", self.name, rc)
        self.last_gripper = value

    def stop(self) -> None:
        if self.handle is not None:
            try:
                self.sdk.lib.rm_set_arm_stop(self.handle)
            except Exception:
                pass

    def _require_handle(self) -> None:
        if self.handle is None:
            raise RuntimeError(f"{self.name} arm is not connected")


@dataclasses.dataclass
class MockArm:
    name: str
    joint_dof: int
    last_gripper: float
    joints: np.ndarray | None = None

    def connect(self) -> None:
        if self.joints is None:
            self.joints = np.zeros(self.joint_dof, dtype=np.float32)
        logging.info("Using mock %s arm", self.name)

    def close(self) -> None:
        pass

    def read_joints(self) -> np.ndarray:
        assert self.joints is not None
        return self.joints.astype(np.float32, copy=True)

    def read_gripper(self, args: Args) -> float:
        del args
        return self.last_gripper

    def command_joints(self, joints: np.ndarray, args: Args) -> None:
        del args
        self.joints = np.asarray(joints, dtype=np.float32)[: self.joint_dof].copy()

    def command_gripper(self, value: float, args: Args) -> None:
        del args
        self.last_gripper = float(np.clip(value, 0.0, 1.0))

    def stop(self) -> None:
        pass


@dataclasses.dataclass
class Rm2Robot(Robot):
    """RM2 SDK adapter exposing the robot-independent runtime boundary."""

    left: Any
    right: Any
    args: Args
    robot_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    action_spec: ActionSpec = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.action_spec = ActionSpec(
            action_dim=action_dim(self.args),
            state_dim=action_dim(self.args),
            joint_dof_per_arm=self.args.joint_dof,
            joint_unit=self.args.policy_joint_unit,
            camera_roles=("left_color", "right_color", "head_color"),
    )

    def read_state(self) -> RobotState:
        values = read_state(self.left, self.right, self.args, robot_lock=self.robot_lock)
        timestamp_ns = time.monotonic_ns()
        return RobotState(
            values=values,
            timestamp_monotonic=timestamp_ns / 1e9,
            timestamp_monotonic_ns=timestamp_ns,
        )

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return execute_action_transition(
            previous,
            target,
            self.left,
            self.right,
            self.args,
            robot_lock=self.robot_lock,
        )

    def reset(self) -> None:
        maybe_reset_arms(self.left, self.right, self.args, self.robot_lock)

    def close(self) -> None:
        for arm in (self.left, self.right):
            try:
                arm.stop()
            finally:
                close_arm(arm)


def resolve_existing_path(
    path: pathlib.Path | None, candidates: list[pathlib.Path | None], label: str
) -> pathlib.Path:
    search_paths = [candidate.expanduser() for candidate in (path, *candidates) if candidate is not None]
    seen: set[pathlib.Path] = set()
    unique_paths: list[pathlib.Path] = []
    for candidate in search_paths:
        candidate = candidate.resolve(strict=False)
        if candidate not in seen:
            unique_paths.append(candidate)
            seen.add(candidate)
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(candidate) for candidate in unique_paths)
    raise FileNotFoundError(
        f"{label} not found. Pass it explicitly with --rm-config/--rm-sdk-lib, or set "
        f"RM_SDK_ROOT/RM_CONFIG/RM_SDK_LIB.\nSearched:\n  {searched}"
    )


def resolve_rm_dependency_paths(args: Args) -> Args:
    sdk_root_env = env_path("RM_SDK_ROOT")
    config_candidates = [env_path("RM_CONFIG")]
    library_candidates = [env_path("RM_SDK_LIB")]
    if sdk_root_env is not None:
        config_candidates.append(sdk_root_env / "build/config.ini")
        library_candidates.extend(
            [
                sdk_root_env / "Robotic_Arm/lib/libapi_cpp.so",
                sdk_root_env / "build/libapi_cpp.so",
            ]
        )
    rm_config = resolve_existing_path(
        args.rm_config,
        config_candidates,
        "RM config.ini",
    )
    rm_sdk_lib = resolve_existing_path(
        args.rm_sdk_lib,
        library_candidates,
        "RM SDK library",
    )
    logging.info("Using RM config: %s", rm_config)
    logging.info("Using RM SDK library: %s", rm_sdk_lib)
    return dataclasses.replace(args, rm_config=rm_config, rm_sdk_lib=rm_sdk_lib)


def load_rm_connection(args: Args) -> tuple[str, str, int]:
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(args.rm_config, encoding="utf-8")
    left_ip = args.left_ip or parser.get("Left_Arm", "slave_ip")
    right_ip = args.right_ip or parser.get("Right_Arm", "slave_ip")
    port = args.arm_port or parser.getint("Left_Arm", "slave_port")
    return left_ip.strip(), right_ip.strip(), int(port)

def make_cameras(args: Args) -> dict[str, Camera]:
    if args.camera_backend == "black":
        return {
            "left_color": BlackCamera("left_color", width=args.camera_width, height=args.camera_height),
            "right_color": BlackCamera("right_color", width=args.camera_width, height=args.camera_height),
            "head_color": BlackCamera("head_color", width=args.camera_width, height=args.camera_height),
        }
    if args.camera_backend == "realsense":
        serials = {
            "left_color": args.cam_left_serial,
            "right_color": args.cam_right_serial,
            "head_color": args.cam_head_serial,
        }
        missing = ", ".join(name for name, serial in serials.items() if not serial)
        missing_message = None
        if missing:
            missing_message = (
                "RealSense backend needs explicit serial numbers for all RM2 cameras. "
                f"Missing: {missing}. Pass --cam-left-serial, --cam-right-serial and --cam-head-serial."
            )
        return make_realsense_cameras(
            serials,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fallback_backends="ros/black",
            require_serials=True,
            missing_message=missing_message,
        )
    init_ros_node()
    return {
        "left_color": ROSImageCamera("left_color", args.cam_left_topic, args.cam_left_info_topic),
        "right_color": ROSImageCamera("right_color", args.cam_right_topic, args.cam_right_info_topic),
        "head_color": ROSImageCamera("head_color", args.cam_head_topic, args.cam_head_info_topic),
    }


def create_rm_arms(args: Args) -> tuple[Any, Any]:
    if args.robot_backend == "mock":
        left = MockArm(
            "left",
            args.joint_dof,
            float(args.init_left_gripper),
            np.asarray(args.init_left_joints[: args.joint_dof], dtype=np.float32),
        )
        right = MockArm(
            "right",
            args.joint_dof,
            float(args.init_right_gripper),
            np.asarray(args.init_right_joints[: args.joint_dof], dtype=np.float32),
        )
        left.connect()
        right.connect()
        return left, right
    left_ip, right_ip, port = load_rm_connection(args)
    sdk = RmSdk(args.rm_sdk_lib)
    left = RmArm("left", sdk, left_ip, port, args.joint_dof, float(args.init_left_gripper))
    right = RmArm("right", sdk, right_ip, port, args.joint_dof, float(args.init_right_gripper))
    left.connect()
    right.connect()
    return left, right


def close_arm(arm: Any | None) -> None:
    if arm is not None:
        arm.close()


def action_dim(args: Args) -> int:
    return 2 * (args.joint_dof + 1)


def robot_joints_to_policy(joints_deg: np.ndarray, args: Args) -> np.ndarray:
    joints_deg = np.asarray(joints_deg, dtype=np.float32)
    if args.policy_joint_unit == "rad":
        return np.deg2rad(joints_deg).astype(np.float32)
    return joints_deg.astype(np.float32, copy=True)


def policy_joints_to_robot(joints: np.ndarray, args: Args) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if args.policy_joint_unit == "rad":
        return np.rad2deg(joints).astype(np.float32)
    return joints.astype(np.float32, copy=True)


def deg_limit_to_policy(limit_deg: float, args: Args) -> float:
    if args.policy_joint_unit == "rad":
        return float(np.deg2rad(limit_deg))
    return float(limit_deg)


def read_state(left: RmArm, right: RmArm, args: Args, *, robot_lock: threading.Lock | None = None) -> np.ndarray:
    def _read() -> np.ndarray:
        return np.concatenate(
            [
                robot_joints_to_policy(left.read_joints(), args),
                robot_joints_to_policy(right.read_joints(), args),
                np.asarray([left.read_gripper(args)], dtype=np.float32),
                np.asarray([right.read_gripper(args)], dtype=np.float32),
            ]
        ).astype(np.float32)

    if robot_lock is None:
        return _read()
    with robot_lock:
        return _read()


def prepare_observation(
    cameras: dict[str, Camera],
    left: RmArm,
    right: RmArm,
    args: Args,
    *,
    robot_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    return capture_observation_snapshot(cameras, left, right, args, robot_lock=robot_lock).to_policy_observation()


def capture_observation_snapshot(
    cameras: dict[str, Camera],
    left: RmArm,
    right: RmArm,
    args: Args,
    *,
    robot_lock: threading.Lock | None = None,
) -> ObservationSnapshot:
    return capture_observation(
        cameras,
        read_state=lambda: read_state(left, right, args, robot_lock=robot_lock),
        prompt=args.prompt,
        resize_size=args.resize_size,
        timeout=args.camera_timeout,
        image_masks={name: np.bool_(not isinstance(camera, BlackCamera)) for name, camera in cameras.items()},
        include_camera_params=True,
    )


def response_to_action_chunk(response: dict[str, Any], args: Args) -> np.ndarray:
    chunk = np.asarray(response["actions"], dtype=np.float32)
    dim = action_dim(args)
    if chunk.ndim != 2 or chunk.shape[1] < dim:
        raise RuntimeError(f"Expected action chunk [T, >= {dim}], got {chunk.shape}")
    if len(chunk) < args.replan_steps:
        raise RuntimeError(f"Policy returned {len(chunk)} actions, replan_steps={args.replan_steps}")
    return chunk[: args.replan_steps, :dim].copy()


def action_to_targets(action: np.ndarray, args: Args) -> tuple[np.ndarray, float, np.ndarray, float]:
    action = np.asarray(action, dtype=np.float32)
    n = args.joint_dof
    if action.shape[-1] < action_dim(args):
        raise ValueError(f"Expected at least {action_dim(args)} action dims, got {action.shape[-1]}")
    return action[:n], float(action[2 * n]), action[n : 2 * n], float(action[2 * n + 1])


def joint_mask(args: Args) -> np.ndarray:
    mask = np.ones(action_dim(args), dtype=bool)
    mask[2 * args.joint_dof :] = False
    return mask


def smooth_action(action: np.ndarray, last_action: np.ndarray | None, args: Args) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if last_action is None:
        return action
    last_action = np.asarray(last_action, dtype=np.float32)
    jm = joint_mask(args)
    gm = ~jm
    joint_alpha = float(np.clip(args.action_smoothing, 0.0, 0.98))
    grip_alpha = float(np.clip(args.gripper_smoothing, 0.0, 0.98))
    if joint_alpha > 0:
        action[jm] = joint_alpha * last_action[jm] + (1.0 - joint_alpha) * action[jm]
    if grip_alpha > 0:
        action[gm] = grip_alpha * last_action[gm] + (1.0 - grip_alpha) * action[gm]
    return action


def stabilize_action(action: np.ndarray, last_action: np.ndarray | None, args: Args) -> np.ndarray:
    action = smooth_action(action, last_action, args)
    step_limit = deg_limit_to_policy(args.max_action_step_deg, args)
    if last_action is None or step_limit <= 0:
        return action
    action = action.copy()
    jm = joint_mask(args)
    delta = action[jm] - last_action[jm]
    action[jm] = last_action[jm] + np.clip(delta, -step_limit, step_limit)
    return action


def limit_joint_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return target
    return current + np.clip(target - current, -max_step, max_step)


def limit_action_to_feedback(action: np.ndarray, left: RmArm, right: RmArm, args: Args) -> np.ndarray:
    if args.dry_run:
        return np.asarray(action, dtype=np.float32).copy()
    step_limit = deg_limit_to_policy(args.max_joint_step_deg, args)
    if step_limit <= 0:
        return np.asarray(action, dtype=np.float32).copy()
    action = np.asarray(action, dtype=np.float32).copy()
    lj, lg, rj, rg = action_to_targets(action, args)
    lj = limit_joint_step(lj, robot_joints_to_policy(left.read_joints(), args), step_limit)
    rj = limit_joint_step(rj, robot_joints_to_policy(right.read_joints(), args), step_limit)
    n = args.joint_dof
    action[:n] = lj
    action[n : 2 * n] = rj
    action[2 * n] = lg
    action[2 * n + 1] = rg
    return action


def _send_action_unlocked(
    action: np.ndarray, left: RmArm, right: RmArm, args: Args, *, send_gripper: bool = True
) -> None:
    lj, lg, rj, rg = action_to_targets(action, args)
    left.last_gripper = float(np.clip(lg, 0.0, 1.0))
    right.last_gripper = float(np.clip(rg, 0.0, 1.0))
    if args.dry_run:
        logging.info("dry-run left=%s lg=%.3f right=%s rg=%.3f", lj, left.last_gripper, rj, right.last_gripper)
        return
    left.command_joints(policy_joints_to_robot(lj, args), args)
    right.command_joints(policy_joints_to_robot(rj, args), args)
    if args.command_gripper and send_gripper:
        left.command_gripper(left.last_gripper, args)
        right.command_gripper(right.last_gripper, args)


def execute_action_transition(
    start_action: np.ndarray | None,
    target_action: np.ndarray,
    left: RmArm,
    right: RmArm,
    args: Args,
    *,
    robot_lock: threading.Lock,
) -> np.ndarray:
    with robot_lock:
        limited_target = limit_action_to_feedback(target_action, left, right, args)

    steps = interpolation_steps(args)
    if start_action is None or steps <= 1:
        with robot_lock:
            _send_action_unlocked(limited_target, left, right, args)
        return limited_target

    start_action = np.asarray(start_action, dtype=np.float32)
    interval_s = 1.0 / args.fps / steps
    next_t = time.monotonic()
    for i in range(1, steps + 1):
        ratio = i / steps
        command = start_action + ratio * (limited_target - start_action)
        send_gripper = args.command_gripper_every_step or i == steps
        with robot_lock:
            _send_action_unlocked(command, left, right, args, send_gripper=send_gripper)
        next_t += interval_s
        if i < steps:
            sleep_until(next_t)
    return limited_target


def interpolation_steps(args: Args) -> int:
    if args.dry_run or not args.interpolate_actions or args.command_rate_hz <= args.fps:
        return 1
    return max(1, round(args.command_rate_hz / args.fps))


def maybe_reset_arms(left: RmArm, right: RmArm, args: Args, robot_lock: threading.Lock) -> None:
    if not args.reset_on_start:
        return
    action = np.concatenate(
        [
            np.asarray(args.init_left_joints[: args.joint_dof], dtype=np.float32),
            np.asarray(args.init_right_joints[: args.joint_dof], dtype=np.float32),
            np.asarray([args.init_left_gripper], dtype=np.float32),
            np.asarray([args.init_right_gripper], dtype=np.float32),
        ]
    )
    with robot_lock:
        _send_action_unlocked(action, left, right, args)


@dataclasses.dataclass
class Rm2InferenceAdapter:
    robot: Rm2Robot
    cameras: dict[str, Camera]
    args: Args
    name: str = "rm2"
    last_observation_snapshot: ObservationSnapshot | None = dataclasses.field(default=None, init=False, repr=False)

    def observe(self) -> dict[str, Any]:
        self.last_observation_snapshot = capture_observation_snapshot(
            self.cameras,
            self.robot.left,
            self.robot.right,
            self.args,
            robot_lock=self.robot.robot_lock,
        )
        return self.last_observation_snapshot.to_policy_observation()

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        if replan_steps != self.args.replan_steps:
            raise ValueError("RM2 replan_steps must match its runtime config")
        return response_to_action_chunk(response, self.args)

    def initial_action(self) -> np.ndarray:
        return self.robot.read_state().values

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        return stabilize_action(action, previous, self.args)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return self.robot.execute_transition(previous, target)

    def infer_only_metadata(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {"camera_params": observation["camera_params"]}

    def profile(self, stage: str, elapsed_s: float) -> None:
        if self.args.profile_timing:
            logging.info("%s profile %s=%.3fs", self.name, stage, elapsed_s)

    def infer_only_interval_s(self) -> float:
        return 0.0


def _adapter(
    cameras: dict[str, Camera], left: Any, right: Any, args: Args, robot_lock: threading.Lock
) -> Rm2InferenceAdapter:
    return Rm2InferenceAdapter(Rm2Robot(left, right, args, robot_lock), cameras, args)


def run_infer_only(
    client: websocket_client_policy.WebsocketClientPolicy,
    cameras: dict[str, Camera],
    left: Any,
    right: Any,
    args: Args,
    robot_lock: threading.Lock,
) -> None:
    run_generic_infer_only(client, _adapter(cameras, left, right, args, robot_lock), InferenceLoopConfig.from_args(args))


def run_sync_loop(
    client: websocket_client_policy.WebsocketClientPolicy,
    cameras: dict[str, Camera],
    left: Any,
    right: Any,
    args: Args,
    robot_lock: threading.Lock,
) -> None:
    run_generic_sync_loop(client, _adapter(cameras, left, right, args, robot_lock), InferenceLoopConfig.from_args(args))


def run_rtc_loop(
    client: websocket_client_policy.WebsocketClientPolicy,
    cameras: dict[str, Camera],
    left: Any,
    right: Any,
    args: Args,
    robot_lock: threading.Lock,
) -> None:
    run_generic_rtc_loop(client, _adapter(cameras, left, right, args, robot_lock), InferenceLoopConfig.from_args(args))


def validate_args(args: Args) -> None:
    if args.joint_dof <= 0 or args.joint_dof > 7:
        raise ValueError("joint_dof must be in [1, 7]")
    if len(args.init_left_joints) < args.joint_dof or len(args.init_right_joints) < args.joint_dof:
        raise ValueError("init joint tuples must contain at least joint_dof values")
    if args.command_rate_hz <= 0:
        raise ValueError("command_rate_hz must be positive")
    InferenceLoopConfig.from_args(args).validate()


def create_robot(args: Args) -> Rm2Robot:
    if args.robot_backend == "rm" and (args.rm_config is None or args.rm_sdk_lib is None):
        args = resolve_rm_dependency_paths(args)
    left, right = create_rm_arms(args)
    return Rm2Robot(left, right, args)


def main(args: Args) -> None:
    validate_args(args)
    if args.robot_backend == "rm":
        args = resolve_rm_dependency_paths(args)
    loop_config = InferenceLoopConfig.from_args(args)
    robot: Rm2Robot | None = None
    cameras: dict[str, Camera] = {}
    try:
        if args.reset_only:
            robot = create_robot(args)
            robot.reset()
            return

        host, port = parse_server_url(args.server_url)
        logging.info("Connecting to OpenPI server at %s%s", host, f":{port}" if port is not None else "")
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=args.api_key)
        logging.info("Server metadata: %s", client.get_server_metadata())

        logging.info("Creating cameras with backend=%s", args.camera_backend)
        cameras = make_cameras(args)
        logging.info("Created cameras: %s", tuple(cameras))

        robot = create_robot(args)
        robot.reset()
        adapter = Rm2InferenceAdapter(robot, cameras, args)
        if loop_config.infer_only:
            run_generic_infer_only(client, adapter, loop_config)
        elif loop_config.use_rtc:
            run_generic_rtc_loop(client, adapter, loop_config)
        else:
            run_generic_sync_loop(client, adapter, loop_config)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        close_cameras(cameras)
        if robot is not None:
            robot.close()


register_robot("rm2", create_robot)


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
