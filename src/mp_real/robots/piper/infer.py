from __future__ import annotations

import dataclasses
import importlib
import logging
import pathlib
import threading
import time
from typing import Any, Literal

import numpy as np
import tyro

from mp_real.common.camera import Camera, close_cameras
from mp_real.common.camera import make_camera as make_common_camera
from mp_real.common.runtime import parse_server_url, sleep_until
from mp_real.policy_client import websocket_client_policy as _websocket_client_policy
from mp_real.pose.models import (
    MoveToRecordedStatePlan,
    PoseMoveProgress,
    PoseMoveResult,
    PoseValidationIssue,
    PoseValidationReport,
    RecordedPoseTarget,
)
from mp_real.robots.base import Robot
from mp_real.robots.registry import register_robot
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import decode_action_chunk_for_spec
from mp_real.runtime.inference import run_infer_only as run_generic_infer_only
from mp_real.runtime.inference import run_rtc_loop as run_generic_rtc_loop
from mp_real.runtime.inference import run_sync_loop as run_generic_sync_loop
from mp_real.runtime.models import ActionSpec, ObservationSnapshot, RobotState, VectorField
from mp_real.runtime.observation import capture_observation

CameraBackend = Literal["realsense", "v4l2", "black"]
ArmCommandMode = Literal["move_j", "move_js", "auto"]

_JOINT_ACTION_MASK = np.asarray(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)
_GRIPPER_ACTION_MASK = np.logical_not(_JOINT_ACTION_MASK)


def _vector_fields() -> tuple[VectorField, ...]:
    fields: list[VectorField] = []
    for arm in ("left", "right"):
        fields.extend(VectorField(f"{arm}_joint_{index}", "rad", "joint_position") for index in range(1, 7))
        fields.append(VectorField(f"{arm}_gripper", "normalized_0_open_1", "gripper_open_fraction"))
    return tuple(fields)


@dataclasses.dataclass
class Args:
    """Run Piper dual-arm inference against an OpenPI websocket policy server."""

    # Websocket policy server URL.
    server_url: str = "ws://127.0.0.1:8000"
    # Optional API key for the policy server.
    api_key: str | None = None
    # Language instruction sent to the policy.
    prompt: str = "perform the task"

    # Left and right Piper CAN channels.
    left_can: str = "can_left"
    right_can: str = "can_right"

    # Camera backends. Current hardware uses RealSense for head and black placeholders for wrist cameras.
    cam_head_backend: CameraBackend = "realsense"
    cam_left_wrist_backend: CameraBackend = "v4l2"
    cam_right_wrist_backend: CameraBackend = "v4l2"
    # Camera selector: RealSense serial number for realsense, /dev/video* path for v4l2, ignored for black.
    cam_head: str = "261222074970"
    cam_left_wrist: str = "/dev/left-camera"
    cam_right_wrist: str = "/dev/right-camera"
    # Camera frame width requested from physical cameras and placeholders.
    camera_width: int = 640
    # Camera frame height requested from physical cameras and placeholders.
    camera_height: int = 480
    # Camera frame rate requested from RealSense cameras.
    camera_fps: int = 30
    # Per-frame camera read timeout in seconds.
    camera_timeout: float = 2.0

    # Control rate for executing actions.
    fps: float = 10.0
    # Number of actions consumed from each returned action chunk.
    replan_steps: int = 5
    # Stop after this many control steps. None means run until interrupted.
    max_steps: int | None = None
    # Image resize target expected by pi0.5.
    resize_size: int = 224

    # Do not send commands to the arms, but still read observations and query the server.
    dry_run: bool = False
    # Only fetch action chunks from the server. This never enables, resets, or commands the arms.
    infer_only: bool = False
    # Number of fresh action chunks to fetch when infer_only is true.
    infer_only_chunks: int = 1
    # Optional .npz path for saving infer_only action chunks and states.
    infer_only_output: pathlib.Path | None = None
    # Enable arms on startup.
    enable_on_start: bool = True
    # Seconds to wait for arm.enable() to succeed. Set <= 0 to wait forever.
    enable_timeout_s: float = 10.0
    # Move both arms to the init pose before connecting the policy server.
    reset_on_start: bool = True
    # Only reset both arms to the init pose, then exit without connecting the policy server or cameras.
    reset_only: bool = False
    # Seconds to wait for the reset joint motion to finish. Set <= 0 to skip waiting.
    reset_timeout_s: float = 8.0
    # Piper speed percentage used for joint moves.
    speed_percent: int = 60
    # Arm command API. move_j is planned position control; move_js is follower mode and needs conservative smoothing.
    arm_command: ArmCommandMode = "move_j"

    # Initial state used for reset and as gripper fallback before feedback is available.
    init_left_joints: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_right_joints: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    init_left_gripper: float = 1.0
    init_right_gripper: float = 1.0

    # Map policy gripper values [0, 1] to AGX gripper angle mode.
    gripper_closed_deg: float = 0.0
    gripper_open_deg: float = 100.0
    gripper_force: float = 1.0

    # Limit the per-step joint target jump in radians. Set <= 0 to disable.
    max_joint_step: float = 0.35
    # Use a real-time chunking producer so inference runs in the background while the robot keeps executing.
    use_rtc: bool = True
    # Control-step stride between RTC replans. 0 means use replan_steps, which is the most stable setting.
    rtc_replan_stride: int = 0
    # How many control steps before the next stride boundary to prefetch. 0 means half of rtc_replan_stride.
    rtc_prefetch_steps: int = 0
    # Exponential weight factor for fusing overlapping RTC chunks; larger values favor newer chunks more strongly.
    rtc_exp_weight: float = 0.0
    # EMA applied to joint action targets before execution. 0 disables, larger values are smoother but laggier.
    action_smoothing: float = 0.1
    # EMA applied to gripper targets before execution. Usually keep this lower than joint smoothing.
    gripper_smoothing: float = 0.0
    # Ignore tiny joint target changes relative to the previous command. Set <= 0 to disable.
    joint_deadband: float = 0.0
    # Limit joint target changes relative to the previous command. Set <= 0 to disable.
    max_action_step: float = 0.03
    # Send interpolated hardware setpoints between policy actions.
    interpolate_actions: bool = True
    # Hardware setpoint rate used by interpolation. Piper follower control is commonly run much faster than policy FPS.
    command_rate_hz: float = 100.0
    # Send gripper commands at every interpolated setpoint instead of only the final setpoint.
    command_gripper_every_step: bool = False
    # Re-send the previous command while RTC waits for the first/next available action.
    hold_last_action: bool = True
    # Log producer inference time and control-loop timing.
    log_timing: bool = True


@dataclasses.dataclass
class PiperArm:
    name: str
    arm: Any
    gripper: Any
    last_gripper: float


@dataclasses.dataclass
class PiperRobot(Robot):
    """Piper SDK adapter exposing the robot-independent runtime boundary."""

    left: PiperArm
    right: PiperArm
    args: Args
    robot_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    action_spec: ActionSpec = dataclasses.field(
        default_factory=lambda: ActionSpec(
            action_dim=14,
            state_dim=14,
            joint_dof_per_arm=6,
            joint_unit="rad",
            camera_roles=("cam_head", "cam_left_wrist", "cam_right_wrist"),
            state_fields=_vector_fields(),
            action_fields=_vector_fields(),
        )
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
        with self.robot_lock:
            maybe_reset_arms(self.left, self.right, self.args)

    def configure_runtime(self, config: object) -> None:
        if not isinstance(config, Args):
            raise TypeError(f"Expected Piper Args, got {type(config).__name__}")
        with self.robot_lock:
            self.left.arm.set_speed_percent(config.speed_percent)
            self.right.arm.set_speed_percent(config.speed_percent)
            self.args = config

    # PoseControlCapability is deliberately optional; these methods are not
    # part of Robot and are only reached after the Web/CLI asks for the
    # high-risk recorded-state workflow.
    def get_current_pose_state(self) -> RobotState:
        return self.read_state()

    def validate_pose_target(self, target: RecordedPoseTarget) -> PoseValidationReport:
        del target
        issues: list[PoseValidationIssue] = []
        # No workspace FK or SDK stop primitive is assumed.  A deployed Piper
        # integration must expose both before a real move is eligible.
        if not all(callable(getattr(bundle.arm, "stop", None)) for bundle in (self.left, self.right)):
            issues.append(PoseValidationIssue("stop_motion_unsupported", "Piper SDK stop() is unavailable"))
        issues.append(
            PoseValidationIssue(
                "workspace_validation_unavailable",
                "Piper workspace validation is not configured for this SDK deployment",
            )
        )
        issues.extend(
            (
                PoseValidationIssue(
                    "joint_limit_validation_unavailable",
                    "Piper joint-limit validation is not configured for this SDK deployment",
                ),
                PoseValidationIssue(
                    "health_validation_unavailable",
                    "Piper health validation is not configured for this SDK deployment",
                ),
            )
        )
        return PoseValidationReport(tuple(issues))

    def plan_move_to_state(self, plan: MoveToRecordedStatePlan) -> MoveToRecordedStatePlan:
        if plan.target_state.shape != (self.action_spec.state_dim,):
            raise ValueError("Piper pose plan dimension does not match its ActionSpec")
        return plan

    def execute_pose_plan(self, plan, *, stop_event, on_progress=None) -> PoseMoveResult:
        previous = self.get_current_pose_state()
        for waypoint in plan.waypoints:
            if stop_event.is_set():
                return PoseMoveResult(plan.plan_id, "aborted", previous, None, "stop requested")
            cycle_started_ns = time.monotonic_ns()
            error = float(np.max(np.abs(previous.values - waypoint.target)))
            if error > plan.constraints.max_tracking_error:
                return PoseMoveResult(plan.plan_id, "failed", previous, error, "tracking error exceeded")
            with self.robot_lock:
                _send_action_unlocked(waypoint.target, self.left, self.right, self.args)
            if stop_event.wait(plan.constraints.control_period_s):
                return PoseMoveResult(plan.plan_id, "aborted", self.get_current_pose_state(), None, "stop requested")
            previous = self.get_current_pose_state()
            elapsed_s = (time.monotonic_ns() - cycle_started_ns) / 1e9
            if elapsed_s > plan.constraints.control_period_s + plan.constraints.max_control_overrun_s:
                return PoseMoveResult(plan.plan_id, "failed", previous, None, "control cycle overrun")
            if on_progress is not None:
                on_progress(
                    PoseMoveProgress(
                        plan.plan_id,
                        waypoint.index,
                        len(plan.waypoints),
                        previous.values.copy(),
                        waypoint.target.copy(),
                        float(np.max(np.abs(previous.values - waypoint.target))),
                        time.monotonic_ns(),
                    )
                )
        return PoseMoveResult(plan.plan_id, "reached", previous, 0.0)

    def stop_pose_motion(self) -> None:
        errors: list[BaseException] = []
        with self.robot_lock:
            for bundle in (self.left, self.right):
                stop = getattr(bundle.arm, "stop", None)
                if not callable(stop):
                    errors.append(RuntimeError(f"{bundle.name} Piper arm does not expose stop()"))
                    continue
                try:
                    stop()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    def verify_target_reached(self, plan: MoveToRecordedStatePlan) -> PoseMoveResult:
        current = self.get_current_pose_state()
        error = float(np.max(np.abs(current.values - plan.target_state)))
        status = "reached" if error <= plan.constraints.tracking_tolerance else "failed"
        return PoseMoveResult(plan.plan_id, status, current, error, None if status == "reached" else "tracking error")

    def close(self) -> None:
        close_arm(self.left)
        close_arm(self.right)


def import_pyagxarm() -> tuple[Any, Any, Any, Any]:
    try:
        pyagxarm = importlib.import_module("pyAgxArm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyAgxArm or one of its dependencies is not importable. Keep pyAgxArm next to mp-real and run "
            "`uv sync --extra piper` from mp-real. Expected layout: parent/{mp-real,pyAgxArm}."
        ) from exc
    return pyagxarm.AgxArmFactory, pyagxarm.ArmModel, pyagxarm.PiperFW, pyagxarm.create_agx_arm_config


def short_repr(value: Any, *, limit: int = 500) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def arm_diagnostics(arm: Any) -> str:
    diagnostics: list[str] = []
    for label, method_name in (
        ("firmware", "get_firmware"),
        ("status", "get_arm_status"),
        ("joints", "get_joint_angles"),
    ):
        method = getattr(arm, method_name, None)
        if method is None:
            continue
        try:
            value = method()
            value = getattr(value, "msg", value)
            diagnostics.append(f"{label}={short_repr(value)}")
        except Exception as exc:
            diagnostics.append(f"{label}=<error {exc}>")
    return "; ".join(diagnostics) if diagnostics else "no diagnostics available"


def enable_arm_with_timeout(arm: Any, name: str, timeout_s: float) -> None:
    logging.info("Enabling %s arm", name)
    start_t = time.monotonic()
    last_log_t = start_t
    attempts = 0

    while True:
        attempts += 1
        try:
            if arm.enable():
                logging.info("Enabled %s arm after %d attempt(s)", name, attempts)
                return
        except Exception as exc:
            logging.warning("%s arm enable attempt failed: %s", name, exc)

        now = time.monotonic()
        elapsed = now - start_t
        if timeout_s > 0 and elapsed >= timeout_s:
            raise TimeoutError(
                f"Timed out enabling {name} arm after {elapsed:.1f}s and {attempts} attempt(s). "
                f"Diagnostics: {arm_diagnostics(arm)}"
            )

        if now - last_log_t >= 1.0:
            logging.info("Waiting for %s arm enable... %.1fs elapsed; %s", name, elapsed, arm_diagnostics(arm))
            last_log_t = now
        time.sleep(0.05)


def create_piper_arm(name: str, channel: str, args: Args, *, initial_gripper: float) -> PiperArm:
    agx_arm_factory, arm_model, piper_fw, create_config = import_pyagxarm()
    logging.info("Creating %s arm config on %s", name, channel)
    cfg = create_config(
        robot=arm_model.PIPER,
        firmeware_version=piper_fw.DEFAULT,
        interface="socketcan",
        channel=channel,
    )
    logging.info("Creating %s arm object", name)
    arm = agx_arm_factory.create_arm(cfg)
    logging.info("Connecting %s arm on %s", name, channel)
    arm.connect()
    logging.info("Setting %s arm speed=%d", name, args.speed_percent)
    arm.set_speed_percent(args.speed_percent)
    logging.info("Setting %s arm motion mode", name)
    arm.set_motion_mode(arm.OPTIONS.MOTION_MODE.J)
    logging.info("Setting %s arm installation position", name)
    arm.set_installation_pos(arm.OPTIONS.INSTALLATION_POS.HORIZONTAL)

    if args.enable_on_start and not args.infer_only:
        enable_arm_with_timeout(arm, name, args.enable_timeout_s)
    else:
        logging.info("Skipping %s arm enable", name)

    logging.info("Initializing %s gripper", name)
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    logging.info("Connected %s arm on %s", name, channel)
    return PiperArm(name=name, arm=arm, gripper=gripper, last_gripper=float(initial_gripper))


def close_arm(bundle: PiperArm | None) -> None:
    if bundle is None:
        return
    try:
        bundle.arm.disconnect()
    except Exception as exc:
        logging.warning("Failed to disconnect %s arm: %s", bundle.name, exc)


def make_camera(name: str, backend: CameraBackend, selector: str, args: Args) -> Camera:
    hint = f"Pass --cam-{name.removeprefix('cam_').replace('_', '-')} to choose explicitly."
    return make_common_camera(
        name,
        backend,
        selector,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        fallback_backends="v4l2/black",
        multiple_devices_hint=hint,
    )


def make_cameras(args: Args) -> dict[str, Camera]:
    return {
        "cam_head": make_camera("cam_head", args.cam_head_backend, args.cam_head, args),
        "cam_left_wrist": make_camera(
            "cam_left_wrist",
            args.cam_left_wrist_backend,
            args.cam_left_wrist,
            args,
        ),
        "cam_right_wrist": make_camera(
            "cam_right_wrist",
            args.cam_right_wrist_backend,
            args.cam_right_wrist,
            args,
        ),
    }


def read_joint_angles(bundle: PiperArm) -> np.ndarray:
    joint_msg = bundle.arm.get_joint_angles()
    if joint_msg is None:
        raise RuntimeError(f"{bundle.name} joint feedback is not available")
    joints = np.asarray(joint_msg.msg, dtype=np.float32)
    if joints.shape != (6,):
        raise RuntimeError(f"{bundle.name} joint feedback has shape {joints.shape}, expected (6,)")
    return joints


def read_gripper(bundle: PiperArm, args: Args) -> float:
    status = bundle.gripper.get_gripper_status()
    if status is None:
        return bundle.last_gripper

    mode = getattr(status.msg, "mode", None)
    value = getattr(status.msg, "value", None)
    if mode == "angle" and value is not None:
        denom = args.gripper_open_deg - args.gripper_closed_deg
        if abs(denom) > 1e-6:
            bundle.last_gripper = float(np.clip((float(value) - args.gripper_closed_deg) / denom, 0.0, 1.0))
    return bundle.last_gripper


def _read_state_unlocked(left: PiperArm, right: PiperArm, args: Args) -> np.ndarray:
    left_joints = read_joint_angles(left)
    right_joints = read_joint_angles(right)
    return np.concatenate(
        [
            left_joints,
            np.asarray([read_gripper(left, args)], dtype=np.float32),
            right_joints,
            np.asarray([read_gripper(right, args)], dtype=np.float32),
        ]
    ).astype(np.float32)


def read_state(
    left: PiperArm,
    right: PiperArm,
    args: Args,
    *,
    robot_lock: threading.Lock | None = None,
) -> np.ndarray:
    if robot_lock is None:
        return _read_state_unlocked(left, right, args)
    with robot_lock:
        return _read_state_unlocked(left, right, args)


def prepare_observation(
    cameras: dict[str, Camera],
    left: PiperArm,
    right: PiperArm,
    args: Args,
    *,
    robot_lock: threading.Lock | None = None,
) -> dict:
    return capture_observation_snapshot(cameras, left, right, args, robot_lock=robot_lock).to_policy_observation()


def capture_observation_snapshot(
    cameras: dict[str, Camera],
    left: PiperArm,
    right: PiperArm,
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
        image_masks={
            "cam_head": np.bool_(args.cam_head_backend != "black"),
            "cam_left_wrist": np.bool_(args.cam_left_wrist_backend != "black"),
            "cam_right_wrist": np.bool_(args.cam_right_wrist_backend != "black"),
        },
    )


def action_to_targets(action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] < 14:
        raise ValueError(f"Expected at least 14 action dims, got {action.shape[-1]}")
    return action[:6], float(action[6]), action[7:13], float(action[13])


def response_to_action_chunk(response: dict, args: Args) -> np.ndarray:
    return decode_action_chunk_for_spec(
        response,
        action_spec=ActionSpec(
            action_dim=14,
            state_dim=14,
            joint_dof_per_arm=6,
            joint_unit="rad",
            camera_roles=("cam_head", "cam_left_wrist", "cam_right_wrist"),
        ),
        replan_steps=args.replan_steps,
    )


def gripper_to_deg(value: float, args: Args) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return args.gripper_closed_deg + value * (args.gripper_open_deg - args.gripper_closed_deg)


def limit_joint_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return target
    return current + np.clip(target - current, -max_step, max_step)


def smooth_action(action: np.ndarray, last_action: np.ndarray | None, args: Args) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if last_action is None:
        return action

    last_action = np.asarray(last_action, dtype=np.float32)
    joint_alpha = float(np.clip(args.action_smoothing, 0.0, 0.98))
    if joint_alpha > 0:
        action[_JOINT_ACTION_MASK] = (
            joint_alpha * last_action[_JOINT_ACTION_MASK] + (1.0 - joint_alpha) * action[_JOINT_ACTION_MASK]
        )

    gripper_alpha = float(np.clip(args.gripper_smoothing, 0.0, 0.98))
    if gripper_alpha > 0:
        action[_GRIPPER_ACTION_MASK] = (
            gripper_alpha * last_action[_GRIPPER_ACTION_MASK] + (1.0 - gripper_alpha) * action[_GRIPPER_ACTION_MASK]
        )
    return action


def stabilize_action(action: np.ndarray, last_action: np.ndarray | None, args: Args) -> np.ndarray:
    action = smooth_action(action, last_action, args)
    if last_action is None:
        return action

    last_action = np.asarray(last_action, dtype=np.float32)
    joint_delta = action[_JOINT_ACTION_MASK] - last_action[_JOINT_ACTION_MASK]

    if args.joint_deadband > 0:
        small = np.abs(joint_delta) < args.joint_deadband
        joint_values = action[_JOINT_ACTION_MASK]
        joint_values[small] = last_action[_JOINT_ACTION_MASK][small]
        action[_JOINT_ACTION_MASK] = joint_values
        joint_delta = action[_JOINT_ACTION_MASK] - last_action[_JOINT_ACTION_MASK]

    if args.max_action_step > 0:
        action[_JOINT_ACTION_MASK] = last_action[_JOINT_ACTION_MASK] + np.clip(
            joint_delta,
            -args.max_action_step,
            args.max_action_step,
        )
    return action


def move_arm_joints(bundle: PiperArm, joints: np.ndarray, args: Args) -> None:
    joints_list = joints.tolist()
    if args.arm_command == "move_js":
        bundle.arm.move_js(joints_list)
        return
    if args.arm_command == "auto" and hasattr(bundle.arm, "move_js"):
        bundle.arm.move_js(joints_list)
        return
    bundle.arm.move_j(joints_list)


def limit_action_to_feedback_unlocked(action: np.ndarray, left: PiperArm, right: PiperArm, args: Args) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if args.max_joint_step > 0:
        left_joints, left_gripper, right_joints, right_gripper = action_to_targets(action)
        left_joints = limit_joint_step(left_joints, read_joint_angles(left), args.max_joint_step)
        right_joints = limit_joint_step(right_joints, read_joint_angles(right), args.max_joint_step)
        action[:6] = left_joints
        action[6] = left_gripper
        action[7:13] = right_joints
        action[13] = right_gripper
    return action


def _send_action_unlocked(
    action: np.ndarray,
    left: PiperArm,
    right: PiperArm,
    args: Args,
    *,
    send_gripper: bool = True,
) -> None:
    left_joints, left_gripper, right_joints, right_gripper = action_to_targets(action)

    left.last_gripper = float(np.clip(left_gripper, 0.0, 1.0))
    right.last_gripper = float(np.clip(right_gripper, 0.0, 1.0))

    if args.dry_run:
        logging.info(
            "dry-run action left=%s lg=%.3f right=%s rg=%.3f",
            np.array2string(left_joints, precision=3),
            left.last_gripper,
            np.array2string(right_joints, precision=3),
            right.last_gripper,
        )
        return

    move_arm_joints(left, left_joints, args)
    move_arm_joints(right, right_joints, args)
    if send_gripper:
        left.gripper.move_gripper_deg(gripper_to_deg(left.last_gripper, args), force=args.gripper_force)
        right.gripper.move_gripper_deg(gripper_to_deg(right.last_gripper, args), force=args.gripper_force)


def _execute_action_unlocked(action: np.ndarray, left: PiperArm, right: PiperArm, args: Args) -> None:
    limited_action = limit_action_to_feedback_unlocked(action, left, right, args)
    _send_action_unlocked(limited_action, left, right, args)


def execute_action(
    action: np.ndarray,
    left: PiperArm,
    right: PiperArm,
    args: Args,
    *,
    robot_lock: threading.Lock | None = None,
) -> None:
    if robot_lock is None:
        _execute_action_unlocked(action, left, right, args)
        return
    with robot_lock:
        _execute_action_unlocked(action, left, right, args)


def interpolation_steps(args: Args) -> int:
    if args.dry_run or not args.interpolate_actions or args.command_rate_hz <= args.fps:
        return 1
    return max(1, round(args.command_rate_hz / args.fps))


def execute_action_transition(
    start_action: np.ndarray | None,
    target_action: np.ndarray,
    left: PiperArm,
    right: PiperArm,
    args: Args,
    *,
    robot_lock: threading.Lock | None = None,
) -> np.ndarray:
    if robot_lock is None:
        limited_target = limit_action_to_feedback_unlocked(target_action, left, right, args)
    else:
        with robot_lock:
            limited_target = limit_action_to_feedback_unlocked(target_action, left, right, args)

    if start_action is None or interpolation_steps(args) <= 1:
        execute_action(limited_target, left, right, args, robot_lock=robot_lock)
        return limited_target

    start_action = np.asarray(start_action, dtype=np.float32)
    steps = interpolation_steps(args)
    interval_s = 1.0 / args.fps / steps
    next_t = time.monotonic()
    for i in range(1, steps + 1):
        ratio = i / steps
        command = start_action + ratio * (limited_target - start_action)
        send_gripper = args.command_gripper_every_step or i == steps
        if robot_lock is None:
            _send_action_unlocked(command, left, right, args, send_gripper=send_gripper)
        else:
            with robot_lock:
                _send_action_unlocked(command, left, right, args, send_gripper=send_gripper)
        next_t += interval_s
        if i < steps:
            sleep_until(next_t)
    return limited_target


def wait_for_arm_idle(bundle: PiperArm, timeout_s: float) -> bool:
    if timeout_s <= 0:
        return True

    start_t = time.monotonic()
    while time.monotonic() - start_t <= timeout_s:
        status = bundle.arm.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        time.sleep(0.05)
    return False


def maybe_reset_arms(left: PiperArm, right: PiperArm, args: Args) -> None:
    if not args.reset_on_start:
        return

    left.last_gripper = float(np.clip(args.init_left_gripper, 0.0, 1.0))
    right.last_gripper = float(np.clip(args.init_right_gripper, 0.0, 1.0))
    if args.dry_run:
        logging.info("dry-run reset requested; not moving arms")
        return

    logging.info(
        "Resetting Piper arms before connecting policy server: left=%s right=%s",
        args.init_left_joints,
        args.init_right_joints,
    )
    left.arm.move_j(list(args.init_left_joints))
    right.arm.move_j(list(args.init_right_joints))
    left.gripper.move_gripper_deg(gripper_to_deg(left.last_gripper, args), force=args.gripper_force)
    right.gripper.move_gripper_deg(gripper_to_deg(right.last_gripper, args), force=args.gripper_force)

    if args.reset_timeout_s <= 0:
        return

    time.sleep(0.5)
    left_ready = wait_for_arm_idle(left, args.reset_timeout_s)
    right_ready = wait_for_arm_idle(right, args.reset_timeout_s)
    if not left_ready or not right_ready:
        raise TimeoutError(
            "Timed out waiting for reset motion to finish "
            f"(left_ready={left_ready}, right_ready={right_ready}, timeout={args.reset_timeout_s:.1f}s)"
        )
    logging.info("Piper reset motion finished")


@dataclasses.dataclass
class PiperInferenceAdapter:
    robot: PiperRobot
    cameras: dict[str, Camera]
    args: Args
    name: str = "piper"
    last_observation_snapshot: ObservationSnapshot | None = dataclasses.field(default=None, init=False, repr=False)

    def observe(self) -> dict[str, Any]:
        return self.capture_observation_snapshot().to_policy_observation()

    def capture_observation_snapshot(self) -> ObservationSnapshot:
        self.last_observation_snapshot = capture_observation_snapshot(
            self.cameras,
            self.robot.left,
            self.robot.right,
            self.args,
            robot_lock=self.robot.robot_lock,
        )
        return self.last_observation_snapshot

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        if replan_steps != self.args.replan_steps:
            raise ValueError("Piper replan_steps must match its runtime config")
        return response_to_action_chunk(response, self.args)

    def initial_action(self) -> np.ndarray:
        return self.robot.read_state().values

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        return stabilize_action(action, previous, self.args)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return self.robot.execute_transition(previous, target)

    def infer_only_metadata(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 1.0 / self.args.fps


def _adapter(
    cameras: dict[str, Camera], left: PiperArm, right: PiperArm, args: Args, robot_lock: threading.Lock | None = None
) -> PiperInferenceAdapter:
    return PiperInferenceAdapter(PiperRobot(left, right, args, robot_lock or threading.Lock()), cameras, args)


def run_infer_only(
    client: _websocket_client_policy.WebsocketClientPolicy,
    cameras: dict[str, Camera],
    left: PiperArm,
    right: PiperArm,
    args: Args,
) -> None:
    run_generic_infer_only(client, _adapter(cameras, left, right, args), InferenceLoopConfig.from_args(args))


def run_sync_loop(
    client: _websocket_client_policy.WebsocketClientPolicy,
    cameras: dict[str, Camera],
    left: PiperArm,
    right: PiperArm,
    args: Args,
    robot_lock: threading.Lock,
) -> None:
    run_generic_sync_loop(client, _adapter(cameras, left, right, args, robot_lock), InferenceLoopConfig.from_args(args))


def run_rtc_loop(
    client: _websocket_client_policy.WebsocketClientPolicy,
    cameras: dict[str, Camera],
    left: PiperArm,
    right: PiperArm,
    args: Args,
    robot_lock: threading.Lock,
) -> None:
    run_generic_rtc_loop(client, _adapter(cameras, left, right, args, robot_lock), InferenceLoopConfig.from_args(args))


def create_robot(args: Args) -> PiperRobot:
    left = create_piper_arm("left", args.left_can, args, initial_gripper=args.init_left_gripper)
    try:
        right = create_piper_arm("right", args.right_can, args, initial_gripper=args.init_right_gripper)
    except Exception:
        close_arm(left)
        raise
    return PiperRobot(left, right, args)


def main(args: Args) -> None:
    loop_config = InferenceLoopConfig.from_args(args)
    loop_config.validate()
    if args.command_rate_hz <= 0:
        raise ValueError("command_rate_hz must be positive")

    robot: PiperRobot | None = None
    cameras: dict[str, Camera] = {}

    try:
        robot = create_robot(args)

        if args.reset_only:
            robot.reset()
            return

        if not args.infer_only:
            robot.reset()

        host, port = parse_server_url(args.server_url)
        logging.info("Connecting to OpenPI server at %s%s", host, f":{port}" if port is not None else "")
        client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=args.api_key)
        logging.info("Server metadata: %s", client.get_server_metadata())

        cameras = make_cameras(args)
        adapter = PiperInferenceAdapter(robot, cameras, args)
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


register_robot("piper", create_robot)


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


if __name__ == "__main__":
    cli()
