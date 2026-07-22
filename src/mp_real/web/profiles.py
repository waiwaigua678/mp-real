from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mp_real.common.camera import Camera, close_cameras
from mp_real.robots.base import Robot
from mp_real.robots.piper import infer as infer_piper
from mp_real.robots.registry import create_robot
from mp_real.robots.rm2 import infer as infer_rm2
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.models import ActionSpec, VectorField
from mp_real.safety.models import RobotSafetyProfile
from mp_real.web.runtime import CachedFrameObservationSource, WebInferenceAdapter


@dataclasses.dataclass(frozen=True)
class RobotWebCapabilities:
    supports_reset: bool
    supports_move_to_recorded_state: bool
    supports_trajectory_replay: bool
    supports_gripper: bool
    control_modes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RobotWebProfile:
    """Robot-owned wiring consumed by the shared Web lifecycle.

    The profile deliberately contains only construction and conversion logic.
    It neither replaces ``Robot``/``ActionSpec`` nor introduces a second policy
    loop; its adapter delegates all robot-specific decoding and stabilization
    to the existing robot modules.
    """

    robot_name: str
    default_args: Callable[[], Any]
    create_cameras: Callable[[Any], dict[str, Camera]]
    action_spec_for_args: Callable[[Any], ActionSpec]
    camera_roles_for_args: Callable[[Any], tuple[str, ...]]
    camera_masks_for_args: Callable[[Any], Mapping[str, np.bool_]]
    make_adapter: Callable[
        [
            Robot,
            Any,
            Callable[[], tuple[Mapping[str, np.ndarray], Mapping[str, Any] | None]],
            Callable[[str, float], None],
        ],
        WebInferenceAdapter,
    ]
    make_loop_config: Callable[[Any], InferenceLoopConfig]
    safety_profile_for_args: Callable[[Any], RobotSafetyProfile]
    validate_args: Callable[[Any], None]
    configure_reset: Callable[[Any], Any]
    baseline_config_for_args: Callable[[Any], Mapping[str, Mapping[str, Any]]]
    include_camera_params: bool
    capabilities: RobotWebCapabilities
    initialize_cameras_before_robot: bool = False

    def create_robot(self, args: Any) -> Robot:
        return create_robot(self.robot_name, args)


def _piper_default_args() -> infer_piper.Args:
    args = infer_piper.Args()
    args.init_left_joints = (0.0,) * 6
    args.init_right_joints = (0.0,) * 6
    args.init_left_gripper = 1.0
    args.init_right_gripper = 1.0
    return args


def _piper_action_spec(args: infer_piper.Args) -> ActionSpec:
    del args
    fields = tuple(
        field
        for arm in ("left", "right")
        for field in (
            *(VectorField(f"{arm}_joint_{index}", "rad", "joint_position") for index in range(1, 7)),
            VectorField(f"{arm}_gripper", "normalized_0_open_1", "gripper_open_fraction"),
        )
    )
    return ActionSpec(
        action_dim=14,
        state_dim=14,
        joint_dof_per_arm=6,
        joint_unit="rad",
        camera_roles=("cam_head", "cam_left_wrist", "cam_right_wrist"),
        state_fields=fields,
        action_fields=fields,
        capabilities={
            "supports_reset": True,
            "supports_move_to_recorded_state": True,
            "supports_trajectory_replay": False,
            "supports_gripper": True,
        },
    )


def _piper_camera_masks(args: infer_piper.Args) -> Mapping[str, np.bool_]:
    return {
        "cam_head": np.bool_(args.cam_head_backend != "black"),
        "cam_left_wrist": np.bool_(args.cam_left_wrist_backend != "black"),
        "cam_right_wrist": np.bool_(args.cam_right_wrist_backend != "black"),
    }


def _piper_adapter(
    robot: Robot,
    args: infer_piper.Args,
    read_images: Callable[[], tuple[Mapping[str, np.ndarray], Mapping[str, Any] | None]],
    profile_callback: Callable[[str, float], None],
) -> WebInferenceAdapter:
    source = CachedFrameObservationSource(
        robot=robot,
        read_images=read_images,
        image_masks=_piper_camera_masks(args),
        prompt=args.prompt,
    )

    def decode(response: dict[str, Any], replan_steps: int) -> np.ndarray:
        if replan_steps != args.replan_steps:
            raise ValueError("Piper replan_steps must match the Web runtime configuration")
        return infer_piper.response_to_action_chunk(response, args)

    return WebInferenceAdapter(
        name="piper",
        robot=robot,
        observation_source=source,
        decode_chunk=decode,
        stabilize=lambda action, previous: infer_piper.stabilize_action(action, previous, args),
        infer_interval_s=1.0 / args.fps,
        profile_callback=profile_callback,
    )


def _piper_loop_config(args: infer_piper.Args) -> InferenceLoopConfig:
    config = InferenceLoopConfig.from_args(args)
    if args.infer_only and args.max_steps is not None:
        config = dataclasses.replace(config, infer_only_chunks=args.max_steps)
    return config


def _piper_validate(args: infer_piper.Args) -> None:
    if args.command_rate_hz <= 0:
        raise ValueError("command_rate_hz must be positive")
    infer_piper.safety_profile_from_args(args, _piper_action_spec(args))
    _piper_loop_config(args).validate()


def _piper_reset_args(args: infer_piper.Args) -> infer_piper.Args:
    return dataclasses.replace(
        args,
        reset_on_start=True,
        init_left_joints=(0.0,) * 6,
        init_right_joints=(0.0,) * 6,
        init_left_gripper=1.0,
        init_right_gripper=1.0,
    )


def _piper_baseline_config(args: infer_piper.Args) -> Mapping[str, Mapping[str, Any]]:
    """Piper-owned Baseline categories; shared evaluation code stays vendor-neutral."""
    return {
        "camera_config": {
            "roles": _piper_action_spec(args).camera_roles,
            "backends": {
                "cam_head": args.cam_head_backend,
                "cam_left_wrist": args.cam_left_wrist_backend,
                "cam_right_wrist": args.cam_right_wrist_backend,
            },
            "selectors": {
                "cam_head": args.cam_head,
                "cam_left_wrist": args.cam_left_wrist,
                "cam_right_wrist": args.cam_right_wrist,
            },
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "timeout_s": args.camera_timeout,
        },
        "robot_config": {
            "transports": {"left": args.left_can, "right": args.right_can},
            "enable_on_start": args.enable_on_start,
            "reset_on_start": args.reset_on_start,
            "speed_percent": args.speed_percent,
            "command": args.arm_command,
            "initial_joints": {"left": args.init_left_joints, "right": args.init_right_joints},
            "initial_grippers": {"left": args.init_left_gripper, "right": args.init_right_gripper},
        },
        "safety_config": {
            "policy": args.safety_policy.value
            if hasattr(args.safety_policy, "value")
            else str(args.safety_policy),
            "profile_path": str(args.safety_profile_path) if args.safety_profile_path is not None else None,
            "hardware_motion_enabled": args.hardware_motion_enabled,
            "development_override": {
                "enabled": str(args.safety_policy) == "development_override",
                "operator": args.safety_override_operator,
                "reason": args.safety_override_reason,
            },
            "max_joint_step": args.max_joint_step,
            "max_action_step": args.max_action_step,
            "joint_deadband": args.joint_deadband,
            "action_smoothing": args.action_smoothing,
            "gripper_smoothing": args.gripper_smoothing,
            "interpolate_actions": args.interpolate_actions,
            "command_rate_hz": args.command_rate_hz,
            "command_gripper_every_step": args.command_gripper_every_step,
            "hold_last_action": args.hold_last_action,
            "gripper": {
                "closed_deg": args.gripper_closed_deg,
                "open_deg": args.gripper_open_deg,
                "force": args.gripper_force,
            },
        },
    }


def _rm2_action_spec(args: infer_rm2.Args) -> ActionSpec:
    dimension = infer_rm2.action_dim(args)
    fields = infer_rm2._vector_fields(args)
    return ActionSpec(
        action_dim=dimension,
        state_dim=dimension,
        joint_dof_per_arm=args.joint_dof,
        joint_unit=args.policy_joint_unit,
        camera_roles=("left_color", "right_color", "head_color"),
        state_fields=fields,
        action_fields=fields,
        capabilities={
            "supports_reset": True,
            "supports_move_to_recorded_state": True,
            "supports_trajectory_replay": False,
            "supports_gripper": True,
        },
    )


def _rm2_camera_masks(args: infer_rm2.Args) -> Mapping[str, np.bool_]:
    return {role: np.bool_(args.camera_backend != "black") for role in _rm2_action_spec(args).camera_roles}


def _rm2_adapter(
    robot: Robot,
    args: infer_rm2.Args,
    read_images: Callable[[], tuple[Mapping[str, np.ndarray], Mapping[str, Any] | None]],
    profile_callback: Callable[[str, float], None],
) -> WebInferenceAdapter:
    source = CachedFrameObservationSource(
        robot=robot,
        read_images=read_images,
        image_masks=_rm2_camera_masks(args),
        prompt=args.prompt,
        state_transform=lambda state: infer_rm2.policy_state_from_feedback(state, args),
    )

    def decode(response: dict[str, Any], replan_steps: int) -> np.ndarray:
        if replan_steps != args.replan_steps:
            raise ValueError("RM2 replan_steps must match the Web runtime configuration")
        return infer_rm2.response_to_action_chunk(response, args)

    def profile(stage: str, elapsed_s: float) -> None:
        profile_callback(stage, elapsed_s)

    return WebInferenceAdapter(
        name="rm2",
        robot=robot,
        observation_source=source,
        decode_chunk=decode,
        stabilize=lambda action, previous: infer_rm2.stabilize_action(action, previous, args),
        metadata_keys=("camera_params",),
        profile_callback=profile,
    )


def _rm2_reset_args(args: infer_rm2.Args) -> infer_rm2.Args:
    return dataclasses.replace(args, reset_on_start=True)


def _rm2_baseline_config(args: infer_rm2.Args) -> Mapping[str, Mapping[str, Any]]:
    """RM2-owned Baseline categories; no Piper layout leaks into shared code."""
    return {
        "camera_config": {
            "roles": _rm2_action_spec(args).camera_roles,
            "backend": args.camera_backend,
            "topics": {
                "left": args.cam_left_topic,
                "right": args.cam_right_topic,
                "head": args.cam_head_topic,
            },
            "serials": {"left": args.cam_left_serial, "right": args.cam_right_serial, "head": args.cam_head_serial},
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "timeout_s": args.camera_timeout,
        },
        "robot_config": {
            "backend": args.robot_backend,
            "connection": {"left_ip": args.left_ip, "right_ip": args.right_ip, "port": args.arm_port},
            "joint_dof": args.joint_dof,
            "policy_joint_unit": args.policy_joint_unit,
            "policy_gripper_unit": args.policy_gripper_unit,
            "reset_on_start": args.reset_on_start,
            "speed_percent": args.speed_percent,
            "command": args.arm_command,
            "command_left_arm": args.command_left_arm,
            "command_right_arm": args.command_right_arm,
            "use_static_left_state": args.use_static_left_state,
            "static_left_joints": args.static_left_joints,
            "initial_joints": {"left": args.init_left_joints, "right": args.init_right_joints},
            "initial_grippers": {"left": args.init_left_gripper, "right": args.init_right_gripper},
        },
        "safety_config": {
            "policy": args.safety_policy.value
            if hasattr(args.safety_policy, "value")
            else str(args.safety_policy),
            "profile_path": str(args.safety_profile_path) if args.safety_profile_path is not None else None,
            "hardware_motion_enabled": args.hardware_motion_enabled,
            "development_override": {
                "enabled": str(args.safety_policy) == "development_override",
                "operator": args.safety_override_operator,
                "reason": args.safety_override_reason,
            },
            "max_joint_step_deg": args.max_joint_step_deg,
            "max_action_step_deg": args.max_action_step_deg,
            "action_smoothing": args.action_smoothing,
            "gripper_smoothing": args.gripper_smoothing,
            "interpolate_actions": args.interpolate_actions,
            "command_rate_hz": args.command_rate_hz,
            "command_gripper_every_step": args.command_gripper_every_step,
            "hold_last_action": args.hold_last_action,
            "gripper": {
                "min": args.gripper_min,
                "max": args.gripper_max,
                "timeout": args.gripper_timeout,
                "async": args.async_gripper,
                "command_rate_hz": args.gripper_command_rate_hz,
                "command_deadband": args.gripper_command_deadband,
                "flush_timeout_s": args.gripper_flush_timeout,
            },
        },
    }


PIPER_WEB_PROFILE = RobotWebProfile(
    robot_name="piper",
    default_args=_piper_default_args,
    create_cameras=infer_piper.make_cameras,
    action_spec_for_args=_piper_action_spec,
    camera_roles_for_args=lambda args: _piper_action_spec(args).camera_roles,
    camera_masks_for_args=_piper_camera_masks,
    make_adapter=_piper_adapter,
    make_loop_config=_piper_loop_config,
    safety_profile_for_args=infer_piper.safety_profile_from_args,
    validate_args=_piper_validate,
    configure_reset=_piper_reset_args,
    baseline_config_for_args=_piper_baseline_config,
    include_camera_params=False,
    capabilities=RobotWebCapabilities(True, True, False, True, ("sync", "rtc", "infer_only")),
)

RM2_WEB_PROFILE = RobotWebProfile(
    robot_name="rm2",
    default_args=infer_rm2.Args,
    create_cameras=infer_rm2.make_cameras,
    action_spec_for_args=_rm2_action_spec,
    camera_roles_for_args=lambda args: _rm2_action_spec(args).camera_roles,
    camera_masks_for_args=_rm2_camera_masks,
    make_adapter=_rm2_adapter,
    make_loop_config=InferenceLoopConfig.from_args,
    safety_profile_for_args=infer_rm2.safety_profile_from_args,
    validate_args=infer_rm2.validate_args,
    configure_reset=_rm2_reset_args,
    baseline_config_for_args=_rm2_baseline_config,
    include_camera_params=True,
    capabilities=RobotWebCapabilities(True, True, False, True, ("sync", "rtc", "infer_only")),
    initialize_cameras_before_robot=True,
)

WEB_PROFILES: dict[str, RobotWebProfile] = {
    PIPER_WEB_PROFILE.robot_name: PIPER_WEB_PROFILE,
    RM2_WEB_PROFILE.robot_name: RM2_WEB_PROFILE,
}


def get_web_profile(robot_name: str) -> RobotWebProfile:
    try:
        return WEB_PROFILES[robot_name]
    except KeyError as exc:
        available = ", ".join(sorted(WEB_PROFILES))
        raise ValueError(f"Unknown robot profile {robot_name!r}; available: {available}") from exc


def clone_default_args(profile: RobotWebProfile) -> Any:
    return copy.deepcopy(profile.default_args())


def close_profile_cameras(cameras: Mapping[str, Camera]) -> None:
    close_cameras(cameras)
