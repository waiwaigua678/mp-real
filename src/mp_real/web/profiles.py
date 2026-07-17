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
    validate_args: Callable[[Any], None]
    configure_reset: Callable[[Any], Any]
    include_camera_params: bool
    capabilities: RobotWebCapabilities

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


def _rm2_action_spec(args: infer_rm2.Args) -> ActionSpec:
    dimension = infer_rm2.action_dim(args)
    fields = tuple(
        [
            *(
                VectorField(f"left_joint_{index}", args.policy_joint_unit, "joint_position")
                for index in range(1, args.joint_dof + 1)
            ),
            *(
                VectorField(f"right_joint_{index}", args.policy_joint_unit, "joint_position")
                for index in range(1, args.joint_dof + 1)
            ),
            VectorField("left_gripper", "normalized_0_closed_1_open", "gripper_open_fraction"),
            VectorField("right_gripper", "normalized_0_closed_1_open", "gripper_open_fraction"),
        ]
    )
    return ActionSpec(
        action_dim=dimension,
        state_dim=dimension,
        joint_dof_per_arm=args.joint_dof,
        joint_unit=args.policy_joint_unit,
        camera_roles=("left_color", "right_color", "head_color"),
        state_fields=fields,
        action_fields=fields,
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


PIPER_WEB_PROFILE = RobotWebProfile(
    robot_name="piper",
    default_args=_piper_default_args,
    create_cameras=infer_piper.make_cameras,
    action_spec_for_args=_piper_action_spec,
    camera_roles_for_args=lambda args: _piper_action_spec(args).camera_roles,
    camera_masks_for_args=_piper_camera_masks,
    make_adapter=_piper_adapter,
    make_loop_config=_piper_loop_config,
    validate_args=_piper_validate,
    configure_reset=_piper_reset_args,
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
    validate_args=infer_rm2.validate_args,
    configure_reset=_rm2_reset_args,
    include_camera_params=True,
    capabilities=RobotWebCapabilities(True, True, False, True, ("sync", "rtc", "infer_only")),
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
