"""Dry-run-first CLI for inspecting a recorded robot-state target."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource
from mp_real.data.pose import recorded_pose_target
from mp_real.pose.config import load_pose_mapping_config
from mp_real.pose.controller import PoseMoveController
from mp_real.pose.models import MoveToRecordedStatePlan, PoseMotionConstraints
from mp_real.pose.validation import MoveToStateValidator
from mp_real.robots.pose import PoseControlCapability
from mp_real.robots.registry import create_robot
from mp_real.web.profiles import get_web_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan a safe move to RecordedSample.observation.state")
    parser.add_argument("--robot", required=True, choices=("piper", "rm2"))
    parser.add_argument("--dataset", required=True, type=Path, help="Local LeRobot v2.1 dataset directory")
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--sample-index", required=True, type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and print only (the default)")
    mode.add_argument(
        "--execute", action="store_true", help="Create a robot after validation and execute a confirmed plan"
    )
    parser.add_argument("--confirm-plan-hash", help="Exact plan hash printed by this command; required with --execute")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--keep-gripper", action="store_true")
    parser.add_argument("--config", type=Path, help="Versioned explicit JSON state mapping")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.episode_index < 0 or args.sample_index < 0:
        raise ValueError("episode-index and sample-index must be non-negative")
    if not 0 < args.speed_scale <= 1:
        raise ValueError("speed-scale must be in (0, 1]")

    source = LeRobotV21EpisodeSource(args.dataset)
    try:
        target = recorded_pose_target(
            source,
            dataset_id=args.dataset.name,
            episode_index=args.episode_index,
            sample_index=args.sample_index,
        )
        profile = get_web_profile(args.robot)
        robot_args = profile.default_args()
        spec = profile.action_spec_for_args(robot_args)
        mapping = load_pose_mapping_config(args.config) if args.config is not None else None
        validated = MoveToStateValidator(args.robot, spec, mapping_config=mapping).validate(target)
        print(f"dataset={target.dataset_id} episode={target.episode_index} sample={target.sample_index}")
        print(f"recorded_robot={target.robot_name} state_schema={target.state_schema}")
        print(f"target_state={target.state_values.tolist()}")
        for issue in (*validated.report.issues, *validated.report.warnings):
            print(f"{issue.severity.upper()} {issue.code}: {issue.message}")
        if not validated.report.valid:
            return 2
        if not args.execute:
            print("dry-run complete: no Robot, camera, or PolicyClient was created")
            return 0

        # This branch is deliberately unreachable without an explicit command
        # and a second exact plan-hash acknowledgement.  It never resets or
        # enables a robot implicitly.
        robot_args.reset_on_start = False
        robot_args.enable_on_start = False
        if hasattr(robot_args, "speed_percent"):
            robot_args.speed_percent = min(int(robot_args.speed_percent), 10)
        robot = create_robot(args.robot, robot_args)
        try:
            if not isinstance(robot, PoseControlCapability):
                raise RuntimeError(f"{args.robot} does not implement PoseControlCapability")
            constraints = PoseMotionConstraints(
                max_joint_velocity=PoseMotionConstraints().max_joint_velocity * args.speed_scale,
                max_joint_acceleration=PoseMotionConstraints().max_joint_acceleration * args.speed_scale,
                max_joint_step=PoseMotionConstraints().max_joint_step * args.speed_scale,
                keep_gripper=args.keep_gripper,
            )
            capability_report = robot.validate_pose_target(target)
            capability_report.require_valid()
            plan = robot.plan_move_to_state(
                MoveToRecordedStatePlan.build(
                    target=target,
                    current_state=robot.get_current_pose_state(),
                    target_state=validated.values,
                    gripper_indices=validated.gripper_indices,
                    mapped_joint_names=validated.field_names,
                    conversions=validated.mappings,
                    constraints=constraints,
                    safety_warnings=("vendor command speed capped at 10 percent",),
                    mapping_fingerprint=validated.report.mapping_fingerprint,
                )
            )
            print(f"plan_hash={plan.plan_hash}")
            if args.confirm_plan_hash != plan.plan_hash:
                raise RuntimeError("--execute requires --confirm-plan-hash matching the newly revalidated plan")
            controller = PoseMoveController(robot)
            controller.start(plan)
            if not controller.join(
                timeout=plan.expected_duration_s + constraints.verify_timeout_s + 5.0, raise_on_error=True
            ):
                controller.stop(wait=True, timeout=2.0)
                raise TimeoutError("pose controller did not stop before timeout")
            print(f"move result: {controller.result()}")
            return 0
        finally:
            robot.close()
    finally:
        source.close()


def cli() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    cli()
