"""Command-line entry point for policy-free trajectory replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource
from mp_real.replay.controller import RobotReplayController
from mp_real.replay.models import (
    ReplayAcknowledgementStrategy,
    ReplayConstraints,
    ReplayMode,
    ReplayTimingMode,
    json_safe,
)
from mp_real.replay.planning import ReplayPlanner
from mp_real.replay.recording import ReplayRecordingConfig, ReplayRecordWriter
from mp_real.web.profiles import get_web_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely replay a validated LeRobot v2.1 trajectory on a real robot")
    parser.add_argument("--robot", required=True, choices=("piper", "rm2"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--start-sample", type=int)
    parser.add_argument("--end-sample", type=int)
    parser.add_argument(
        "--mode", choices=tuple(mode.value for mode in ReplayMode), default=ReplayMode.COMMAND_REPLAY.value
    )
    parser.add_argument(
        "--timing",
        choices=tuple(mode.value for mode in ReplayTimingMode),
        default=ReplayTimingMode.RECORDED_TIMESTAMPS.value,
    )
    parser.add_argument("--fps", type=float, help="Required for --timing fixed")
    parser.add_argument("--speed-scale", type=float, default=0.1)
    parser.add_argument(
        "--acknowledgement-strategy",
        choices=tuple(strategy.value for strategy in ReplayAcknowledgementStrategy),
        default=ReplayAcknowledgementStrategy.FEEDBACK_THRESHOLD.value,
    )
    parser.add_argument("--feedback-poll-interval-s", type=float, default=0.01)
    parser.add_argument("--acknowledgement-timeout-s", type=float, default=1.0)
    parser.add_argument("--feedback-freshness-timeout-s", type=float)
    parser.add_argument("--follower-window-samples", type=int, default=1)
    parser.add_argument("--state-trajectory-settle-cycles", type=int, default=2)
    parser.add_argument("--sustained-tracking-error-limit", type=int, default=3)
    parser.add_argument("--extreme-tracking-error", type=float)
    parser.add_argument("--joint-max-step", type=float)
    parser.add_argument("--joint-max-velocity", type=float)
    parser.add_argument("--joint-max-acceleration", type=float)
    parser.add_argument("--joint-tracking-error", type=float)
    parser.add_argument("--gripper-max-step", type=float)
    parser.add_argument("--gripper-tracking-threshold", type=float)
    parser.add_argument(
        "--record-root",
        type=Path,
        default=Path("recordings/replay"),
        help="Directory for the explicit replay record produced by --execute",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the offline plan only (the default)")
    parser.add_argument("--execute", action="store_true", help="Create the robot and run the reviewed plan")
    parser.add_argument(
        "--confirm-plan-hash",
        help="Required with --execute; exact hash printed by the dry-run/plan report",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--dry-run and --execute are mutually exclusive")
    profile = get_web_profile(args.robot)
    robot_args = profile.default_args()
    source = LeRobotV21EpisodeSource(args.dataset)
    constraints = ReplayConstraints(
        acknowledgement_strategy=ReplayAcknowledgementStrategy(args.acknowledgement_strategy),
        feedback_poll_interval_s=args.feedback_poll_interval_s,
        acknowledgement_timeout_s=args.acknowledgement_timeout_s,
        feedback_freshness_timeout_s=args.feedback_freshness_timeout_s,
        follower_window_samples=args.follower_window_samples,
        state_trajectory_settle_cycles=args.state_trajectory_settle_cycles,
        sustained_tracking_error_limit=args.sustained_tracking_error_limit,
        extreme_tracking_error=args.extreme_tracking_error,
        joint_max_step=args.joint_max_step,
        joint_max_velocity=args.joint_max_velocity,
        joint_max_acceleration=args.joint_max_acceleration,
        joint_tracking_error=args.joint_tracking_error,
        gripper_max_step=args.gripper_max_step,
        gripper_tracking_threshold=args.gripper_tracking_threshold,
    )
    try:
        result = ReplayPlanner(source).plan(
            robot_name=args.robot,
            target_action_spec=profile.action_spec_for_args(robot_args),
            episode_index=args.episode_index,
            start_sample=args.start_sample,
            end_sample=args.end_sample,
            mode=ReplayMode(args.mode),
            timing_mode=ReplayTimingMode(args.timing),
            fps=args.fps,
            speed_scale=args.speed_scale,
            constraints=constraints,
        )
        print(json.dumps(json_safe(result.report), ensure_ascii=False, indent=2))
        if not result.report.valid or result.plan is None:
            return 2
        result.plan.require_integrity()
        if not args.execute:
            return 0
        result.plan.require_integrity(check_expiration=True)
        if args.confirm_plan_hash != result.plan.plan_hash:
            print("--execute requires --confirm-plan-hash matching the reviewed plan", file=sys.stderr)
            return 2
        # Do not allow an invocation intended for replay to reset or speed up
        # a robot.  Vendor validation still decides whether movement is safe.
        robot_args.reset_on_start = False
        if hasattr(robot_args, "enable_on_start"):
            robot_args.enable_on_start = False
        if hasattr(robot_args, "speed_percent"):
            robot_args.speed_percent = min(int(robot_args.speed_percent), 10)
        robot = profile.create_robot(robot_args)
        recorder = ReplayRecordWriter(ReplayRecordingConfig(args.record_root), result.plan)
        recorder.start()
        try:
            controller = RobotReplayController(
                robot,
                result.plan,
                record_callback=recorder.emit,
                thread_name=f"{args.robot}-replay",
            )
            controller.prepare()
            if not controller.join(timeout=30.0) or controller.cursor().state.value != "armed":
                print(json.dumps(json_safe(controller.cursor()), ensure_ascii=False, indent=2), file=sys.stderr)
                return 3
            controller.confirm_and_start(args.confirm_plan_hash)
            if not controller.join(timeout=result.plan.expected_duration_s + 30.0):
                controller.stop(emergency=True, wait=True, timeout=5.0)
                print("replay worker did not stop before timeout", file=sys.stderr)
                return 4
            print(json.dumps(json_safe(controller.cursor()), ensure_ascii=False, indent=2))
            return 0 if controller.cursor().state.value == "completed" else 4
        finally:
            recorder.stop(result=controller.cursor().state.value if "controller" in locals() else "connection_error")
            robot.close()
    finally:
        source.close()


if __name__ == "__main__":  # pragma: no cover - console script
    raise SystemExit(cli())
