from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from mp_real.data.lerobot_v21 import LeRobotV21EpisodeRecorder
from mp_real.data.models import EpisodeRecordingContext, RecorderConfig
from mp_real.runtime.events import ControlStepRecorded
from mp_real.runtime.models import ActionSpec, VectorField


def _spec(*, state_dim: int, action_dim: int, camera_count: int) -> ActionSpec:
    state_fields = tuple(VectorField(f"state_{index}", "unit", "unknown") for index in range(state_dim))
    action_fields = tuple(VectorField(f"action_{index}", "unit", "unknown") for index in range(action_dim))
    return ActionSpec(
        action_dim,
        state_dim,
        0,
        "unknown",
        tuple(f"cam_{index}" for index in range(camera_count)),
        state_fields=state_fields,
        action_fields=action_fields,
        action_mode="joint_position_target",
    )


def _event(
    *,
    episode_id: str,
    step: int,
    spec: ActionSpec,
    image: np.ndarray,
    timestamp_ns: int,
) -> ControlStepRecorded:
    images = {role: image for role in spec.camera_roles}
    return ControlStepRecorded(
        runtime_id="recorder-soak",
        session_id="recorder-soak",
        episode_id=episode_id,
        generation_id=1,
        step=step,
        monotonic_timestamp_ns=timestamp_ns,
        payload={
            "control_step_id": step,
            "observation_id": step,
            "policy_observation_id": step,
            "state": np.full(spec.state_dim, step % 17, dtype=np.float32),
            "images": images,
            "camera_frame_ids": {role: step for role in spec.camera_roles},
            "camera_timestamps_ns": {role: timestamp_ns for role in spec.camera_roles},
            "max_camera_skew_ns": 0,
            "selected_raw_action": np.zeros(spec.action_dim, dtype=np.float32),
            "stabilized_target_action": np.zeros(spec.action_dim, dtype=np.float32),
            "executed_action": np.zeros(spec.action_dim, dtype=np.float32),
            "action_sent_timestamp_ns": timestamp_ns,
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a hardware-free recorder memory soak with fake camera frames.")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cameras", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--telemetry-part-size-steps", type=int, default=900)
    parser.add_argument("--queue-size", type=int, default=2048)
    parser.add_argument("--report-every-steps", type=int, default=900)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--no-save-telemetry", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_minutes <= 0 or args.fps <= 0:
        parser.error("duration and fps must be positive")
    if args.cameras <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("camera count and image dimensions must be positive")

    root_parent = args.output_root or Path(tempfile.mkdtemp(prefix="mp-real-recorder-soak-"))
    root = root_parent / f"soak-{int(time.time())}"
    spec = _spec(state_dim=args.state_dim, action_dim=args.action_dim, camera_count=args.cameras)
    recorder = LeRobotV21EpisodeRecorder(
        RecorderConfig(
            root,
            root.name,
            "fake",
            args.fps,
            spec,
            save_video=args.save_video,
            queue_size=args.queue_size,
            save_telemetry=not args.no_save_telemetry,
            telemetry_part_size_steps=args.telemetry_part_size_steps,
        )
    )
    total_steps = int(args.duration_minutes * 60.0 * args.fps)
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    started = time.monotonic()
    recorder.start()
    try:
        recorder.begin_episode(EpisodeRecordingContext(0, "episode-0", "soak", "recorder-soak", 1))
        for step in range(total_steps):
            timestamp_ns = time.monotonic_ns()
            recorder.emit(_event(episode_id="episode-0", step=step, spec=spec, image=image, timestamp_ns=timestamp_ns))
            if (step + 1) % args.report_every_steps == 0:
                recorder.flush(timeout=30.0)
                print(json.dumps(dataclasses.asdict(recorder.metrics()), sort_keys=True), flush=True)
        recorder.end_episode(labels={"result": "SUCCESS", "soak_duration_minutes": args.duration_minutes})
        if not recorder.stop(timeout=120.0):
            raise TimeoutError("recorder did not stop")
        recorder.raise_if_failed()
    except BaseException:
        recorder.stop(finalize=False, timeout=30.0)
        raise
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "dataset_root": str(root),
                "elapsed_s": elapsed,
                "steps": total_steps,
                "metrics": dataclasses.asdict(recorder.metrics()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
