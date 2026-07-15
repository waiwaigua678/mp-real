from __future__ import annotations

import argparse
import json
from pathlib import Path

from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource, validate_lerobot_v21_dataset


def inspect_cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local LeRobot v2.1 dataset without hardware access")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    source = LeRobotV21EpisodeSource(args.dataset)
    try:
        metadata = source.get_dataset_metadata()
        info = metadata.info
        payload = {
            "path": str(metadata.root),
            "codebase_version": info["codebase_version"],
            "fps": info["fps"],
            "features": info["features"],
            "state_shape": [metadata.action_spec.state_dim],
            "action_shape": [metadata.action_spec.action_dim],
            "camera_roles": list(metadata.camera_roles),
            "episode_count": len(source.list_episodes()),
            "episodes": [dataclass_to_json(item) for item in source.list_episodes()],
            "tasks": sorted({task for item in source.list_episodes() for task in item.tasks}),
            "mp_real_extensions": metadata.is_mp_real,
            "status": metadata.status.value,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    finally:
        source.close()


def validate_cli() -> None:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot v2.1 dataset without hardware access")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--skip-video-check", action="store_true")
    args = parser.parse_args()
    report = validate_lerobot_v21_dataset(args.dataset, check_videos=not args.skip_video_check)
    print(
        json.dumps(
            {
                "path": str(report.root),
                "valid": report.valid,
                "episodes_checked": report.episodes_checked,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report.valid:
        raise SystemExit(1)


def dataclass_to_json(value: object) -> dict[str, object]:
    return {
        "episode_index": getattr(value, "episode_index"),
        "length": getattr(value, "length"),
        "tasks": list(getattr(value, "tasks")),
        "status": getattr(value, "status").value,
        "labels": getattr(value, "labels"),
    }
