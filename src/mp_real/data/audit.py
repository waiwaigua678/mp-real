from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.data.deps import MissingOptionalDependencyError


def audit_lerobot_alignment(root: Path | str, *, episode_index: int | None = None) -> dict[str, Any]:
    from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource

    source = LeRobotV21EpisodeSource(root)
    try:
        episodes = (
            [source.get_episode_metadata(episode_index)]
            if episode_index is not None
            else list(source.list_episodes())
        )
        episode_reports = []
        for episode in episodes:
            episode_reports.append(_audit_episode(source, episode.episode_index))
        totals = {
            "same_policy_observation_pairs": sum(
                len(item["same_policy_observation_pairs"]) for item in episode_reports
            ),
            "repeated_camera_frame_pairs": sum(
                len(item["repeated_camera_frame_pairs"]) for item in episode_reports
            ),
            "identical_state_pairs": sum(len(item["identical_state_pairs"]) for item in episode_reports),
        }
        return {
            "path": str(Path(root)),
            "risk_only": True,
            "message": "Repeated observation, camera frame, or state values are audit signals, not automatic errors.",
            "totals": totals,
            "episodes": episode_reports,
        }
    finally:
        source.close()


def _audit_episode(source: Any, episode_index: int) -> dict[str, Any]:
    same_observations = []
    repeated_states = []
    previous_sample = None
    length = source.get_length(episode_index)
    for index in range(length):
        sample = source.get_sample(episode_index, index, include_images=False)
        if previous_sample is not None:
            previous_policy_id = previous_sample.telemetry.get("policy_observation_id")
            policy_id = sample.telemetry.get("policy_observation_id")
            previous_step_id = previous_sample.telemetry.get("control_step_id")
            step_id = sample.telemetry.get("control_step_id")
            if policy_id is not None and policy_id == previous_policy_id and step_id != previous_step_id:
                same_observations.append(
                    {
                        "previous_sample": previous_sample.index,
                        "sample": sample.index,
                        "policy_observation_id": _json_scalar(policy_id),
                    }
                )
            if np.array_equal(sample.state, previous_sample.state):
                repeated_states.append({"previous_sample": previous_sample.index, "sample": sample.index})
        previous_sample = sample

    repeated_frames = []
    telemetry = source.get_episode_telemetry(episode_index, keys=("camera_roles", "camera_frame_ids"))
    roles = [str(role) for role in telemetry.get("camera_roles", ())]
    frame_ids = np.asarray(telemetry.get("camera_frame_ids", np.empty((0, 0))), dtype=np.int64)
    if frame_ids.ndim == 2:
        for row_index in range(1, len(frame_ids)):
            for role_index, role in enumerate(roles[: frame_ids.shape[1]]):
                if frame_ids[row_index, role_index] == frame_ids[row_index - 1, role_index]:
                    repeated_frames.append(
                        {
                            "previous_sample": row_index - 1,
                            "sample": row_index,
                            "role": role,
                            "frame_id": int(frame_ids[row_index, role_index]),
                        }
                    )
    return {
        "episode_index": episode_index,
        "length": length,
        "same_policy_observation_pairs": same_observations,
        "repeated_camera_frame_pairs": repeated_frames,
        "identical_state_pairs": repeated_states,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def cli() -> None:
    parser = argparse.ArgumentParser(description="Audit mp-real LeRobot observation/action alignment risk signals")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode-index", type=int)
    args = parser.parse_args()
    try:
        report = audit_lerobot_alignment(args.dataset, episode_index=args.episode_index)
    except MissingOptionalDependencyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    cli()
