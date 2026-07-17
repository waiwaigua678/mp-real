"""Read-only conversion of a recorded sample into a pose target."""

from __future__ import annotations

import hashlib
from typing import Any

from mp_real.data.models import RecordedEpisodeSource
from mp_real.pose.models import RecordedPoseTarget


def recorded_pose_target(
    source: RecordedEpisodeSource,
    *,
    dataset_id: str,
    episode_index: int,
    sample_index: int,
) -> RecordedPoseTarget:
    """Build a target solely from ``RecordedSample.state``.

    This function intentionally never reads ``sample.action`` or telemetry
    actions.  The source metadata fingerprint allows a later connection phase
    to reject a changed dataset/sample before movement.
    """

    dataset = source.get_dataset_metadata()
    episode = source.get_episode_metadata(episode_index)
    state, timestamp = source.get_pose_state_sample(episode_index, sample_index)
    spec = source.get_action_spec()
    metadata: dict[str, Any] = {
        "dataset_status": dataset.status.value,
        "episode_status": episode.status.value,
        "dataset_root_fingerprint": hashlib.sha256(str(dataset.root).encode("utf-8")).hexdigest(),
        "dataset_name": str(dataset.root.name),
        "is_mp_real": dataset.is_mp_real,
    }
    return RecordedPoseTarget(
        dataset_id=dataset_id,
        episode_index=episode_index,
        sample_index=sample_index,
        robot_name=str(dataset.info.get("robot_type", "unknown")),
        state_schema=spec.state_field_names,
        state_values=state,
        state_fields=spec.state_fields,
        joint_unit=spec.joint_unit,
        timestamp=timestamp,
        source_metadata=metadata,
        action_spec=spec,
    )
