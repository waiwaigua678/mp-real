from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mp_real.common.camera import Camera
from mp_real.common.image import preprocess_image
from mp_real.runtime.models import CameraSample, ObservationSnapshot, RobotState


def capture_observation(
    cameras: Mapping[str, Camera],
    *,
    read_state: Callable[[], np.ndarray],
    prompt: str,
    resize_size: int,
    timeout: float,
    image_masks: Mapping[str, np.bool_],
    include_camera_params: bool = False,
) -> ObservationSnapshot:
    """Read a coherent best-effort camera/state snapshot with source timestamps."""

    capture_started_ns = time.monotonic_ns()
    images: dict[str, CameraSample] = {}
    camera_params: dict[str, Mapping[str, Any] | None] | None = {} if include_camera_params else None
    for name, camera in cameras.items():
        frame = camera.read_frame(timeout=timeout)
        images[name] = CameraSample(
            image=preprocess_image(frame.image, resize_size),
            timestamp_monotonic=frame.timestamp_monotonic,
            camera_timestamp=frame.camera_timestamp,
            info=frame.info,
            frame_id=frame.frame_id,
            timestamp_monotonic_ns=frame.timestamp_monotonic_ns,
            source_sequence=frame.source_sequence,
            capture_latency_ns=frame.capture_latency_ns,
        )
        if camera_params is not None:
            camera_params[name] = frame.info if frame.info is not None else camera.camera_info()

    state_value = read_state()
    state_finished_ns = time.monotonic_ns()
    if isinstance(state_value, RobotState):
        state = state_value
    else:
        state = RobotState(
            values=np.asarray(state_value, dtype=np.float32),
            timestamp_monotonic=state_finished_ns / 1e9,
            timestamp_monotonic_ns=state_finished_ns,
        )
    capture_finished_ns = time.monotonic_ns()
    return ObservationSnapshot(
        images=images,
        image_masks=dict(image_masks),
        state=state,
        prompt=prompt,
        camera_params=camera_params,
        captured_at_monotonic=capture_finished_ns / 1e9,
        capture_started_ns=capture_started_ns,
        capture_finished_ns=capture_finished_ns,
        state_timestamp_ns=state.timestamp_monotonic_ns,
        camera_frame_ids={name: sample.frame_id for name, sample in images.items()},
        camera_timestamps_ns={name: sample.timestamp_monotonic_ns for name, sample in images.items()},
    )
