from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mp_real.common.camera import Camera
from mp_real.common.image import preprocess_image
from mp_real.runtime.models import CameraSample, ObservationSnapshot, RobotState


def build_observation_snapshot(
    images: Mapping[str, np.ndarray | CameraSample],
    *,
    state: np.ndarray | RobotState,
    prompt: str,
    resize_size: int,
    image_masks: Mapping[str, np.bool_],
    camera_params: Mapping[str, Mapping[str, Any] | None] | None = None,
    captured_at_ns: int | None = None,
) -> ObservationSnapshot:
    """Build the shared policy observation from already acquired values.

    Live capture and recorded-data evaluation both enter through this function.
    Keeping image conversion here prevents an offline evaluator from drifting
    from deployment's resize/pad preprocessing or policy wire schema.
    """

    now_ns = time.monotonic_ns() if captured_at_ns is None else int(captured_at_ns)
    samples: dict[str, CameraSample] = {}
    for name, value in images.items():
        if isinstance(value, CameraSample):
            samples[name] = dataclasses.replace(value, image=preprocess_image(value.image, resize_size))
        else:
            samples[name] = CameraSample(
                image=preprocess_image(np.asarray(value), resize_size),
                timestamp_monotonic=now_ns / 1e9,
                timestamp_monotonic_ns=now_ns,
            )
    if isinstance(state, RobotState):
        robot_state = state
    else:
        robot_state = RobotState(
            values=np.asarray(state, dtype=np.float32),
            timestamp_monotonic=now_ns / 1e9,
            timestamp_monotonic_ns=now_ns,
        )
    return ObservationSnapshot(
        images=samples,
        image_masks=dict(image_masks),
        state=robot_state,
        prompt=prompt,
        camera_params=camera_params,
        captured_at_monotonic=now_ns / 1e9,
        capture_started_ns=now_ns,
        capture_finished_ns=now_ns,
        state_timestamp_ns=robot_state.timestamp_monotonic_ns,
        camera_frame_ids={name: sample.frame_id for name, sample in samples.items()},
        camera_timestamps_ns={name: sample.timestamp_monotonic_ns for name, sample in samples.items()},
    )


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
            image=np.asarray(frame.image),
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
    snapshot = build_observation_snapshot(
        images,
        state=state,
        prompt=prompt,
        resize_size=resize_size,
        image_masks=image_masks,
        camera_params=camera_params,
        captured_at_ns=capture_finished_ns,
    )
    # Preserve the live acquisition span while using the same builder as
    # recorded observations for all policy-visible fields.
    return dataclasses.replace(snapshot, capture_started_ns=capture_started_ns)
