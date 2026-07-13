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

    images: dict[str, CameraSample] = {}
    camera_params: dict[str, Mapping[str, Any] | None] | None = {} if include_camera_params else None
    for name, camera in cameras.items():
        frame = camera.read_frame(timeout=timeout)
        images[name] = CameraSample(
            image=preprocess_image(frame.image, resize_size),
            timestamp_monotonic=frame.timestamp_monotonic,
            camera_timestamp=frame.camera_timestamp,
            info=frame.info,
        )
        if camera_params is not None:
            camera_params[name] = frame.info if frame.info is not None else camera.camera_info()

    state_timestamp = time.monotonic()
    state = RobotState(values=np.asarray(read_state(), dtype=np.float32), timestamp_monotonic=state_timestamp)
    return ObservationSnapshot(
        images=images,
        image_masks=dict(image_masks),
        state=state,
        prompt=prompt,
        camera_params=camera_params,
    )
