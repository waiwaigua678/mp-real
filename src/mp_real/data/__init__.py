"""LeRobot v2.1 recording, reading, validation, and catalog primitives.

The package namespace itself stays lightweight.  Data/video implementations
load PyArrow and PyAV only when the corresponding feature is used.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DatasetMetadata",
    "DataViewSession",
    "EpisodeMetadata",
    "EpisodeRecordingContext",
    "EpisodeStatus",
    "EpisodeReader",
    "FakeRecordedEpisodeSource",
    "LeRobotV21EpisodeRecorder",
    "LeRobotV21EpisodeSource",
    "RecordedDataCatalog",
    "RecordedEpisodeSource",
    "RecordedSample",
    "RecorderConfig",
    "PlaybackCursor",
    "TimelineIndex",
    "ValidationReport",
    "ViewCursor",
    "validate_lerobot_v21_dataset",
]

_EXPORT_MODULES = {
    "RecordedDataCatalog": "mp_real.data.catalog",
    "LeRobotV21EpisodeRecorder": "mp_real.data.lerobot_v21",
    "LeRobotV21EpisodeSource": "mp_real.data.lerobot_v21",
    "ValidationReport": "mp_real.data.lerobot_v21",
    "validate_lerobot_v21_dataset": "mp_real.data.lerobot_v21",
    "DatasetMetadata": "mp_real.data.models",
    "EpisodeMetadata": "mp_real.data.models",
    "EpisodeRecordingContext": "mp_real.data.models",
    "EpisodeStatus": "mp_real.data.models",
    "FakeRecordedEpisodeSource": "mp_real.data.models",
    "RecordedEpisodeSource": "mp_real.data.models",
    "RecordedSample": "mp_real.data.models",
    "RecorderConfig": "mp_real.data.models",
    "DataViewSession": "mp_real.data.view",
    "EpisodeReader": "mp_real.data.view",
    "PlaybackCursor": "mp_real.data.view",
    "TimelineIndex": "mp_real.data.view",
    "ViewCursor": "mp_real.data.view",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
