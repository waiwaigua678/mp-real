"""LeRobot v2.1 recording, reading, validation, and catalog primitives.

This package deliberately has no dependency on a robot SDK, Web handlers, or
the OpenPI training stack.  It reads/writes the file format directly so an
mp-real controller can remain lightweight while external standard datasets
remain first-class inputs.
"""

from mp_real.data.catalog import RecordedDataCatalog
from mp_real.data.lerobot_v21 import (
    LeRobotV21EpisodeRecorder,
    LeRobotV21EpisodeSource,
    ValidationReport,
    validate_lerobot_v21_dataset,
)
from mp_real.data.models import (
    DatasetMetadata,
    EpisodeMetadata,
    EpisodeRecordingContext,
    EpisodeStatus,
    FakeRecordedEpisodeSource,
    RecordedEpisodeSource,
    RecordedSample,
    RecorderConfig,
)

__all__ = [
    "DatasetMetadata",
    "EpisodeMetadata",
    "EpisodeRecordingContext",
    "EpisodeStatus",
    "FakeRecordedEpisodeSource",
    "LeRobotV21EpisodeRecorder",
    "LeRobotV21EpisodeSource",
    "RecordedDataCatalog",
    "RecordedEpisodeSource",
    "RecordedSample",
    "RecorderConfig",
    "ValidationReport",
    "validate_lerobot_v21_dataset",
]
