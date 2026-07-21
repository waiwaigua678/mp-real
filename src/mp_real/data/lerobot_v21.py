from __future__ import annotations

import dataclasses
import json
import os
import queue
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.data.constants import CODEBASE_VERSION, MP_REAL_SCHEMA_VERSION
from mp_real.data.deps import OptionalModule, require_av, require_pyarrow
from mp_real.data.models import (
    DatasetMetadata,
    EpisodeMetadata,
    EpisodeRecordingContext,
    EpisodeStatus,
    RecordedEpisodeSource,
    RecordedSample,
    RecorderConfig,
    RecorderMetrics,
)
from mp_real.runtime.events import (
    ActionExecuted,
    ActionSelected,
    ActionStabilized,
    ChunkReceived,
    ControlStepRecorded,
    InferenceFinished,
    ObservationCaptured,
    RuntimeEvent,
    RuntimeEventSink,
)
from mp_real.runtime.models import ActionSpec, VectorField

RECORDER_VERSION = "h2-telemetry-parts-v3"
DEFAULT_CHUNK_SIZE = 1000
TIMESTAMP_TOLERANCE_S = 1e-4
_STOP = object()
STANDARD_ACTION_SOURCE = "executed_action"

av = OptionalModule("av", feature="LeRobot video recording/reading", extra="recording")
pa = OptionalModule("pyarrow", feature="LeRobot Parquet recording/reading", extra="recording")
pq = OptionalModule(
    "pyarrow.parquet",
    package="pyarrow",
    feature="LeRobot Parquet recording/reading",
    extra="recording",
)


class DatasetValidationError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    root: Path
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    episodes_checked: int


@dataclasses.dataclass
class _OnlineStats:
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    count: int = 0

    def update(self, value: np.ndarray, *, channel_stats: bool = False) -> None:
        array = np.asarray(value, dtype=np.float64)
        if channel_stats:
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image for image stats, got {array.shape}")
            array = np.moveaxis(array, -1, 0).reshape(3, -1) / 255.0
            minimum = array.min(axis=1).reshape(3, 1, 1)
            maximum = array.max(axis=1).reshape(3, 1, 1)
            total = array.sum(axis=1).reshape(3, 1, 1)
            total_sq = np.square(array).sum(axis=1).reshape(3, 1, 1)
            count = array.shape[1]
        else:
            array = array.reshape(-1)
            minimum = array.copy()
            maximum = array.copy()
            total = array.copy()
            total_sq = np.square(array)
            count = 1
        if self.minimum is None:
            self.minimum, self.maximum = minimum, maximum
            self.total, self.total_sq, self.count = total, total_sq, count
            return
        self.minimum = np.minimum(self.minimum, minimum)
        self.maximum = np.maximum(self.maximum, maximum)
        self.total = self.total + total
        self.total_sq = self.total_sq + total_sq
        self.count += count

    def to_json(self) -> dict[str, list]:
        if self.minimum is None or self.total is None or self.total_sq is None or self.maximum is None:
            raise RuntimeError("Cannot serialize empty statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [self.count],
        }


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)
        stream.write("\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, default=_json_default, separators=(",", ":"))
        stream.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _episode_chunk(episode_index: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    return episode_index // chunk_size


def _integer(value: object, default: int = 0) -> int:
    return default if value is None else int(value)


def _data_path(episode_index: int) -> Path:
    return Path("data") / f"chunk-{_episode_chunk(episode_index):03d}" / f"episode_{episode_index:06d}.parquet"


def _video_path(episode_index: int, role: str) -> Path:
    return (
        Path("videos")
        / f"chunk-{_episode_chunk(episode_index):03d}"
        / f"observation.images.{role}"
        / f"episode_{episode_index:06d}.mp4"
    )


def _telemetry_path(episode_index: int) -> Path:
    return Path("telemetry") / f"chunk-{_episode_chunk(episode_index):03d}" / f"episode_{episode_index:06d}.npz"


def _telemetry_dir_path(episode_index: int) -> Path:
    return Path("telemetry") / f"chunk-{_episode_chunk(episode_index):03d}" / f"episode_{episode_index:06d}"


def _telemetry_index_path(episode_index: int) -> Path:
    return _telemetry_dir_path(episode_index) / "index.json"


def _telemetry_part_name(part_index: int) -> str:
    return f"part_{part_index:06d}.npz"


def _telemetry_part_path(episode_index: int, part_index: int) -> Path:
    return _telemetry_dir_path(episode_index) / _telemetry_part_name(part_index)


def _field_names(fields: tuple[VectorField, ...], dimension: int, prefix: str) -> list[str]:
    return list(field.name for field in fields) if fields else [f"{prefix}_{index}" for index in range(dimension)]


def _feature(dtype: str, shape: list[int], names: object) -> dict[str, object]:
    return {"dtype": dtype, "shape": shape, "names": names}


def _build_features(
    config: RecorderConfig,
    camera_shapes: Mapping[str, tuple[int, int, int]],
    *,
    require_camera_shapes: bool = True,
) -> dict[str, dict[str, object]]:
    spec = config.action_spec
    features: dict[str, dict[str, object]] = {
        "observation.state": _feature(
            "float32",
            [spec.state_dim],
            [_field_names(spec.state_fields, spec.state_dim, "state")],
        ),
        "action": _feature(
            "float32",
            [spec.action_dim],
            [_field_names(spec.action_fields, spec.action_dim, "action")],
        ),
        "timestamp": _feature("float32", [1], None),
        "frame_index": _feature("int64", [1], None),
        "episode_index": _feature("int64", [1], None),
        "index": _feature("int64", [1], None),
        "task_index": _feature("int64", [1], None),
        "mp_real.selected_raw_action": _feature("float32", [spec.action_dim], None),
        "mp_real.stabilized_action": _feature("float32", [spec.action_dim], None),
        "mp_real.timestamp_monotonic_ns": _feature("int64", [1], None),
        "mp_real.inference_latency_ns": _feature("int64", [1], None),
        "mp_real.control_cycle_ns": _feature("int64", [1], None),
        "mp_real.camera_skew_ns": _feature("int64", [1], None),
        "mp_real.control_step_id": _feature("int64", [1], None),
        "mp_real.observation_id": _feature("int64", [1], None),
        "mp_real.policy_observation_id": _feature("int64", [1], None),
        "mp_real.policy_request_id": _feature("int64", [1], None),
        "mp_real.chunk_id": _feature("int64", [1], None),
        "mp_real.chunk_cursor": _feature("int64", [1], None),
        "mp_real.action_sent_timestamp_ns": _feature("int64", [1], None),
    }
    if config.save_video:
        for role in spec.camera_roles:
            if role not in camera_shapes:
                if require_camera_shapes:
                    raise DatasetValidationError(f"No image shape was captured for camera role {role!r}")
                continue
            height, width, channels = camera_shapes[role]
            features[f"observation.images.{role}"] = _feature(
                "video", [channels, height, width], ["channels", "height", "width"]
            )
    return features


def _build_info(
    config: RecorderConfig,
    camera_shapes: Mapping[str, tuple[int, int, int]],
    *,
    require_camera_shapes: bool = True,
    control_step_aligned: bool = False,
) -> dict[str, object]:
    semantics = (
        "control_step_observation_action"
        if control_step_aligned
        else "legacy_observation_action_provenance_unknown"
    )
    spec = config.action_spec
    replay_metadata = {
        "action_source": STANDARD_ACTION_SOURCE,
        "action_mode": spec.action_mode,
        "joint_unit": spec.joint_unit,
        "arm_count": spec.arm_count,
        "gripper_indices": list(spec.gripper_indices),
        "state_names": list(spec.state_field_names),
        "action_names": list(spec.action_field_names),
    }
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": config.robot_name,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": config.fps,
        "splits": {},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        if config.save_video
        else None,
        "features": _build_features(config, camera_shapes, require_camera_shapes=require_camera_shapes),
        "mp_real": {
            "schema_version": MP_REAL_SCHEMA_VERSION,
            "recorder_version": RECORDER_VERSION,
            "recording_semantics": semantics,
            "control_step_aligned": control_step_aligned,
            "policy_observation_reuse_possible": not control_step_aligned,
            "schema_status": "current" if control_step_aligned else "legacy_or_unknown",
            "action_spec": spec.to_dict(),
            "session_id": config.session_id,
            "operator": config.operator,
            "policy_label": config.policy_label,
            "runtime_config": dict(config.runtime_config),
            "telemetry": {
                "enabled": config.save_telemetry,
                "layout": "parts" if config.save_telemetry else "disabled",
                "part_size_steps": config.telemetry_part_size_steps if config.save_telemetry else 0,
            },
            # The standard LeRobot ``action`` field is populated only from
            # ActionExecuted events.  Naming that provenance explicitly is
            # required before a future real-robot command replay may consume
            # the dataset; raw model chunks remain telemetry only.
            "replay": replay_metadata,
        },
    }


def _recording_schema_payload(
    config: RecorderConfig,
    *,
    recording_semantics: str,
    control_step_aligned: bool,
) -> dict[str, object]:
    spec = config.action_spec
    return {
        "schema_version": MP_REAL_SCHEMA_VERSION,
        "recorder_version": RECORDER_VERSION,
        "recording_semantics": recording_semantics,
        "control_step_aligned": control_step_aligned,
        "policy_observation_reuse_possible": not control_step_aligned,
        "schema_status": "current" if control_step_aligned else "legacy_or_unknown",
        "action_source": STANDARD_ACTION_SOURCE,
        "action_mode": spec.action_mode,
        "joint_unit": spec.joint_unit,
        "arm_count": spec.arm_count,
        "gripper_indices": list(spec.gripper_indices),
        "robot_name": config.robot_name,
        "action_spec": spec.to_dict(),
        "state_field_order": _field_names(spec.state_fields, spec.state_dim, "state"),
        "action_field_order": _field_names(spec.action_fields, spec.action_dim, "action"),
        "camera_roles": list(spec.camera_roles),
        "max_camera_age_ns": config.max_camera_age_ns,
        "telemetry": {
            "enabled": config.save_telemetry,
            "layout": "parts" if config.save_telemetry else "disabled",
            "part_size_steps": config.telemetry_part_size_steps if config.save_telemetry else 0,
        },
    }


def _arrow_schema(features: Mapping[str, Mapping[str, object]]) -> pa.Schema:
    fields: list[pa.Field] = []
    for name, feature in features.items():
        if feature["dtype"] == "video":
            continue
        dtype = str(feature["dtype"])
        shape = list(feature["shape"])
        scalar = getattr(pa, dtype)()
        arrow_type = scalar if shape == [1] else pa.list_(scalar, list_size=int(shape[0]))
        fields.append(pa.field(name, arrow_type))
    return pa.schema(fields)


@dataclasses.dataclass(frozen=True)
class _QueuedEvent:
    event: RuntimeEvent
    image_bytes: int
    telemetry_bytes: int


class _BoundedCache:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[object, object] = OrderedDict()
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def get(self, key: object) -> object | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: object, value: object) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
            self.evictions += 1

    def pop(self, key: object, default: object | None = None) -> object | None:
        return self._items.pop(key, default)

    def discard_where(self, predicate: Callable[[object], bool]) -> None:
        for key in tuple(self._items):
            if predicate(key):
                self._items.pop(key, None)

    def clear(self) -> None:
        self._items.clear()


def _payload_array_bytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, np.generic):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_payload_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return sum(_payload_array_bytes(item) for item in value)
    return 0


def _event_image_bytes(event: RuntimeEvent) -> int:
    images = event.payload.get("images")
    return _payload_array_bytes(images) if isinstance(images, Mapping) else 0


def _event_telemetry_bytes(event: RuntimeEvent) -> int:
    return max(0, _payload_array_bytes(event.payload) - _event_image_bytes(event))


def _percentile(values: tuple[int, ...], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return int(ordered[index])


class _TelemetryPartWriter:
    """Incremental mp-real telemetry writer with bounded in-memory buffers."""

    _INDEX_VERSION = 1

    def __init__(self, root: Path, config: RecorderConfig, context: EpisodeRecordingContext) -> None:
        self.root = root
        self.config = config
        self.context = context
        self.directory = root / _telemetry_dir_path(context.episode_index)
        self._part_size_steps = config.telemetry_part_size_steps
        self._part_index = 0
        self._parts: list[dict[str, object]] = []
        self._next_frame_start = 0
        self._buffered_bytes = 0
        self._dropped_frame_count = 0
        self._dropped_event_count = 0
        self._camera_frame_ids: list[list[int]] = []
        self._camera_timestamps: list[list[int]] = []
        self._camera_reused: list[list[bool]] = []
        self._camera_ages: list[list[int]] = []
        self._control_step_ids: list[int] = []
        self._policy_observation_ids: list[int] = []
        self._policy_request_ids: list[int] = []
        self._control_chunk_ids: list[int] = []
        self._control_chunk_cursors: list[int] = []
        self._action_sent_timestamps: list[int] = []
        self._raw_chunks: list[np.ndarray] = []
        self._raw_chunk_request_ids: list[int] = []
        self._raw_chunk_ids: list[int] = []
        self._raw_chunk_cursors: list[int] = []
        self._raw_chunk_observation_ids: list[int] = []
        self._policy_generation_ids: list[int] = []
        self._safety_flags: list[str] = []

    @property
    def buffered_bytes(self) -> int:
        return self._buffered_bytes

    @property
    def part_count(self) -> int:
        return len(self._parts)

    def set_drop_counts(self, *, dropped_frame_count: int, dropped_event_count: int) -> None:
        self._dropped_frame_count = int(dropped_frame_count)
        self._dropped_event_count = int(dropped_event_count)

    def record_chunk(self, event: RuntimeEvent) -> bool:
        raw_chunk = event.payload.get("raw_action_chunk")
        if raw_chunk is None:
            return True
        array = np.asarray(raw_chunk, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.config.action_spec.action_dim:
            return False
        copied = array.copy()
        self._raw_chunks.append(copied)
        self._raw_chunk_request_ids.append(-1 if event.request_id is None else event.request_id)
        self._raw_chunk_ids.append(-1 if event.chunk_id is None else event.chunk_id)
        self._raw_chunk_cursors.append(_integer(event.payload.get("chunk_cursor"), -1))
        self._raw_chunk_observation_ids.append(_integer(event.payload.get("observation_id"), -1))
        self._buffered_bytes += int(copied.nbytes) + 32
        return True

    def record_frame(
        self,
        *,
        frame_index: int,
        event: RuntimeEvent,
        current_ids: list[int],
        current_timestamps: list[int],
        reused: list[bool],
        ages: list[int],
        control_step_id: int,
        policy_observation_id: int,
        policy_request_id: int,
        control_chunk_id: int,
        chunk_cursor: int,
        action_sent_timestamp_ns: int,
        safety_flags: str,
    ) -> None:
        if frame_index != self._next_frame_start + len(self._control_step_ids):
            self.flush()
            self._next_frame_start = frame_index
        self._camera_frame_ids.append(current_ids)
        self._camera_timestamps.append(current_timestamps)
        self._camera_reused.append(reused)
        self._camera_ages.append(ages)
        self._control_step_ids.append(control_step_id)
        self._policy_observation_ids.append(policy_observation_id)
        self._policy_request_ids.append(policy_request_id)
        self._control_chunk_ids.append(control_chunk_id)
        self._control_chunk_cursors.append(chunk_cursor)
        self._action_sent_timestamps.append(action_sent_timestamp_ns)
        self._policy_generation_ids.append(event.generation_id)
        self._safety_flags.append(safety_flags)
        camera_count = len(self.config.action_spec.camera_roles)
        self._buffered_bytes += 8 * (5 * camera_count + 8) + len(safety_flags.encode("utf-8"))
        if len(self._control_step_ids) >= self._part_size_steps:
            self.flush()

    def flush(self) -> None:
        frame_count = len(self._control_step_ids)
        if frame_count == 0 and not self._raw_chunks:
            self._write_index()
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        part_name = _telemetry_part_name(self._part_index)
        part_path = self.directory / part_name
        chunk_count = len(self._raw_chunks)
        max_chunk_length = max((len(chunk) for chunk in self._raw_chunks), default=0)
        action_dim = self.config.action_spec.action_dim
        raw_chunks = np.full((chunk_count, max_chunk_length, action_dim), np.nan, dtype=np.float32)
        for index, chunk in enumerate(self._raw_chunks):
            raw_chunks[index, : len(chunk)] = chunk
        temporary = part_path.with_name(part_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                start_control_step=np.asarray(self._control_step_ids[:1] or [-1], dtype=np.int64),
                end_control_step=np.asarray(self._control_step_ids[-1:] or [-1], dtype=np.int64),
                raw_action_chunk=raw_chunks,
                raw_action_chunk_length=np.asarray([len(chunk) for chunk in self._raw_chunks], dtype=np.int64),
                request_id=np.asarray(self._raw_chunk_request_ids, dtype=np.int64),
                chunk_id=np.asarray(self._raw_chunk_ids, dtype=np.int64),
                chunk_cursor=np.asarray(self._raw_chunk_cursors, dtype=np.int64),
                observation_id=np.asarray(self._raw_chunk_observation_ids, dtype=np.int64),
                camera_roles=np.asarray(self.config.action_spec.camera_roles),
                camera_frame_ids=np.asarray(self._camera_frame_ids, dtype=np.int64),
                camera_timestamps_ns=np.asarray(self._camera_timestamps, dtype=np.int64),
                camera_frame_reused=np.asarray(self._camera_reused, dtype=np.bool_),
                camera_age_ns=np.asarray(self._camera_ages, dtype=np.int64),
                control_step_id=np.asarray(self._control_step_ids, dtype=np.int64),
                policy_observation_id=np.asarray(self._policy_observation_ids, dtype=np.int64),
                policy_request_id=np.asarray(self._policy_request_ids, dtype=np.int64),
                control_chunk_id=np.asarray(self._control_chunk_ids, dtype=np.int64),
                control_chunk_cursor=np.asarray(self._control_chunk_cursors, dtype=np.int64),
                action_sent_timestamp_ns=np.asarray(self._action_sent_timestamps, dtype=np.int64),
                policy_generation_id=np.asarray(self._policy_generation_ids, dtype=np.int64),
                safety_flags=np.asarray(self._safety_flags, dtype=str),
                dropped_frame_count=np.asarray(self._dropped_frame_count, dtype=np.int64),
                dropped_event_count=np.asarray(self._dropped_event_count, dtype=np.int64),
            )
        os.replace(temporary, part_path)
        start_frame = self._next_frame_start
        end_frame = self._next_frame_start + frame_count
        self._parts.append(
            {
                "path": part_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "start_control_step": self._control_step_ids[0] if self._control_step_ids else None,
                "end_control_step": self._control_step_ids[-1] if self._control_step_ids else None,
                "raw_chunk_count": chunk_count,
                "raw_observation_ids": list(self._raw_chunk_observation_ids),
                "raw_chunk_ids": list(self._raw_chunk_ids),
            }
        )
        self._part_index += 1
        self._next_frame_start = end_frame
        self._clear_buffers()
        self._write_index()

    def close(self) -> None:
        self.flush()

    def _clear_buffers(self) -> None:
        self._camera_frame_ids.clear()
        self._camera_timestamps.clear()
        self._camera_reused.clear()
        self._camera_ages.clear()
        self._control_step_ids.clear()
        self._policy_observation_ids.clear()
        self._policy_request_ids.clear()
        self._control_chunk_ids.clear()
        self._control_chunk_cursors.clear()
        self._action_sent_timestamps.clear()
        self._raw_chunks.clear()
        self._raw_chunk_request_ids.clear()
        self._raw_chunk_ids.clear()
        self._raw_chunk_cursors.clear()
        self._raw_chunk_observation_ids.clear()
        self._policy_generation_ids.clear()
        self._safety_flags.clear()
        self._buffered_bytes = 0

    def _write_index(self) -> None:
        if not self.directory.exists():
            return
        _write_json(
            self.directory / "index.json",
            {
                "schema_version": MP_REAL_SCHEMA_VERSION,
                "index_version": self._INDEX_VERSION,
                "layout": "parts",
                "episode_index": self.context.episode_index,
                "camera_roles": list(self.config.action_spec.camera_roles),
                "action_dim": self.config.action_spec.action_dim,
                "part_size_steps": self._part_size_steps,
                "total_parts": len(self._parts),
                "total_frames": sum(int(part["frame_count"]) for part in self._parts),
                "dropped_frame_count": self._dropped_frame_count,
                "dropped_event_count": self._dropped_event_count,
                "parts": list(self._parts),
            },
        )


class _VideoWriter:
    def __init__(self, path: Path, *, fps: float, image_shape: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width, channels = image_shape
        if channels != 3:
            raise DatasetValidationError(f"Only RGB cameras are supported, got {channels} channels")
        self.path = path
        self._container = av.open(str(path), mode="w")
        frame_rate = Fraction(str(fps)).limit_denominator(1_000_000)
        try:
            self._stream = self._container.add_stream("libx264", rate=frame_rate)
        except av.error.FFmpegError:
            self._stream = self._container.add_stream("mpeg4", rate=frame_rate)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"

    def write(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> dict[str, object]:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            return {
                "video.height": stream.codec_context.height,
                "video.width": stream.codec_context.width,
                "video.codec": stream.codec_context.name,
                "video.pix_fmt": stream.codec_context.format.name if stream.codec_context.format else None,
                "video.is_depth_map": False,
                "video.fps": float(stream.average_rate) if stream.average_rate else None,
                "video.channels": 3,
                "has_audio": False,
            }


class _SequentialVideoReader:
    """One bounded, thread-safe decoder for sequential viewer frame reads.

    Episode playback normally asks for frame ``n`` followed by ``n + 1``.  A
    fresh PyAV container for every request turns that into repeated decoding
    from frame zero.  This reader holds only the last decoded RGB frame and a
    decoder cursor, so sequential reads are O(1) decoded frames while random
    backwards seeks deliberately restart from the beginning.  The latter is
    less common and keeps cache memory and seek assumptions bounded.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._container: Any | None = None
        self._frames: Iterator[Any] | None = None
        self._next_index = 0
        self._last_frame: np.ndarray | None = None
        self._last_index: int | None = None
        self._exhausted = False

    def frame_at(self, frame_index: int) -> tuple[np.ndarray | None, int | None]:
        if frame_index < 0:
            raise IndexError(frame_index)
        with self._lock:
            if self._last_index == frame_index and self._last_frame is not None:
                return self._last_frame.copy(), self._last_index
            if self._container is None or (self._last_index is not None and frame_index < self._last_index):
                self._restart_locked()
            while self._next_index <= frame_index and not self._exhausted:
                assert self._frames is not None
                try:
                    frame = next(self._frames)
                except StopIteration:
                    self._exhausted = True
                    break
                self._last_frame = frame.to_ndarray(format="rgb24")
                self._last_index = self._next_index
                self._next_index += 1
            if self._last_frame is None:
                return None, None
            return self._last_frame.copy(), self._last_index

    def close(self) -> None:
        with self._lock:
            if self._container is not None:
                self._container.close()
            self._container = None
            self._frames = None
            self._last_frame = None
            self._last_index = None
            self._next_index = 0
            self._exhausted = False

    def _restart_locked(self) -> None:
        self.close()
        self._container = av.open(str(self._path))
        self._frames = iter(self._container.decode(video=0))


class _EpisodeWriter:
    def __init__(self, root: Path, config: RecorderConfig, context: EpisodeRecordingContext, task_index: int) -> None:
        self.root = root
        self.config = config
        self.context = context
        self.task_index = task_index
        self.frame_count = 0
        self._global_start = 0
        self._camera_shapes: dict[str, tuple[int, int, int]] = {}
        self._parquet_writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._features: dict[str, dict[str, object]] | None = None
        self._video_writers: dict[str, _VideoWriter] = {}
        self._stats: dict[str, _OnlineStats] = defaultdict(_OnlineStats)
        self._last_camera_ids: dict[str, int] = {}
        self._control_step_frame_count = 0
        self._telemetry = _TelemetryPartWriter(root, config, context) if config.save_telemetry else None
        self._invalid_reason: str | None = None
        self._dropped_frame_count = 0
        self._dropped_event_count = 0
        self._closed = False

    @property
    def camera_shapes(self) -> Mapping[str, tuple[int, int, int]]:
        return self._camera_shapes

    @property
    def control_step_aligned(self) -> bool:
        return self.frame_count > 0 and self._control_step_frame_count == self.frame_count

    @property
    def telemetry_part_count(self) -> int:
        return 0 if self._telemetry is None else self._telemetry.part_count

    @property
    def buffered_telemetry_bytes(self) -> int:
        return 0 if self._telemetry is None else self._telemetry.buffered_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def set_global_start(self, value: int) -> None:
        self._global_start = value

    def record_chunk(self, event: RuntimeEvent) -> None:
        if self._telemetry is None:
            return
        if not self._telemetry.record_chunk(event):
            self._dropped_event_count += 1

    def record_frame(
        self,
        event: RuntimeEvent,
        observation: Mapping[str, object],
        selected: np.ndarray | None,
        stabilized: np.ndarray | None,
        inference_latency_ns: int,
    ) -> None:
        state = np.asarray(observation["state"], dtype=np.float32)
        action = np.asarray(event.payload["executed_action"], dtype=np.float32)
        if state.shape != (self.config.action_spec.state_dim,) or action.shape != (self.config.action_spec.action_dim,):
            self._dropped_frame_count += 1
            self._invalid_reason = "state_or_action_shape_mismatch"
            return
        images = observation.get("images", {})
        if not isinstance(images, Mapping):
            self._dropped_frame_count += 1
            self._invalid_reason = "observation_images_missing"
            return
        converted_images: dict[str, np.ndarray] = {}
        for role in self.config.action_spec.camera_roles:
            image = images.get(role)
            if image is None:
                self._dropped_frame_count += 1
                self._invalid_reason = f"missing_camera:{role}"
                return
            converted = np.asarray(image, dtype=np.uint8)
            if converted.ndim != 3 or converted.shape[-1] != 3:
                self._dropped_frame_count += 1
                self._invalid_reason = f"invalid_camera_shape:{role}:{converted.shape}"
                return
            shape = tuple(int(value) for value in converted.shape)
            prior_shape = self._camera_shapes.setdefault(role, shape)
            if prior_shape != shape:
                self._dropped_frame_count += 1
                self._invalid_reason = f"camera_shape_changed:{role}"
                return
            converted_images[role] = converted.copy()

        if self._parquet_writer is None:
            self._features = _build_features(self.config, self._camera_shapes)
            self._schema = _arrow_schema(self._features)
            parquet_path = self.root / _data_path(self.context.episode_index)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            self._parquet_writer = pq.ParquetWriter(parquet_path, self._schema, compression="zstd")
            if self.config.save_video:
                for role, image in converted_images.items():
                    self._video_writers[role] = _VideoWriter(
                        self.root / _video_path(self.context.episode_index, role),
                        fps=self.config.fps,
                        image_shape=tuple(int(value) for value in image.shape),
                    )

        frame_index = self.frame_count
        camera_ids = observation.get("camera_frame_ids", {})
        camera_timestamps = observation.get("camera_timestamps_ns", {})
        if not isinstance(camera_ids, Mapping) or not isinstance(camera_timestamps, Mapping):
            self._dropped_frame_count += 1
            self._invalid_reason = "camera_telemetry_missing"
            return
        current_ids = [int(camera_ids.get(role, -1)) for role in self.config.action_spec.camera_roles]
        current_timestamps = [
            int(camera_timestamps.get(role, 0)) for role in self.config.action_spec.camera_roles
        ]
        reused = [
            self._last_camera_ids.get(role) == frame_id
            for role, frame_id in zip(self.config.action_spec.camera_roles, current_ids)
        ]
        camera_age_payload = event.payload.get("camera_age_ns", {})
        if isinstance(camera_age_payload, Mapping):
            ages = [
                max(0, int(camera_age_payload.get(role, event.monotonic_timestamp_ns - timestamp)))
                for role, timestamp in zip(self.config.action_spec.camera_roles, current_timestamps)
            ]
        else:
            ages = [max(0, event.monotonic_timestamp_ns - timestamp) for timestamp in current_timestamps]
        if any(age > self.config.max_camera_age_ns for age in ages):
            self._invalid_reason = "camera_age_exceeded"

        selected_payload = event.payload.get("selected_raw_action", action)
        stabilized_payload = event.payload.get("stabilized_target_action", action)
        selected = selected_payload if selected is None else selected
        stabilized = stabilized_payload if stabilized is None else stabilized
        selected = np.asarray(selected, dtype=np.float32)
        stabilized = np.asarray(stabilized, dtype=np.float32)
        if selected.shape != action.shape or stabilized.shape != action.shape:
            self._dropped_frame_count += 1
            self._invalid_reason = "action_telemetry_shape_mismatch"
            return
        control_step_id = _integer(event.payload.get("control_step_id"), -1)
        policy_observation_id = _integer(event.payload.get("policy_observation_id"), -1)
        policy_request_id = _integer(event.payload.get("policy_request_id", event.request_id), -1)
        control_chunk_id = _integer(event.payload.get("chunk_id", event.chunk_id), -1)
        chunk_cursor = _integer(event.payload.get("chunk_cursor"), -1)
        action_sent_timestamp_ns = _integer(event.payload.get("action_sent_timestamp_ns"), 0)
        row = {
            "observation.state": [state.tolist()],
            "action": [action.tolist()],
            "timestamp": [np.float32(frame_index / self.config.fps)],
            "frame_index": [frame_index],
            "episode_index": [self.context.episode_index],
            "index": [self._global_start + frame_index],
            "task_index": [self.task_index],
            "mp_real.selected_raw_action": [selected.tolist()],
            "mp_real.stabilized_action": [stabilized.tolist()],
            "mp_real.timestamp_monotonic_ns": [event.monotonic_timestamp_ns],
            "mp_real.inference_latency_ns": [inference_latency_ns],
            "mp_real.control_cycle_ns": [_integer(event.payload.get("control_cycle_ns"))],
            "mp_real.camera_skew_ns": [_integer(observation.get("max_camera_skew_ns"))],
            "mp_real.control_step_id": [control_step_id],
            "mp_real.observation_id": [_integer(observation.get("observation_id"), -1)],
            "mp_real.policy_observation_id": [policy_observation_id],
            "mp_real.policy_request_id": [policy_request_id],
            "mp_real.chunk_id": [control_chunk_id],
            "mp_real.chunk_cursor": [chunk_cursor],
            "mp_real.action_sent_timestamp_ns": [action_sent_timestamp_ns],
        }
        assert self._schema is not None and self._parquet_writer is not None
        table = pa.Table.from_pydict(row, schema=self._schema)
        self._parquet_writer.write_table(table)
        for role, writer in self._video_writers.items():
            writer.write(converted_images[role])

        self._stats["observation.state"].update(state)
        self._stats["action"].update(action)
        for name, value in row.items():
            if name not in {"observation.state", "action"}:
                self._stats[name].update(np.asarray(value[0]))
        for role, image in converted_images.items():
            self._stats[f"observation.images.{role}"].update(image, channel_stats=True)
            self._last_camera_ids[role] = int(camera_ids[role])
        if isinstance(event, ControlStepRecorded):
            self._control_step_frame_count += 1
        safety_flags = event.payload.get("safety_flags", ())
        encoded_safety_flags = json.dumps(safety_flags, ensure_ascii=False, sort_keys=True, default=_json_default)
        if self._telemetry is not None:
            self._telemetry.record_frame(
                frame_index=frame_index,
                event=event,
                current_ids=current_ids,
                current_timestamps=current_timestamps,
                reused=reused,
                ages=ages,
                control_step_id=control_step_id,
                policy_observation_id=policy_observation_id,
                policy_request_id=policy_request_id,
                control_chunk_id=control_chunk_id,
                chunk_cursor=chunk_cursor,
                action_sent_timestamp_ns=action_sent_timestamp_ns,
                safety_flags=encoded_safety_flags,
            )
        self.frame_count += 1

    def note_drop(self, *, frame: bool = False) -> None:
        self._dropped_event_count += 1
        if frame:
            self._dropped_frame_count += 1

    def flush(self) -> None:
        if self._telemetry is not None:
            self._telemetry.set_drop_counts(
                dropped_frame_count=self._dropped_frame_count,
                dropped_event_count=self._dropped_event_count,
            )
            self._telemetry.flush()

    def abort(self) -> tuple[dict[str, object], dict[str, object], str | None]:
        video_info = self._close_resources(finalize_telemetry=True)
        return ({name: stats.to_json() for name, stats in self._stats.items()}, video_info, self._invalid_reason)

    def close(self) -> tuple[dict[str, object], dict[str, object], str | None]:
        video_info = self._close_resources(finalize_telemetry=True)
        return ({name: stats.to_json() for name, stats in self._stats.items()}, video_info, self._invalid_reason)

    def _close_resources(self, *, finalize_telemetry: bool) -> dict[str, object]:
        if self._closed:
            return {}
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        video_info: dict[str, object] = {}
        for role, writer in tuple(self._video_writers.items()):
            video_info[role] = writer.close()
        self._video_writers.clear()
        if finalize_telemetry and self._telemetry is not None:
            self._telemetry.set_drop_counts(
                dropped_frame_count=self._dropped_frame_count,
                dropped_event_count=self._dropped_event_count,
            )
            self._telemetry.close()
        self._closed = True
        return video_info


class LeRobotV21EpisodeRecorder(RuntimeEventSink):
    """Bounded, asynchronous LeRobot v2.1 episode writer.

    ``emit`` only performs a non-blocking queue operation.  Parquet writes,
    MP4 encoding, JSONL updates, and finalization all run on one explicit
    non-daemon worker whose exception is exposed by ``join``/``raise_if_failed``.
    """

    requires_observation_images = True

    def __init__(self, config: RecorderConfig, *, on_failure: Callable[[BaseException], None] | None = None) -> None:
        self.config = config
        self._on_failure = on_failure
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.queue_size)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._stop_event = threading.Event()
        self._failure: BaseException | None = None
        self._active: EpisodeRecordingContext | None = None
        self._dropped_event_count = 0
        self._dropped_frame_count = 0
        self._queue_high_watermark = 0
        self._cache_entry_count = 0
        self._cache_high_watermark = 0
        self._cache_eviction_count = 0
        self._buffered_image_bytes = 0
        self._buffered_telemetry_bytes = 0
        self._writer_buffered_telemetry_bytes = 0
        self._written_frame_count = 0
        self._telemetry_part_count = 0
        self._active_writer_count = 0
        self._writer_latency_ns: deque[int] = deque(maxlen=2048)
        self._final_root = config.dataset_root.resolve()
        self._work_root = self._final_root.with_name(self._final_root.name + ".inprogress")

    @property
    def dropped_event_count(self) -> int:
        with self._lock:
            return self._dropped_event_count

    @property
    def dropped_frame_count(self) -> int:
        with self._lock:
            return self._dropped_frame_count

    @property
    def queue_high_watermark(self) -> int:
        with self._lock:
            return self._queue_high_watermark

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self.config.queue_size

    def metrics(self) -> RecorderMetrics:
        with self._lock:
            latencies = tuple(self._writer_latency_ns)
            buffered_image_bytes = self._buffered_image_bytes
            buffered_telemetry_bytes = self._buffered_telemetry_bytes + self._writer_buffered_telemetry_bytes
            return RecorderMetrics(
                queue_size=self._queue.qsize(),
                queue_capacity=self.config.queue_size,
                queue_high_watermark=self._queue_high_watermark,
                cache_entry_count=self._cache_entry_count,
                cache_high_watermark=self._cache_high_watermark,
                cache_eviction_count=self._cache_eviction_count,
                buffered_image_bytes=buffered_image_bytes,
                buffered_telemetry_bytes=buffered_telemetry_bytes,
                written_frame_count=self._written_frame_count,
                dropped_frame_count=self._dropped_frame_count,
                dropped_event_count=self._dropped_event_count,
                writer_latency_p50_ns=_percentile(latencies, 0.50),
                writer_latency_p95_ns=_percentile(latencies, 0.95),
                current_memory_estimate_bytes=buffered_image_bytes + buffered_telemetry_bytes,
                telemetry_part_count=self._telemetry_part_count,
                active_writer_count=self._active_writer_count,
            )

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    @property
    def work_root(self) -> Path:
        return self._work_root

    @property
    def final_root(self) -> Path:
        return self._final_root

    def start(self) -> None:
        require_pyarrow("LeRobot v2.1 recording")
        if self.config.save_video:
            require_av("LeRobot v2.1 video recording")
        with self._lock:
            if self._started:
                return
            if self._final_root.exists() or self._work_root.exists():
                raise FileExistsError(f"Recording dataset already exists: {self._final_root}")
            self._started = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"recording-{self.config.dataset_name}",
                daemon=False,
            )
            self._thread.start()
        self._put(("start",), block=True)

    def begin_episode(self, context: EpisodeRecordingContext) -> None:
        with self._lock:
            if not self._started or self._stopping:
                raise RuntimeError("Recorder is not accepting episodes")
            if self._active is not None:
                raise RuntimeError("An episode is already active")
            self._active = context
        self._put(("begin", context), block=True)

    def end_episode(self, *, labels: Mapping[str, object] | None = None) -> None:
        with self._lock:
            context = self._active
            self._active = None
        if context is not None:
            self._put(("end", context, dict(labels or {})), block=True)

    def set_episode_label(self, episode_index: int, label: Mapping[str, object]) -> None:
        self._put(("label", episode_index, dict(label)), block=True)

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            active = self._active
        if active is None or event.episode_id != active.episode_id:
            return
        queued = _QueuedEvent(event, _event_image_bytes(event), _event_telemetry_bytes(event))
        if self._put(("event", queued), block=False):
            with self._lock:
                self._buffered_image_bytes += queued.image_bytes
                self._buffered_telemetry_bytes += queued.telemetry_bytes
            return
        abort_error: RuntimeError | None = None
        with self._lock:
            self._dropped_event_count += 1
            if isinstance(event, (ObservationCaptured, ActionExecuted, ControlStepRecorded)):
                self._dropped_frame_count += 1
            if self.config.drop_policy == "abort" and self._failure is None:
                abort_error = RuntimeError("Recorder queue is full")
                self._failure = abort_error
        if abort_error is not None and self._on_failure is not None:
            try:
                self._on_failure(abort_error)
            except BaseException:
                pass

    def flush(self, *, timeout: float | None = 5.0) -> bool:
        marker = threading.Event()
        self._put(("flush", marker), block=True)
        return marker.wait(timeout)

    def stop(self, *, finalize: bool = True, timeout: float | None = 5.0) -> bool:
        with self._lock:
            if not self._started or self._stopping:
                return self.join(timeout=timeout, raise_on_error=False)
            self._stopping = True
            self._stop_event.set()
        self._put(("stop", finalize), block=True)
        return self.join(timeout=timeout, raise_on_error=False)

    def join(self, *, timeout: float | None = 5.0, raise_on_error: bool = True) -> bool:
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        complete = thread is None or not thread.is_alive()
        if complete and raise_on_error:
            self.raise_if_failed()
        return complete

    def raise_if_failed(self) -> None:
        failure = self.failure
        if failure is not None:
            raise failure

    def _put(self, item: object, *, block: bool) -> bool:
        try:
            if block:
                self._queue.put(item, timeout=5.0)
            else:
                self._queue.put_nowait(item)
            with self._lock:
                self._queue_high_watermark = max(self._queue_high_watermark, self._queue.qsize())
            return True
        except queue.Full:
            return False

    def _release_queued_event(self, queued: _QueuedEvent) -> None:
        with self._lock:
            self._buffered_image_bytes = max(0, self._buffered_image_bytes - queued.image_bytes)
            self._buffered_telemetry_bytes = max(0, self._buffered_telemetry_bytes - queued.telemetry_bytes)

    def _note_writer_latency(self, started_ns: int) -> None:
        with self._lock:
            self._writer_latency_ns.append(max(0, time.monotonic_ns() - started_ns))

    def _set_cache_metrics(self, caches: tuple[_BoundedCache, ...]) -> None:
        entry_count = sum(len(cache) for cache in caches)
        eviction_count = sum(cache.evictions for cache in caches)
        with self._lock:
            self._cache_entry_count = entry_count
            self._cache_high_watermark = max(self._cache_high_watermark, entry_count)
            self._cache_eviction_count = eviction_count

    def _set_writer_metrics(self, episodes: Mapping[int, _EpisodeWriter]) -> None:
        with self._lock:
            self._active_writer_count = sum(not writer.closed for writer in episodes.values())
            self._written_frame_count = sum(writer.frame_count for writer in episodes.values())
            self._writer_buffered_telemetry_bytes = sum(
                writer.buffered_telemetry_bytes for writer in episodes.values()
            )
            self._telemetry_part_count = sum(writer.telemetry_part_count for writer in episodes.values())

    def _run(self) -> None:
        episodes: dict[int, _EpisodeWriter] = {}
        contexts: dict[int, EpisodeRecordingContext] = {}
        episode_stats: dict[int, dict[str, object]] = {}
        video_info: dict[str, object] = {}
        labels: dict[int, dict[str, object]] = {}
        persisted_labels: set[int] = set()
        observations = _BoundedCache(self.config.max_observation_cache_entries)
        selected_actions = _BoundedCache(self.config.max_incomplete_event_entries)
        stabilized_actions = _BoundedCache(self.config.max_incomplete_event_entries)
        control_step_recorded_steps = _BoundedCache(self.config.max_incomplete_event_entries)
        inference_latency_ns = _BoundedCache(self.config.max_inference_latency_entries)
        caches = (
            observations,
            selected_actions,
            stabilized_actions,
            control_step_recorded_steps,
            inference_latency_ns,
        )
        global_index = 0
        tasks: dict[str, int] = {}
        try:
            while True:
                item = self._queue.get()
                try:
                    event_started_ns = time.monotonic_ns()
                    queued_event: _QueuedEvent | None = None
                    event: RuntimeEvent | None = None
                    kind = item[0]  # type: ignore[index]
                    if kind == "start":
                        self._work_root.mkdir(parents=True, exist_ok=False)
                        _write_json(
                            self._work_root / "meta" / "info.json",
                            _build_info(self.config, {}, require_camera_shapes=False),
                        )
                        _write_json(
                            self._work_root / "meta" / "mp_real" / "schema.json",
                            _recording_schema_payload(
                                self.config,
                                recording_semantics="legacy_observation_action_provenance_unknown",
                                control_step_aligned=False,
                            ),
                        )
                        _append_jsonl(
                            self._work_root / "meta" / "mp_real" / "sessions.jsonl",
                            {
                                "session_id": self.config.session_id,
                                "dataset_name": self.config.dataset_name,
                                "robot_name": self.config.robot_name,
                                "operator": self.config.operator,
                                "policy_label": self.config.policy_label,
                                "created_at_monotonic_ns": time.monotonic_ns(),
                                "runtime_config": dict(self.config.runtime_config),
                            },
                        )
                    elif kind == "begin":
                        context = item[1]  # type: ignore[index]
                        assert isinstance(context, EpisodeRecordingContext)
                        if context.episode_index in episodes:
                            raise ValueError(f"Duplicate episode index: {context.episode_index}")
                        task_index = tasks.setdefault(context.task, len(tasks))
                        writer = _EpisodeWriter(self._work_root, self.config, context, task_index)
                        writer.set_global_start(global_index)
                        episodes[context.episode_index] = writer
                        contexts[context.episode_index] = context
                    elif kind == "event":
                        queued_event = item[1]  # type: ignore[index]
                        assert isinstance(queued_event, _QueuedEvent)
                        event = queued_event.event
                        event_episode = next(
                            (context for context in contexts.values() if context.episode_id == event.episode_id), None
                        )
                        if event_episode is None:
                            continue
                        writer = episodes[event_episode.episode_index]
                        _append_jsonl(
                            self._work_root
                            / "meta"
                            / "mp_real"
                            / "events"
                            / f"episode_{event_episode.episode_index:06d}.jsonl",
                            {
                                "event_id": event.event_id,
                                "event_type": event.event_type,
                                "session_id": event.session_id,
                                "episode_id": event.episode_id,
                                "generation_id": event.generation_id,
                                "request_id": event.request_id,
                                "chunk_id": event.chunk_id,
                                "step": event.step,
                                "monotonic_timestamp_ns": event.monotonic_timestamp_ns,
                                "wall_timestamp_iso": event.wall_timestamp_iso,
                                "payload": event.payload,
                            },
                        )
                        if isinstance(event, ControlStepRecorded):
                            step_key = (event_episode.episode_index, int(event.step or -1))
                            control_step_recorded_steps.put(step_key, True)
                            selected_actions.pop(step_key, None)
                            stabilized_actions.pop(step_key, None)
                            policy_observation_id = _integer(event.payload.get("policy_observation_id"), -1)
                            control_observation_id = _integer(event.payload.get("observation_id"), -1)
                            writer.record_frame(
                                event,
                                event.payload,
                                None,
                                None,
                                int(inference_latency_ns.pop(policy_observation_id, 0) or 0),
                            )
                            observations.pop(policy_observation_id, None)
                            observations.pop(control_observation_id, None)
                        elif isinstance(event, ObservationCaptured):
                            observation_id = _integer(event.payload.get("observation_id"), -1)
                            if observation_id >= 0:
                                observations.put(observation_id, event.payload)
                        elif isinstance(event, InferenceFinished):
                            observation_id = _integer(event.payload.get("observation_id"), -1)
                            if observation_id >= 0:
                                inference_latency_ns.put(
                                    observation_id,
                                    _integer(event.payload.get("inference_latency_ns")),
                                )
                        elif isinstance(event, ChunkReceived):
                            writer.record_chunk(event)
                        elif isinstance(event, ActionSelected):
                            selected_actions.put(
                                (event_episode.episode_index, int(event.step or -1)),
                                np.asarray(event.payload.get("selected_raw_action"), dtype=np.float32),
                            )
                        elif isinstance(event, ActionStabilized):
                            stabilized_actions.put(
                                (event_episode.episode_index, int(event.step or -1)),
                                np.asarray(event.payload.get("stabilized_target_action"), dtype=np.float32),
                            )
                        elif isinstance(event, ActionExecuted):
                            step_key = (event_episode.episode_index, int(event.step or -1))
                            if step_key in control_step_recorded_steps:
                                selected_actions.pop(step_key, None)
                                stabilized_actions.pop(step_key, None)
                                continue
                            observation_id = _integer(event.payload.get("observation_id"), -1)
                            observation = observations.pop(observation_id, None)
                            if observation is None:
                                writer.note_drop(frame=True)
                            else:
                                assert isinstance(observation, Mapping)
                                writer.record_frame(
                                    event,
                                    observation,
                                    selected_actions.pop(step_key, None),  # type: ignore[arg-type]
                                    stabilized_actions.pop(step_key, None),  # type: ignore[arg-type]
                                    int(inference_latency_ns.pop(observation_id, 0) or 0),
                                )
                                observation = None
                    elif kind == "end":
                        context, label = item[1], item[2]  # type: ignore[index]
                        assert isinstance(context, EpisodeRecordingContext)
                        writer = episodes[context.episode_index]
                        stats, episode_video_info, invalid_reason = writer.close()
                        episode_stats[context.episode_index] = stats
                        video_info.update(episode_video_info)
                        global_index += writer.frame_count
                        labels[context.episode_index] = dict(label)
                        if invalid_reason:
                            labels[context.episode_index].setdefault("result", "INVALID")
                            labels[context.episode_index].setdefault("failure_reason", invalid_reason)
                        observations.clear()
                        inference_latency_ns.clear()
                        selected_actions.discard_where(
                            lambda key, episode_index=context.episode_index: (
                                isinstance(key, tuple) and key[:1] == (episode_index,)
                            )
                        )
                        stabilized_actions.discard_where(
                            lambda key, episode_index=context.episode_index: (
                                isinstance(key, tuple) and key[:1] == (episode_index,)
                            )
                        )
                        control_step_recorded_steps.discard_where(
                            lambda key, episode_index=context.episode_index: (
                                isinstance(key, tuple) and key[:1] == (episode_index,)
                            )
                        )
                    elif kind == "label":
                        episode_index = int(item[1])  # type: ignore[index]
                        context = contexts.get(episode_index)
                        if context is None:
                            raise ValueError(f"Cannot label unknown episode index: {episode_index}")
                        labels[episode_index] = dict(item[2])  # type: ignore[index]
                        _append_jsonl(
                            self._work_root / "meta" / "mp_real" / "episode_labels.jsonl",
                            {
                                "session_id": context.session_id or self.config.session_id,
                                "episode_index": episode_index,
                                "source_episode_id": context.episode_id,
                                "created_at": time.time(),
                                **labels[episode_index],
                            },
                        )
                        persisted_labels.add(episode_index)
                    elif kind == "flush":
                        for writer in episodes.values():
                            writer.flush()
                        item[1].set()  # type: ignore[index]
                    elif kind == "stop":
                        if item[1]:
                            self._finalize(
                                episodes,
                                contexts,
                                episode_stats,
                                video_info,
                                labels,
                                persisted_labels,
                                tasks,
                            )
                        else:
                            for episode_index, writer in episodes.items():
                                stats, episode_video_info, invalid_reason = writer.abort()
                                if writer.frame_count > 0 and episode_index not in episode_stats:
                                    episode_stats[episode_index] = stats
                                video_info.update(episode_video_info)
                                if invalid_reason:
                                    labels.setdefault(episode_index, {})
                                    labels[episode_index].setdefault("result", "INVALID")
                                    labels[episode_index].setdefault("failure_reason", invalid_reason)
                            self._write_incomplete_dataset(
                                episodes,
                                contexts,
                                episode_stats,
                                video_info,
                                labels,
                                persisted_labels,
                                tasks,
                            )
                            self._write_recovery("Recorder stopped without finalization")
                        return
                    else:
                        raise RuntimeError(f"Unknown recorder command: {kind!r}")
                finally:
                    if queued_event is not None:
                        self._release_queued_event(queued_event)
                        self._note_writer_latency(event_started_ns)
                    event = None
                    queued_event = None
                    self._set_cache_metrics(caches)
                    self._set_writer_metrics(episodes)
                    self._queue.task_done()
        except BaseException as exc:
            for writer in episodes.values():
                try:
                    writer.abort()
                except BaseException:
                    pass
            self._write_recovery(f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._failure = exc
            if self._on_failure is not None:
                try:
                    self._on_failure(exc)
                except BaseException:
                    pass

    def _write_recovery(self, message: str) -> None:
        try:
            if self._work_root.exists():
                _write_json(
                    self._work_root / "meta" / "mp_real" / "recovery.json",
                    {
                        "status": "INCOMPLETE",
                        "message": message,
                        "timestamp_monotonic_ns": time.monotonic_ns(),
                        "dropped_event_count": self.dropped_event_count,
                        "dropped_frame_count": self.dropped_frame_count,
                        "queue_high_watermark": self.queue_high_watermark,
                    },
                )
        except BaseException:
            return

    def _write_incomplete_dataset(
        self,
        episodes: Mapping[int, _EpisodeWriter],
        contexts: Mapping[int, EpisodeRecordingContext],
        episode_stats: Mapping[int, Mapping[str, object]],
        video_info: Mapping[str, object],
        labels: Mapping[int, Mapping[str, object]],
        persisted_labels: set[int],
        tasks: Mapping[str, int],
    ) -> None:
        non_empty = {index: writer for index, writer in episodes.items() if writer.frame_count > 0}
        if not non_empty:
            return
        camera_shapes: dict[str, tuple[int, int, int]] = {}
        for writer in non_empty.values():
            camera_shapes.update(writer.camera_shapes)
        control_step_aligned = all(writer.control_step_aligned for writer in non_empty.values())
        info = _build_info(
            self.config,
            camera_shapes,
            require_camera_shapes=False,
            control_step_aligned=control_step_aligned,
        )
        info["total_episodes"] = len(non_empty)
        info["total_frames"] = sum(writer.frame_count for writer in non_empty.values())
        info["total_tasks"] = len(tasks)
        info["total_videos"] = (
            len(non_empty) * len(self.config.action_spec.camera_roles) if self.config.save_video else 0
        )
        info["total_chunks"] = len({_episode_chunk(index) for index in non_empty})
        info["splits"] = {"train": f"0:{len(non_empty)}"}
        if self.config.save_video:
            for role, metadata in video_info.items():
                if f"observation.images.{role}" in info["features"]:
                    info["features"][f"observation.images.{role}"]["info"] = metadata  # type: ignore[index]
        _write_json(self._work_root / "meta" / "info.json", info)
        _write_json(
            self._work_root / "meta" / "mp_real" / "schema.json",
            _recording_schema_payload(
                self.config,
                recording_semantics=str(info["mp_real"]["recording_semantics"]),  # type: ignore[index]
                control_step_aligned=control_step_aligned,
            ),
        )
        for task, task_index in sorted(tasks.items(), key=lambda item: item[1]):
            _append_jsonl(self._work_root / "meta" / "tasks.jsonl", {"task_index": task_index, "task": task})
        for episode_index, writer in sorted(non_empty.items()):
            context = contexts[episode_index]
            _append_jsonl(
                self._work_root / "meta" / "episodes.jsonl",
                {"episode_index": episode_index, "tasks": [context.task], "length": writer.frame_count},
            )
            _append_jsonl(
                self._work_root / "meta" / "episodes_stats.jsonl",
                {"episode_index": episode_index, "stats": episode_stats.get(episode_index, {})},
            )
            if episode_index not in persisted_labels:
                _append_jsonl(
                    self._work_root / "meta" / "mp_real" / "episode_labels.jsonl",
                    {
                        "session_id": context.session_id or self.config.session_id,
                        "episode_index": episode_index,
                        "source_episode_id": context.episode_id,
                        "created_at": time.time(),
                        **dict(labels.get(episode_index, {})),
                    },
                )
        _write_json(self._work_root / "meta" / "stats.json", _aggregate_stats(episode_stats.values()))

    def _finalize(
        self,
        episodes: Mapping[int, _EpisodeWriter],
        contexts: Mapping[int, EpisodeRecordingContext],
        episode_stats: Mapping[int, Mapping[str, object]],
        video_info: Mapping[str, object],
        labels: Mapping[int, Mapping[str, object]],
        persisted_labels: set[int],
        tasks: Mapping[str, int],
    ) -> None:
        if any(writer.frame_count == 0 for writer in episodes.values()):
            raise DatasetValidationError("Cannot finalize a dataset with an empty episode")
        camera_shapes: dict[str, tuple[int, int, int]] = {}
        for writer in episodes.values():
            camera_shapes.update(writer.camera_shapes)
        control_step_aligned = all(writer.control_step_aligned for writer in episodes.values())
        info = _build_info(self.config, camera_shapes, control_step_aligned=control_step_aligned)
        info["total_episodes"] = len(episodes)
        info["total_frames"] = sum(writer.frame_count for writer in episodes.values())
        info["total_tasks"] = len(tasks)
        info["total_videos"] = (
            len(episodes) * len(self.config.action_spec.camera_roles) if self.config.save_video else 0
        )
        info["total_chunks"] = len({_episode_chunk(index) for index in episodes})
        info["splits"] = {"train": f"0:{len(episodes)}"}
        if self.config.save_video:
            for role, metadata in video_info.items():
                info["features"][f"observation.images.{role}"]["info"] = metadata  # type: ignore[index]
        _write_json(self._work_root / "meta" / "info.json", info)
        _write_json(
            self._work_root / "meta" / "mp_real" / "schema.json",
            _recording_schema_payload(
                self.config,
                recording_semantics=str(info["mp_real"]["recording_semantics"]),  # type: ignore[index]
                control_step_aligned=control_step_aligned,
            ),
        )
        for task, task_index in sorted(tasks.items(), key=lambda item: item[1]):
            _append_jsonl(self._work_root / "meta" / "tasks.jsonl", {"task_index": task_index, "task": task})
        for episode_index, writer in sorted(episodes.items()):
            context = contexts[episode_index]
            _append_jsonl(
                self._work_root / "meta" / "episodes.jsonl",
                {"episode_index": episode_index, "tasks": [context.task], "length": writer.frame_count},
            )
            _append_jsonl(
                self._work_root / "meta" / "episodes_stats.jsonl",
                {"episode_index": episode_index, "stats": episode_stats[episode_index]},
            )
            if episode_index not in persisted_labels:
                label = {
                    "session_id": context.session_id or self.config.session_id,
                    "episode_index": episode_index,
                    "source_episode_id": context.episode_id,
                    "created_at": time.time(),
                    **dict(labels.get(episode_index, {})),
                }
                _append_jsonl(self._work_root / "meta" / "mp_real" / "episode_labels.jsonl", label)
        _write_json(self._work_root / "meta" / "stats.json", _aggregate_stats(episode_stats.values()))
        _write_json(
            self._work_root / "meta" / "mp_real" / "recording_summary.json",
            {
                "dropped_event_count": self.dropped_event_count,
                "dropped_frame_count": self.dropped_frame_count,
                "queue_high_watermark": self.queue_high_watermark,
                "queue_capacity": self.queue_capacity,
                "metrics": dataclasses.asdict(self.metrics()),
            },
        )
        self._write_recovery("finalization complete")
        recovery = self._work_root / "meta" / "mp_real" / "recovery.json"
        recovery.unlink(missing_ok=True)
        os.replace(self._work_root, self._final_root)


def _aggregate_stats(values: Iterator[Mapping[str, object]]) -> dict[str, object]:
    stats_by_feature: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for episode_stats in values:
        for feature, stats in episode_stats.items():
            stats_by_feature[feature].append(stats)  # type: ignore[arg-type]
    result: dict[str, object] = {}
    for feature, stats_items in stats_by_feature.items():
        counts = np.asarray([item["count"][0] for item in stats_items], dtype=np.float64)
        means = [np.asarray(item["mean"], dtype=np.float64) for item in stats_items]
        stds = [np.asarray(item["std"], dtype=np.float64) for item in stats_items]
        minimum = np.minimum.reduce([np.asarray(item["min"], dtype=np.float64) for item in stats_items])
        maximum = np.maximum.reduce([np.asarray(item["max"], dtype=np.float64) for item in stats_items])
        total_count = counts.sum()
        mean = sum(mean * count for mean, count in zip(means, counts)) / total_count
        second = sum(
            (np.square(std) + np.square(mean_item)) * count
            for std, mean_item, count in zip(stds, means, counts)
        )
        variance = np.maximum(second / total_count - np.square(mean), 0.0)
        result[feature] = {
            "min": minimum.tolist(),
            "max": maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(total_count)],
        }
    return result


_FRAME_TELEMETRY_KEYS = frozenset(
    {
        "camera_frame_ids",
        "camera_timestamps_ns",
        "camera_frame_reused",
        "camera_age_ns",
        "control_step_id",
        "policy_observation_id",
        "policy_request_id",
        "control_chunk_id",
        "control_chunk_cursor",
        "action_sent_timestamp_ns",
        "policy_generation_id",
        "safety_flags",
    }
)
_RAW_CHUNK_TELEMETRY_KEYS = frozenset(
    {
        "raw_action_chunk",
        "raw_action_chunk_length",
        "request_id",
        "chunk_id",
        "chunk_cursor",
        "observation_id",
    }
)


def _concat_raw_action_chunks(chunks: list[np.ndarray], action_dim: int) -> np.ndarray:
    chunk_count = sum(int(chunk.shape[0]) for chunk in chunks)
    max_horizon = max((int(chunk.shape[1]) for chunk in chunks if chunk.ndim == 3), default=0)
    result = np.full((chunk_count, max_horizon, action_dim), np.nan, dtype=np.float32)
    offset = 0
    for chunk in chunks:
        if chunk.ndim != 3:
            continue
        count, horizon, width = chunk.shape
        if width != action_dim:
            continue
        result[offset : offset + count, :horizon] = chunk
        offset += count
    return result[:offset]


class LeRobotV21EpisodeSource(RecordedEpisodeSource):
    """Read standard LeRobot v2.1 and optional mp-real extensions without mutation."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve(strict=True)
        self._info = _load_json(self._root / "meta" / "info.json")
        if self._info.get("codebase_version") != CODEBASE_VERSION:
            raise DatasetValidationError(
                f"Expected LeRobot {CODEBASE_VERSION}, found {self._info.get('codebase_version')!r}"
            )
        self._episodes = {
            int(item["episode_index"]): item for item in _load_jsonl(self._root / "meta" / "episodes.jsonl")
        }
        self._tasks = {
            int(item["task_index"]): str(item["task"])
            for item in _load_jsonl(self._root / "meta" / "tasks.jsonl")
        }
        self._labels = {
            int(item["episode_index"]): item
            for item in _load_jsonl(self._root / "meta" / "mp_real" / "episode_labels.jsonl")
        }
        # Keep Parquet metadata/handles, never a decoded full-episode table.
        # A recording may have a very long episode, and the viewer only needs a
        # bounded row batch for a selected sample or curve aggregation.
        self._parquet_files: dict[int, pq.ParquetFile] = {}
        # Keep a small LRU of sequential decoders.  Each decoder retains only
        # its most recently decoded frame, and is closed on eviction or source
        # shutdown so browsing many episodes cannot accumulate video handles.
        self._video_readers: OrderedDict[tuple[int, str], _SequentialVideoReader] = OrderedDict()
        self._video_reader_lock = threading.RLock()
        self._video_reader_capacity = 6
        self._status = (
            EpisodeStatus.INCOMPLETE
            if self._root.name.endswith(".inprogress")
            or (self._root / "meta" / "mp_real" / "recovery.json").exists()
            else EpisodeStatus.COMPLETE
        )
        self._action_spec = _action_spec_from_recording_schema(self._root, self._info)

    def list_episodes(self) -> tuple[EpisodeMetadata, ...]:
        return tuple(self.get_episode_metadata(index) for index in sorted(self._episodes))

    def get_dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            root=self._root,
            info=self._info,
            status=self._status,
            is_mp_real=(self._root / "meta" / "mp_real" / "schema.json").is_file(),
            action_spec=self._action_spec,
            camera_roles=self.get_camera_roles(),
        )

    def get_episode_metadata(self, episode_index: int) -> EpisodeMetadata:
        item = self._episodes[episode_index]
        declared_length = int(item["length"])
        actual_length = self.get_length(episode_index)
        return EpisodeMetadata(
            episode_index=episode_index,
            length=actual_length,
            tasks=tuple(str(task) for task in item.get("tasks", ())),
            status=EpisodeStatus.INCOMPLETE if actual_length < declared_length else self._status,
            labels=self._labels.get(episode_index),
        )

    def get_action_spec(self) -> ActionSpec:
        return self._action_spec

    def get_state_schema(self) -> tuple[str, ...]:
        return self._action_spec.state_field_names

    def get_camera_roles(self) -> tuple[str, ...]:
        return tuple(
            key.removeprefix("observation.images.")
            for key, value in self._info["features"].items()
            if value.get("dtype") in {"video", "image"}
        )

    def get_length(self, episode_index: int) -> int:
        declared_length = int(self._episodes[episode_index]["length"])
        path = self._root / _data_path(episode_index)
        if not path.is_file():
            return 0
        return min(declared_length, self._parquet_file(episode_index).metadata.num_rows)

    def get_task(self, episode_index: int, task_index: int) -> str:
        """Return the recorded task selected by a sample's standard task index."""
        if episode_index not in self._episodes:
            raise KeyError(episode_index)
        try:
            return self._tasks[task_index]
        except KeyError as exc:
            raise DatasetValidationError(
                f"episode {episode_index} references unknown task_index {task_index}"
            ) from exc

    def get_sample(self, episode_index: int, index: int, *, include_images: bool = True) -> RecordedSample:
        if index < 0 or index >= self.get_length(episode_index):
            raise IndexError(index)
        row = self.get_row(episode_index, index)
        images = {
            role: self._video_frame(episode_index, role, int(row["frame_index"]))
            for role in self.get_camera_roles()
            if include_images and self._video_file(episode_index, role).is_file()
        }
        telemetry = {key.removeprefix("mp_real."): value for key, value in row.items() if key.startswith("mp_real.")}
        return RecordedSample(
            episode_index=int(row["episode_index"]),
            frame_index=int(row["frame_index"]),
            index=int(row["index"]),
            timestamp=float(row["timestamp"]),
            task_index=int(row["task_index"]),
            state=np.asarray(row["observation.state"], dtype=np.float32),
            action=np.asarray(row["action"], dtype=np.float32),
            images=images,
            telemetry=telemetry,
        )

    def get_pose_state_sample(self, episode_index: int, index: int) -> tuple[np.ndarray, float]:
        """Read exactly the recorded state and timestamp for a pose target."""
        if index < 0 or index >= self.get_length(episode_index):
            raise IndexError(index)
        row = self.get_row(episode_index, index, columns=("timestamp", "observation.state"))
        return np.asarray(row["observation.state"], dtype=np.float32), float(row["timestamp"])

    def get_sample_at_timestamp(self, episode_index: int, timestamp: float) -> RecordedSample:
        length = self.get_length(episode_index)
        if length <= 0:
            raise IndexError(f"episode {episode_index} has no readable samples")
        index = min(max(round(timestamp * float(self._info["fps"])), 0), length - 1)
        return self.get_sample(episode_index, index)

    def iter_samples(self, episode_index: int) -> Iterator[RecordedSample]:
        for index in range(self.get_length(episode_index)):
            yield self.get_sample(episode_index, index)

    def close(self) -> None:
        self._parquet_files.clear()
        with self._video_reader_lock:
            for reader in self._video_readers.values():
                reader.close()
            self._video_readers.clear()

    def get_row(self, episode_index: int, index: int, *, columns: tuple[str, ...] | None = None) -> dict[str, Any]:
        """Read one row through a bounded Parquet batch, without table caching."""
        file = self._parquet_file(episode_index)
        if index < 0 or index >= file.metadata.num_rows:
            raise IndexError(index)
        available = set(file.schema_arrow.names)
        selected = [column for column in columns if column in available] if columns is not None else None
        row_group = 0
        offset = index
        while row_group < file.num_row_groups:
            row_count = file.metadata.row_group(row_group).num_rows
            if offset < row_count:
                break
            offset -= row_count
            row_group += 1
        if row_group >= file.num_row_groups:
            raise IndexError(index)
        batch_size = 512
        for batch in file.iter_batches(batch_size=batch_size, row_groups=[row_group], columns=selected):
            if offset < batch.num_rows:
                return batch.slice(offset, 1).to_pylist()[0]
            offset -= batch.num_rows
        raise IndexError(index)

    def iter_rows(
        self,
        episode_index: int,
        *,
        columns: tuple[str, ...] | None = None,
        batch_size: int = 1024,
    ) -> Iterator[dict[str, Any]]:
        """Yield Parquet rows in bounded batches for offline metrics/curves."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        file = self._parquet_file(episode_index)
        available = set(file.schema_arrow.names)
        selected = [column for column in columns if column in available] if columns is not None else None
        for batch in file.iter_batches(batch_size=batch_size, columns=selected):
            yield from batch.to_pylist()

    def get_column_names(self, episode_index: int) -> tuple[str, ...]:
        return tuple(self._parquet_file(episode_index).schema_arrow.names)

    def get_camera_frame(self, episode_index: int, role: str, frame_index: int) -> np.ndarray | None:
        frame, _ = self.get_camera_frame_with_index(episode_index, role, frame_index)
        return frame

    def get_camera_frame_with_index(
        self, episode_index: int, role: str, frame_index: int
    ) -> tuple[np.ndarray | None, int | None]:
        """Return an exact frame or the last older frame when a video is short."""
        if not self.has_camera_video(episode_index, role):
            return None, None
        return self._video_reader(episode_index, role).frame_at(frame_index)

    def has_camera_video(self, episode_index: int, role: str) -> bool:
        return role in self.get_camera_roles() and self._video_file(episode_index, role).is_file()

    def get_episode_telemetry(
        self, episode_index: int, *, keys: tuple[str, ...] | None = None
    ) -> dict[str, np.ndarray]:
        """Read optional mp-real telemetry without requiring it for standard data."""
        index = self._telemetry_index(episode_index)
        if index is not None:
            return self._read_episode_telemetry_parts(episode_index, index, keys=keys)
        path = self._root / _telemetry_path(episode_index)
        if not path.is_file():
            return {}
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files if keys is None else (name for name in keys if name in archive.files)
            return {name: np.asarray(archive[name]) for name in names}

    def get_sample_telemetry(
        self,
        episode_index: int,
        sample_index: int,
        *,
        keys: tuple[str, ...] | None = None,
        raw_observation_id: int | None = None,
    ) -> dict[str, Any]:
        """Read one sample's telemetry from at most the relevant part files."""

        index = self._telemetry_index(episode_index)
        if index is None:
            telemetry = self.get_episode_telemetry(episode_index, keys=keys)
            return self._slice_legacy_sample_telemetry(
                telemetry,
                sample_index,
                keys=keys,
                raw_observation_id=raw_observation_id,
            )
        requested = set(keys) if keys is not None else None
        result: dict[str, Any] = {}
        if requested is None or "camera_roles" in requested:
            result["camera_roles"] = np.asarray(index.get("camera_roles", ()))
        if requested is None or "dropped_frame_count" in requested:
            result["dropped_frame_count"] = np.asarray(index.get("dropped_frame_count", 0), dtype=np.int64)
        if requested is None or "dropped_event_count" in requested:
            result["dropped_event_count"] = np.asarray(index.get("dropped_event_count", 0), dtype=np.int64)

        sample_part = self._part_for_sample(index, sample_index)
        if sample_part is not None:
            local_index = sample_index - int(sample_part["start_frame"])
            with np.load(self._telemetry_part_file(episode_index, sample_part), allow_pickle=False) as archive:
                for name in _FRAME_TELEMETRY_KEYS:
                    if requested is not None and name not in requested:
                        continue
                    if name in archive.files and local_index < len(archive[name]):
                        result[name] = np.asarray(archive[name][local_index])
        raw_part = self._part_for_raw_observation(index, raw_observation_id)
        if raw_part is not None:
            with np.load(self._telemetry_part_file(episode_index, raw_part), allow_pickle=False) as archive:
                for name in _RAW_CHUNK_TELEMETRY_KEYS:
                    if requested is None or name in requested:
                        if name in archive.files:
                            result[name] = np.asarray(archive[name])
        return result

    def _parquet_file(self, episode_index: int) -> pq.ParquetFile:
        file = self._parquet_files.get(episode_index)
        if file is None:
            path = self._root / _data_path(episode_index)
            file = pq.ParquetFile(path)
            self._parquet_files[episode_index] = file
        return file

    def _video_file(self, episode_index: int, role: str) -> Path:
        video_path = self._info.get("video_path")
        if not video_path:
            return Path("")
        return self._root / str(video_path).format(
            episode_chunk=_episode_chunk(episode_index),
            video_key=f"observation.images.{role}",
            episode_index=episode_index,
        )

    def _video_frame(self, episode_index: int, role: str, index: int) -> np.ndarray:
        frame, rendered_frame_index = self._video_reader(episode_index, role).frame_at(index)
        if frame is None or rendered_frame_index != index:
            raise IndexError(f"Video {self._video_file(episode_index, role)} contains no frame {index}")
        return frame

    def _video_reader(self, episode_index: int, role: str) -> _SequentialVideoReader:
        key = (episode_index, role)
        with self._video_reader_lock:
            reader = self._video_readers.get(key)
            if reader is not None:
                self._video_readers.move_to_end(key)
                return reader
            reader = _SequentialVideoReader(self._video_file(episode_index, role))
            self._video_readers[key] = reader
            while len(self._video_readers) > self._video_reader_capacity:
                _, evicted = self._video_readers.popitem(last=False)
                evicted.close()
            return reader

    def _telemetry_index(self, episode_index: int) -> dict[str, Any] | None:
        path = self._root / _telemetry_index_path(episode_index)
        if not path.is_file():
            return None
        return _load_json(path)

    def _telemetry_part_file(self, episode_index: int, part: Mapping[str, object]) -> Path:
        return self._root / _telemetry_dir_path(episode_index) / str(part["path"])

    def _read_episode_telemetry_parts(
        self,
        episode_index: int,
        index: Mapping[str, Any],
        *,
        keys: tuple[str, ...] | None,
    ) -> dict[str, np.ndarray]:
        requested = set(keys) if keys is not None else None
        result: dict[str, np.ndarray] = {}
        if requested is None or "camera_roles" in requested:
            result["camera_roles"] = np.asarray(index.get("camera_roles", ()))
        if requested is None or "dropped_frame_count" in requested:
            result["dropped_frame_count"] = np.asarray(index.get("dropped_frame_count", 0), dtype=np.int64)
        if requested is None or "dropped_event_count" in requested:
            result["dropped_event_count"] = np.asarray(index.get("dropped_event_count", 0), dtype=np.int64)
        if requested is None or "telemetry_part_count" in requested:
            result["telemetry_part_count"] = np.asarray(index.get("total_parts", 0), dtype=np.int64)
        arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for part in index.get("parts", ()):
            if not isinstance(part, Mapping):
                continue
            path = self._telemetry_part_file(episode_index, part)
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as archive:
                names = archive.files if requested is None else (name for name in requested if name in archive.files)
                for name in names:
                    if name in {"camera_roles", "dropped_frame_count", "dropped_event_count"}:
                        continue
                    arrays[name].append(np.asarray(archive[name]))
        for name, values in arrays.items():
            if name == "raw_action_chunk":
                result[name] = _concat_raw_action_chunks(values, self._action_spec.action_dim)
            elif name in {"start_control_step", "end_control_step"}:
                result[name] = np.concatenate([value.reshape(-1) for value in values], axis=0)
            else:
                result[name] = np.concatenate(values, axis=0) if values else np.asarray([])
        return result

    @staticmethod
    def _part_for_sample(index: Mapping[str, Any], sample_index: int) -> Mapping[str, object] | None:
        for part in index.get("parts", ()):
            if not isinstance(part, Mapping):
                continue
            if int(part.get("start_frame", -1)) <= sample_index < int(part.get("end_frame", -1)):
                return part
        return None

    @staticmethod
    def _part_for_raw_observation(
        index: Mapping[str, Any],
        raw_observation_id: int | None,
    ) -> Mapping[str, object] | None:
        if raw_observation_id is None or raw_observation_id < 0:
            return None
        for part in index.get("parts", ()):
            if not isinstance(part, Mapping):
                continue
            raw_ids = part.get("raw_observation_ids", ())
            if isinstance(raw_ids, list) and int(raw_observation_id) in {int(value) for value in raw_ids}:
                return part
        return None

    @staticmethod
    def _slice_legacy_sample_telemetry(
        telemetry: Mapping[str, np.ndarray],
        sample_index: int,
        *,
        keys: tuple[str, ...] | None,
        raw_observation_id: int | None,
    ) -> dict[str, Any]:
        requested = set(keys) if keys is not None else None
        result: dict[str, Any] = {}
        for name, value in telemetry.items():
            if requested is not None and name not in requested:
                continue
            if name in _FRAME_TELEMETRY_KEYS and sample_index < len(value):
                result[name] = np.asarray(value[sample_index])
            elif name in {"camera_roles", "dropped_frame_count", "dropped_event_count"}:
                result[name] = np.asarray(value)
        if raw_observation_id is not None:
            raw_ids = telemetry.get("observation_id")
            if raw_ids is not None and np.any(np.asarray(raw_ids) == int(raw_observation_id)):
                for name in _RAW_CHUNK_TELEMETRY_KEYS:
                    if requested is None or name in requested:
                        if name in telemetry:
                            result[name] = np.asarray(telemetry[name])
        return result


def _action_spec_from_recording_schema(root: Path, info: Mapping[str, Any]) -> ActionSpec:
    """Prefer mp-real's exact ActionSpec snapshot over lossy LeRobot fields."""
    schema_path = root / "meta" / "mp_real" / "schema.json"
    if schema_path.is_file():
        try:
            schema_payload = _load_json(schema_path)
            payload = schema_payload["action_spec"]
            if isinstance(payload, Mapping):
                payload = _legacy_action_spec_payload(payload, schema_payload)
                return ActionSpec.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            # An invalid optional extension is represented by the conservative
            # unknown schema below and will be rejected by pose validation.
            pass
    mp_real_info = info.get("mp_real", {})
    if isinstance(mp_real_info, Mapping):
        try:
            payload = mp_real_info["action_spec"]
            if isinstance(payload, Mapping):
                payload = _legacy_action_spec_payload(payload, mp_real_info)
                return ActionSpec.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            pass
    return _action_spec_from_info(info)


def _legacy_action_spec_payload(payload: Mapping[str, Any], container: Mapping[str, Any]) -> Mapping[str, Any]:
    schema_version = int(container.get("schema_version", 0) or 0)
    control_step_aligned = bool(container.get("control_step_aligned", False))
    if schema_version >= MP_REAL_SCHEMA_VERSION and control_step_aligned:
        return payload
    adjusted = dict(payload)
    adjusted.setdefault("action_mode", "unknown")
    return adjusted


def _action_spec_from_info(info: Mapping[str, Any]) -> ActionSpec:
    features = info["features"]
    state_feature = features["observation.state"]
    action_feature = features["action"]
    state_dim = int(state_feature["shape"][0])
    action_dim = int(action_feature["shape"][0])
    state_names = _read_feature_names(state_feature, state_dim, "state")
    action_names = _read_feature_names(action_feature, action_dim, "action")
    mp_real_info = info.get("mp_real", {})
    replay = mp_real_info.get("replay", {}) if isinstance(mp_real_info, Mapping) else {}
    action_mode = "unknown"
    arm_count = 0
    gripper_indices: tuple[int, ...] = ()
    if isinstance(replay, Mapping):
        action_mode = str(replay.get("action_mode") or action_mode)
        if replay.get("arm_count") is not None:
            arm_count = int(replay["arm_count"])
        gripper_indices = tuple(int(item) for item in replay.get("gripper_indices", ()))
    return ActionSpec(
        action_dim=action_dim,
        state_dim=state_dim,
        joint_dof_per_arm=0,
        joint_unit="unknown",
        camera_roles=tuple(
            key.removeprefix("observation.images.")
            for key, value in features.items()
            if value.get("dtype") in {"video", "image"}
        ),
        state_fields=tuple(VectorField(name, "unknown", "unknown") for name in state_names),
        action_fields=tuple(VectorField(name, "unknown", "unknown") for name in action_names),
        action_mode=action_mode,
        arm_count=arm_count,
        gripper_indices=gripper_indices,
    )


def _read_feature_names(feature: Mapping[str, Any], dimension: int, prefix: str) -> list[str]:
    names = feature.get("names")
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if isinstance(names, list) and len(names) == dimension and all(isinstance(name, str) for name in names):
        return list(names)
    return [f"{prefix}_{index}" for index in range(dimension)]


def validate_lerobot_v21_dataset(root: Path | str, *, check_videos: bool = True) -> ValidationReport:
    path = Path(root).expanduser().resolve(strict=True)
    errors: list[str] = []
    warnings: list[str] = []
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        return ValidationReport(path, False, ("Missing meta/info.json",), (), 0)
    try:
        info = _load_json(info_path)
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationReport(path, False, (f"Cannot parse info.json: {exc}",), (), 0)
    if info.get("codebase_version") != CODEBASE_VERSION:
        errors.append(f"codebase_version must be {CODEBASE_VERSION}")
    required_features = {
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    features = info.get("features", {})
    missing_features = required_features - set(features)
    if missing_features:
        errors.append("Missing features: " + ", ".join(sorted(missing_features)))
    fps = float(info.get("fps", 0))
    if fps <= 0:
        errors.append("fps must be positive")
    episodes = _load_jsonl(path / "meta" / "episodes.jsonl")
    tasks = {int(item["task_index"]) for item in _load_jsonl(path / "meta" / "tasks.jsonl")}
    expected_global_index = 0
    camera_roles = [
        key.removeprefix("observation.images.")
        for key, value in features.items()
        if value.get("dtype") == "video"
    ]
    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        episode_index = int(episode["episode_index"])
        parquet_path = path / _data_path(episode_index)
        if not parquet_path.is_file():
            errors.append(f"episode {episode_index}: missing parquet")
            continue
        try:
            table = pq.read_table(parquet_path)
        except (OSError, pa.ArrowException) as exc:
            errors.append(f"episode {episode_index}: cannot read parquet: {exc}")
            continue
        length = int(episode.get("length", -1))
        if len(table) != length:
            errors.append(f"episode {episode_index}: metadata length {length} != parquet rows {len(table)}")
        missing_columns = required_features - set(table.column_names)
        if missing_columns:
            errors.append(f"episode {episode_index}: missing columns {sorted(missing_columns)}")
            continue
        rows = table.select(
            ["timestamp", "frame_index", "episode_index", "index", "task_index", "observation.state", "action"]
        ).to_pylist()
        for local_index, row in enumerate(rows):
            expected_timestamp = local_index / fps
            if abs(float(row["timestamp"]) - expected_timestamp) > TIMESTAMP_TOLERANCE_S:
                errors.append(f"episode {episode_index}: frame {local_index} timestamp is not aligned to fps")
                break
            if int(row["frame_index"]) != local_index:
                errors.append(f"episode {episode_index}: non-contiguous frame_index")
                break
            if int(row["episode_index"]) != episode_index:
                errors.append(f"episode {episode_index}: row episode_index mismatch")
                break
            if int(row["index"]) != expected_global_index + local_index:
                errors.append(f"episode {episode_index}: non-contiguous global index")
                break
            if tasks and int(row["task_index"]) not in tasks:
                errors.append(f"episode {episode_index}: unknown task_index")
                break
            if len(row["observation.state"]) != int(features["observation.state"]["shape"][0]):
                errors.append(f"episode {episode_index}: invalid observation.state shape")
                break
            if len(row["action"]) != int(features["action"]["shape"][0]):
                errors.append(f"episode {episode_index}: invalid action shape")
                break
        expected_global_index += len(table)
        if check_videos:
            for role in camera_roles:
                video_file = path / _video_path(episode_index, role)
                if not video_file.is_file():
                    errors.append(f"episode {episode_index}: missing video for {role}")
                    continue
                try:
                    with av.open(str(video_file)) as container:
                        count = sum(1 for _ in container.decode(video=0))
                    if count != len(table):
                        errors.append(
                            f"episode {episode_index}: {role} has {count} video frames, expected {len(table)}"
                        )
                except av.error.FFmpegError as exc:
                    errors.append(f"episode {episode_index}: cannot read video {role}: {exc}")
    if info.get("total_episodes") != len(episodes):
        errors.append("info total_episodes does not match episodes.jsonl")
    if info.get("total_frames") != expected_global_index:
        errors.append("info total_frames does not match parquet rows")
    if not (path / "meta" / "episodes_stats.jsonl").is_file():
        errors.append("Missing meta/episodes_stats.jsonl")
    if not (path / "meta" / "stats.json").is_file():
        errors.append("Missing meta/stats.json")
    schema_path = path / "meta" / "mp_real" / "schema.json"
    schema_payload = _load_json(schema_path) if schema_path.is_file() else {}
    telemetry_config = schema_payload.get("telemetry", {}) if isinstance(schema_payload, Mapping) else {}
    telemetry_enabled = not isinstance(telemetry_config, Mapping) or telemetry_config.get("enabled", True)
    if schema_path.is_file() and telemetry_enabled and not (path / "telemetry").is_dir():
        warnings.append("mp-real schema is present but telemetry directory is missing")
    mp_real_info = info.get("mp_real", {})
    if isinstance(mp_real_info, Mapping):
        schema_version = int(mp_real_info.get("schema_version", 0) or 0)
        if schema_version and schema_version < MP_REAL_SCHEMA_VERSION:
            warnings.append("mp-real recording schema predates control-step-aligned semantics")
        if schema_version >= MP_REAL_SCHEMA_VERSION and not bool(mp_real_info.get("control_step_aligned", False)):
            warnings.append(
                "mp-real recording is not marked control_step_aligned; observation/action semantics are unknown"
            )
    else:
        warnings.append("non-mp-real or legacy dataset: recording semantics are unknown")
    return ValidationReport(path, not errors, tuple(errors), tuple(warnings), len(episodes))
