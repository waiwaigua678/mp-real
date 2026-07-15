from __future__ import annotations

import dataclasses
import json
import os
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mp_real.data.models import (
    DatasetMetadata,
    EpisodeMetadata,
    EpisodeRecordingContext,
    EpisodeStatus,
    RecordedEpisodeSource,
    RecordedSample,
    RecorderConfig,
)
from mp_real.runtime.events import (
    ActionExecuted,
    ActionSelected,
    ActionStabilized,
    ChunkReceived,
    InferenceFinished,
    ObservationCaptured,
    RuntimeEvent,
    RuntimeEventSink,
)
from mp_real.runtime.models import ActionSpec, VectorField

CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
TIMESTAMP_TOLERANCE_S = 1e-4
_STOP = object()


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
        "mp_real.observation_id": _feature("int64", [1], None),
        "mp_real.chunk_cursor": _feature("int64", [1], None),
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
) -> dict[str, object]:
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


class _VideoWriter:
    def __init__(self, path: Path, *, fps: float, image_shape: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width, channels = image_shape
        if channels != 3:
            raise DatasetValidationError(f"Only RGB cameras are supported, got {channels} channels")
        self.path = path
        self._container = av.open(str(path), mode="w")
        try:
            self._stream = self._container.add_stream("libx264", rate=fps)
        except av.error.FFmpegError:
            self._stream = self._container.add_stream("mpeg4", rate=fps)
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
        self._camera_frame_ids: list[list[int]] = []
        self._camera_timestamps: list[list[int]] = []
        self._camera_reused: list[list[bool]] = []
        self._camera_ages: list[list[int]] = []
        self._raw_chunks: list[np.ndarray] = []
        self._raw_chunk_request_ids: list[int] = []
        self._raw_chunk_ids: list[int] = []
        self._raw_chunk_cursors: list[int] = []
        self._raw_chunk_observation_ids: list[int] = []
        self._policy_generation_ids: list[int] = []
        self._safety_flags: list[str] = []
        self._invalid_reason: str | None = None
        self._dropped_frame_count = 0
        self._dropped_event_count = 0

    @property
    def camera_shapes(self) -> Mapping[str, tuple[int, int, int]]:
        return self._camera_shapes

    def set_global_start(self, value: int) -> None:
        self._global_start = value

    def record_chunk(self, event: RuntimeEvent) -> None:
        raw_chunk = event.payload.get("raw_action_chunk")
        if raw_chunk is None:
            return
        array = np.asarray(raw_chunk, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.config.action_spec.action_dim:
            self._dropped_event_count += 1
            return
        self._raw_chunks.append(array.copy())
        self._raw_chunk_request_ids.append(-1 if event.request_id is None else event.request_id)
        self._raw_chunk_ids.append(-1 if event.chunk_id is None else event.chunk_id)
        self._raw_chunk_cursors.append(_integer(event.payload.get("chunk_cursor"), -1))
        self._raw_chunk_observation_ids.append(_integer(event.payload.get("observation_id"), -1))

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
        ages = [max(0, event.monotonic_timestamp_ns - timestamp) for timestamp in current_timestamps]
        if any(age > self.config.max_camera_age_ns for age in ages):
            self._invalid_reason = "camera_age_exceeded"

        selected = action if selected is None else np.asarray(selected, dtype=np.float32)
        stabilized = action if stabilized is None else np.asarray(stabilized, dtype=np.float32)
        if selected.shape != action.shape or stabilized.shape != action.shape:
            self._dropped_frame_count += 1
            self._invalid_reason = "action_telemetry_shape_mismatch"
            return
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
            "mp_real.observation_id": [_integer(observation.get("observation_id"), -1)],
            "mp_real.chunk_cursor": [_integer(event.payload.get("chunk_cursor"), -1)],
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
        self._camera_frame_ids.append(current_ids)
        self._camera_timestamps.append(current_timestamps)
        self._camera_reused.append(reused)
        self._camera_ages.append(ages)
        self._policy_generation_ids.append(event.generation_id)
        safety_flags = event.payload.get("safety_flags", ())
        self._safety_flags.append(
            json.dumps(safety_flags, ensure_ascii=False, sort_keys=True, default=_json_default)
        )
        self.frame_count += 1

    def note_drop(self, *, frame: bool = False) -> None:
        self._dropped_event_count += 1
        if frame:
            self._dropped_frame_count += 1

    def close(self) -> tuple[dict[str, object], dict[str, object], str | None]:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        video_info: dict[str, object] = {}
        for role, writer in self._video_writers.items():
            video_info[role] = writer.close()
        telemetry_path = self.root / _telemetry_path(self.context.episode_index)
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_count = len(self._raw_chunks)
        max_chunk_length = max((len(chunk) for chunk in self._raw_chunks), default=0)
        action_dim = self.config.action_spec.action_dim
        raw_chunks = np.full((chunk_count, max_chunk_length, action_dim), np.nan, dtype=np.float32)
        for index, chunk in enumerate(self._raw_chunks):
            raw_chunks[index, : len(chunk)] = chunk
        np.savez_compressed(
            telemetry_path,
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
            policy_generation_id=np.asarray(self._policy_generation_ids, dtype=np.int64),
            safety_flags=np.asarray(self._safety_flags),
            dropped_frame_count=np.asarray(self._dropped_frame_count, dtype=np.int64),
            dropped_event_count=np.asarray(self._dropped_event_count, dtype=np.int64),
        )
        return ({name: stats.to_json() for name, stats in self._stats.items()}, video_info, self._invalid_reason)


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
        self._failure: BaseException | None = None
        self._active: EpisodeRecordingContext | None = None
        self._dropped_event_count = 0
        self._dropped_frame_count = 0
        self._queue_high_watermark = 0
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
        with self._lock:
            if self._started:
                return
            if self._final_root.exists() or self._work_root.exists():
                raise FileExistsError(f"Recording dataset already exists: {self._final_root}")
            self._started = True
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
        if not self._put(("event", event), block=False):
            with self._lock:
                self._dropped_event_count += 1
                if isinstance(event, (ObservationCaptured, ActionExecuted)):
                    self._dropped_frame_count += 1

    def flush(self, *, timeout: float | None = 5.0) -> bool:
        marker = threading.Event()
        self._put(("flush", marker), block=True)
        return marker.wait(timeout)

    def stop(self, *, finalize: bool = True, timeout: float | None = 5.0) -> bool:
        with self._lock:
            if not self._started or self._stopping:
                return self.join(timeout=timeout, raise_on_error=False)
            self._stopping = True
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

    def _run(self) -> None:
        episodes: dict[int, _EpisodeWriter] = {}
        contexts: dict[int, EpisodeRecordingContext] = {}
        episode_stats: dict[int, dict[str, object]] = {}
        video_info: dict[str, object] = {}
        labels: dict[int, dict[str, object]] = {}
        observations: dict[int, Mapping[str, object]] = {}
        selected_actions: dict[tuple[int, int], np.ndarray] = {}
        stabilized_actions: dict[tuple[int, int], np.ndarray] = {}
        inference_latency_ns: dict[int, int] = {}
        global_index = 0
        tasks: dict[str, int] = {}
        try:
            while True:
                item = self._queue.get()
                try:
                    kind = item[0]  # type: ignore[index]
                    if kind == "start":
                        self._work_root.mkdir(parents=True, exist_ok=False)
                        _write_json(
                            self._work_root / "meta" / "info.json",
                            _build_info(self.config, {}, require_camera_shapes=False),
                        )
                        _write_json(
                            self._work_root / "meta" / "mp_real" / "schema.json",
                            {
                                "schema_version": 1,
                                "action_source": "executed_action",
                                "robot_name": self.config.robot_name,
                                "action_spec": dataclasses.asdict(self.config.action_spec),
                                "state_field_order": _field_names(
                                    self.config.action_spec.state_fields, self.config.action_spec.state_dim, "state"
                                ),
                                "action_field_order": _field_names(
                                    self.config.action_spec.action_fields, self.config.action_spec.action_dim, "action"
                                ),
                                "camera_roles": list(self.config.action_spec.camera_roles),
                                "max_camera_age_ns": self.config.max_camera_age_ns,
                            },
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
                        event = item[1]  # type: ignore[index]
                        assert isinstance(event, RuntimeEvent)
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
                        if isinstance(event, ObservationCaptured):
                            observation_id = _integer(event.payload.get("observation_id"), -1)
                            if observation_id >= 0:
                                observations[observation_id] = event.payload
                        elif isinstance(event, InferenceFinished):
                            observation_id = _integer(event.payload.get("observation_id"), -1)
                            if observation_id >= 0:
                                inference_latency_ns[observation_id] = _integer(
                                    event.payload.get("inference_latency_ns")
                                )
                        elif isinstance(event, ChunkReceived):
                            writer.record_chunk(event)
                        elif isinstance(event, ActionSelected):
                            selected_actions[(event_episode.episode_index, int(event.step or -1))] = np.asarray(
                                event.payload.get("selected_raw_action"), dtype=np.float32
                            )
                        elif isinstance(event, ActionStabilized):
                            stabilized_actions[(event_episode.episode_index, int(event.step or -1))] = np.asarray(
                                event.payload.get("stabilized_target_action"), dtype=np.float32
                            )
                        elif isinstance(event, ActionExecuted):
                            observation_id = _integer(event.payload.get("observation_id"), -1)
                            observation = observations.get(observation_id)
                            if observation is None:
                                writer.note_drop(frame=True)
                            else:
                                step_key = (event_episode.episode_index, int(event.step or -1))
                                writer.record_frame(
                                    event,
                                    observation,
                                    selected_actions.pop(step_key, None),
                                    stabilized_actions.pop(step_key, None),
                                    inference_latency_ns.get(observation_id, 0),
                                )
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
                    elif kind == "label":
                        labels[int(item[1])] = dict(item[2])  # type: ignore[index]
                    elif kind == "flush":
                        item[1].set()  # type: ignore[index]
                    elif kind == "stop":
                        if item[1]:
                            self._finalize(episodes, contexts, episode_stats, video_info, labels, tasks)
                        else:
                            self._write_recovery("Recorder stopped without finalization")
                        return
                    else:
                        raise RuntimeError(f"Unknown recorder command: {kind!r}")
                finally:
                    self._queue.task_done()
        except BaseException as exc:
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

    def _finalize(
        self,
        episodes: Mapping[int, _EpisodeWriter],
        contexts: Mapping[int, EpisodeRecordingContext],
        episode_stats: Mapping[int, Mapping[str, object]],
        video_info: Mapping[str, object],
        labels: Mapping[int, Mapping[str, object]],
        tasks: Mapping[str, int],
    ) -> None:
        if any(writer.frame_count == 0 for writer in episodes.values()):
            raise DatasetValidationError("Cannot finalize a dataset with an empty episode")
        camera_shapes: dict[str, tuple[int, int, int]] = {}
        for writer in episodes.values():
            camera_shapes.update(writer.camera_shapes)
        info = _build_info(self.config, camera_shapes)
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
        self._table_cache: dict[int, pa.Table] = {}
        self._status = (
            EpisodeStatus.INCOMPLETE
            if self._root.name.endswith(".inprogress")
            or (self._root / "meta" / "mp_real" / "recovery.json").exists()
            else EpisodeStatus.COMPLETE
        )
        self._action_spec = _action_spec_from_info(self._info)

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
        return EpisodeMetadata(
            episode_index=episode_index,
            length=int(item["length"]),
            tasks=tuple(str(task) for task in item.get("tasks", ())),
            status=self._status,
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
        return int(self._episodes[episode_index]["length"])

    def get_sample(self, episode_index: int, index: int) -> RecordedSample:
        table = self._table(episode_index)
        if index < 0 or index >= len(table):
            raise IndexError(index)
        row = table.slice(index, 1).to_pylist()[0]
        images = {
            role: self._video_frame(episode_index, role, index)
            for role in self.get_camera_roles()
            if self._video_file(episode_index, role).is_file()
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

    def get_sample_at_timestamp(self, episode_index: int, timestamp: float) -> RecordedSample:
        index = min(max(round(timestamp * float(self._info["fps"])), 0), self.get_length(episode_index) - 1)
        return self.get_sample(episode_index, index)

    def iter_samples(self, episode_index: int) -> Iterator[RecordedSample]:
        for index in range(self.get_length(episode_index)):
            yield self.get_sample(episode_index, index)

    def close(self) -> None:
        self._table_cache.clear()

    def _table(self, episode_index: int) -> pa.Table:
        table = self._table_cache.get(episode_index)
        if table is None:
            path = self._root / _data_path(episode_index)
            table = pq.read_table(path)
            self._table_cache[episode_index] = table
        return table

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
        path = self._video_file(episode_index, role)
        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index == index:
                    return frame.to_ndarray(format="rgb24")
        raise IndexError(f"Video {path} contains no frame {index}")


def _action_spec_from_info(info: Mapping[str, Any]) -> ActionSpec:
    features = info["features"]
    state_feature = features["observation.state"]
    action_feature = features["action"]
    state_dim = int(state_feature["shape"][0])
    action_dim = int(action_feature["shape"][0])
    state_names = _read_feature_names(state_feature, state_dim, "state")
    action_names = _read_feature_names(action_feature, action_dim, "action")
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
    if (path / "meta" / "mp_real" / "schema.json").is_file() and not (path / "telemetry").is_dir():
        warnings.append("mp-real schema is present but telemetry directory is missing")
    return ValidationReport(path, not errors, tuple(errors), tuple(warnings), len(episodes))
