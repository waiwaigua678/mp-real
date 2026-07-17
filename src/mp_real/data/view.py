"""Read-only, bounded offline viewing primitives for LeRobot v2.1 episodes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.data.catalog import CatalogDataset, RecordedDataCatalog
from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource
from mp_real.data.models import EpisodeMetadata, RecordedSample
from mp_real.data.pose import recorded_pose_target
from mp_real.evaluation.metrics import compute_episode_metrics


class DataViewError(ValueError):
    """A user-safe error returned by the offline viewer API."""


@dataclasses.dataclass(frozen=True)
class TimelineIndex:
    """Sample-index timeline; percentage is intentionally presentation-only."""

    length: int
    fps: float
    first_timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("timeline length must be positive")
        if self.fps <= 0:
            raise ValueError("timeline fps must be positive")

    def clamp_index(self, sample_index: int) -> int:
        return min(max(int(sample_index), 0), self.length - 1)

    def progress_ratio(self, sample_index: int) -> float:
        if self.length == 1:
            return 0.0
        return self.clamp_index(sample_index) / (self.length - 1)

    def index_for_progress(self, progress_ratio: float) -> int:
        if not math.isfinite(progress_ratio):
            raise ValueError("progress_ratio must be finite")
        return self.clamp_index(round(min(max(progress_ratio, 0.0), 1.0) * (self.length - 1)))

    def estimated_index_for_timestamp(self, timestamp: float) -> int:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        return self.clamp_index(round((timestamp - self.first_timestamp) * self.fps))


@dataclasses.dataclass(frozen=True)
class ViewCursor:
    """A freely movable cursor for data inspection only.

    This deliberately has a distinct type from the future motion-capable
    ``RobotReplayCursor``.  Moving it only changes an in-memory view state.
    """

    dataset_id: str
    episode_index: int
    sample_index: int
    timestamp: float
    progress_ratio: float
    playing: bool
    playback_rate: float

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if self.episode_index < 0 or self.sample_index < 0:
            raise ValueError("episode_index and sample_index must be non-negative")
        if self.playback_rate not in {0.25, 0.5, 1.0, 2.0}:
            raise ValueError("playback_rate must be one of 0.25, 0.5, 1.0, 2.0")


@dataclasses.dataclass
class PlaybackCursor:
    """Clock state used by a viewer; it has no robot reference or side effect."""

    cursor: ViewCursor
    fractional_sample: float = 0.0

    def advance(self, elapsed_s: float, timeline: TimelineIndex) -> ViewCursor:
        if not self.cursor.playing:
            return self.cursor
        self.fractional_sample += max(0.0, elapsed_s) * timeline.fps * self.cursor.playback_rate
        steps = int(self.fractional_sample)
        if steps <= 0:
            return self.cursor
        self.fractional_sample -= steps
        index = timeline.clamp_index(self.cursor.sample_index + steps)
        playing = index < timeline.length - 1
        self.cursor = dataclasses.replace(
            self.cursor,
            sample_index=index,
            progress_ratio=timeline.progress_ratio(index),
            timestamp=timeline.first_timestamp + index / timeline.fps,
            playing=playing,
        )
        return self.cursor


class EpisodeReader:
    """A LeRobot episode facade that keeps source reads bounded and read-only."""

    def __init__(self, source: LeRobotV21EpisodeSource) -> None:
        self.source = source

    def metadata(self, episode_index: int) -> EpisodeMetadata:
        return self.source.get_episode_metadata(episode_index)

    def timeline(self, episode_index: int) -> TimelineIndex:
        length = self.source.get_length(episode_index)
        if length <= 0:
            raise DataViewError(f"episode {episode_index} has no readable samples")
        first = self.source.get_row(episode_index, 0, columns=("timestamp",))
        info = self.source.get_dataset_metadata().info
        return TimelineIndex(length=length, fps=float(info["fps"]), first_timestamp=float(first["timestamp"]))

    def sample(self, episode_index: int, sample_index: int) -> RecordedSample:
        return self.source.get_sample(episode_index, sample_index, include_images=False)

    def nearest_sample_index(self, episode_index: int, timestamp: float) -> int:
        timeline = self.timeline(episode_index)
        guess = timeline.estimated_index_for_timestamp(timestamp)
        # LeRobot timestamps are normally exact fps ticks.  Inspect nearby
        # rows as well so an imported dataset with a shifted first timestamp
        # still returns the nearest actual sample without loading the episode.
        candidates = range(max(0, guess - 2), min(timeline.length, guess + 3))
        return min(
            candidates,
            key=lambda index: abs(
                float(self.source.get_row(episode_index, index, columns=("timestamp",))["timestamp"]) - timestamp
            ),
        )


@dataclasses.dataclass(frozen=True)
class _DatasetEntry:
    dataset_id: str
    dataset: CatalogDataset


class _ExtremeDownsampler:
    """Bounded min/max envelope sampler that preserves visible extrema."""

    def __init__(self, length: int, max_points: int, event_indices: Iterable[int] = ()) -> None:
        self._length = max(1, length)
        self._max_points = max(8, max_points)
        candidates = sorted(
            {
                candidate
                for event_index in event_indices
                for candidate in (event_index - 1, event_index, event_index + 1)
                if 0 <= candidate < self._length
            }
        )
        # A recorder can emit several events per sample.  Keep event markers
        # useful without allowing them to defeat the API's explicit point cap.
        event_budget = min(len(candidates), max(0, self._max_points // 3))
        if event_budget and len(candidates) > event_budget:
            candidates = [
                candidates[round(position * (len(candidates) - 1) / (event_budget - 1))]
                for position in range(event_budget)
            ] if event_budget > 1 else [candidates[len(candidates) // 2]]
        self._bucket_count = max(1, (self._max_points - 2 - len(candidates)) // 2)
        self._minimum: list[tuple[int, float] | None] = [None] * self._bucket_count
        self._maximum: list[tuple[int, float] | None] = [None] * self._bucket_count
        self._forced_indices = set(candidates)
        self._forced: dict[int, float] = {}
        self._first: tuple[int, float] | None = None
        self._last: tuple[int, float] | None = None

    def add(self, index: int, value: object) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        point = (index, number)
        if self._first is None:
            self._first = point
        self._last = point
        bucket = min(self._bucket_count - 1, index * self._bucket_count // self._length)
        minimum = self._minimum[bucket]
        maximum = self._maximum[bucket]
        if minimum is None or number < minimum[1]:
            self._minimum[bucket] = point
        if maximum is None or number > maximum[1]:
            self._maximum[bucket] = point
        if index in self._forced_indices:
            self._forced[index] = number

    def points(self) -> list[list[float | int]]:
        points: dict[int, float] = dict(self._forced)
        for point in (self._first, self._last, *self._minimum, *self._maximum):
            if point is not None:
                points[point[0]] = point[1]
        return [[index, points[index]] for index in sorted(points)]


def downsample_series(
    values: Sequence[float] | np.ndarray,
    *,
    max_points: int,
    event_indices: Iterable[int] = (),
) -> list[list[float | int]]:
    """Downsample one curve while retaining endpoints, extrema and events."""
    sampler = _ExtremeDownsampler(len(values), max_points, event_indices)
    for index, value in enumerate(values):
        sampler.add(index, value)
    return sampler.points()


class DataViewSession:
    """Catalog-backed offline data session with no hardware dependencies."""

    def __init__(self, storage_roots: Sequence[Path | str]) -> None:
        self._catalog = RecordedDataCatalog(storage_roots)
        self._sources: dict[str, LeRobotV21EpisodeSource] = {}
        self._selection: ViewCursor | None = None
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            for source in self._sources.values():
                source.close()
            self._sources.clear()
            self._selection = None

    def datasets(self) -> list[dict[str, Any]]:
        entries = self._dataset_entries()
        return [
            {
                "dataset_id": entry.dataset_id,
                "name": entry.dataset.name,
                "robot_name": entry.dataset.robot_name,
                "status": entry.dataset.status.value,
                "is_mp_real": entry.dataset.is_mp_real,
                "episode_count": entry.dataset.episode_count,
            }
            for entry in entries
        ]

    def episodes(self, dataset_id: str) -> list[dict[str, Any]]:
        reader = self._reader(dataset_id)
        return [
            {
                "episode_index": item.episode_index,
                "length": item.length,
                "tasks": list(item.tasks),
                "status": item.status.value,
                "labels": item.labels,
            }
            for item in reader.source.list_episodes()
        ]

    def episode_metadata(self, dataset_id: str, episode_index: int) -> dict[str, Any]:
        reader = self._reader(dataset_id)
        metadata = reader.metadata(episode_index)
        timeline = reader.timeline(episode_index)
        dataset = reader.source.get_dataset_metadata()
        return {
            "dataset_id": dataset_id,
            "episode_index": episode_index,
            "length": timeline.length,
            "fps": timeline.fps,
            "duration_s": (timeline.length - 1) / timeline.fps,
            "tasks": list(metadata.tasks),
            "status": metadata.status.value,
            "labels": metadata.labels,
            "is_mp_real": dataset.is_mp_real,
            "robot_name": dataset.info.get("robot_type", "unknown"),
            "camera_roles": list(dataset.camera_roles),
            "state_fields": list(dataset.action_spec.state_field_names),
            "action_fields": list(dataset.action_spec.action_field_names),
            "action_dim": dataset.action_spec.action_dim,
            "state_dim": dataset.action_spec.state_dim,
        }

    def sample(self, dataset_id: str, episode_index: int, sample_index: int) -> dict[str, Any]:
        reader = self._reader(dataset_id)
        timeline = reader.timeline(episode_index)
        index = timeline.clamp_index(sample_index)
        sample = reader.sample(episode_index, index)
        metadata = reader.metadata(episode_index)
        source = reader.source
        telemetry = dict(sample.telemetry)
        extension = self._telemetry_for_sample(source, episode_index, index, telemetry)
        telemetry.update(extension)
        cursor = self._cursor(
            dataset_id, episode_index, index, sample, timeline, playing=False, playback_rate=1.0
        )
        cameras = self._camera_metadata(source, episode_index, index, sample.frame_index, extension)
        return {
            "cursor": dataclasses.asdict(cursor),
            "frame_index": sample.frame_index,
            "global_index": sample.index,
            "task": metadata.tasks[0] if metadata.tasks else None,
            "state": sample.state.tolist(),
            "action": sample.action.tolist(),
            "selected_raw_action": _vector_json(telemetry.get("selected_raw_action")),
            "stabilized_action": _vector_json(telemetry.get("stabilized_action")),
            "executed_action": sample.action.tolist() if source.get_dataset_metadata().is_mp_real else None,
            "raw_action": _vector_json(telemetry.get("raw_action")),
            "timestamp_monotonic_ns": _integer_or_none(telemetry.get("timestamp_monotonic_ns")),
            "inference_latency_ns": _integer_or_none(telemetry.get("inference_latency_ns")),
            "control_cycle_ns": _integer_or_none(telemetry.get("control_cycle_ns")),
            "camera_skew_ns": _integer_or_none(telemetry.get("camera_skew_ns")),
            "chunk_cursor": _integer_or_none(telemetry.get("chunk_cursor")),
            "chunk_id": _integer_or_none(telemetry.get("chunk_id")),
            "safety_flags": telemetry.get("safety_flags", []),
            "labels": metadata.labels,
            "cameras": cameras,
        }

    def sample_at_timestamp(self, dataset_id: str, episode_index: int, timestamp: float) -> dict[str, Any]:
        reader = self._reader(dataset_id)
        return self.sample(dataset_id, episode_index, reader.nearest_sample_index(episode_index, timestamp))

    def camera_frame(
        self, dataset_id: str, episode_index: int, sample_index: int, role: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reader = self._reader(dataset_id)
        timeline = reader.timeline(episode_index)
        index = timeline.clamp_index(sample_index)
        sample = reader.sample(episode_index, index)
        frame, rendered_frame_index = reader.source.get_camera_frame_with_index(
            episode_index, role, sample.frame_index
        )
        if frame is None:
            raise DataViewError(f"camera frame is missing for role {role!r}")
        camera = self._camera_metadata(reader.source, episode_index, index, sample.frame_index, {})[role]
        camera["rendered_frame_index"] = rendered_frame_index
        if rendered_frame_index is not None and rendered_frame_index != sample.frame_index:
            camera["frame_reused"] = True
        return frame, camera

    def curves(
        self,
        dataset_id: str,
        episode_index: int,
        *,
        series: Sequence[str],
        max_points: int = 600,
    ) -> dict[str, Any]:
        requested = tuple(dict.fromkeys(series)) or ("action",)
        invalid = set(requested) - {
            "state",
            "action",
            "selected_raw_action",
            "stabilized_action",
            "executed_action",
            "inference_latency",
            "control_cycle",
            "camera_skew",
            "action_jump",
            "jerk",
            "safety_modification",
        }
        if invalid:
            raise DataViewError("unknown curve series: " + ", ".join(sorted(invalid)))
        if not 16 <= max_points <= 4000:
            raise DataViewError("max_points must be between 16 and 4000")
        reader = self._reader(dataset_id)
        timeline = reader.timeline(episode_index)
        source = reader.source
        spec = source.get_action_spec()
        names: dict[str, tuple[str, ...]] = {
            "state": spec.state_field_names,
            "action": spec.action_field_names,
            "selected_raw_action": spec.action_field_names,
            "stabilized_action": spec.action_field_names,
            "executed_action": spec.action_field_names,
        }
        event_indices = [event["sample_index"] for event in self.runtime_events(dataset_id, episode_index)["events"]]
        samplers: dict[str, _ExtremeDownsampler] = {}
        for group in requested:
            if group in names:
                if group == "executed_action" and not source.get_dataset_metadata().is_mp_real:
                    continue
                for dimension, name in enumerate(names[group]):
                    samplers[f"{group}.{name}"] = _ExtremeDownsampler(timeline.length, max_points, event_indices)
            else:
                samplers[group] = _ExtremeDownsampler(timeline.length, max_points, event_indices)

        columns = (
            "frame_index",
            "timestamp",
            "observation.state",
            "action",
            "mp_real.selected_raw_action",
            "mp_real.stabilized_action",
            "mp_real.inference_latency_ns",
            "mp_real.control_cycle_ns",
            "mp_real.camera_skew_ns",
        )
        previous_action: np.ndarray | None = None
        previous_velocity: np.ndarray | None = None
        previous_timestamp: float | None = None
        for local_index, row in enumerate(source.iter_rows(episode_index, columns=columns)):
            state = _as_vector(row.get("observation.state"))
            action = _as_vector(row.get("action"))
            selected = _as_vector(row.get("mp_real.selected_raw_action"))
            stabilized = _as_vector(row.get("mp_real.stabilized_action"))
            for group, vector in (
                ("state", state),
                ("action", action),
                ("selected_raw_action", selected),
                ("stabilized_action", stabilized),
                ("executed_action", action if source.get_dataset_metadata().is_mp_real else None),
            ):
                if group not in requested or vector is None:
                    continue
                for dimension, value in enumerate(vector):
                    name = names[group][dimension]
                    samplers[f"{group}.{name}"].add(local_index, value)
            scalar_mapping = {
                "inference_latency": _as_float(row.get("mp_real.inference_latency_ns"), scale=1e-6),
                "control_cycle": _as_float(row.get("mp_real.control_cycle_ns"), scale=1e-6),
                "camera_skew": _as_float(row.get("mp_real.camera_skew_ns"), scale=1e-6),
            }
            for group, value in scalar_mapping.items():
                if group in samplers and value is not None:
                    samplers[group].add(local_index, value)
            timestamp = _as_float(row.get("timestamp"))
            if action is not None and previous_action is not None:
                if "action_jump" in samplers:
                    samplers["action_jump"].add(local_index, float(np.linalg.norm(action - previous_action)))
                if timestamp is not None and previous_timestamp is not None and timestamp > previous_timestamp:
                    velocity = (action - previous_action) / (timestamp - previous_timestamp)
                    if previous_velocity is not None and "jerk" in samplers:
                        acceleration = (velocity - previous_velocity) / (timestamp - previous_timestamp)
                        jerk = float(np.linalg.norm(acceleration) / (timestamp - previous_timestamp))
                        samplers["jerk"].add(local_index, jerk)
                    previous_velocity = velocity
            if action is not None and stabilized is not None and "safety_modification" in samplers:
                samplers["safety_modification"].add(local_index, float(np.linalg.norm(stabilized - action)))
            previous_action = action if action is not None else previous_action
            previous_timestamp = timestamp if timestamp is not None else previous_timestamp
        return {
            "episode_index": episode_index,
            "length": timeline.length,
            "series": [
                {"id": name, "label": name, "points": sampler.points()}
                for name, sampler in sorted(samplers.items())
                if sampler.points()
            ],
        }

    def runtime_events(self, dataset_id: str, episode_index: int, *, limit: int = 2000) -> dict[str, Any]:
        if limit <= 0 or limit > 10000:
            raise DataViewError("event limit must be between 1 and 10000")
        reader = self._reader(dataset_id)
        timeline = reader.timeline(episode_index)
        events: list[dict[str, Any]] = []
        stored_limit = max(0, limit - 2)
        stored_truncated = False
        path = _event_log_path(reader.source.get_dataset_metadata().root, episode_index)
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if len(events) >= stored_limit:
                        stored_truncated = True
                        break
                    step = payload.get("step")
                    index = timeline.clamp_index(int(step)) if isinstance(step, int) and step >= 0 else 0
                    events.append(
                        {
                            "event_id": f"stored-{len(events)}",
                            "type": str(payload.get("event_type", "runtime_event")),
                            "sample_index": index,
                            "timestamp_monotonic_ns": payload.get("monotonic_timestamp_ns"),
                            "description": payload.get("description")
                            or str(payload.get("event_type", "runtime event")),
                        }
                    )
        metadata = reader.metadata(episode_index)
        if metadata.labels:
            result = metadata.labels.get("result")
            if result:
                events.append(
                    {
                        "event_id": "label",
                        "type": f"{str(result).lower()}_label",
                        "sample_index": timeline.length - 1,
                        "timestamp_monotonic_ns": None,
                        "description": f"人工结果：{result}",
                    }
                )
        events.append(
            {
                "event_id": "episode-end",
                "type": "episode_end",
                "sample_index": timeline.length - 1,
                "timestamp_monotonic_ns": None,
                "description": "Episode end",
            }
        )
        return {"events": events[-limit:], "truncated": stored_truncated}

    def metrics(self, dataset_id: str, episode_index: int) -> dict[str, Any]:
        reader = self._reader(dataset_id)
        source = reader.source
        telemetry = source.get_episode_telemetry(
            episode_index,
            keys=("dropped_frame_count", "dropped_event_count"),
        )
        dropped_frames = _array_scalar(telemetry.get("dropped_frame_count"))
        dropped_events = _array_scalar(telemetry.get("dropped_event_count"))
        columns = (
            "timestamp",
            "action",
            "mp_real.selected_raw_action",
            "mp_real.stabilized_action",
            "mp_real.inference_latency_ns",
            "mp_real.control_cycle_ns",
            "mp_real.camera_skew_ns",
        )
        return {
            "episode_index": episode_index,
            "metrics": compute_episode_metrics(
                source.iter_rows(episode_index, columns=columns),
                dropped_frame_count=dropped_frames,
                dropped_event_count=dropped_events,
            ),
        }

    def selection(self) -> dict[str, Any]:
        with self._lock:
            return {"selection": dataclasses.asdict(self._selection) if self._selection is not None else None}

    def select(
        self,
        dataset_id: str,
        episode_index: int,
        sample_index: int,
        *,
        playing: bool = False,
        playback_rate: float = 1.0,
    ) -> dict[str, Any]:
        reader = self._reader(dataset_id)
        timeline = reader.timeline(episode_index)
        sample = reader.sample(episode_index, timeline.clamp_index(sample_index))
        cursor = self._cursor(
            dataset_id,
            episode_index,
            timeline.clamp_index(sample_index),
            sample,
            timeline,
            playing=playing,
            playback_rate=playback_rate,
        )
        with self._lock:
            self._selection = cursor
        return {"selection": dataclasses.asdict(cursor)}

    def pose_target(self, dataset_id: str, episode_index: int, sample_index: int):
        """Return a state-only target for a separate, authenticated control flow.

        This does not acquire a Robot or make a motion; the caller must still
        perform its own schema and live-state revalidation.
        """
        reader = self._reader(dataset_id)
        timeline = reader.timeline(episode_index)
        return recorded_pose_target(
            reader.source,
            dataset_id=dataset_id,
            episode_index=episode_index,
            sample_index=timeline.clamp_index(sample_index),
        )

    def _dataset_entries(self) -> tuple[_DatasetEntry, ...]:
        entries = []
        for dataset in self._catalog.scan():
            digest = hashlib.sha256(str(dataset.root).encode("utf-8")).hexdigest()[:16]
            entries.append(_DatasetEntry(f"ds_{digest}", dataset))
        return tuple(entries)

    def _reader(self, dataset_id: str) -> EpisodeReader:
        if not dataset_id.startswith("ds_"):
            raise DataViewError("unknown dataset_id")
        entries = {entry.dataset_id: entry.dataset for entry in self._dataset_entries()}
        try:
            dataset = entries[dataset_id]
        except KeyError as exc:
            raise DataViewError("unknown dataset_id") from exc
        with self._lock:
            source = self._sources.get(dataset_id)
            if source is None:
                source = LeRobotV21EpisodeSource(dataset.root)
                self._sources[dataset_id] = source
        return EpisodeReader(source)

    @staticmethod
    def _cursor(
        dataset_id: str,
        episode_index: int,
        sample_index: int,
        sample: RecordedSample,
        timeline: TimelineIndex,
        *,
        playing: bool,
        playback_rate: float,
    ) -> ViewCursor:
        return ViewCursor(
            dataset_id=dataset_id,
            episode_index=episode_index,
            sample_index=sample_index,
            timestamp=sample.timestamp,
            progress_ratio=timeline.progress_ratio(sample_index),
            playing=playing,
            playback_rate=playback_rate,
        )

    @staticmethod
    def _camera_metadata(
        source: LeRobotV21EpisodeSource,
        episode_index: int,
        sample_index: int,
        frame_index: int,
        extension: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        telemetry = extension.get("camera", {})
        result: dict[str, dict[str, Any]] = {}
        for role in source.get_camera_roles():
            values = telemetry.get(role, {}) if isinstance(telemetry, Mapping) else {}
            result[role] = {
                "role": role,
                "frame_index": frame_index,
                "frame_id": values.get("frame_id", frame_index) if isinstance(values, Mapping) else frame_index,
                "camera_timestamp_ns": values.get("timestamp_ns") if isinstance(values, Mapping) else None,
                "frame_reused": values.get("reused") if isinstance(values, Mapping) else None,
                "camera_age_ns": values.get("age_ns") if isinstance(values, Mapping) else None,
                "missing": not source.has_camera_video(episode_index, role),
                "sample_index": sample_index,
            }
        return result

    @staticmethod
    def _telemetry_for_sample(
        source: LeRobotV21EpisodeSource,
        episode_index: int,
        sample_index: int,
        row_telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        telemetry = source.get_episode_telemetry(
            episode_index,
            keys=(
                "camera_roles",
                "camera_frame_ids",
                "camera_timestamps_ns",
                "camera_frame_reused",
                "camera_age_ns",
                "safety_flags",
                "observation_id",
                "raw_action_chunk",
                "raw_action_chunk_length",
                "chunk_id",
            ),
        )
        if not telemetry:
            return {}
        result: dict[str, Any] = {}
        roles = [str(role) for role in telemetry.get("camera_roles", ())]
        for key, target in (
            ("camera_frame_ids", "frame_id"),
            ("camera_timestamps_ns", "timestamp_ns"),
            ("camera_frame_reused", "reused"),
            ("camera_age_ns", "age_ns"),
        ):
            values = telemetry.get(key)
            if values is None or sample_index >= len(values):
                continue
            camera = result.setdefault("camera", {})
            for role, value in zip(roles, values[sample_index]):
                camera.setdefault(role, {})[target] = _json_scalar(value)
        flags = telemetry.get("safety_flags")
        if flags is not None and sample_index < len(flags):
            try:
                result["safety_flags"] = json.loads(str(flags[sample_index]))
            except json.JSONDecodeError:
                result["safety_flags"] = [str(flags[sample_index])]
        observation_id = row_telemetry.get("observation_id")
        raw_observations = telemetry.get("observation_id")
        raw_chunks = telemetry.get("raw_action_chunk")
        raw_lengths = telemetry.get("raw_action_chunk_length")
        if observation_id is not None and raw_observations is not None and raw_chunks is not None:
            matches = np.flatnonzero(np.asarray(raw_observations) == int(observation_id))
            if matches.size:
                chunk_index = int(matches[-1])
                cursor = int(row_telemetry.get("chunk_cursor", 0))
                length = int(raw_lengths[chunk_index]) if raw_lengths is not None else len(raw_chunks[chunk_index])
                if 0 <= cursor < length:
                    result["raw_action"] = np.asarray(raw_chunks[chunk_index][cursor]).tolist()
                    chunk_ids = telemetry.get("chunk_id")
                    if chunk_ids is not None and chunk_index < len(chunk_ids):
                        result["chunk_id"] = _json_scalar(chunk_ids[chunk_index])
        return result


def _event_log_path(root: Path, episode_index: int) -> Path:
    return root / "meta" / "mp_real" / "events" / f"episode_{episode_index:06d}.jsonl"


def _as_vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.ndim == 1 and np.all(np.isfinite(array)) else None


def _as_float(value: object, *, scale: float = 1.0) -> float | None:
    if value is None:
        return None
    try:
        number = float(value) * scale
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector_json(value: object) -> list[float] | None:
    vector = _as_vector(value)
    return vector.tolist() if vector is not None else None


def _integer_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _array_scalar(value: np.ndarray | None) -> int:
    if value is None:
        return 0
    array = np.asarray(value).reshape(-1)
    return int(array[0]) if array.size else 0


def _json_scalar(value: object) -> Any:
    return value.item() if isinstance(value, np.generic) else value
