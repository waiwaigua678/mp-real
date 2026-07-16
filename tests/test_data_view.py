from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mp_real.data.catalog import RecordedDataCatalog
from mp_real.data.lerobot_v21 import LeRobotV21EpisodeRecorder
from mp_real.data.models import EpisodeRecordingContext, RecorderConfig
from mp_real.data.view import DataViewSession as _DataViewSession
from mp_real.data.view import PlaybackCursor, TimelineIndex, ViewCursor, downsample_series
from mp_real.runtime.events import ActionExecuted, ActionSelected, ActionStabilized, ChunkReceived, ObservationCaptured
from mp_real.runtime.models import ActionSpec, VectorField


def _spec(*, dimensions: int = 2, roles: tuple[str, ...] = ("head", "wrist")) -> ActionSpec:
    fields = tuple(VectorField(f"joint_{index}", "rad", "joint_position") for index in range(dimensions))
    return ActionSpec(dimensions, dimensions, 0, "rad", roles, state_fields=fields, action_fields=fields)


def _event(event_type, episode_id: str, step: int, payload: dict) -> object:
    return event_type(
        runtime_id="viewer-test",
        session_id="session",
        episode_id=episode_id,
        generation_id=1,
        step=step,
        monotonic_timestamp_ns=1_000_000_000 + step * 100_000_000,
        payload=payload,
    )


def _write_dataset(
    root: Path,
    *,
    spec: ActionSpec | None = None,
    robot_name: str = "piper",
    frames: int = 5,
    save_video: bool = True,
) -> Path:
    action_spec = spec or _spec()
    recorder = LeRobotV21EpisodeRecorder(
        RecorderConfig(
            root,
            root.name,
            robot_name,
            10,
            action_spec,
            save_video=save_video,
            queue_size=max(256, frames * 8),
        )
    )
    recorder.start()
    recorder.begin_episode(EpisodeRecordingContext(0, "episode-0", "pick-and-place", "session", 1))
    for index in range(frames):
        state = np.full(action_spec.state_dim, index, dtype=np.float32)
        selected = np.arange(action_spec.action_dim, dtype=np.float32) + index / 10
        stabilized = selected - 0.05
        executed = stabilized - 0.05
        images = {
            role: np.full((8, 10, 3), min(255, 20 + index * 10), dtype=np.uint8)
            for role in action_spec.camera_roles
        }
        observation_id = index + 1
        recorder.emit(
            _event(
                ObservationCaptured,
                "episode-0",
                index,
                {
                    "observation_id": observation_id,
                    "state": state,
                    "images": images,
                    "camera_frame_ids": {role: index + 10 for role in action_spec.camera_roles},
                    "camera_timestamps_ns": {
                        role: 1_000_000_000 + index * 100_000_000 for role in action_spec.camera_roles
                    },
                    "max_camera_skew_ns": 0,
                },
            )
        )
        recorder.emit(
            _event(
                ChunkReceived,
                "episode-0",
                index,
                {"observation_id": observation_id, "raw_action_chunk": selected.reshape(1, -1)},
            )
        )
        recorder.emit(
            _event(
                ActionSelected,
                "episode-0",
                index,
                {"observation_id": observation_id, "selected_raw_action": selected},
            )
        )
        recorder.emit(
            _event(
                ActionStabilized,
                "episode-0",
                index,
                {"observation_id": observation_id, "stabilized_target_action": stabilized},
            )
        )
        recorder.emit(
            _event(ActionExecuted, "episode-0", index, {"observation_id": observation_id, "executed_action": executed})
        )
    recorder.end_episode(labels={"result": "SUCCESS"})
    if not recorder.stop(timeout=30):
        raise AssertionError("recorder did not stop")
    return root


class OfflineDataViewTests(unittest.TestCase):
    def test_catalog_sample_timestamp_camera_metrics_and_selection_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write_dataset(Path(directory) / "piper")
            session = _DataViewSession([Path(directory)])
            self.addCleanup(session.close)
            datasets = session.datasets()
            self.assertEqual(len(datasets), 1)
            dataset_id = datasets[0]["dataset_id"]
            self.assertEqual(session.episodes(dataset_id)[0]["length"], 5)
            metadata = session.episode_metadata(dataset_id, 0)
            self.assertEqual(metadata["action_fields"], ["joint_0", "joint_1"])

            sample = session.sample(dataset_id, 0, 3)
            self.assertEqual(sample["cursor"]["sample_index"], 3)
            self.assertEqual(sample["frame_index"], 3)
            self.assertTrue(np.allclose(sample["selected_raw_action"], [0.3, 1.3]))
            self.assertEqual(sample["cameras"]["head"]["frame_id"], 13)
            frame, camera = session.camera_frame(dataset_id, 0, 3, "head")
            self.assertEqual(frame.shape, (8, 10, 3))
            self.assertEqual(camera["sample_index"], 3)
            nearest = session.sample_at_timestamp(dataset_id, 0, 0.31)
            self.assertEqual(nearest["cursor"]["sample_index"], 3)
            self.assertEqual(session.metrics(dataset_id, 0)["metrics"]["frame_count"], 5)
            events = session.runtime_events(dataset_id, 0)["events"]
            self.assertTrue(any(event["type"] == "episode_end" for event in events))
            selected = session.select(dataset_id, 0, 3)
            self.assertEqual(selected["selection"]["sample_index"], 3)

    def test_timeline_progress_drag_playback_rates_and_peak_downsampling(self) -> None:
        timeline = TimelineIndex(length=101, fps=10, first_timestamp=2.0)
        self.assertEqual(timeline.index_for_progress(0.73), 73)
        self.assertEqual(timeline.estimated_index_for_timestamp(7.0), 50)
        for rate in (0.25, 0.5, 1.0, 2.0):
            view = ViewCursor("dataset", 0, 0, 2.0, 0.0, True, rate)
            cursor = PlaybackCursor(view).advance(1.0, timeline)
            self.assertEqual(cursor.sample_index, int(10 * rate))
        paused = PlaybackCursor(ViewCursor("dataset", 0, 7, 2.7, 0.07, False, 1.0)).advance(2.0, timeline)
        self.assertEqual(paused.sample_index, 7)
        points = downsample_series([0.0] * 50 + [100.0] + [0.0] * 50, max_points=16, event_indices=(75,))
        indices = {int(point[0]) for point in points}
        self.assertIn(0, indices)
        self.assertIn(100, indices)
        self.assertIn(50, indices)
        self.assertTrue({74, 75, 76} & indices)

    def test_missing_video_missing_telemetry_incomplete_and_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _write_dataset(Path(directory) / "missing")
            (root / "videos" / "chunk-000" / "observation.images.wrist" / "episode_000000.mp4").unlink()
            (root / "telemetry").rename(root / "telemetry-hidden")
            incomplete = root.rename(Path(directory) / "missing.inprogress")
            session = _DataViewSession([Path(directory)])
            self.addCleanup(session.close)
            dataset_id = session.datasets()[0]["dataset_id"]
            self.assertEqual(session.episodes(dataset_id)[0]["status"], "incomplete")
            sample = session.sample(dataset_id, 0, 0)
            self.assertTrue(sample["cameras"]["wrist"]["missing"])
            self.assertEqual(sample["cameras"]["head"]["frame_id"], 0)
            self.assertIsNotNone(sample["selected_raw_action"])
            with self.assertRaises(ValueError):
                RecordedDataCatalog([Path(directory)]).resolve_dataset_path(Path(directory), "../outside")
            self.assertTrue(incomplete.is_dir())

    def test_large_episode_is_iterated_in_bounded_batches_and_action_schema_is_dynamic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            piper = _write_dataset(
                Path(directory) / "piper",
                spec=_spec(dimensions=2),
                frames=1025,
                save_video=False,
            )
            _write_dataset(
                Path(directory) / "rm2",
                spec=_spec(dimensions=5, roles=("left",)),
                robot_name="rm2",
                frames=8,
                save_video=False,
            )
            session = _DataViewSession([Path(directory)])
            self.addCleanup(session.close)
            datasets = {item["name"]: item["dataset_id"] for item in session.datasets()}
            self.assertEqual(session.episode_metadata(datasets["rm2"], 0)["action_dim"], 5)
            curves = session.curves(datasets["piper"], 0, series=("state", "action"), max_points=16)
            self.assertEqual(len(curves["series"]), 4)
            self.assertTrue(all(len(item["points"]) <= 16 for item in curves["series"]))
            self.assertTrue(piper.is_dir())


if __name__ == "__main__":
    unittest.main()
