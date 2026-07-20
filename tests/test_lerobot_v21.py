from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np
import pyarrow.parquet as pq

from mp_real.data.catalog import RecordedDataCatalog
from mp_real.data.lerobot_v21 import LeRobotV21EpisodeRecorder, LeRobotV21EpisodeSource, validate_lerobot_v21_dataset
from mp_real.data.models import EpisodeRecordingContext, EpisodeStatus, RecorderConfig
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.events import (
    ChunkReceived,
    ControlStepRecorded,
    ObservationCaptured,
    RuntimeEventHooks,
    RuntimeEventIdentity,
)
from mp_real.runtime.inference import run_sync_loop
from mp_real.runtime.models import ActionSpec, CameraSample, ObservationSnapshot, RobotState, VectorField
from mp_real.web.profiles import PIPER_WEB_PROFILE, RM2_WEB_PROFILE


def _spec() -> ActionSpec:
    fields = (
        VectorField("left_joint_1", "rad", "joint_position"),
        VectorField("left_gripper", "normalized_0_open_1", "gripper_open_fraction"),
    )
    return ActionSpec(2, 2, 1, "rad", ("head", "wrist"), state_fields=fields, action_fields=fields)


def _event(event_type, *, episode_id: str, step: int | None = None, payload: dict) -> object:
    return event_type(
        runtime_id="test-runtime",
        session_id="session-1",
        episode_id=episode_id,
        generation_id=1,
        step=step,
        monotonic_timestamp_ns=1_000_000_000 + (step or 0) * 100_000_000,
        payload=payload,
    )


class LeRobotV21Tests(unittest.TestCase):
    def _write_dataset(
        self,
        root: Path,
        *,
        action_spec: ActionSpec | None = None,
        robot_name: str = "fake-dual",
    ) -> Path:
        action_spec = action_spec or _spec()
        recorder = LeRobotV21EpisodeRecorder(
            RecorderConfig(
                dataset_root=root,
                dataset_name="golden",
                robot_name=robot_name,
                fps=10.0,
                action_spec=action_spec,
                queue_size=256,
                session_id="session-1",
            )
        )
        recorder.start()
        for episode_index, task in enumerate(("pick", "place")):
            episode_id = f"episode-{episode_index}"
            recorder.begin_episode(EpisodeRecordingContext(episode_index, episode_id, task, "session-1", 1))
            for frame_index in range(3):
                images = {
                    role: np.full((8, 10, 3), 20 * index + frame_index * 10, dtype=np.uint8)
                    for index, role in enumerate(action_spec.camera_roles)
                }
                selected_action = np.linspace(0.1, 0.9, action_spec.action_dim, dtype=np.float32)
                stabilized_action = selected_action - 0.05
                executed_action = stabilized_action - 0.05
                policy_observation_id = episode_index * 10 + frame_index + 1
                control_observation_id = episode_index * 100 + frame_index + 1
                recorder.emit(
                    _event(
                        ChunkReceived,
                        episode_id=episode_id,
                        payload={
                            "observation_id": policy_observation_id,
                            "raw_action_chunk": selected_action.reshape(1, -1),
                        },
                    )
                )
                recorder.emit(
                    _event(
                        ControlStepRecorded,
                        episode_id=episode_id,
                        step=frame_index,
                        payload={
                            "control_step_id": frame_index,
                            "observation_id": control_observation_id,
                            "policy_observation_id": policy_observation_id,
                            "state": np.full(action_spec.state_dim, frame_index, dtype=np.float32),
                            "images": images,
                            "camera_frame_ids": {
                                role: frame_index + 1 for role in action_spec.camera_roles
                            },
                            "camera_timestamps_ns": {
                                role: 1_000_000_000 for role in action_spec.camera_roles
                            },
                            "max_camera_skew_ns": 0,
                            "chunk_cursor": 0,
                            "selected_raw_action": selected_action,
                            "stabilized_target_action": stabilized_action,
                            "executed_action": executed_action,
                            "action_sent_timestamp_ns": 1_000_000_000 + frame_index,
                        },
                    )
                )
            recorder.end_episode(labels={"result": "SUCCESS", "operator": "tester"})
        self.assertTrue(recorder.stop(timeout=20))
        return root

    def test_recorder_reader_validator_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_dataset(Path(directory) / "golden")
            report = validate_lerobot_v21_dataset(root)
            self.assertTrue(report.valid, report.errors)
            source = LeRobotV21EpisodeSource(root)
            self.assertEqual(source.get_dataset_metadata().status, EpisodeStatus.COMPLETE)
            self.assertEqual(source.get_action_spec().action_field_names, ("left_joint_1", "left_gripper"))
            self.assertEqual([episode.length for episode in source.list_episodes()], [3, 3])
            sample = source.get_sample(1, 2)
            self.assertEqual(sample.frame_index, 2)
            self.assertEqual(sample.images["head"].shape, (8, 10, 3))
            self.assertTrue(np.allclose(sample.action, [0.0, 0.8]))
            source.close()
            catalog = RecordedDataCatalog([directory])
            datasets = catalog.scan()
            self.assertEqual(len(datasets), 1)
            self.assertEqual(len(catalog.list_episodes(result="SUCCESS")), 2)
            with self.assertRaises(ValueError):
                catalog.resolve_dataset_path(directory, "../outside")

    def test_reader_accepts_external_standard_v21_without_mp_real_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_dataset(Path(directory) / "native")
            external = Path(directory) / "external"
            shutil.copytree(root, external)
            shutil.rmtree(external / "meta" / "mp_real")
            shutil.rmtree(external / "telemetry")
            info_path = external / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["features"] = {key: value for key, value in info["features"].items() if not key.startswith("mp_real.")}
            info_path.write_text(json.dumps(info), encoding="utf-8")
            for parquet_file in (external / "data").rglob("*.parquet"):
                table = pq.read_table(parquet_file)
                pq.write_table(
                    table.select([name for name in table.column_names if not name.startswith("mp_real.")]),
                    parquet_file,
                )
            source = LeRobotV21EpisodeSource(external)
            self.assertFalse(source.get_dataset_metadata().is_mp_real)
            self.assertEqual(source.get_sample(0, 0).telemetry, {})
            self.assertTrue(validate_lerobot_v21_dataset(external).valid)

    def test_reader_reuses_a_video_decoder_for_sequential_camera_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_dataset(Path(directory) / "sequential")
            source = LeRobotV21EpisodeSource(root)
            try:
                with mock.patch("mp_real.data.lerobot_v21.av.open", wraps=av.open) as open_video:
                    for frame_index in range(3):
                        frame, rendered = source.get_camera_frame_with_index(0, "head", frame_index)
                        self.assertIsNotNone(frame)
                        self.assertEqual(rendered, frame_index)
                    self.assertEqual(open_video.call_count, 1)

                    frame, rendered = source.get_camera_frame_with_index(0, "head", 1)
                    self.assertIsNotNone(frame)
                    self.assertEqual(rendered, 1)
                    self.assertEqual(open_video.call_count, 2)
            finally:
                source.close()

    def test_piper_and_rm2_action_schemas_are_dynamic_and_ordered(self) -> None:
        piper = PIPER_WEB_PROFILE.action_spec_for_args(PIPER_WEB_PROFILE.default_args())
        rm2_args = RM2_WEB_PROFILE.default_args()
        rm2_args.joint_dof = 7
        rm2 = RM2_WEB_PROFILE.action_spec_for_args(rm2_args)
        self.assertEqual(piper.action_field_names[6], "left_gripper")
        self.assertEqual(piper.action_field_names[7], "right_joint_1")
        self.assertEqual(rm2.action_dim, 16)
        self.assertEqual(rm2.action_field_names[6], "left_joint_7")
        self.assertEqual(rm2.action_field_names[7], "right_joint_1")
        self.assertEqual(rm2.action_field_names[-2:], ("left_gripper", "right_gripper"))

        with tempfile.TemporaryDirectory() as directory:
            for name, action_spec in (("piper", piper), ("rm2", rm2)):
                root = self._write_dataset(Path(directory) / name, action_spec=action_spec, robot_name=name)
                report = validate_lerobot_v21_dataset(root)
                self.assertTrue(report.valid, report.errors)
                source = LeRobotV21EpisodeSource(root)
                self.assertEqual(source.get_action_spec().action_field_names, action_spec.action_field_names)

    def test_openpi_lerobot_loader_reads_generated_standard_dataset(self) -> None:
        openpi_python = Path(os.environ.get("OPENPI_PYTHON", "/home/pc4/0x0219/openpi/.venv/bin/python"))
        if not openpi_python.is_file():
            self.skipTest("OpenPI's pinned LeRobot interpreter is not available")
        with tempfile.TemporaryDirectory() as directory:
            root = self._write_dataset(Path(directory) / "openpi-golden")
            environment = os.environ.copy()
            environment["HF_HOME"] = str(Path(directory) / "huggingface")
            code = """
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import sys
root = sys.argv[1]
metadata = LeRobotDatasetMetadata('mp-real/golden', root=root)
dataset = LeRobotDataset('mp-real/golden', root=root, download_videos=False)
sample = dataset[0]
assert metadata.total_episodes == 2
assert len(dataset) == 6
assert sample['action'].shape[-1] == 2
assert sample['observation.state'].shape[-1] == 2
assert 'observation.images.head' in sample
"""
            result = subprocess.run(
                [str(openpi_python), "-c", code, str(root)],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_queue_full_never_blocks_emit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = LeRobotV21EpisodeRecorder(
                RecorderConfig(Path(directory) / "full", "full", "fake", 10, _spec(), queue_size=1)
            )
            recorder.start()
            recorder.begin_episode(EpisodeRecordingContext(0, "episode", "task"))
            for _ in range(100):
                recorder.emit(_event(ObservationCaptured, episode_id="episode", payload={"observation_id": 1}))
            self.assertGreater(recorder.dropped_event_count, 0)
            recorder.stop(finalize=False)

    def test_shared_runtime_records_executed_actions_with_observation_provenance(self) -> None:
        class Adapter:
            name = "recording-fake"

            def __init__(self) -> None:
                self.step = 0
                self.last_observation_snapshot: ObservationSnapshot | None = None

            def observe(self) -> dict:
                now_ns = 2_000_000_000 + self.step * 100_000_000
                self.last_observation_snapshot = ObservationSnapshot(
                    images={
                        "head": CameraSample(
                            np.full((8, 10, 3), self.step, dtype=np.uint8),
                            now_ns / 1e9,
                            frame_id=self.step + 1,
                            timestamp_monotonic_ns=now_ns,
                        ),
                        "wrist": CameraSample(
                            np.full((8, 10, 3), 50 + self.step, dtype=np.uint8),
                            now_ns / 1e9,
                            frame_id=self.step + 1,
                            timestamp_monotonic_ns=now_ns,
                        ),
                    },
                    image_masks={"head": np.bool_(True), "wrist": np.bool_(True)},
                    state=RobotState(np.asarray([self.step, 0.0], dtype=np.float32), now_ns / 1e9, now_ns),
                    prompt="record",
                    capture_started_ns=now_ns,
                    capture_finished_ns=now_ns,
                )
                self.step += 1
                return self.last_observation_snapshot.to_policy_observation()

            def decode_action_chunk(self, response: dict, replan_steps: int) -> np.ndarray:
                return np.asarray(response["actions"], dtype=np.float32)[:replan_steps]

            def initial_action(self) -> np.ndarray:
                return np.zeros(2, dtype=np.float32)

            def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
                del previous
                return np.asarray(action, dtype=np.float32)

            def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
                del previous
                return np.asarray(target, dtype=np.float32) + np.asarray([0.1, 0.0], dtype=np.float32)

            def infer_only_metadata(self, observation: dict) -> dict:
                del observation
                return {}

            def profile(self, stage: str, elapsed_s: float) -> None:
                del stage, elapsed_s

            def infer_only_interval_s(self) -> float:
                return 0.0

        class Policy:
            def infer(self, observation: dict) -> dict:
                del observation
                return {"actions": np.asarray([[0.5, 0.25], [0.75, 0.5]], dtype=np.float32)}

        with tempfile.TemporaryDirectory() as directory:
            recorder = LeRobotV21EpisodeRecorder(
                RecorderConfig(Path(directory) / "runtime", "runtime", "fake", 10, _spec(), session_id="session")
            )
            recorder.start()
            recorder.begin_episode(EpisodeRecordingContext(0, "episode", "record", "session", 1))
            run_sync_loop(
                Policy(),
                Adapter(),
                InferenceLoopConfig(10, 2, 2, False, 0, 0, 0.0, False, False, 1, None, "record", False),
                hooks=RuntimeEventHooks(recorder, RuntimeEventIdentity("runtime", 1, "session", "episode")),
            )
            recorder.end_episode()
            recorder.stop(timeout=20)
            source = LeRobotV21EpisodeSource(Path(directory) / "runtime")
            self.assertEqual(source.get_length(0), 2)
            self.assertTrue(np.allclose(source.get_sample(0, 0).action, [0.6, 0.25]))
            first = source.get_sample(0, 0, include_images=False)
            second = source.get_sample(0, 1, include_images=False)
            self.assertEqual(first.telemetry["policy_observation_id"], second.telemetry["policy_observation_id"])
            self.assertNotEqual(first.telemetry["control_step_id"], second.telemetry["control_step_id"])
            self.assertGreater(second.telemetry["observation_id"], first.telemetry["observation_id"])
            self.assertFalse(np.array_equal(first.state, second.state))
            metadata = source.get_dataset_metadata().info["mp_real"]
            self.assertTrue(metadata["control_step_aligned"])
            self.assertEqual(metadata["recording_semantics"], "control_step_observation_action")


if __name__ == "__main__":
    unittest.main()
