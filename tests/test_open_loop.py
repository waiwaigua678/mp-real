from __future__ import annotations

import inspect
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from mp_real.data.models import FakeRecordedEpisodeSource, RecordedSample
from mp_real.evaluation.open_loop.alignment import ActionAlignment
from mp_real.evaluation.open_loop.evaluator import OpenLoopEvaluator
from mp_real.evaluation.open_loop.jobs import OpenLoopEvaluationJobManager, OpenLoopJobState
from mp_real.evaluation.open_loop.metrics import compute_open_loop_metrics
from mp_real.evaluation.open_loop.models import (
    AlignmentMode,
    OpenLoopEvaluationConfig,
)
from mp_real.evaluation.open_loop.source import OpenLoopInputError
from mp_real.runtime.models import ActionSpec, VectorField


def _spec() -> ActionSpec:
    fields = (
        VectorField("joint_1", "rad", "joint_position"),
        VectorField("gripper", "normalized_0_closed_1_open", "gripper_open_fraction"),
    )
    return ActionSpec(
        action_dim=2,
        state_dim=2,
        joint_dof_per_arm=1,
        joint_unit="rad",
        camera_roles=("head",),
        state_fields=fields,
        action_fields=fields,
    )


def _samples(*, count: int = 3, image: np.ndarray | None = None, state_dim: int = 2, action_dim: int = 2):
    image = np.full((5, 7, 3), 10, dtype=np.uint8) if image is None else image
    return tuple(
        RecordedSample(
            episode_index=0,
            frame_index=index,
            index=index,
            timestamp=index * 0.1,
            task_index=0,
            state=np.full(state_dim, index, dtype=np.float32),
            action=np.asarray((index, float(index % 2)), dtype=np.float32)[:action_dim],
            images={"head": image},
            telemetry={"chunk_cursor": 0},
        )
        for index in range(count)
    )


class _Policy:
    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.metadata = {"name": "fake"}
        self.calls = 0
        self.delay_s = delay_s
        self.closed = False

    def infer(self, observation: dict) -> dict:
        del observation
        if self.delay_s:
            threading.Event().wait(self.delay_s)
        value = float(self.calls)
        self.calls += 1
        return {"actions": np.full((2, 2), value, dtype=np.float32)}

    def set_timeout(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s

    def close(self) -> None:
        self.closed = True


class OpenLoopEvaluatorTests(unittest.TestCase):
    def _source(self, samples: tuple[RecordedSample, ...] | None = None) -> FakeRecordedEpisodeSource:
        return FakeRecordedEpisodeSource(_spec(), {0: samples or _samples()})

    def _config(self, root: Path, **changes: object) -> OpenLoopEvaluationConfig:
        values: dict[str, object] = {
            "dataset": root / "source",
            "episode_indices": (0,),
            "policy_url": "ws://fake",
            "policy_label": "fake-policy",
            "output_dir": root / "result",
            "prompt_override": "pick",
            "replan_steps": 2,
        }
        values.update(changes)
        return OpenLoopEvaluationConfig(**values)  # type: ignore[arg-type]

    def test_teacher_forced_warmup_is_discarded_and_tail_is_masked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = _Policy()
            source = self._source()
            result = OpenLoopEvaluator(
                self._config(root), source=source, policy_factory=lambda *args: policy
            ).run()
            self.assertEqual(result.status, "complete")
            with np.load(root / "result" / "predictions" / "episode_000000.npz") as output:
                # call 0 is warmup; the first persisted live chunk is call 1.
                self.assertTrue(np.allclose(output["predicted_chunks"][0], 1.0))
                self.assertTrue(output["valid_mask"][2, 0])
                self.assertFalse(output["valid_mask"][2, 1])
                self.assertEqual(output["target_sample_index"][0, 1], 1)
            self.assertTrue(policy.closed)

    def test_all_alignment_modes_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = _samples()
            sample = ActionAlignment(samples, self._config(root), fps=10)
            self.assertEqual(sample.align(1, 1).target_sample_index, 2)
            timestamp_config = self._config(
                root,
                alignment_mode=AlignmentMode.TIMESTAMP_ALIGNMENT,
                max_timestamp_error_s=0.001,
            )
            timestamp = ActionAlignment(samples, timestamp_config, fps=10)
            self.assertTrue(timestamp.align(1, 1).valid)
            absolute_config = self._config(
                root,
                alignment_mode=AlignmentMode.ABSOLUTE_CONTROL_STEP_ALIGNMENT,
                allow_frame_index_as_control_step=True,
            )
            absolute = ActionAlignment(samples, absolute_config, fps=10)
            self.assertEqual(absolute.align(1, 1).target_sample_index, 2)
            with self.assertRaises(OpenLoopInputError):
                ActionAlignment(
                    samples,
                    self._config(root, alignment_mode=AlignmentMode.ABSOLUTE_CONTROL_STEP_ALIGNMENT),
                    fps=10,
                )

    def test_formal_input_rejects_missing_state_and_target_and_marks_video_only_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_state = list(_samples())
            missing_state[0] = RecordedSample(
                **{**missing_state[0].__dict__, "state": np.zeros(1, dtype=np.float32)}
            )
            evaluator = OpenLoopEvaluator(self._config(root), source=self._source(tuple(missing_state)))
            status = evaluator.preview_status(0)
            self.assertTrue(status["incomplete_observation"])
            self.assertFalse(status["formal_action_metrics"])
            missing_target = list(_samples())
            missing_target[0] = RecordedSample(
                **{**missing_target[0].__dict__, "action": np.empty(0, dtype=np.float32)}
            )
            with self.assertRaises(OpenLoopInputError):
                from mp_real.evaluation.open_loop.source import resolve_target

                resolve_target(self._source(tuple(missing_target)), missing_target[0], self._config(root))
            video_only = list(_samples(image=None))
            video_only[0] = RecordedSample(
                **{**video_only[0].__dict__, "images": {"head": None}}
            )
            status = OpenLoopEvaluator(self._config(root), source=self._source(tuple(video_only))).preview_status(0)
            self.assertTrue(status["incomplete_observation"])

    def test_dynamic_action_spec_gripper_and_latency_metrics(self) -> None:
        spec = _spec()
        predicted = np.asarray([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 1.0]]], dtype=np.float32)
        targets = predicted.copy()
        valid = np.ones((3, 1), dtype=bool)
        metrics = compute_open_loop_metrics(
            predicted,
            targets,
            valid,
            target_indices=np.asarray([[0], [1], [2]]),
            source_timestamps=np.asarray([0.0, 0.1, 0.2]),
            target_timestamps=np.asarray([[0.0], [0.1], [0.2]]),
            action_spec=spec,
        )
        self.assertEqual(metrics["per_dimension"][0]["name"], "joint_1")
        self.assertEqual(metrics["gripper"]["classification_accuracy"], 1.0)
        self.assertEqual(metrics["horizons"][0]["valid_sample_count"], 3)

    def test_source_arrays_are_not_modified_and_open_loop_module_has_no_robot_import(self) -> None:
        samples = _samples()
        before_state = samples[0].state.copy()
        before_image = np.asarray(samples[0].images["head"]).copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            OpenLoopEvaluator(
                self._config(root), source=self._source(samples), policy_factory=lambda *args: _Policy()
            ).run()
        np.testing.assert_array_equal(samples[0].state, before_state)
        np.testing.assert_array_equal(samples[0].images["head"], before_image)
        import mp_real.evaluation.open_loop.evaluator as evaluator_module

        self.assertNotIn("mp_real.robots", inspect.getsource(evaluator_module))

    def test_policy_results_are_isolated_and_queued_job_can_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = OpenLoopEvaluationJobManager(
                root,
                source_factory=lambda path: self._source(),
                policy_factory=lambda *args: _Policy(delay_s=0.02),
            )
            try:
                first = manager.submit(self._config(root, policy_label="a"))
                second = manager.submit(self._config(root, policy_label="b"))
                self.assertNotEqual(first["output_dir"], second["output_dir"])
                stopped = manager.stop(second["job_id"])
                self.assertEqual(stopped["state"], OpenLoopJobState.CANCELLED.value)
            finally:
                manager.close(timeout=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
