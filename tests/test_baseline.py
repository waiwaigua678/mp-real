from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from mp_real.evaluation.baseline import BaselineReferenceWriter, BaselineService, BaselineStore
from mp_real.runtime.models import ActionSpec, VectorField


def _spec() -> ActionSpec:
    fields = (
        VectorField("joint_1", "rad", "joint_position"),
        VectorField("gripper", "normalized_0_closed_1_open", "gripper_open_fraction"),
    )
    return ActionSpec(2, 2, 1, "rad", ("head",), state_fields=fields, action_fields=fields)


def _runtime() -> dict:
    return {
        "robot": "fake",
        "server_url": "ws://policy",
        "prompt": "pick",
        "fps": 10.0,
        "replan_steps": 2,
        "max_steps": 20,
        "use_rtc": True,
        "rtc_replan_stride": 2,
        "rtc_prefetch_steps": 1,
        "rtc_exp_weight": 0.25,
        "policy_warmup_enabled": True,
        "policy_warmup_requests": 2,
        "policy_warmup_timeout_s": 5.0,
        "policy_inference_timeout_s": 1.0,
        "camera_roles": ["head"],
        "camera_backend": "black",
        "max_action_step": 0.1,
        "joint_deadband": 0.0,
        "action_smoothing": 0.1,
        "action_spec": _spec().to_dict(),
        "api_key": "must-not-persist",
        "policy_metadata": {"model": "fake"},
    }


def _payload(name: str = "baseline") -> dict:
    return {
        "name": name,
        "robot_name": "fake",
        "task_name": "pick",
        "prompt": "pick",
        "policy_label": "checkpoint-a",
        "planned_episodes": 3,
        "max_episode_duration_s": 30.0,
        "tags": ["smoke"],
    }


class BaselineTests(unittest.TestCase):
    def test_action_spec_round_trip_and_legacy_constructor(self) -> None:
        spec = _spec()
        restored = ActionSpec.from_dict(spec.to_dict())
        self.assertEqual(restored, spec)
        self.assertEqual(restored.action_names, ("joint_1", "gripper"))
        self.assertEqual(restored.arm_count, 1)
        self.assertEqual(restored.gripper_indices, (1,))
        legacy = ActionSpec.from_dict(
            {
                "action_dim": 2,
                "state_dim": 2,
                "joint_dof_per_arm": 1,
                "joint_unit": "rad",
                "state_names": ["state_joint", "state_gripper"],
                "action_names": ["action_joint", "action_gripper"],
            }
        )
        self.assertEqual(legacy.state_names, ("state_joint", "state_gripper"))
        self.assertEqual(legacy.action_names, ("action_joint", "action_gripper"))

    def test_create_clone_diff_compare_and_attach(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = BaselineService(BaselineStore(Path(directory)))
            payload = _payload()
            payload["policy_metadata"] = {"model": "fake", "authorization": "must-not-persist-either"}
            runtime = _runtime()
            runtime["policy_metadata"] = payload["policy_metadata"]
            created = service.create_from_runtime(payload, runtime_config=runtime, git_commit="a" * 40)
            self.assertIsNone(created.checkpoint_hash)
            self.assertIsNone(created.runtime_config.get("api_key"))
            self.assertIsNone(created.policy_metadata.get("authorization"))
            serialized = (Path(directory) / f"{created.baseline_id}.json").read_text()
            self.assertNotIn("must-not-persist", serialized)
            derived = service.clone(
                created.baseline_id,
                {"name": "rtc-off", "rtc_config": {"use_rtc": False}},
                derived_reason="disable RTC only",
            )
            diff = service.diff(created.baseline_id, derived.baseline_id)
            self.assertTrue(any(item.category == "RTC" for item in diff.items))
            snapshot = {
                "evaluation_id": "session-1",
                "state": "COMPLETED",
                "summary": {
                    "completed_episodes": 3,
                    "result_counts": {"SUCCESS": 1, "FAILURE": 1, "INVALID": 1},
                    "failure_reason_counts": {"GRASP_FAILED": 1},
                },
                "episodes": [
                    {
                        "result": "SUCCESS",
                        "started_at_monotonic_ns": 1_000_000_000,
                        "stopped_at_monotonic_ns": 3_000_000_000,
                    },
                    {
                        "result": "FAILURE",
                        "started_at_monotonic_ns": 4_000_000_000,
                        "stopped_at_monotonic_ns": 7_000_000_000,
                    },
                ],
                "recording": {"dataset_root": "recordings/run"},
            }
            attached = service.attach_evaluation(created.baseline_id, snapshot)
            comparison = service.compare((attached.baseline_id, derived.baseline_id))
            rate = comparison["results"][0]["live"]["success_rate"]
            self.assertEqual(rate["numerator"], 1)
            self.assertEqual(rate["denominator"], 2)
            self.assertEqual(rate["sample_size"], 3)
            self.assertEqual(comparison["results"][0]["live"]["average_duration_s"], 2.5)

            evaluation_runtime = _runtime()
            evaluation_runtime["git_commit"] = "c" * 40
            from_evaluation = service.create_from_evaluation(
                {
                    "evaluation_id": "session-2",
                    "state": "COMPLETED",
                    "config": {
                        "name": "original evaluation",
                        "robot_name": "fake",
                        "task_name": "pick",
                        "prompt": "pick",
                        "policy_label": "checkpoint-a",
                        "planned_episodes": 2,
                        "max_episode_seconds": 20.0,
                        "runtime_config_snapshot": evaluation_runtime,
                        "action_spec_snapshot": _spec().to_dict(),
                    },
                    "summary": {"completed_episodes": 0, "result_counts": {}},
                    "recording": {"dataset_root": "recordings/from-evaluation"},
                },
                name="from evaluation",
            )
            self.assertEqual(from_evaluation.name, "from evaluation")
            self.assertEqual(from_evaluation.evaluation_runs[0].evaluation_id, "session-2")

    def test_open_loop_and_background_writer_are_durable_and_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = BaselineService(BaselineStore(root / "baselines"))
            baseline = service.create_from_runtime(_payload(), runtime_config=_runtime(), git_commit="b" * 40)
            result = root / "open-loop"
            result.mkdir()
            (result / "config.json").write_text(
                json.dumps(
                    {
                        "evaluation_id": "open-1",
                        "config_fingerprint": "fingerprint",
                        "source_dataset": "recordings/source",
                        "source_action_spec": _spec().to_dict(),
                        "state_schema": ["joint_1", "gripper"],
                        "episodes": [0, 1],
                        "config": {
                            "target_source": "action",
                            "alignment_mode": "sample_index",
                            "selected_camera_roles": ["head"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (result / "summary.json").write_text(
                json.dumps({"evaluation_id": "open-1", "status": "complete", "valid_prediction_count": 4}),
                encoding="utf-8",
            )
            writer = BaselineReferenceWriter(service, queue_size=2)
            self.addCleanup(writer.close)
            clone_job = writer.submit_clone(
                baseline.baseline_id,
                {"name": "writer-derived"},
                reason="validate asynchronous clone",
            )
            self._wait_for_job(writer, clone_job["job_id"])
            self.assertEqual(writer.job_status(clone_job["job_id"])["state"], "complete")
            job = writer.submit_open_loop(baseline.baseline_id, result)
            self._wait_for_job(writer, job["job_id"])
            self.assertEqual(writer.job_status(job["job_id"])["state"], "complete")
            attached = service.get(baseline.baseline_id)
            self.assertEqual(len(attached.open_loop_runs), 1)
            self.assertEqual(
                attached.open_loop_runs[0].summary["comparison_contract"]["alignment_mode"], "sample_index"
            )

    @staticmethod
    def _wait_for_job(writer: BaselineReferenceWriter, job_id: str) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if writer.job_status(job_id)["finished"]:
                return
            time.sleep(0.01)
        raise AssertionError(f"Timed out waiting for Baseline job {job_id}")


if __name__ == "__main__":
    unittest.main()
