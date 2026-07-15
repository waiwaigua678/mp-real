from __future__ import annotations

import dataclasses
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.data.lerobot_v21 import validate_lerobot_v21_dataset
from mp_real.evaluation.service import EvaluationConflict, EvaluationRuntimeLease, EvaluationService
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.controller import RuntimeController
from mp_real.runtime.events import SafetyRejected
from mp_real.runtime.models import ActionSpec, ObservationSnapshot, RobotState
from mp_real.runtime.startup import PolicyStartupConfig
from mp_real.web.server import PiperWebHandler, PiperWebServer


class _Robot:
    action_spec = ActionSpec(2, 2, 1, "rad", ())

    def __init__(self) -> None:
        self.executed = 0
        self.closed = False

    def read_state(self) -> RobotState:
        now_ns = time.monotonic_ns()
        return RobotState(np.zeros(2, dtype=np.float32), now_ns / 1e9, now_ns)

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        self.executed += 1
        return np.asarray(target, dtype=np.float32).copy()

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Adapter:
    name = "evaluation-fake"

    def __init__(self, robot: _Robot) -> None:
        self._robot = robot
        self.last_observation_snapshot: ObservationSnapshot | None = None

    def observe(self) -> dict[str, Any]:
        state = self._robot.read_state()
        self.last_observation_snapshot = ObservationSnapshot(
            images={}, image_masks={}, state=state, prompt="evaluation"
        )
        return self.last_observation_snapshot.to_policy_observation()

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        return self._robot.action_spec.validate_chunk(response["actions"])[:replan_steps]

    def initial_action(self) -> np.ndarray:
        return self._robot.read_state().values

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return np.asarray(action, dtype=np.float32).copy()

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        return self._robot.execute_transition(previous, target)

    def infer_only_metadata(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


class _Policy:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.timeout = 1.0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        if self.error is not None:
            raise self.error
        return {"actions": np.asarray([[1.0, 2.0]], dtype=np.float32)}

    def set_timeout(self, timeout_s: float) -> None:
        self.timeout = timeout_s

    def close(self) -> None:
        pass


def _loop_config(prompt: str) -> InferenceLoopConfig:
    return InferenceLoopConfig(
        fps=100.0,
        replan_steps=1,
        max_steps=None,
        use_rtc=False,
        rtc_replan_stride=0,
        rtc_prefetch_steps=0,
        rtc_exp_weight=0.0,
        hold_last_action=True,
        infer_only=False,
        infer_only_chunks=1,
        infer_only_output=None,
        prompt=prompt,
        log_timing=False,
    )


class _Broker:
    def __init__(self, policy: _Policy, *, robot_name: str = "fake") -> None:
        self.robot = _Robot()
        self.robot_name = robot_name
        self.policy = policy
        self.service: EvaluationService | None = None
        self.controller: RuntimeController | None = None
        self.owner: str | None = None
        self.normal_control_running = False

    def acquire_evaluation_control(self, evaluation_id: str) -> EvaluationRuntimeLease:
        if self.normal_control_running:
            raise EvaluationConflict("Normal deployment already owns robot control")
        if self.owner is not None:
            raise EvaluationConflict("Evaluation already owns robot control")
        assert self.service is not None
        if self.controller is None:
            self.controller = RuntimeController(
                self.robot,
                _Adapter(self.robot),
                self.policy,
                _loop_config("evaluation"),
                event_sink=self.service,
            )
        self.owner = evaluation_id
        controller = self.controller

        def release() -> None:
            if self.owner == evaluation_id:
                self.owner = None

        return EvaluationRuntimeLease(
            controller=controller,
            runtime_config_snapshot={"robot": self.robot_name, "prompt": "evaluation", "fps": 100.0},
            action_spec_snapshot=dataclasses.asdict(self.robot.action_spec),
            robot_name=self.robot_name,
            make_adapter=lambda prompt: _Adapter(self.robot),
            make_loop_config=_loop_config,
            make_startup_config=lambda: PolicyStartupConfig(
                warmup_timeout_s=0.5,
                inference_timeout_s=0.5,
            ),
            release=release,
        )

    def start_normal_deployment(self) -> None:
        if self.owner is not None:
            raise EvaluationConflict("Evaluation owns robot control")
        self.normal_control_running = True

    def close(self) -> None:
        if self.controller is not None:
            self.controller.close()


def _wait_for_state(service: EvaluationService, expected: str, *, timeout: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = service.current()
        if status is not None and status["state"] == expected:
            return status
        time.sleep(0.005)
    raise AssertionError(f"Timed out waiting for {expected}; current={service.current()}")


class EvaluationServiceTests(unittest.TestCase):
    def _service(
        self, *, policy: _Policy | None = None, robot_name: str = "fake"
    ) -> tuple[EvaluationService, _Broker]:
        broker = _Broker(policy or _Policy(), robot_name=robot_name)
        service = EvaluationService(broker)
        broker.service = service
        self.addCleanup(broker.close)
        self.addCleanup(service.shutdown)
        return service, broker

    @staticmethod
    def _create(
        service: EvaluationService,
        *,
        robot_name: str | None = None,
        planned_episodes: int = 3,
        max_episode_seconds: float = 1.0,
    ) -> None:
        service.create(
            {
                "name": "three rounds",
                "task_name": "pick-place",
                "planned_episodes": planned_episodes,
                "max_episode_seconds": max_episode_seconds,
                "reset_mode": "manual",
                "result_mode": "manual",
                **({"robot_name": robot_name} if robot_name is not None else {}),
            }
        )

    def _warm_and_reset(self, service: EvaluationService) -> None:
        service.warmup()
        _wait_for_state(service, "WAITING_RESET")
        service.reset_ready()
        _wait_for_state(service, "READY")

    def _run_and_stop(self, service: EvaluationService) -> None:
        service.start_episode()
        _wait_for_state(service, "RUNNING")
        service.stop_episode()
        _wait_for_state(service, "WAITING_RESULT")

    def test_complete_three_episode_manual_flow(self) -> None:
        service, broker = self._service()
        self._create(service)
        self._warm_and_reset(service)

        for index in range(3):
            self._run_and_stop(service)
            status = service.label({"result": "SUCCESS"})
            if index < 2:
                self.assertEqual(status["state"], "WAITING_RESET")
                service.reset_ready()
            else:
                self.assertEqual(status["state"], "COMPLETED")

        status = service.current()
        assert status is not None
        self.assertEqual(status["summary"]["completed_episodes"], 3)
        self.assertEqual(status["summary"]["success_rate"], 1.0)
        self.assertIsNone(broker.owner)
        self.assertEqual(service.complete()["state"], "COMPLETED")

    def test_piper_and_rm2_fake_robots_complete_three_labeled_episodes(self) -> None:
        for robot_name in ("piper", "rm2"):
            with self.subTest(robot_name=robot_name):
                service, _ = self._service(robot_name=robot_name)
                self._create(service, robot_name=robot_name)
                self._warm_and_reset(service)
                for index in range(3):
                    self._run_and_stop(service)
                    status = service.label({"result": "SUCCESS"})
                    if index < 2:
                        self.assertEqual(status["state"], "WAITING_RESET")
                        service.reset_ready()
                self.assertEqual(service.current_or_raise()["state"], "COMPLETED")

    def test_safety_abort_stops_before_manual_result(self) -> None:
        service, broker = self._service()
        self._create(service, planned_episodes=1)
        self._warm_and_reset(service)
        service.start_episode()
        running = _wait_for_state(service, "RUNNING")
        assert broker.controller is not None
        active = running["active_episode"]
        service.emit(
            SafetyRejected(
                runtime_id=broker.controller.runtime_id,
                session_id=running["session_id"],
                episode_id=active["episode_id"],
                generation_id=active["generation_id"],
            )
        )
        status = _wait_for_state(service, "WAITING_RESULT")
        self.assertEqual(status["current_episode"]["stop_trigger"], "SAFETY_ABORT")
        self.assertFalse(broker.controller.status().running)

    def test_warmup_failure_enters_error_without_actions(self) -> None:
        service, broker = self._service(policy=_Policy(error=RuntimeError("policy unavailable")))
        self._create(service)
        service.warmup()
        status = _wait_for_state(service, "ERROR")

        self.assertIn("policy unavailable", status["error"])
        self.assertEqual(broker.robot.executed, 0)
        self.assertIsNone(broker.owner)

    def test_episode_timeout_stops_before_waiting_for_manual_result(self) -> None:
        service, _ = self._service()
        self._create(service, planned_episodes=1, max_episode_seconds=0.03)
        self._warm_and_reset(service)
        service.start_episode()
        status = _wait_for_state(service, "WAITING_RESULT")

        self.assertEqual(status["active_episode"]["stop_trigger"], "TIMEOUT")
        self.assertEqual(status["active_episode"]["result"], None)

    def test_manual_abort(self) -> None:
        service, broker = self._service()
        self._create(service)
        self._warm_and_reset(service)

        status = service.abort()
        self.assertEqual(status["state"], "ABORTED")
        self.assertIsNone(broker.owner)

    def test_invalid_is_excluded_from_success_rate(self) -> None:
        service, _ = self._service()
        self._create(service, planned_episodes=2)
        self._warm_and_reset(service)
        self._run_and_stop(service)
        service.label({"result": "INVALID"})
        service.reset_ready()
        self._run_and_stop(service)
        status = service.label({"result": "SUCCESS"})

        self.assertEqual(status["state"], "COMPLETED")
        self.assertEqual(status["summary"]["success_rate_denominator"], 1)
        self.assertEqual(status["summary"]["success_rate"], 1.0)

    def test_duplicate_label_is_rejected(self) -> None:
        service, _ = self._service()
        self._create(service, planned_episodes=2)
        self._warm_and_reset(service)
        self._run_and_stop(service)
        service.label({"result": "SUCCESS"})

        with self.assertRaises(EvaluationConflict) as raised:
            service.label({"result": "SUCCESS"})
        self.assertEqual(raised.exception.legal_operations, ("reset-ready", "abort"))

    def test_two_concurrent_clients_only_accept_one_label(self) -> None:
        service, _ = self._service()
        self._create(service, planned_episodes=2)
        self._warm_and_reset(service)
        self._run_and_stop(service)
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def submit_label() -> None:
            barrier.wait()
            try:
                service.label({"result": "SUCCESS"})
                outcomes.append("accepted")
            except EvaluationConflict:
                outcomes.append("conflict")

        clients = [threading.Thread(target=submit_label, daemon=False) for _ in range(2)]
        for client in clients:
            client.start()
        for client in clients:
            client.join(timeout=1.0)
        self.assertCountEqual(outcomes, ["accepted", "conflict"])
        self.assertEqual(service.current_or_raise()["summary"]["completed_episodes"], 1)

    def test_duplicate_start_is_rejected_without_second_generation(self) -> None:
        service, broker = self._service()
        self._create(service)
        self._warm_and_reset(service)
        service.start_episode()
        with self.assertRaises(EvaluationConflict):
            service.start_episode()
        status = _wait_for_state(service, "RUNNING")
        assert broker.controller is not None
        self.assertEqual(status["active_episode"]["generation_id"], broker.controller.status().generation_id)
        service.stop_episode()
        _wait_for_state(service, "WAITING_RESULT")

    def test_recorder_flush_failure_marks_system_error_and_excludes_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broker = _Broker(_Policy())
            service = EvaluationService(broker, recording_root=directory)
            broker.service = service
            self.addCleanup(broker.close)
            self.addCleanup(service.shutdown)
            service.create(
                {
                    "name": "flush failure",
                    "task_name": "pick-place",
                    "planned_episodes": 1,
                    "max_episode_seconds": 1.0,
                    "save_data": True,
                    "save_video": False,
                }
            )
            self._warm_and_reset(service)
            self._run_and_stop(service)
            recorder = service._recorder
            assert recorder is not None
            recorder.flush = lambda *, timeout=5.0: False  # type: ignore[method-assign]
            service.label({"result": "SUCCESS"})
            status = _wait_for_state(service, "ERROR")
            self.assertIn("Recorder did not flush", status["last_error"])
            self.assertEqual(status["summary"]["success_rate_denominator"], 0)
            service.shutdown()

    def test_save_data_finalizes_lerobot_dataset_off_the_control_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broker = _Broker(_Policy())
            service = EvaluationService(broker, recording_root=directory)
            broker.service = service
            self.addCleanup(broker.close)
            self.addCleanup(service.shutdown)
            service.create(
                {
                    "name": "recorded evaluation",
                    "task_name": "pick-place",
                    "planned_episodes": 1,
                    "max_episode_seconds": 1.0,
                    "save_data": True,
                    "save_video": False,
                }
            )
            self._warm_and_reset(service)
            self._run_and_stop(service)
            status = service.label({"result": "SUCCESS"})
            self.assertEqual(status["state"], "SAVING")
            status = _wait_for_state(service, "COMPLETED", timeout=5.0)
            deadline = time.monotonic() + 5.0
            datasets = []
            while time.monotonic() < deadline:
                datasets = [path for path in Path(directory).iterdir() if not path.name.endswith(".inprogress")]
                if datasets:
                    break
                time.sleep(0.01)
            self.assertEqual(
                len(datasets),
                1,
                {"status": service.current(), "entries": [path.name for path in Path(directory).iterdir()]},
            )
            report = validate_lerobot_v21_dataset(datasets[0], check_videos=False)
            self.assertTrue(report.valid, report.errors)
            label_path = datasets[0] / "meta/mp_real/episode_labels.jsonl"
            labels = [json.loads(line) for line in label_path.read_text().splitlines()]
            self.assertEqual(labels[0]["result"], "SUCCESS")
            self.assertEqual(labels[0]["prompt"], "evaluation")
            info = json.loads((datasets[0] / "meta/info.json").read_text())
            evaluation_snapshot = info["mp_real"]["runtime_config"]["evaluation"]
            self.assertEqual(evaluation_snapshot["task_name"], "pick-place")
            self.assertEqual(evaluation_snapshot["action_spec_snapshot"]["action_dim"], 2)

    def test_save_data_abort_without_episode_keeps_recovery_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broker = _Broker(_Policy())
            service = EvaluationService(broker, recording_root=directory)
            broker.service = service
            self.addCleanup(broker.close)
            self.addCleanup(service.shutdown)
            service.create(
                {
                    "name": "aborted recording",
                    "task_name": "pick-place",
                    "planned_episodes": 1,
                    "max_episode_seconds": 1.0,
                    "save_data": True,
                }
            )
            self.assertEqual(service.abort()["state"], "ABORTED")
            deadline = time.monotonic() + 2.0
            recovery_files: list[Path] = []
            while time.monotonic() < deadline:
                recovery_files = list(Path(directory).glob("*.inprogress/meta/mp_real/recovery.json"))
                if recovery_files:
                    break
                time.sleep(0.01)
            self.assertEqual(len(recovery_files), 1)
            self.assertEqual(json.loads(recovery_files[0].read_text())["status"], "INCOMPLETE")

    def test_illegal_transition_reports_current_legal_operations(self) -> None:
        service, _ = self._service()
        self._create(service)

        with self.assertRaises(EvaluationConflict) as raised:
            service.reset_ready()
        self.assertEqual(raised.exception.legal_operations, ("warmup", "abort"))

    def test_normal_deployment_and_evaluation_control_are_mutually_exclusive(self) -> None:
        service, broker = self._service()
        self._create(service)
        with self.assertRaises(EvaluationConflict):
            broker.start_normal_deployment()

        service.abort()
        broker.start_normal_deployment()
        self.assertTrue(broker.normal_control_running)

    def test_http_illegal_transition_returns_409_with_legal_operations(self) -> None:
        service, _ = self._service()

        class _ApiRuntime:
            evaluation_service = service

        server = PiperWebServer(("127.0.0.1", 0), PiperWebHandler, _ApiRuntime())
        server_thread = threading.Thread(target=server.serve_forever, name="evaluation-api-test", daemon=False)
        server_thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server_thread.join)
        self.addCleanup(server.shutdown)
        address, port = server.server_address
        connection = http.client.HTTPConnection(address, port, timeout=2.0)
        self.addCleanup(connection.close)

        payload = {
            "name": "http state",
            "task_name": "pick-place",
            "planned_episodes": 1,
            "max_episode_seconds": 1.0,
        }
        connection.request("POST", "/api/evaluations", json.dumps(payload), {"Content-Type": "application/json"})
        created = connection.getresponse()
        self.assertEqual(created.status, 200)
        created.read()

        connection.request("GET", "/api/evaluations/current")
        restored = connection.getresponse()
        restored_body = json.loads(restored.read())
        self.assertEqual(restored.status, 200)
        self.assertEqual(restored_body["evaluation"]["state"], "PREPARING")
        self.assertEqual(restored_body["evaluation"]["session_id"], restored_body["evaluation"]["evaluation_id"])

        connection.request("POST", "/api/evaluations/current/reset-ready", "{}", {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        self.assertEqual(response.status, 409)
        self.assertEqual(body["legal_operations"], ["warmup", "abort"])


if __name__ == "__main__":
    unittest.main()
