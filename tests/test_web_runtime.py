from __future__ import annotations

import http.client
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from mp_real.common.camera import BlackCamera
from mp_real.pose.models import PoseMoveProgress, PoseMoveResult, PoseValidationReport
from mp_real.robots.piper import infer as infer_piper
from mp_real.runtime.models import ActionSpec, RobotState
from mp_real.web.profiles import PIPER_WEB_PROFILE
from mp_real.web.server import (
    CAMERA_NAMES,
    ApiError,
    PiperWebHandler,
    PiperWebRuntime,
    PiperWebServer,
    _default_args,
)


class _FakeWebRobot:
    action_spec = ActionSpec(14, 14, 6, "rad", CAMERA_NAMES)

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []
        self.reset_count = 0
        self.closed = False
        self.runtime_configs: list[Any] = []

    def read_state(self) -> RobotState:
        return RobotState(np.zeros(14, dtype=np.float32), time.monotonic())

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        executed = np.asarray(target, dtype=np.float32).copy()
        self.executed.append(executed)
        return executed

    def reset(self) -> None:
        self.reset_count += 1

    def configure_runtime(self, config: object) -> None:
        self.runtime_configs.append(config)

    def close(self) -> None:
        self.closed = True


class _FakePoseWebRobot(_FakeWebRobot):
    def __init__(self, action_spec: ActionSpec) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.state = np.zeros(action_spec.state_dim, dtype=np.float32)
        self.pose_stops = 0

    def read_state(self) -> RobotState:
        return RobotState(self.state.copy(), time.monotonic())

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        executed = super().execute_transition(previous, target)
        self.state = executed.copy()
        return executed

    def get_current_pose_state(self) -> RobotState:
        return self.read_state()

    def validate_pose_target(self, target: object) -> PoseValidationReport:
        del target
        return PoseValidationReport()

    def plan_move_to_state(self, plan: object) -> object:
        return plan

    def execute_pose_plan(
        self, plan: object, *, stop_event: threading.Event, on_progress: Any = None
    ) -> PoseMoveResult:
        for waypoint in plan.waypoints:
            if stop_event.is_set():
                return PoseMoveResult(plan.plan_id, "aborted", self.read_state(), None, "stopped")
            self.state = waypoint.target.copy()
            if on_progress is not None:
                on_progress(
                    PoseMoveProgress(
                        plan.plan_id,
                        waypoint.index,
                        len(plan.waypoints),
                        self.state.copy(),
                        waypoint.target.copy(),
                        0.0,
                        time.monotonic_ns(),
                    )
                )
        return PoseMoveResult(plan.plan_id, "reached", self.read_state(), 0.0)

    def stop_pose_motion(self) -> None:
        self.pose_stops += 1

    def verify_target_reached(self, plan: object) -> PoseMoveResult:
        error = float(np.max(np.abs(self.state - plan.target_state)))
        return PoseMoveResult(plan.plan_id, "reached" if error <= 0.05 else "failed", self.read_state(), error)


class _FakeWebPolicy:
    metadata = {"model": "fake"}

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.closed = False
        self.timeout = 3.0
        self.timeout_history: list[float] = []

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        return {"actions": np.zeros((1, 14), dtype=np.float32)}

    def set_timeout(self, timeout_s: float) -> None:
        self.timeout = timeout_s
        self.timeout_history.append(timeout_s)

    def close(self) -> None:
        self.closed = True


class _TrackedCamera(BlackCamera):
    def __init__(self, name: str, *, fail_reads: bool = False) -> None:
        super().__init__(name, width=8, height=6)
        self.fail_reads = fail_reads
        self.closed = False

    def read(self, *, timeout: float = 2.0) -> np.ndarray:
        if self.fail_reads:
            raise RuntimeError(f"{self.name} camera failed")
        return super().read(timeout=timeout)

    def read_frame(self, *, timeout: float = 2.0):
        if self.fail_reads:
            raise RuntimeError(f"{self.name} camera failed")
        return super().read_frame(timeout=timeout)

    def close(self) -> None:
        self.closed = True


class _SlowFirstPolicy(_FakeWebPolicy):
    def __init__(self, delay_s: float, *, actions: list[float] | None = None) -> None:
        super().__init__()
        self.delay_s = delay_s
        self.actions = actions or [0.0]

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        call_index = len(self.observations) - 1
        if call_index == 0:
            time.sleep(self.delay_s)
            if self.delay_s > self.timeout:
                raise TimeoutError(f"fake policy exceeded {self.timeout:.3f}s")
        value = self.actions[min(call_index, len(self.actions) - 1)]
        return {"actions": np.full((1, 14), value, dtype=np.float32)}


class _BlockingWarmupPolicy(_FakeWebPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.released = threading.Event()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.observations.append(observation)
        self.entered.set()
        self.released.wait(timeout=2.0)
        if self.closed:
            raise TimeoutError("fake policy closed during warmup")
        return {"actions": np.zeros((1, 14), dtype=np.float32)}

    def close(self) -> None:
        super().close()
        self.released.set()


def _runtime(args: Any, robot: _FakeWebRobot, policy: _FakeWebPolicy) -> PiperWebRuntime:
    return PiperWebRuntime(
        args,
        robot_factory=lambda name, config: robot,
        policy_client_factory=lambda server_url, api_key, timeout: policy,
        camera_factory=lambda config: {
            name: BlackCamera(name, width=config.camera_width, height=config.camera_height) for name in CAMERA_NAMES
        },
    )


def _wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for runtime state")


class WebRuntimeTests(unittest.TestCase):
    def test_data_view_imports_and_removes_runtime_only_roots_without_hardware(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = _write_dataset(root / "recorded", frames=1, save_video=False)
            (root / "empty").mkdir()
            args = _default_args(camera_profile="black")
            calls = {"robot": 0, "policy": 0, "camera": 0}

            def unexpected_robot(name: str, config: Any) -> _FakeWebRobot:
                del name, config
                calls["robot"] += 1
                self.fail("DATA_VIEW directory import must not create a robot")

            def unexpected_policy(server_url: str, api_key: str | None, timeout: float) -> _FakeWebPolicy:
                del server_url, api_key, timeout
                calls["policy"] += 1
                self.fail("DATA_VIEW directory import must not create a policy client")

            def unexpected_cameras(config: Any) -> dict[str, _TrackedCamera]:
                del config
                calls["camera"] += 1
                self.fail("DATA_VIEW directory import must not create cameras")

            runtime = PiperWebRuntime(
                args,
                robot_factory=unexpected_robot,
                policy_client_factory=unexpected_policy,
                camera_factory=unexpected_cameras,
            )
            self.addCleanup(runtime.disconnect)
            runtime.update_config({"runtime_mode": "data_view"})

            self.assertEqual(runtime.data_view_datasets(), [])
            self.assertFalse(runtime.data_view_status()["ready"])
            imported = runtime.data_view_import_root({"path": str(recorded)})
            self.assertTrue(imported["added"])
            self.assertEqual(imported["root"]["origin"], "web")
            self.assertEqual(imported["root"]["dataset_count"], 1)
            self.assertNotIn(str(recorded), json.dumps(imported))
            self.assertTrue(imported["data_view"]["ready"])
            self.assertEqual(len(runtime.data_view_datasets()), 1)
            self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 0})
            self.assertEqual(runtime.resource_manager.snapshot(), {})

            duplicate = runtime.data_view_import_root({"path": str(recorded)})
            self.assertFalse(duplicate["added"])
            with self.assertRaisesRegex(ApiError, "No readable LeRobot"):
                runtime.data_view_import_root({"path": str(root / "empty")})
            with self.assertRaisesRegex(ApiError, "overlaps"):
                runtime.data_view_import_root({"path": str(root)})

            removed = runtime.data_view_remove_root({"root_id": imported["root"]["root_id"]})
            self.assertTrue(removed["removed"])
            self.assertFalse(removed["data_view"]["ready"])
            self.assertEqual(runtime.data_view_datasets(), [])
            self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 0})
            self.assertEqual(runtime.resource_manager.snapshot(), {})

    def test_data_view_root_replacement_waits_for_an_inflight_browse(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_dataset(root / "first", frames=1, save_video=False)
            second = _write_dataset(root / "second", frames=1, save_video=False)
            runtime = PiperWebRuntime(
                _default_args(camera_profile="black"),
                recorded_data_roots=(first,),
                robot_factory=lambda name, config: self.fail("DATA_VIEW browsing must not create a robot"),
                policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                    "DATA_VIEW browsing must not create a policy client"
                ),
                camera_factory=lambda config: self.fail("DATA_VIEW browsing must not create cameras"),
            )
            self.addCleanup(runtime.disconnect)
            runtime.update_config({"runtime_mode": "data_view"})
            old_viewer = runtime._recorded_data_view
            self.assertIsNotNone(old_viewer)
            assert old_viewer is not None
            entered = threading.Event()
            release = threading.Event()
            closed = threading.Event()
            original_datasets = old_viewer.datasets
            original_close = old_viewer.close

            def blocking_datasets() -> list[dict[str, Any]]:
                entered.set()
                release.wait(timeout=2.0)
                return original_datasets()

            def tracked_close() -> None:
                closed.set()
                original_close()

            old_viewer.datasets = blocking_datasets  # type: ignore[method-assign]
            old_viewer.close = tracked_close  # type: ignore[method-assign]
            browse_errors: list[BaseException] = []

            def browse() -> None:
                try:
                    runtime.data_view_datasets()
                except BaseException as exc:
                    browse_errors.append(exc)

            browse_thread = threading.Thread(target=browse, name="data-view-browse-lease-test", daemon=False)
            browse_thread.start()
            self.assertTrue(entered.wait(timeout=1.0))
            try:
                imported = runtime.data_view_import_root({"path": str(second)})
                self.assertTrue(imported["added"])
                self.assertFalse(closed.is_set(), "the previous catalog closed while a request still used it")
            finally:
                release.set()
                browse_thread.join(timeout=2.0)
            self.assertFalse(browse_thread.is_alive())
            self.assertEqual(browse_errors, [])
            self.assertTrue(closed.is_set())
            self.assertEqual(len(runtime.data_view_datasets()), 2)

    def test_data_view_dataset_import_http_endpoint_is_authenticated_and_hides_path(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = _write_dataset(root / "recorded", frames=1, save_video=False)
            runtime = PiperWebRuntime(
                _default_args(camera_profile="black"),
                robot_factory=lambda name, config: self.fail("dataset import must not create a robot"),
                policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                    "dataset import must not create a policy client"
                ),
                camera_factory=lambda config: self.fail("dataset import must not create cameras"),
            )
            self.addCleanup(runtime.disconnect)
            runtime.update_config({"runtime_mode": "data_view"})
            server = PiperWebServer(("127.0.0.1", 0), PiperWebHandler, runtime, access_key="data-view-key")
            server_thread = threading.Thread(
                target=server.serve_forever,
                name="data-view-import-http-test",
                daemon=False,
            )
            server_thread.start()
            self.addCleanup(server_thread.join, 2.0)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            address, port = server.server_address
            connection = http.client.HTTPConnection(address, port, timeout=2.0)
            self.addCleanup(connection.close)
            body = json.dumps({"path": str(recorded)})

            connection.request("POST", "/api/data-view/datasets", body, {"Content-Type": "application/json"})
            denied = connection.getresponse()
            self.assertEqual(denied.status, 401)
            denied.read()

            connection.request(
                "POST",
                "/api/data-view/datasets",
                body,
                {"Content-Type": "application/json", "X-Motrix-Key": "data-view-key"},
            )
            created = connection.getresponse()
            created_body = json.loads(created.read())
            self.assertEqual(created.status, 201)
            self.assertTrue(created_body["added"])
            self.assertNotIn(str(recorded), json.dumps(created_body))
            self.assertEqual(created_body["root"]["dataset_count"], 1)

            connection.request("GET", "/api/data-view/datasets")
            listed = connection.getresponse()
            listed_body = json.loads(listed.read())
            self.assertEqual(listed.status, 200)
            self.assertEqual(len(listed_body["datasets"]), 1)

            connection.request(
                "POST",
                "/api/data-view/datasets/remove",
                json.dumps({"root_id": created_body["root"]["root_id"]}),
                {"Content-Type": "application/json", "X-Motrix-Key": "data-view-key"},
            )
            removed = connection.getresponse()
            removed_body = json.loads(removed.read())
            self.assertEqual(removed.status, 200)
            self.assertTrue(removed_body["removed"])
            self.assertFalse(removed_body["data_view"]["ready"])

    def test_trajectory_replay_plans_offline_then_runs_without_policy_client(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            args = _default_args(camera_profile="black")
            fake = _FakePoseWebRobot(PIPER_WEB_PROFILE.action_spec_for_args(args))
            _write_dataset(Path(directory) / "recorded", spec=fake.action_spec, frames=1, save_video=False)
            factories = {"robot": 0, "policy": 0}

            def robot_factory(name: str, config: Any) -> _FakePoseWebRobot:
                del name, config
                factories["robot"] += 1
                return fake

            def policy_factory(server_url: str, api_key: str | None, timeout: float) -> _FakeWebPolicy:
                del server_url, api_key, timeout
                factories["policy"] += 1
                return _FakeWebPolicy()

            runtime = PiperWebRuntime(
                args,
                robot_factory=robot_factory,
                policy_client_factory=policy_factory,
                camera_factory=lambda config: {},
                recorded_data_roots=(Path(directory),),
            )
            dataset_id = runtime._recorded_data_view.datasets()[0]["dataset_id"]
            runtime.replay_plan(
                {
                    "dataset_id": dataset_id,
                    "episode_index": 0,
                    "mode": "state",
                    "timing_mode": "fixed",
                    "fps": 100.0,
                    "speed_scale": 0.1,
                }
            )
            _wait_until(lambda: runtime.replay_status()["state"] in {"validated", "error"})
            self.assertEqual(runtime.replay_status()["state"], "validated", runtime.replay_status())
            self.assertEqual(factories, {"robot": 0, "policy": 0})

            runtime.replay_connect()
            _wait_until(lambda: runtime.replay_status()["state"] in {"armed", "error"})
            replay = runtime.replay_status()
            self.assertEqual(replay["state"], "armed", replay)
            self.assertTrue(replay["view_cursor_locked"])
            self.assertEqual(factories, {"robot": 1, "policy": 0})

            runtime.replay_start(replay["plan"]["plan_hash"])
            _wait_until(
                lambda: runtime.replay_status()["state"] in {"completed", "error", "aborted"}
                and not runtime.replay_status()["view_cursor_locked"]
            )
            self.assertEqual(runtime.replay_status()["state"], "completed", runtime.replay_status())
            progress = runtime.replay_status()["progress"]
            self.assertEqual(progress["sent"], 1.0)
            self.assertEqual(progress["feedback"], 1.0)
            self.assertEqual(progress["acknowledged"], 1.0)
            self.assertEqual(progress["displayed"], progress["acknowledged"])
            self.assertFalse(runtime.replay_status()["view_cursor_locked"])
            self.assertGreater(len(fake.executed), 0)
            self.assertEqual(factories["policy"], 0)
            self.assertEqual(runtime.resource_manager.snapshot(), {})

    def test_recorded_pose_handoff_never_resets_and_uses_fresh_policy_start(self) -> None:
        # Reuse the recorder fixture to exercise the state-only target read.
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            args = _default_args(camera_profile="black")
            args.enable_on_start = False
            args.reset_on_start = False
            args.use_rtc = False
            args.replan_steps = 1
            args.max_steps = 1
            args.fps = 1000.0
            spec = PIPER_WEB_PROFILE.action_spec_for_args(args)
            _write_dataset(Path(directory) / "recorded", spec=spec, robot_name="piper", frames=2, save_video=False)
            robot = _FakePoseWebRobot(spec)
            policy = _SlowFirstPolicy(0.0, actions=[9.0, 2.0, 3.0])
            runtime = PiperWebRuntime(
                args,
                robot_factory=lambda name, config: robot,
                policy_client_factory=lambda server_url, api_key, timeout: policy,
                camera_factory=lambda config: {
                    name: BlackCamera(name, width=config.camera_width, height=config.camera_height)
                    for name in CAMERA_NAMES
                },
                recorded_data_roots=(Path(directory),),
            )
            self.addCleanup(runtime.disconnect)
            dataset_id = runtime._recorded_data_view.datasets()[0]["dataset_id"]

            selected = runtime.pose_select({"dataset_id": dataset_id, "episode_index": 0, "sample_index": 0})
            self.assertEqual(selected["phase"], "offline_preflighted")
            runtime.pose_connect()
            _wait_until(lambda: runtime.pose_status()["phase"] in {"awaiting_move_confirmation", "failed"})
            plan = runtime.pose_status()["plan"]
            self.assertIsNotNone(plan)
            runtime.pose_execute(plan["plan_hash"])
            _wait_until(lambda: runtime.pose_status()["phase"] == "reached")
            runtime.pose_prepare_deployment(plan["plan_hash"])
            _wait_until(lambda: runtime.pose_status()["phase"] in {"awaiting_deployment_confirmation", "failed"})
            self.assertEqual(runtime.pose_status()["phase"], "awaiting_deployment_confirmation")
            runtime.pose_start_deployment(plan["plan_hash"])
            _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"}, timeout=5.0)

            self.assertEqual(robot.reset_count, 0)
            self.assertGreaterEqual(len(policy.observations), 3)  # warmup, discarded prefetch, fresh final prefetch
            np.testing.assert_array_equal(
                robot.executed,
                [
                    infer_piper.stabilize_action(
                        np.full(spec.action_dim, 3.0, dtype=np.float32),
                        np.zeros(spec.action_dim, dtype=np.float32),
                        args,
                    )
                ],
            )

    def test_piper_web_runs_once_with_fake_robot_and_policy(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = False
        args.replan_steps = 1
        args.max_steps = 1
        args.fps = 1000.0
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"})
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        status = runtime.status()

        self.assertFalse(status["running"])
        self.assertEqual(status["phase"], "stopped")
        self.assertEqual(status["step"], 1)
        self.assertEqual(len(robot.executed), 1)
        self.assertEqual(len(policy.observations), 2)
        self.assertEqual(len(robot.runtime_configs), 1)
        self.assertEqual(set(policy.observations[0]["images"]), set(CAMERA_NAMES))

        runtime.disconnect()
        self.assertTrue(robot.closed)
        self.assertTrue(policy.closed)

    def test_piper_web_infer_only_uses_controller_without_executing(self) -> None:
        args = _default_args(camera_profile="black")
        args.infer_only = True
        args.infer_only_chunks = 3
        args.replan_steps = 1
        args.max_steps = 2
        args.fps = 1000.0
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"})
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=2.0, raise_on_error=True))
        status = runtime.status()

        self.assertEqual(status["step"], 2)
        self.assertEqual(len(policy.observations), 3)
        self.assertEqual(robot.executed, [])
        self.assertEqual(robot.reset_count, 0)
        runtime.disconnect()

    def test_piper_web_rtc_switch_uses_shared_runtime(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = True
        args.hold_last_action = False
        args.replan_steps = 1
        args.max_steps = 1
        args.fps = 100.0
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "error"}, timeout=5.0)
        controller = runtime._controller
        self.assertIsNotNone(controller)
        self.assertTrue(controller.join(timeout=5.0, raise_on_error=True))

        self.assertEqual(runtime.status()["step"], 1)
        self.assertEqual(len(robot.executed), 1)
        runtime.disconnect()

    def test_camera_preview_creates_only_cameras_and_can_stop_repeatedly(self) -> None:
        args = _default_args(camera_profile="black")
        calls = {"robot": 0, "policy": 0, "camera": 0}
        cameras = {name: _TrackedCamera(name) for name in CAMERA_NAMES}
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: calls.__setitem__("robot", calls["robot"] + 1),
            policy_client_factory=lambda server_url, api_key, timeout: calls.__setitem__("policy", calls["policy"] + 1),
            camera_factory=lambda config: calls.__setitem__("camera", calls["camera"] + 1) or cameras,
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "camera_preview"})

        status = runtime.start()
        self.assertEqual(status["phase"], "previewing")
        _wait_until(lambda: all(runtime.status()["frames"][name]["sequence"] > 0 for name in CAMERA_NAMES))
        self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 1})

        self.assertEqual(runtime.start()["phase"], "previewing")
        runtime.stop(wait=True)
        runtime.stop(wait=True)
        runtime.disconnect()
        self.assertTrue(all(camera.closed for camera in cameras.values()))

    def test_camera_preview_isolates_one_camera_read_failure(self) -> None:
        args = _default_args(camera_profile="black")
        cameras = {
            "cam_head": _TrackedCamera("cam_head", fail_reads=True),
            "cam_left_wrist": _TrackedCamera("cam_left_wrist"),
            "cam_right_wrist": _TrackedCamera("cam_right_wrist"),
        }
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: self.fail("camera preview must not create a robot"),
            policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                "camera preview must not create a policy"
            ),
            camera_factory=lambda config: cameras,
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "camera_preview"})

        runtime.start()
        _wait_until(
            lambda: (
                runtime.status()["frames"]["cam_head"]["error"] is not None
                and runtime.status()["frames"]["cam_left_wrist"]["sequence"] > 0
                and runtime.status()["frames"]["cam_right_wrist"]["sequence"] > 0
            )
        )
        status = runtime.status()
        self.assertGreater(status["frames"]["cam_left_wrist"]["sequence"], 0)
        self.assertGreater(status["frames"]["cam_right_wrist"]["sequence"], 0)
        runtime.disconnect()
        self.assertTrue(all(camera.closed for camera in cameras.values()))
        self.assertIsNone(runtime._camera_thread)

    def test_offline_replay_creates_no_resources(self) -> None:
        args = _default_args(camera_profile="black")
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: self.fail("offline replay must not create a robot"),
            policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                "offline replay must not create a policy"
            ),
            camera_factory=lambda config: self.fail("offline replay must not create cameras"),
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "offline_replay"})

        status = runtime.connect()
        self.assertEqual(status["phase"], "offline_replay")
        self.assertEqual(runtime.start()["phase"], "offline_replay")
        runtime.disconnect()

    def test_data_view_browses_before_connect_without_hardware_resources(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = _default_args(camera_profile="black")
            spec = PIPER_WEB_PROFILE.action_spec_for_args(args)
            _write_dataset(root / "recorded", spec=spec, frames=2, save_video=False)
            calls = {"robot": 0, "policy": 0, "camera": 0}

            def unexpected_robot(name: str, config: Any) -> _FakeWebRobot:
                del name, config
                calls["robot"] += 1
                self.fail("DATA_VIEW browsing must not create a robot")

            def unexpected_policy(server_url: str, api_key: str | None, timeout: float) -> _FakeWebPolicy:
                del server_url, api_key, timeout
                calls["policy"] += 1
                self.fail("DATA_VIEW browsing must not create a policy client")

            def unexpected_cameras(config: Any) -> dict[str, _TrackedCamera]:
                del config
                calls["camera"] += 1
                self.fail("DATA_VIEW browsing must not create cameras")

            runtime = PiperWebRuntime(
                args,
                robot_factory=unexpected_robot,
                policy_client_factory=unexpected_policy,
                camera_factory=unexpected_cameras,
                recorded_data_roots=(root,),
                open_loop_output_root=root / "open-loop-results",
            )
            self.addCleanup(runtime.disconnect)
            runtime.update_config({"runtime_mode": "data_view"})

            before_connect = runtime.data_view_status()
            self.assertTrue(before_connect["ready"])
            self.assertFalse(before_connect["connected"])
            self.assertFalse(before_connect["open_loop_worker_ready"])
            datasets = runtime.data_view_datasets()
            self.assertEqual(len(datasets), 1)
            dataset_id = datasets[0]["dataset_id"]
            metadata = runtime.data_view_episode_metadata(dataset_id, 0)
            self.assertEqual(metadata["action_fields"], list(spec.action_field_names))
            self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 0})
            self.assertEqual(runtime.resource_manager.snapshot(), {})

            connected = runtime.connect()
            self.assertEqual(connected["phase"], "data_view")
            self.assertTrue(connected["data_view"]["connected"])
            self.assertTrue(connected["data_view"]["open_loop_worker_ready"])
            self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 0})
            self.assertEqual(
                set(runtime.resource_manager.snapshot()),
                {"recorded_data:piper:data-view"},
            )
            runtime.disconnect()
            self.assertEqual(runtime.resource_manager.snapshot(), {})

    def test_data_view_open_loop_is_lazy_hardware_free_and_uses_action_field_curve_labels(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = _default_args(camera_profile="black")
            spec = PIPER_WEB_PROFILE.action_spec_for_args(args)
            _write_dataset(root / "recorded", spec=spec, frames=2, save_video=True)

            class _OpenLoopPolicy(_FakeWebPolicy):
                def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
                    self.observations.append(observation)
                    # The open-loop evaluator keeps the production default of
                    # five policy steps per request; its fake must satisfy the
                    # same ActionSpec contract rather than weakening it.
                    return {"actions": np.zeros((5, spec.action_dim), dtype=np.float32)}

            policy = _OpenLoopPolicy()
            calls = {"robot": 0, "camera": 0, "policy": 0}

            def unexpected_robot(name: str, config: Any) -> _FakeWebRobot:
                del name, config
                calls["robot"] += 1
                self.fail("open-loop evaluation must not create a robot")

            def policy_factory(server_url: str, api_key: str | None, timeout: float) -> _FakeWebPolicy:
                self.assertEqual(server_url, "ws://policy.example:8000")
                self.assertEqual(api_key, "evaluation-key")
                self.assertGreater(timeout, 0)
                calls["policy"] += 1
                return policy

            def unexpected_cameras(config: Any) -> dict[str, _TrackedCamera]:
                del config
                calls["camera"] += 1
                self.fail("open-loop evaluation must not create cameras")

            runtime = PiperWebRuntime(
                args,
                robot_factory=unexpected_robot,
                policy_client_factory=policy_factory,
                camera_factory=unexpected_cameras,
                recorded_data_roots=(root,),
                open_loop_output_root=root / "open-loop-results",
            )
            self.addCleanup(runtime.disconnect)
            runtime.update_config({"runtime_mode": "data_view"})
            dataset_id = runtime.data_view_datasets()[0]["dataset_id"]

            queued = runtime.data_view_submit_open_loop(
                {
                    "dataset_id": dataset_id,
                    "episode_index": 0,
                    "policy_url": "ws://policy.example:8000",
                    "policy_label": "fake-policy",
                    "policy_api_key": "evaluation-key",
                }
            )
            self.assertTrue(runtime.status()["data_view"]["connected"])
            self.assertEqual(calls, {"robot": 0, "camera": 0, "policy": 0})
            self.assertTrue(queued["data_view_session_id"])
            self.assertGreater(queued["data_view_generation_id"], 0)

            job_id = queued["job_id"]
            _wait_until(
                lambda: runtime.data_view_open_loop_job(job_id)["state"]
                in {"complete", "partial_error", "cancelled", "error"},
                timeout=10.0,
            )
            job = runtime.data_view_open_loop_job(job_id)
            self.assertEqual(job["state"], "complete", job)
            self.assertGreaterEqual(len(policy.observations), 3)  # warmup plus recorded samples
            self.assertTrue(policy.closed)
            self.assertEqual(calls, {"robot": 0, "camera": 0, "policy": 1})
            self.assertFalse(
                any(key.startswith("policy_client:") for key in runtime.resource_manager.snapshot()),
                runtime.resource_manager.snapshot(),
            )

            report = runtime.data_view_open_loop_report(job_id, 0, include_curves=True)
            curves = report["curves"]
            self.assertEqual(curves[0]["id"], f"prediction.0.{spec.action_field_names[0]}")
            self.assertEqual(curves[0]["label"], f"prediction {spec.action_field_names[0]}")
            self.assertEqual(curves[1]["id"], f"target.0.{spec.action_field_names[0]}")
            self.assertEqual(curves[1]["field_name"], spec.action_field_names[0])
            self.assertEqual(calls["robot"], 0)
            self.assertEqual(calls["camera"], 0)
            runtime.disconnect()
            self.assertEqual(runtime.resource_manager.snapshot(), {})

    def test_data_view_virtual_session_can_arm_existing_replay_lifecycle(self) -> None:
        from tests.test_data_view import _write_dataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = _default_args(camera_profile="black")
            fake = _FakePoseWebRobot(PIPER_WEB_PROFILE.action_spec_for_args(args))
            _write_dataset(root / "recorded", spec=fake.action_spec, frames=1, save_video=False)
            factories = {"robot": 0, "policy": 0, "camera": 0}

            def robot_factory(name: str, config: Any) -> _FakePoseWebRobot:
                del name, config
                factories["robot"] += 1
                return fake

            def unexpected_policy(server_url: str, api_key: str | None, timeout: float) -> _FakeWebPolicy:
                del server_url, api_key, timeout
                factories["policy"] += 1
                self.fail("robot replay must not create a policy client")

            def unexpected_cameras(config: Any) -> dict[str, _TrackedCamera]:
                del config
                factories["camera"] += 1
                self.fail("robot replay must not create cameras")

            runtime = PiperWebRuntime(
                args,
                robot_factory=robot_factory,
                policy_client_factory=unexpected_policy,
                camera_factory=unexpected_cameras,
                recorded_data_roots=(root,),
                open_loop_output_root=root / "open-loop-results",
            )
            self.addCleanup(runtime.disconnect)
            runtime.update_config({"runtime_mode": "data_view"})
            runtime.connect()
            dataset_id = runtime.data_view_datasets()[0]["dataset_id"]

            runtime.replay_plan(
                {
                    "dataset_id": dataset_id,
                    "episode_index": 0,
                    "mode": "state",
                    "timing_mode": "fixed",
                    "fps": 100.0,
                    "speed_scale": 0.1,
                }
            )
            _wait_until(lambda: runtime.replay_status()["state"] in {"validated", "error"})
            self.assertEqual(runtime.replay_status()["state"], "validated", runtime.replay_status())
            self.assertEqual(factories, {"robot": 0, "policy": 0, "camera": 0})

            runtime.replay_connect()
            _wait_until(lambda: runtime.replay_status()["state"] in {"armed", "error"})
            replay = runtime.replay_status()
            self.assertEqual(replay["state"], "armed", replay)
            self.assertTrue(replay["view_cursor_locked"])
            self.assertEqual(factories, {"robot": 1, "policy": 0, "camera": 0})
            leases = runtime.resource_manager.snapshot()
            self.assertIn("recorded_data:piper:data-view", leases)
            self.assertIn("robot_control:piper", leases)

            runtime.replay_stop()
            _wait_until(lambda: not runtime.replay_status()["view_cursor_locked"])
            self.assertEqual(factories["policy"], 0)
            self.assertEqual(factories["camera"], 0)
            runtime.disconnect()
            self.assertEqual(runtime.resource_manager.snapshot(), {})

    def test_deployment_creates_robot_policy_and_cameras(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.max_steps = 1
        args.replan_steps = 1
        args.max_action_step = 0.0
        args.max_joint_step = 0.0
        args.action_smoothing = 0.0
        args.use_rtc = False
        robot = _FakeWebRobot()
        policy = _FakeWebPolicy()
        calls = {"robot": 0, "policy": 0, "camera": 0}
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: calls.__setitem__("robot", calls["robot"] + 1) or robot,
            policy_client_factory=lambda server_url, api_key, timeout: (
                calls.__setitem__("policy", calls["policy"] + 1) or policy
            ),
            camera_factory=lambda config: (
                calls.__setitem__("camera", calls["camera"] + 1)
                or {name: _TrackedCamera(name) for name in CAMERA_NAMES}
            ),
        )
        self.addCleanup(runtime.disconnect)

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] == "stopped")
        self.assertEqual(calls, {"robot": 1, "policy": 1, "camera": 1})

    def test_warmup_discards_its_action_and_prefetches_fresh_first_chunk(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = False
        args.max_steps = 1
        args.replan_steps = 1
        args.max_action_step = 0.0
        args.max_joint_step = 0.0
        args.action_smoothing = 0.0
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03, actions=[9.0, 2.0])
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"policy_inference_timeout_s": 0.01, "policy_warmup_timeout_s": 0.1})

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "warmup_failed"})
        self.assertEqual(runtime.status()["phase"], "stopped")
        np.testing.assert_array_equal(robot.executed, [np.full(14, 2.0, dtype=np.float32)])
        self.assertEqual(len(policy.observations), 2)
        self.assertGreaterEqual(policy.timeout_history[0], 0.1)
        self.assertIn(0.01, policy.timeout_history)
        self.assertIsNotNone(runtime.status()["metrics"]["cold_inference_latency_ms"])
        self.assertIsNotNone(runtime.status()["metrics"]["first_live_inference_latency_ms"])

    def test_warmup_timeout_reports_its_root_cause_without_motion(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.max_steps = 1
        args.replan_steps = 1
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03)
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"policy_warmup_timeout_s": 0.01})

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] == "warmup_failed")
        status = runtime.status()
        self.assertIn("PolicyWarmupTimeout", status["last_error"])
        self.assertEqual(robot.executed, [])

    def test_without_warmup_slow_first_rtc_request_reproduces_the_old_failure(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = True
        args.hold_last_action = False
        args.max_steps = 1
        args.replan_steps = 1
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03)
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config(
            {
                "policy_warmup_enabled": False,
                "policy_prefetch_first_chunk": False,
                "policy_inference_timeout_s": 0.01,
            }
        )

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] == "error", timeout=5.0)
        self.assertIn("TimeoutError: fake policy exceeded", runtime.status()["last_error"])
        self.assertEqual(robot.executed, [])

    def test_rtc_uses_prefetched_chunk_without_waiting_for_cold_inference(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        args.use_rtc = True
        args.hold_last_action = False
        args.max_steps = 1
        args.replan_steps = 1
        args.fps = 100.0
        args.max_action_step = 0.0
        args.max_joint_step = 0.0
        args.action_smoothing = 0.0
        robot = _FakeWebRobot()
        policy = _SlowFirstPolicy(0.03, actions=[8.0, 3.0])
        runtime = _runtime(args, robot, policy)
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"policy_inference_timeout_s": 0.01, "policy_warmup_timeout_s": 0.1})

        runtime.start()
        _wait_until(lambda: runtime.status()["phase"] in {"stopped", "warmup_failed"}, timeout=5.0)
        self.assertEqual(runtime.status()["phase"], "stopped")
        np.testing.assert_array_equal(robot.executed, [np.full(14, 3.0, dtype=np.float32)])

    def test_stop_and_disconnect_during_warmup_close_the_policy_worker(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        robot = _FakeWebRobot()
        policy = _BlockingWarmupPolicy()
        runtime = _runtime(args, robot, policy)

        runtime.start()
        self.assertTrue(policy.entered.wait(timeout=1.0))
        runtime.stop(wait=True)
        _wait_until(lambda: not runtime._startup_active_locked())
        self.assertEqual(robot.executed, [])
        runtime.disconnect()
        self.assertTrue(policy.closed)

    def test_disconnect_during_warmup_releases_the_policy_worker(self) -> None:
        args = _default_args(camera_profile="black")
        args.dry_run = True
        args.enable_on_start = False
        args.reset_on_start = False
        robot = _FakeWebRobot()
        policy = _BlockingWarmupPolicy()
        runtime = _runtime(args, robot, policy)

        runtime.start()
        self.assertTrue(policy.entered.wait(timeout=1.0))
        runtime.disconnect()
        self.assertTrue(policy.closed)
        self.assertFalse(runtime.status()["connected"])
        self.assertFalse(runtime._startup_active_locked())


if __name__ == "__main__":
    unittest.main()
