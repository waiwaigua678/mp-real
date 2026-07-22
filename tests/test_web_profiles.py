from __future__ import annotations

import tempfile
import threading
import time
import unittest
from typing import Any

import numpy as np

from mp_real.common.camera import BlackCamera
from mp_real.runtime.models import ActionSpec, RobotState
from mp_real.web.profiles import PIPER_WEB_PROFILE, RM2_WEB_PROFILE
from mp_real.web.resources import ResourceLeaseConflict, ResourceLeaseManager, ResourceRequest, ResourceType
from mp_real.web.server import ApiError, PiperWebHandler, PiperWebServer, RobotWebRuntime


class _Robot:
    def __init__(self, action_spec: ActionSpec) -> None:
        self.action_spec = action_spec
        self.closed = False
        self.resets = 0

    def read_state(self) -> RobotState:
        now_ns = time.monotonic_ns()
        return RobotState(
            np.zeros(self.action_spec.state_dim, dtype=np.float32),
            now_ns / 1e9,
            now_ns,
        )

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        return np.asarray(target, dtype=np.float32).copy()

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class _Policy:
    metadata = {"model": "profile-test"}

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.closed = False
        self.timeout = 1.0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        return {"actions": np.zeros((1, self.action_dim), dtype=np.float32)}

    def set_timeout(self, timeout_s: float) -> None:
        self.timeout = timeout_s

    def close(self) -> None:
        self.closed = True


class _TrackedBlackCamera(BlackCamera):
    def __init__(self, name: str) -> None:
        super().__init__(name, width=8, height=6)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RetryCloseRobot(_Robot):
    def __init__(self, action_spec: ActionSpec) -> None:
        super().__init__(action_spec)
        self.close_calls = 0
        self.close_complete = False

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise TimeoutError("fake worker is still stopping")
        self.closed = True
        self.close_complete = True


class _ResetFailRetryCloseRobot(_RetryCloseRobot):
    def reset(self) -> None:
        raise RuntimeError("fake reset failed")


class _SlowCloseRobot(_Robot):
    def __init__(self, action_spec: ActionSpec) -> None:
        super().__init__(action_spec)
        self.close_entered = threading.Event()
        self.release_close = threading.Event()
        self.close_calls = 0
        self.concurrent_closes = 0
        self.max_concurrent_closes = 0
        self.close_complete = False
        self._close_tracking_lock = threading.Lock()

    def close(self) -> None:
        with self._close_tracking_lock:
            self.close_calls += 1
            self.concurrent_closes += 1
            self.max_concurrent_closes = max(self.max_concurrent_closes, self.concurrent_closes)
        self.close_entered.set()
        self.release_close.wait(timeout=2.0)
        with self._close_tracking_lock:
            self.concurrent_closes -= 1
            self.closed = True
            self.close_complete = True


def _wait(predicate: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


class RobotWebProfileTests(unittest.TestCase):
    def _preview_runtime(self, profile: Any) -> tuple[RobotWebRuntime, dict[str, int]]:
        args = profile.default_args()
        if profile.robot_name == "piper":
            args.cam_head_backend = "black"
            args.cam_left_wrist_backend = "black"
            args.cam_right_wrist_backend = "black"
        else:
            args.camera_backend = "black"
        calls = {"robot": 0, "policy": 0, "camera": 0}
        runtime = RobotWebRuntime(
            args,
            profile=profile,
            robot_factory=lambda name, config: calls.__setitem__("robot", calls["robot"] + 1),
            policy_client_factory=lambda server_url, api_key, timeout: calls.__setitem__("policy", calls["policy"] + 1),
            camera_factory=lambda config: calls.__setitem__("camera", calls["camera"] + 1)
            or {
                role: BlackCamera(role, width=8, height=6)
                for role in profile.camera_roles_for_args(config)
            },
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "camera_preview"})
        return runtime, calls

    def test_piper_camera_preview_does_not_create_robot_or_policy(self) -> None:
        runtime, calls = self._preview_runtime(PIPER_WEB_PROFILE)
        runtime.start()
        _wait(lambda: all(frame["sequence"] > 0 for frame in runtime.status()["frames"].values()))
        self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 1})
        self.assertEqual(runtime.status()["camera_roles"], ["cam_head", "cam_left_wrist", "cam_right_wrist"])

    def test_rm2_camera_preview_does_not_create_robot_or_policy(self) -> None:
        runtime, calls = self._preview_runtime(RM2_WEB_PROFILE)
        runtime.start()
        _wait(lambda: all(frame["sequence"] > 0 for frame in runtime.status()["frames"].values()))
        self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 1})
        self.assertEqual(runtime.status()["camera_roles"], ["left_color", "right_color", "head_color"])
        self.assertNotIn("cam_head", runtime.status()["frames"])

    def test_profiles_route_deployment_to_their_own_robot_factory(self) -> None:
        for profile in (PIPER_WEB_PROFILE, RM2_WEB_PROFILE):
            with self.subTest(robot=profile.robot_name):
                args = profile.default_args()
                if profile.robot_name == "piper":
                    args.cam_head_backend = args.cam_left_wrist_backend = args.cam_right_wrist_backend = "black"
                    args.enable_on_start = False
                    args.reset_on_start = False
                else:
                    args.camera_backend = "black"
                    args.reset_on_start = False
                args.dry_run = True
                calls: list[str] = []
                robot = _Robot(profile.action_spec_for_args(args))
                runtime = RobotWebRuntime(
                    args,
                    profile=profile,
                    robot_factory=lambda name, config: calls.append(name) or robot,
                    policy_client_factory=lambda server_url, api_key, timeout: _Policy(robot.action_spec.action_dim),
                    camera_factory=lambda config: {
                        role: BlackCamera(role, width=8, height=6)
                        for role in profile.camera_roles_for_args(config)
                    },
                )
                self.addCleanup(runtime.disconnect)
                runtime.connect()
                _wait(lambda: runtime.status()["connected"] or runtime.status()["phase"] == "error")
                self.assertTrue(runtime.status()["connected"], runtime.status()["last_error"])
                self.assertEqual(calls, [profile.robot_name])

    def test_deployment_initializes_resources_in_profile_specific_order(self) -> None:
        expected_orders = {
            "piper": ["robot", "reset", "policy", "camera"],
            "rm2": ["policy", "camera", "robot", "reset"],
        }
        for profile in (PIPER_WEB_PROFILE, RM2_WEB_PROFILE):
            with self.subTest(robot=profile.robot_name):
                args = profile.default_args()
                if profile.robot_name == "piper":
                    args.cam_head_backend = args.cam_left_wrist_backend = args.cam_right_wrist_backend = "black"
                    args.enable_on_start = False
                    args.reset_on_start = False
                else:
                    args.camera_backend = "black"
                    args.reset_on_start = False
                args.dry_run = True
                events: list[str] = []

                class _OrderRobot(_Robot):
                    def reset(self) -> None:
                        events.append("reset")
                        super().reset()

                robot = _OrderRobot(profile.action_spec_for_args(args))

                def make_robot(name: str, config: Any) -> _Robot:
                    del name, config
                    events.append("robot")
                    return robot

                def make_policy(server_url: str, api_key: str | None, timeout: float) -> _Policy:
                    del server_url, api_key, timeout
                    events.append("policy")
                    return _Policy(robot.action_spec.action_dim)

                def make_cameras(config: Any) -> dict[str, BlackCamera]:
                    events.append("camera")
                    return {
                        role: BlackCamera(role, width=8, height=6)
                        for role in profile.camera_roles_for_args(config)
                    }

                runtime = RobotWebRuntime(
                    args,
                    profile=profile,
                    robot_factory=make_robot,
                    policy_client_factory=make_policy,
                    camera_factory=make_cameras,
                )
                self.addCleanup(runtime.disconnect)

                runtime.connect()
                _wait(lambda: runtime.status()["connected"] or runtime.status()["phase"] == "error")

                self.assertTrue(runtime.status()["connected"], runtime.status()["last_error"])
                self.assertEqual(events, expected_orders[profile.robot_name])
                runtime.disconnect()

    def test_rm2_robot_factory_failure_closes_earlier_resources_and_releases_leases(self) -> None:
        args = RM2_WEB_PROFILE.default_args()
        args.camera_backend = "black"
        args.reset_on_start = False
        policy = _Policy(RM2_WEB_PROFILE.action_spec_for_args(args).action_dim)
        cameras = {
            role: _TrackedBlackCamera(role)
            for role in RM2_WEB_PROFILE.camera_roles_for_args(args)
        }

        def fail_robot_factory(name: str, config: Any) -> _Robot:
            del name, config
            raise RuntimeError("rm2 robot factory failed")

        runtime = RobotWebRuntime(
            args,
            profile=RM2_WEB_PROFILE,
            robot_factory=fail_robot_factory,
            policy_client_factory=lambda server_url, api_key, timeout: policy,
            camera_factory=lambda config: cameras,
        )
        self.addCleanup(runtime.disconnect)

        runtime.connect()
        _wait(lambda: runtime.status()["phase"] == "error")
        status = runtime.status()

        self.assertFalse(status["connected"])
        self.assertIn("RuntimeError: rm2 robot factory failed", status["last_error"])
        self.assertTrue(policy.closed)
        self.assertTrue(all(camera.closed for camera in cameras.values()))
        self.assertEqual(status["resource_leases"], {})

    def test_disconnect_retains_cleanup_ownership_until_robot_close_can_be_retried(self) -> None:
        args = RM2_WEB_PROFILE.default_args()
        args.camera_backend = "black"
        args.reset_on_start = False
        robot = _RetryCloseRobot(RM2_WEB_PROFILE.action_spec_for_args(args))
        policy = _Policy(robot.action_spec.action_dim)
        runtime = RobotWebRuntime(
            args,
            profile=RM2_WEB_PROFILE,
            robot_factory=lambda name, config: robot,
            policy_client_factory=lambda server_url, api_key, timeout: policy,
            camera_factory=lambda config: {
                role: _TrackedBlackCamera(role)
                for role in RM2_WEB_PROFILE.camera_roles_for_args(config)
            },
        )
        self.addCleanup(runtime.disconnect)

        runtime.connect()
        _wait(lambda: runtime.status()["connected"] or runtime.status()["phase"] == "error")
        self.assertTrue(runtime.status()["connected"], runtime.status()["last_error"])

        with self.assertRaises(ApiError):
            runtime.disconnect()

        failed = runtime.status()
        self.assertTrue(failed["cleanup_pending"])
        self.assertFalse(failed["can_connect"])
        self.assertNotEqual(failed["resource_leases"], {})
        self.assertEqual(robot.close_calls, 1)

        disconnected = runtime.disconnect()

        self.assertFalse(disconnected["cleanup_pending"])
        self.assertFalse(disconnected["connected"])
        self.assertEqual(disconnected["resource_leases"], {})
        self.assertEqual(robot.close_calls, 2)
        self.assertTrue(robot.closed)

    def test_connect_failure_retains_robot_until_cleanup_retry_succeeds(self) -> None:
        args = RM2_WEB_PROFILE.default_args()
        args.camera_backend = "black"
        robot = _ResetFailRetryCloseRobot(RM2_WEB_PROFILE.action_spec_for_args(args))
        policy = _Policy(robot.action_spec.action_dim)
        runtime = RobotWebRuntime(
            args,
            profile=RM2_WEB_PROFILE,
            robot_factory=lambda name, config: robot,
            policy_client_factory=lambda server_url, api_key, timeout: policy,
            camera_factory=lambda config: {
                role: _TrackedBlackCamera(role)
                for role in RM2_WEB_PROFILE.camera_roles_for_args(config)
            },
        )
        self.addCleanup(runtime.disconnect)

        runtime.connect()
        _wait(lambda: runtime.status()["phase"] == "cleanup_failed")
        failed = runtime.status()

        self.assertTrue(failed["cleanup_pending"])
        self.assertFalse(failed["connected"])
        self.assertFalse(failed["can_connect"])
        self.assertNotEqual(failed["resource_leases"], {})
        self.assertTrue(policy.closed)
        self.assertEqual(robot.close_calls, 1)

        disconnected = runtime.disconnect()

        self.assertFalse(disconnected["cleanup_pending"])
        self.assertEqual(disconnected["resource_leases"], {})
        self.assertEqual(robot.close_calls, 2)
        self.assertTrue(robot.closed)

    def test_concurrent_disconnect_serializes_deployment_resource_close(self) -> None:
        args = RM2_WEB_PROFILE.default_args()
        args.camera_backend = "black"
        args.reset_on_start = False
        robot = _SlowCloseRobot(RM2_WEB_PROFILE.action_spec_for_args(args))
        runtime = RobotWebRuntime(
            args,
            profile=RM2_WEB_PROFILE,
            robot_factory=lambda name, config: robot,
            policy_client_factory=lambda server_url, api_key, timeout: _Policy(robot.action_spec.action_dim),
            camera_factory=lambda config: {
                role: _TrackedBlackCamera(role)
                for role in RM2_WEB_PROFILE.camera_roles_for_args(config)
            },
        )
        self.addCleanup(runtime.disconnect)
        runtime.connect()
        _wait(lambda: runtime.status()["connected"] or runtime.status()["phase"] == "error")
        self.assertTrue(runtime.status()["connected"], runtime.status()["last_error"])

        errors: list[BaseException] = []

        def disconnect() -> None:
            try:
                runtime.disconnect()
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=disconnect)
        second = threading.Thread(target=disconnect)
        first.start()
        self.assertTrue(robot.close_entered.wait(timeout=1.0))
        second.start()
        time.sleep(0.02)
        robot.release_close.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(robot.close_calls, 1)
        self.assertEqual(robot.max_concurrent_closes, 1)
        self.assertFalse(runtime.status()["cleanup_pending"])
        self.assertEqual(runtime.status()["resource_leases"], {})

    def test_robot_switch_is_rejected_while_deployment_cleanup_is_pending(self) -> None:
        args = RM2_WEB_PROFILE.default_args()
        args.camera_backend = "black"
        args.reset_on_start = False
        robot = _RetryCloseRobot(RM2_WEB_PROFILE.action_spec_for_args(args))
        runtime = RobotWebRuntime(
            args,
            profile=RM2_WEB_PROFILE,
            robot_factory=lambda name, config: robot,
            policy_client_factory=lambda server_url, api_key, timeout: _Policy(robot.action_spec.action_dim),
            camera_factory=lambda config: {
                role: _TrackedBlackCamera(role)
                for role in RM2_WEB_PROFILE.camera_roles_for_args(config)
            },
        )
        self.addCleanup(runtime.disconnect)
        runtime.connect()
        _wait(lambda: runtime.status()["connected"] or runtime.status()["phase"] == "error")
        self.assertTrue(runtime.status()["connected"], runtime.status()["last_error"])
        with self.assertRaises(ApiError):
            runtime.disconnect()
        self.assertTrue(runtime.status()["cleanup_pending"])

        server = object.__new__(PiperWebServer)
        server.runtime = runtime
        server._runtime_switch_lock = threading.Lock()

        with self.assertRaisesRegex(ApiError, "cleanup completes"):
            server.select_robot("piper")

        self.assertIs(server.runtime, runtime)
        runtime.disconnect()

    def test_robot_switch_is_rejected_while_policy_connect_is_in_progress(self) -> None:
        args = RM2_WEB_PROFILE.default_args()
        args.camera_backend = "black"
        args.reset_on_start = False
        entered = threading.Event()
        release = threading.Event()
        robot = _Robot(RM2_WEB_PROFILE.action_spec_for_args(args))

        def make_policy(server_url: str, api_key: str | None, timeout: float) -> _Policy:
            del server_url, api_key, timeout
            entered.set()
            release.wait(timeout=2.0)
            return _Policy(robot.action_spec.action_dim)

        runtime = RobotWebRuntime(
            args,
            profile=RM2_WEB_PROFILE,
            robot_factory=lambda name, config: robot,
            policy_client_factory=make_policy,
            camera_factory=lambda config: {
                role: _TrackedBlackCamera(role)
                for role in RM2_WEB_PROFILE.camera_roles_for_args(config)
            },
        )
        self.addCleanup(runtime.disconnect)
        self.addCleanup(release.set)
        runtime.connect()
        self.assertTrue(entered.wait(timeout=1.0))
        self.assertEqual(runtime.status()["phase"], "connecting")

        server = object.__new__(PiperWebServer)
        server.runtime = runtime
        server._runtime_switch_lock = threading.RLock()

        with self.assertRaisesRegex(ApiError, "active runtime"):
            server.select_robot("piper")

        self.assertIs(server.runtime, runtime)
        release.set()
        _wait(lambda: runtime.status()["connected"] or runtime.status()["phase"] == "error")

    def test_connect_post_and_robot_switch_share_one_server_lock(self) -> None:
        class _BlockingRuntime:
            def __init__(self) -> None:
                self.entered = threading.Event()
                self.release = threading.Event()
                self.connected = False

            def connect(self) -> dict[str, Any]:
                self.entered.set()
                self.release.wait(timeout=2.0)
                self.connected = True
                return self.status()

            def status(self) -> dict[str, Any]:
                return {
                    "cleanup_pending": False,
                    "connected": self.connected,
                    "running": False,
                    "stop_requested": False,
                    "can_stop": self.connected,
                }

            def pose_status(self) -> dict[str, str]:
                return {"phase": "idle"}

        runtime = _BlockingRuntime()
        server = object.__new__(PiperWebServer)
        server.runtime = runtime
        server.access_key = None
        server._runtime_switch_lock = threading.RLock()

        handler = object.__new__(PiperWebHandler)
        handler.server = server
        handler.path = "/api/connect"
        handler.headers = {}
        responses: list[dict[str, Any]] = []
        handler._send_json = lambda payload, status=None: responses.append(payload)

        switch_errors: list[BaseException] = []
        request_thread = threading.Thread(target=handler.do_POST)

        def switch_robot() -> None:
            try:
                server.select_robot("piper")
            except BaseException as exc:
                switch_errors.append(exc)

        switch_thread = threading.Thread(target=switch_robot)
        request_thread.start()
        self.assertTrue(runtime.entered.wait(timeout=1.0))
        switch_thread.start()
        time.sleep(0.02)
        self.assertTrue(switch_thread.is_alive())
        runtime.release.set()
        request_thread.join(timeout=2.0)
        switch_thread.join(timeout=2.0)

        self.assertFalse(request_thread.is_alive())
        self.assertFalse(switch_thread.is_alive())
        self.assertEqual(len(responses), 1)
        self.assertEqual(len(switch_errors), 1)
        self.assertIsInstance(switch_errors[0], ApiError)
        self.assertIs(server.runtime, runtime)

    def test_switch_lock_excludes_request_io_and_does_not_block_stop(self) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.stop_called = threading.Event()

            def connect(self) -> dict[str, bool]:
                return {"connected": True}

            def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
                return payload

            def stop(self, *, wait: bool) -> dict[str, bool]:
                self.stop_called.set()
                return {"wait": wait}

        runtime = _Runtime()
        server = object.__new__(PiperWebServer)
        server.runtime = runtime
        server.access_key = None
        server._runtime_switch_lock = threading.RLock()

        def make_handler(path: str) -> PiperWebHandler:
            handler = object.__new__(PiperWebHandler)
            handler.server = server
            handler.path = path
            handler.headers = {}
            return handler

        read_entered = threading.Event()
        release_read = threading.Event()
        config_handler = make_handler("/api/config")

        def read_json() -> dict[str, float]:
            read_entered.set()
            release_read.wait(timeout=2.0)
            return {"fps": 10.0}

        config_handler._read_json = read_json
        config_handler._send_json = lambda payload, status=None: None
        config_thread = threading.Thread(target=config_handler.do_POST)
        config_thread.start()
        self.assertTrue(read_entered.wait(timeout=1.0))
        self.assertTrue(server._runtime_switch_lock.acquire(timeout=0.2))
        server._runtime_switch_lock.release()
        release_read.set()
        config_thread.join(timeout=2.0)
        self.assertFalse(config_thread.is_alive())

        send_entered = threading.Event()
        release_send = threading.Event()
        connect_handler = make_handler("/api/connect")

        def send_json(payload: dict[str, Any], status: Any = None) -> None:
            del payload, status
            send_entered.set()
            release_send.wait(timeout=2.0)

        connect_handler._send_json = send_json
        connect_thread = threading.Thread(target=connect_handler.do_POST)
        connect_thread.start()
        self.assertTrue(send_entered.wait(timeout=1.0))
        self.assertTrue(server._runtime_switch_lock.acquire(timeout=0.2))
        server._runtime_switch_lock.release()
        release_send.set()
        connect_thread.join(timeout=2.0)
        self.assertFalse(connect_thread.is_alive())

        stop_handler = make_handler("/api/stop")
        stop_handler._send_json = lambda payload, status=None: None
        server._runtime_switch_lock.acquire()
        try:
            stop_thread = threading.Thread(target=stop_handler.do_POST)
            stop_thread.start()
            self.assertTrue(runtime.stop_called.wait(timeout=0.2))
        finally:
            server._runtime_switch_lock.release()
        stop_thread.join(timeout=2.0)
        self.assertFalse(stop_thread.is_alive())

    def test_disconnect_waits_for_reset_runtime_operation(self) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.reset_entered = threading.Event()
                self.release_reset = threading.Event()
                self.disconnect_called = threading.Event()

            def reset_arms(self) -> dict[str, bool]:
                self.reset_entered.set()
                self.release_reset.wait(timeout=2.0)
                return {"reset": True}

            def disconnect(self) -> dict[str, bool]:
                self.disconnect_called.set()
                return {"connected": False}

        runtime = _Runtime()
        server = object.__new__(PiperWebServer)
        server.runtime = runtime
        server.access_key = None
        server._runtime_switch_lock = threading.RLock()

        def make_handler(path: str) -> PiperWebHandler:
            handler = object.__new__(PiperWebHandler)
            handler.server = server
            handler.path = path
            handler.headers = {}
            handler._send_json = lambda payload, status=None: None
            return handler

        reset_thread = threading.Thread(target=make_handler("/api/reset").do_POST)
        disconnect_thread = threading.Thread(target=make_handler("/api/disconnect").do_POST)
        reset_thread.start()
        self.assertTrue(runtime.reset_entered.wait(timeout=1.0))
        disconnect_thread.start()
        self.assertFalse(runtime.disconnect_called.wait(timeout=0.1))
        runtime.release_reset.set()
        reset_thread.join(timeout=2.0)
        disconnect_thread.join(timeout=2.0)

        self.assertFalse(reset_thread.is_alive())
        self.assertFalse(disconnect_thread.is_alive())
        self.assertTrue(runtime.disconnect_called.is_set())

    def test_offline_data_mode_does_not_lease_robot_or_policy(self) -> None:
        runtime, calls = self._preview_runtime(RM2_WEB_PROFILE)
        runtime.update_config({"runtime_mode": "offline_replay"})
        status = runtime.connect()
        self.assertEqual(status["phase"], "offline_replay")
        self.assertEqual(calls, {"robot": 0, "policy": 0, "camera": 0})
        leases = status["resource_leases"]
        self.assertEqual(leases, {"recorded_data:rm2": next(iter(leases.values()))})

    def test_profiles_publish_round_trip_action_specs_and_baseline_categories(self) -> None:
        for profile in (PIPER_WEB_PROFILE, RM2_WEB_PROFILE):
            with self.subTest(robot=profile.robot_name):
                args = profile.default_args()
                spec = profile.action_spec_for_args(args)
                self.assertEqual(ActionSpec.from_dict(spec.to_dict()), spec)
                self.assertGreater(spec.arm_count or 0, 0)
                self.assertTrue(spec.capabilities["supports_gripper"])
                categories = profile.baseline_config_for_args(args)
                self.assertEqual(set(categories), {"camera_config", "robot_config", "safety_config"})
                self.assertEqual(tuple(categories["camera_config"]["roles"]), spec.camera_roles)

    def test_web_baseline_create_and_clone_are_profile_owned_and_hardware_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = PIPER_WEB_PROFILE.default_args()
            args.cam_head_backend = args.cam_left_wrist_backend = args.cam_right_wrist_backend = "black"
            runtime = RobotWebRuntime(args, profile=PIPER_WEB_PROFILE, baseline_root=directory)
            self.addCleanup(runtime.shutdown_baselines)
            self.addCleanup(runtime.disconnect)
            job = runtime.create_baseline(
                {
                    "name": "profile baseline",
                    "robot_name": "piper",
                    "task_name": "pick",
                    "prompt": "pick",
                    "policy_label": "fake-policy",
                    "planned_episodes": 3,
                    "max_episode_duration_s": 30.0,
                }
            )
            _wait(lambda: runtime.baseline_job_status(job["job_id"])["finished"])
            created = runtime.baseline_job_status(job["job_id"])
            self.assertEqual(created["state"], "complete", created)
            baseline = runtime.baseline_service.list()[0]
            self.assertEqual(tuple(baseline.camera_config["roles"]), baseline.camera_roles)
            self.assertIn("transports", baseline.robot_config)
            self.assertIn("max_action_step", baseline.safety_config)

            clone = runtime.clone_baseline(
                baseline.baseline_id,
                {"patch": {"name": "profile baseline derived"}, "derived_reason": "name for inspection"},
            )
            _wait(lambda: runtime.baseline_job_status(clone["job_id"])["finished"])
            self.assertEqual(runtime.baseline_job_status(clone["job_id"])["state"], "complete")


class ResourceLeaseManagerTests(unittest.TestCase):
    def test_second_robot_control_owner_is_rejected(self) -> None:
        manager = ResourceLeaseManager()
        first = manager.acquire("deployment-a", [ResourceRequest(ResourceType.ROBOT_CONTROL, "piper")])
        self.addCleanup(first.release)
        with self.assertRaises(ResourceLeaseConflict):
            manager.acquire("replay-b", [ResourceRequest(ResourceType.ROBOT_CONTROL, "piper")])

    def test_open_loop_policy_lease_does_not_claim_robot_control(self) -> None:
        manager = ResourceLeaseManager()
        lease = manager.acquire("open-loop-evaluation", [ResourceRequest(ResourceType.POLICY_CLIENT, "ws://policy")])
        self.addCleanup(lease.release)
        self.assertIsNone(manager.owner_of(ResourceRequest(ResourceType.ROBOT_CONTROL, "piper")))

    def test_control_conflict_matrix_and_non_control_modes(self) -> None:
        manager = ResourceLeaseManager()
        robot = ResourceRequest(ResourceType.ROBOT_CONTROL, "piper")
        policy = ResourceRequest(ResourceType.POLICY_CLIENT, "ws://policy")
        cameras = ResourceRequest(ResourceType.CAMERAS, "piper:all")
        recorded_data = ResourceRequest(ResourceType.RECORDED_DATA, "dataset-a")

        deployment = manager.acquire("normal-deployment", (robot, policy, cameras))
        for blocked_owner in ("robot-replay", "evaluation", "move-to-state", "second-control-session"):
            with self.subTest(blocked_owner=blocked_owner), self.assertRaises(ResourceLeaseConflict):
                manager.acquire(blocked_owner, (robot,))
        deployment.release()

        replay = manager.acquire("robot-replay", (robot,))
        with self.assertRaises(ResourceLeaseConflict):
            manager.acquire("policy-deployment", (robot, policy))
        with self.assertRaises(ResourceLeaseConflict):
            manager.acquire("second-replay", (robot,))
        replay.release()

        pose = manager.acquire("move-to-state", (robot,))
        for blocked_owner in ("normal-deployment", "evaluation"):
            with self.subTest(blocked_owner=blocked_owner), self.assertRaises(ResourceLeaseConflict):
                manager.acquire(blocked_owner, (robot,))
        pose.release()

        # Viewing Baselines needs no lease. Offline data and camera preview can
        # coexist because they do not request the same hardware resource.
        data_view = manager.acquire("offline-data-view", (recorded_data,))
        preview = manager.acquire("camera-preview", (cameras,))
        self.assertEqual(manager.owner_of(recorded_data), "offline-data-view")
        self.assertEqual(manager.owner_of(cameras), "camera-preview")
        preview.release()
        data_view.release()


if __name__ == "__main__":
    unittest.main()
