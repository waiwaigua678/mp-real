from __future__ import annotations

import tempfile
import time
import unittest
from typing import Any

import numpy as np

from mp_real.common.camera import BlackCamera
from mp_real.runtime.models import ActionSpec, RobotState
from mp_real.web.profiles import PIPER_WEB_PROFILE, RM2_WEB_PROFILE
from mp_real.web.resources import ResourceLeaseConflict, ResourceLeaseManager, ResourceRequest, ResourceType
from mp_real.web.server import RobotWebRuntime


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
