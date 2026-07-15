from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
