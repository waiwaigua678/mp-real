from __future__ import annotations

import unittest

from mp_real.common.camera import BlackCamera
from mp_real.evaluation.service import EvaluationConflict
from mp_real.web.server import CAMERA_NAMES, PiperWebRuntime, _default_args


class WebEvaluationResourceTests(unittest.TestCase):
    def test_camera_preview_keeps_its_config_and_rejects_real_robot_evaluation(self) -> None:
        args = _default_args(camera_profile="black")
        cameras = {name: BlackCamera(name, width=8, height=6) for name in CAMERA_NAMES}
        runtime = PiperWebRuntime(
            args,
            robot_factory=lambda name, config: self.fail("camera preview must not create a robot"),
            policy_client_factory=lambda server_url, api_key, timeout: self.fail(
                "camera preview must not create a policy client"
            ),
            camera_factory=lambda config: cameras,
        )
        self.addCleanup(runtime.disconnect)
        runtime.update_config({"runtime_mode": "camera_preview"})
        runtime.start()
        preview_config = runtime.get_config()

        with self.assertRaises(EvaluationConflict):
            runtime.evaluation_service.create(
                {
                    "name": "preview conflict",
                    "task_name": "pick-place",
                    "planned_episodes": 1,
                    "max_episode_seconds": 5.0,
                }
            )

        self.assertEqual(runtime.get_config(), preview_config)
        self.assertIsNone(runtime.evaluation_service.current())
        self.assertEqual(runtime.status()["runtime_mode"], "camera_preview")


if __name__ == "__main__":
    unittest.main()
