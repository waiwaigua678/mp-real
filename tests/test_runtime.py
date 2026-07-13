from __future__ import annotations

import unittest

import numpy as np

from mp_real.common.camera import BlackCamera
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import run_sync_loop
from mp_real.runtime.models import ActionSpec
from mp_real.runtime.observation import capture_observation


class _Policy:
    def infer(self, observation: dict) -> dict:
        del observation
        return {"actions": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}


class _Adapter:
    name = "fake"

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []

    def observe(self) -> dict:
        return {"state": np.zeros(2, dtype=np.float32), "prompt": "test"}

    def decode_action_chunk(self, response: dict, replan_steps: int) -> np.ndarray:
        return ActionSpec(2, 2, 1, "rad", ()).validate_chunk(response["actions"])[:replan_steps]

    def initial_action(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return action

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous
        self.executed.append(target.copy())
        return target.copy()

    def infer_only_metadata(self, observation: dict) -> dict:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


class RuntimeTests(unittest.TestCase):
    def test_sync_loop_consumes_action_chunk(self) -> None:
        adapter = _Adapter()
        config = InferenceLoopConfig(
            fps=1000.0,
            replan_steps=2,
            max_steps=2,
            use_rtc=False,
            rtc_replan_stride=0,
            rtc_prefetch_steps=0,
            rtc_exp_weight=0.0,
            hold_last_action=True,
            infer_only=False,
            infer_only_chunks=1,
            infer_only_output=None,
            prompt="test",
            log_timing=False,
        )
        run_sync_loop(_Policy(), adapter, config)
        np.testing.assert_array_equal(adapter.executed, [[1.0, 2.0], [3.0, 4.0]])

    def test_observation_snapshot_keeps_timestamps(self) -> None:
        snapshot = capture_observation(
            {"head": BlackCamera("head", width=8, height=6)},
            read_state=lambda: np.asarray([1.0, 2.0], dtype=np.float32),
            prompt="test",
            resize_size=4,
            timeout=0.1,
            image_masks={"head": np.bool_(False)},
        )
        self.assertEqual(snapshot.images["head"].image.shape, (4, 4, 3))
        self.assertGreater(snapshot.images["head"].timestamp_monotonic, 0.0)
        self.assertGreater(snapshot.state.timestamp_monotonic, 0.0)
        self.assertEqual(snapshot.to_policy_observation()["prompt"], "test")


if __name__ == "__main__":
    unittest.main()
