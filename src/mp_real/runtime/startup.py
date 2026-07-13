from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

import numpy as np

from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import (
    ActionDecodeError,
    InferenceAdapter,
    InferenceHooks,
    PolicyClient,
    PolicyProtocolError,
    fetch_action_chunk,
)


class PolicyStartupError(RuntimeError):
    """A policy could not be made safe to start the control loop."""


class PolicyWarmupTimeout(PolicyStartupError):
    pass


class PolicyInferenceTimeout(PolicyStartupError):
    pass


class PolicyStartupCancelled(PolicyStartupError):
    pass


@dataclasses.dataclass(frozen=True)
class PolicyStartupConfig:
    warmup_enabled: bool = True
    warmup_requests: int = 1
    warmup_timeout_s: float = 60.0
    inference_timeout_s: float = 3.0
    prefetch_first_chunk: bool = True

    def validate(self) -> None:
        if self.warmup_requests <= 0:
            raise ValueError("policy_warmup_requests must be positive")
        if self.warmup_timeout_s <= 0:
            raise ValueError("policy_warmup_timeout_s must be positive")
        if self.inference_timeout_s <= 0:
            raise ValueError("policy_inference_timeout_s must be positive")


@dataclasses.dataclass(frozen=True)
class PolicyStartupMetrics:
    cold_inference_latency_ms: float | None
    warmup_latency_ms: float | None
    first_live_inference_latency_ms: float | None


@dataclasses.dataclass(frozen=True)
class PreparedPolicyStart:
    initial_chunk: np.ndarray | None
    metrics: PolicyStartupMetrics


def _set_client_timeout(client: PolicyClient, timeout_s: float) -> None:
    set_timeout = getattr(client, "set_timeout", None)
    if callable(set_timeout):
        set_timeout(timeout_s)
    elif hasattr(client, "timeout"):
        setattr(client, "timeout", timeout_s)


class PolicyStartupCoordinator:
    """Warm a policy and prepare a fresh chunk without executing robot actions."""

    def __init__(
        self,
        client: PolicyClient,
        adapter: InferenceAdapter,
        loop_config: InferenceLoopConfig,
        startup_config: PolicyStartupConfig,
        *,
        hooks: InferenceHooks,
        stop_requested: Callable[[], bool],
        on_phase: Callable[[str], None] | None = None,
    ) -> None:
        startup_config.validate()
        self._client = client
        self._adapter = adapter
        self._loop_config = loop_config
        self._startup_config = startup_config
        self._hooks = hooks
        self._stop_requested = stop_requested
        self._on_phase = on_phase

    def prepare(self) -> PreparedPolicyStart:
        cold_inference_latency_ms: float | None = None
        warmup_started_ns: int | None = None
        warming_up = False
        try:
            if self._startup_config.warmup_enabled:
                self._notify_phase("WARMING_UP")
                warming_up = True
                warmup_started_ns = time.monotonic_ns()
                self._hooks.on_policy_warmup_started(self._startup_config.warmup_requests)
                _set_client_timeout(self._client, self._startup_config.warmup_timeout_s)
                for _ in range(self._startup_config.warmup_requests):
                    elapsed_s = self._request("warmup", self._startup_config.warmup_timeout_s)
                    if cold_inference_latency_ms is None:
                        cold_inference_latency_ms = elapsed_s * 1000.0
                warmup_elapsed_s = (time.monotonic_ns() - warmup_started_ns) / 1e9
                self._hooks.on_policy_warmup_finished(warmup_elapsed_s)
                warming_up = False
            warmup_latency_ms = (
                (time.monotonic_ns() - warmup_started_ns) / 1e6 if warmup_started_ns is not None else None
            )

            self._raise_if_stopped()
            initial_chunk: np.ndarray | None = None
            first_live_inference_latency_ms: float | None = None
            if self._startup_config.prefetch_first_chunk and not self._loop_config.infer_only:
                self._notify_phase("PREFETCHING_FIRST_CHUNK")
                _set_client_timeout(self._client, self._startup_config.inference_timeout_s)
                fetched, elapsed_s = self._request_with_chunk("first_live", self._startup_config.inference_timeout_s)
                initial_chunk = fetched.chunk.copy()
                first_live_inference_latency_ms = elapsed_s * 1000.0
                if cold_inference_latency_ms is None:
                    cold_inference_latency_ms = first_live_inference_latency_ms

            _set_client_timeout(self._client, self._startup_config.inference_timeout_s)
            self._raise_if_stopped()
            prepared = PreparedPolicyStart(
                initial_chunk=initial_chunk,
                metrics=PolicyStartupMetrics(
                    cold_inference_latency_ms=cold_inference_latency_ms,
                    warmup_latency_ms=warmup_latency_ms,
                    first_live_inference_latency_ms=first_live_inference_latency_ms,
                ),
            )
            self._hooks.on_policy_ready(prepared.initial_chunk)
            return prepared
        except BaseException as exc:
            if warming_up:
                self._hooks.on_policy_warmup_failed(exc)
            raise

    def _request(self, stage: str, timeout_s: float) -> float:
        _, elapsed_s = self._request_with_chunk(stage, timeout_s)
        return elapsed_s

    def _request_with_chunk(self, stage: str, timeout_s: float):
        self._raise_if_stopped()
        try:
            fetched = fetch_action_chunk(
                self._client,
                self._adapter,
                self._loop_config,
                self._hooks,
                profile_stage=f"policy_{stage}",
                event_stage=stage,
            )
        except TimeoutError as exc:
            self._raise_if_stopped()
            raise self._timeout_error(stage, timeout_s) from exc
        except (ActionDecodeError, PolicyProtocolError):
            self._raise_if_stopped()
            raise
        except BaseException:
            self._raise_if_stopped()
            raise
        elapsed_s = fetched.inference_elapsed_s
        if elapsed_s > timeout_s:
            raise self._timeout_error(stage, timeout_s)
        self._raise_if_stopped()
        return fetched, elapsed_s

    def _raise_if_stopped(self) -> None:
        if self._stop_requested():
            raise PolicyStartupCancelled("Policy startup was cancelled")

    @staticmethod
    def _timeout_error(stage: str, timeout_s: float) -> PolicyStartupError:
        if stage == "warmup":
            return PolicyWarmupTimeout(f"Policy warmup exceeded {timeout_s:.3f}s")
        return PolicyInferenceTimeout(f"Policy {stage} inference exceeded {timeout_s:.3f}s")

    def _notify_phase(self, phase: str) -> None:
        if self._on_phase is not None:
            self._on_phase(phase)
