"""Teacher-forced open-loop evaluator; it has no Robot construction path."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mp_real.data.models import RecordedEpisodeSource, RecordedSample
from mp_real.evaluation.open_loop.alignment import ActionAlignment
from mp_real.evaluation.open_loop.metrics import compute_open_loop_metrics
from mp_real.evaluation.open_loop.models import OpenLoopEvaluationConfig, config_json
from mp_real.evaluation.open_loop.results import PredictionResultWriter
from mp_real.evaluation.open_loop.source import OpenLoopInputError, resolve_image_masks, resolve_prompt, resolve_target
from mp_real.policy_client.client import PolicyClient
from mp_real.runtime.config import InferenceLoopConfig
from mp_real.runtime.inference import InferenceAdapter, decode_action_chunk_for_spec, fetch_action_chunk
from mp_real.runtime.models import ActionSpec, ObservationSnapshot
from mp_real.runtime.observation import build_observation_snapshot
from mp_real.runtime.startup import PolicyStartupConfig, PolicyStartupCoordinator


class OpenLoopEvaluationCancelled(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class OpenLoopRunResult:
    evaluation_id: str
    output_dir: Path
    status: str
    completed_episodes: tuple[int, ...]
    failed_episodes: tuple[int, ...]


class _OfflineInferenceAdapter(InferenceAdapter):
    """InferenceAdapter whose only input is a previously recorded snapshot."""

    name = "open_loop_recorded_episode"

    def __init__(self, action_spec: ActionSpec, loop_config: InferenceLoopConfig) -> None:
        self._action_spec = action_spec
        self._loop_config = loop_config
        self._snapshot: ObservationSnapshot | None = None

    @property
    def last_observation_snapshot(self) -> ObservationSnapshot | None:
        return self._snapshot

    def set_snapshot(self, snapshot: ObservationSnapshot) -> None:
        self._snapshot = snapshot

    def observe(self) -> dict[str, Any]:
        if self._snapshot is None:
            raise RuntimeError("offline inference observation was not set")
        return self._snapshot.to_policy_observation()

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray:
        return decode_action_chunk_for_spec(response, action_spec=self._action_spec, replan_steps=replan_steps)

    def initial_action(self) -> np.ndarray:
        if self._snapshot is None:
            raise RuntimeError("offline inference observation was not set")
        return self._snapshot.state.values.copy()

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        del previous
        return np.asarray(action, dtype=np.float32).copy()

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray:
        del previous, target
        raise AssertionError("open-loop evaluation must never execute a robot action")

    def infer_only_metadata(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        return {}

    def profile(self, stage: str, elapsed_s: float) -> None:
        del stage, elapsed_s

    def infer_only_interval_s(self) -> float:
        return 0.0


class OpenLoopEvaluator:
    """Evaluate policy chunks against real recorded observations only.

    This class intentionally imports no robot registry/profile/SDK.  Its
    adapter's execution method raises unconditionally and the evaluator never
    invokes a control loop.
    """

    def __init__(
        self,
        config: OpenLoopEvaluationConfig,
        *,
        source: RecordedEpisodeSource | None = None,
        policy_factory: Callable[[str, str | None, float, float], Any] | None = None,
        stop_event: threading.Event | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self._source = source
        self._owns_source = source is None
        self._policy_factory = policy_factory or self._default_policy_factory
        self._stop_event = stop_event or threading.Event()
        self._progress_callback = progress_callback
        self._latencies_ms: list[float] = []
        self._timeout_count = 0
        self._protocol_error_count = 0
        self._connection_latency_ms: float | None = None
        self._metadata_latency_ms: float | None = None

    @staticmethod
    def _default_policy_factory(url: str, key: str | None, timeout: float, metadata_timeout: float) -> PolicyClient:
        return PolicyClient(url, key, timeout=timeout, metadata_timeout=metadata_timeout)

    def stop(self) -> None:
        self._stop_event.set()

    def preview_status(self, episode_index: int) -> dict[str, Any]:
        """Classify a viewer-only episode without making a policy request.

        A video-only episode remains usable by the data viewer, but it is
        explicitly excluded from formal open-loop metrics.
        """

        source = self._source or _lerobot_source(self.config.dataset)
        owns_source = self._source is None
        try:
            samples = self._samples(source, episode_index)
            roles = self._roles(source)
            try:
                self._validate_formal_input(samples, source.get_action_spec(), roles, episode_index)
                resolve_prompt(source, episode_index, samples, self.config.prompt_override)
                if samples:
                    resolve_target(source, samples[0], self.config)
            except OpenLoopInputError as exc:
                return {
                    "episode_index": episode_index,
                    "incomplete_observation": True,
                    "formal_action_metrics": False,
                    "reason": str(exc),
                }
            return {"episode_index": episode_index, "incomplete_observation": False, "formal_action_metrics": True}
        finally:
            if owns_source:
                source.close()

    def run(self) -> OpenLoopRunResult:
        source = self._source or _lerobot_source(self.config.dataset)
        self._source = source
        writer = PredictionResultWriter(self.config.output_dir, resume=self.config.resume)
        metadata = source.get_dataset_metadata()
        action_spec = source.get_action_spec()
        episodes = self._episodes(source)
        writer.prepare(self._config_payload(source, episodes))
        completed: list[int] = []
        failed: list[int] = []
        client: Any | None = None
        try:
            self._raise_if_stopped()
            prepared_episode = self._first_runnable_episode(source, episodes)
            if prepared_episode is None:
                raise OpenLoopInputError("no selected episode has readable samples")
            warmup_sample, warmup_prompt, warmup_masks, warmup_roles = prepared_episode
            loop_config = self._loop_config(float(metadata.info["fps"]))
            adapter = _OfflineInferenceAdapter(action_spec, loop_config)
            adapter.set_snapshot(self._snapshot(warmup_sample, warmup_prompt, warmup_masks, warmup_roles))
            connect_started_ns = time.monotonic_ns()
            client = self._policy_factory(
                self.config.policy_url,
                self.config.policy_api_key,
                self.config.connection_timeout_s,
                self.config.metadata_timeout_s,
            )
            fallback_connect_latency_ms = (time.monotonic_ns() - connect_started_ns) / 1e6
            self._connection_latency_ms = getattr(client, "connect_latency_ms", None) or fallback_connect_latency_ms
            self._metadata_latency_ms = getattr(client, "metadata_latency_ms", None)
            startup = PolicyStartupCoordinator(
                client,
                adapter,
                loop_config,
                PolicyStartupConfig(
                    warmup_enabled=self.config.warmup.enabled,
                    warmup_requests=self.config.warmup.requests,
                    warmup_timeout_s=self.config.warmup.timeout_s,
                    inference_timeout_s=self.config.warmup.inference_timeout_s,
                    prefetch_first_chunk=False,
                ),
                hooks=_NoopHooks(),
                stop_requested=self._stop_event.is_set,
            ).prepare()
            policy_metadata = _json_safe(getattr(client, "metadata", {}))
            git_commit = _git_commit()
            for episode_index in episodes:
                self._raise_if_stopped()
                if writer.has_completed_episode(episode_index):
                    completed.append(episode_index)
                    self._progress("episode_skipped", episode_index=episode_index)
                    continue
                try:
                    report, arrays = self._run_episode(
                        source, episode_index, adapter, client, loop_config, policy_metadata, git_commit
                    )
                    writer.write_episode(episode_index, arrays=arrays, report=report)
                    if report["status"] == "complete":
                        completed.append(episode_index)
                    else:
                        failed.append(episode_index)
                except OpenLoopEvaluationCancelled:
                    raise
                except BaseException as exc:
                    failed.append(episode_index)
                    writer.write_episode(
                        episode_index,
                        arrays=_empty_arrays(action_spec),
                        report=self._failed_report(episode_index, exc),
                    )
                self._progress("episode_finished", episode_index=episode_index, completed=completed, failed=failed)
            status = "complete" if not failed else "partial_error"
            summary = self._summary(
                source,
                completed,
                failed,
                policy_metadata,
                startup.metrics,
                fallback_connect_latency_ms,
                status,
            )
            writer.write_summary(summary)
            return OpenLoopRunResult(self.config.evaluation_id, writer.root, status, tuple(completed), tuple(failed))
        except OpenLoopEvaluationCancelled:
            writer.write_summary(
                self._summary(source, completed, failed, {}, None, None, "cancelled")
            )
            return OpenLoopRunResult(
                self.config.evaluation_id, writer.root, "cancelled", tuple(completed), tuple(failed)
            )
        except BaseException as exc:
            self._record_policy_error(exc)
            summary = self._summary(source, completed, failed, {}, None, None, "error")
            summary["errors"] = [{"error_type": type(exc).__name__, "message": str(exc)}]
            writer.write_summary(summary)
            return OpenLoopRunResult(self.config.evaluation_id, writer.root, "error", tuple(completed), tuple(failed))
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            if self._owns_source:
                source.close()

    def _run_episode(
        self,
        source: RecordedEpisodeSource,
        episode_index: int,
        adapter: _OfflineInferenceAdapter,
        client: Any,
        loop_config: InferenceLoopConfig,
        policy_metadata: Mapping[str, Any],
        git_commit: str | None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        samples = self._samples(source, episode_index)
        prompt = resolve_prompt(source, episode_index, samples, self.config.prompt_override)
        spec = source.get_action_spec()
        roles = self._roles(source)
        masks = resolve_image_masks(self.config, roles)
        self._validate_formal_input(samples, spec, roles, episode_index)
        alignment = ActionAlignment(samples, self.config, fps=float(source.get_dataset_metadata().info["fps"]))
        request_count = len(samples)
        arrays = _episode_arrays(request_count, self.config.replan_steps, spec.action_dim)
        arrays["source_sample_index"] = np.asarray([sample.index for sample in samples], dtype=np.int64)
        arrays["frame_index"] = np.asarray([sample.frame_index for sample in samples], dtype=np.int64)
        arrays["source_timestamp"] = np.asarray([sample.timestamp for sample in samples], dtype=np.float64)
        errors: list[dict[str, Any]] = []
        physical_target_fields: set[str] = set()
        completed_requests = 0
        status = "complete"
        for request_index, sample in enumerate(samples):
            self._raise_if_stopped()
            adapter.set_snapshot(self._snapshot(sample, prompt, masks, roles))
            try:
                fetched = fetch_action_chunk(client, adapter, loop_config, _NoopHooks())
            except BaseException as exc:
                status = "error"
                self._record_policy_error(exc)
                errors.append({"request_index": request_index, "error_type": type(exc).__name__, "message": str(exc)})
                break
            arrays["predicted_chunks"][request_index] = fetched.chunk
            arrays["chunk_length"][request_index] = len(fetched.chunk)
            arrays["latency_ns"][request_index] = int(round(fetched.inference_elapsed_s * 1e9))
            self._latencies_ms.append(fetched.inference_elapsed_s * 1000.0)
            for horizon in range(len(fetched.chunk)):
                match = alignment.align(request_index, horizon)
                arrays["target_sample_index"][request_index, horizon] = (
                    -1 if match.target_sample_index is None else match.target_sample_index
                )
                arrays["target_timestamp"][request_index, horizon] = (
                    np.nan if match.target_timestamp is None else match.target_timestamp
                )
                arrays["alignment_error_s"][request_index, horizon] = (
                    np.nan if match.alignment_error_s is None else match.alignment_error_s
                )
                if not match.valid:
                    continue
                assert match.target_sample_index is not None
                target, field = resolve_target(source, samples[match.target_sample_index], self.config)
                arrays["targets"][request_index, horizon] = target
                arrays["valid_mask"][request_index, horizon] = True
                physical_target_fields.add(field)
            completed_requests += 1
            self._progress(
                "sample_finished",
                episode_index=episode_index,
                completed_samples=completed_requests,
                total_samples=request_count,
            )
        metrics = compute_open_loop_metrics(
            arrays["predicted_chunks"],
            arrays["targets"],
            arrays["valid_mask"],
            target_indices=arrays["target_sample_index"],
            source_timestamps=arrays["source_timestamp"],
            target_timestamps=arrays["target_timestamp"],
            action_spec=spec,
        )
        top_errors = _top_errors(arrays, spec, samples, self.config.top_error_count)
        report = {
            "schema_version": 1,
            "evaluation_id": self.config.evaluation_id,
            "episode_index": episode_index,
            "status": status,
            "teacher_forced": True,
            "incomplete_observation": False,
            "target_source": self.config.target_source.value,
            "physical_target_fields": sorted(physical_target_fields),
            "alignment_mode": self.config.alignment_mode.value,
            "response_decoding": {
                "decoder": "decode_action_chunk_for_spec",
                "action_contract": "recorded ActionSpec policy-space values; no robot command conversion",
            },
            "completed_samples": completed_requests,
            "valid_prediction_count": int(arrays["valid_mask"].sum()),
            "metrics": metrics,
            "latency": _latency_summary(arrays["latency_ns"][arrays["latency_ns"] >= 0]),
            "top_errors": top_errors,
            "errors": errors,
            "input": {
                "prompt": prompt,
                "camera_roles": list(roles),
                "image_masks": {name: bool(mask) for name, mask in masks.items()},
                "dataset_fps": float(source.get_dataset_metadata().info["fps"]),
                "action_spec": _action_spec_json(spec),
                "state_schema": list(source.get_state_schema()),
            },
            "interpretation": _interpretation(),
        }
        arrays["metadata_json"] = np.asarray(
            json.dumps(
                {
                    "source_dataset": str(source.get_dataset_metadata().root),
                    "source_episode": episode_index,
                    "policy_label": self.config.policy_label,
                    "policy_metadata": _json_safe(policy_metadata),
                    "git_commit": git_commit,
                    "target_source": self.config.target_source.value,
                    "alignment_mode": self.config.alignment_mode.value,
                    "alignment_config": {
                        "max_timestamp_error_s": self.config.max_timestamp_error_s,
                        "allow_frame_index_as_control_step": self.config.allow_frame_index_as_control_step,
                    },
                    "teacher_forced": True,
                },
                ensure_ascii=False,
            )
        )
        return report, arrays

    def _samples(self, source: RecordedEpisodeSource, episode_index: int) -> list[RecordedSample]:
        length = source.get_length(episode_index)
        if self.config.limit is not None:
            length = min(length, self.config.limit)
        result: list[RecordedSample] = []
        for index in range(length):
            try:
                result.append(source.get_sample(episode_index, index, include_images=True))  # type: ignore[call-arg]
            except TypeError:
                result.append(source.get_sample(episode_index, index))
        return result

    def _first_runnable_episode(
        self, source: RecordedEpisodeSource, episodes: Sequence[int]
    ) -> tuple[RecordedSample, str, Mapping[str, np.bool_], tuple[str, ...]] | None:
        for episode_index in episodes:
            samples = self._samples(source, episode_index)
            if not samples:
                continue
            prompt = resolve_prompt(source, episode_index, samples, self.config.prompt_override)
            roles = self._roles(source)
            masks = resolve_image_masks(self.config, roles)
            self._validate_formal_input(samples, source.get_action_spec(), roles, episode_index)
            return samples[0], prompt, masks, roles
        return None

    def _roles(self, source: RecordedEpisodeSource) -> tuple[str, ...]:
        available = source.get_camera_roles()
        roles = self.config.selected_camera_roles or available
        if not roles:
            raise OpenLoopInputError("formal open-loop evaluation requires at least one selected camera role")
        missing = set(roles) - set(available)
        if missing:
            raise OpenLoopInputError("selected camera roles are absent from dataset: " + ", ".join(sorted(missing)))
        return tuple(roles)

    @staticmethod
    def _validate_formal_input(
        samples: Sequence[RecordedSample], spec: ActionSpec, roles: Sequence[str], episode_index: int
    ) -> None:
        if not samples:
            raise OpenLoopInputError(f"episode {episode_index} has no readable samples")
        if not spec.state_fields:
            raise OpenLoopInputError("formal open-loop evaluation requires a state schema")
        for sample in samples:
            if np.asarray(sample.state).shape != (spec.state_dim,):
                raise OpenLoopInputError(f"episode {episode_index} has missing or invalid observation.state")
            for role in roles:
                image = sample.images.get(role)
                if image is None:
                    raise OpenLoopInputError(
                        f"episode {episode_index} is incomplete_observation=true: missing observation.images.{role}"
                    )

    def _snapshot(
        self,
        sample: RecordedSample,
        prompt: str,
        image_masks: Mapping[str, np.bool_],
        roles: Sequence[str],
    ) -> ObservationSnapshot:
        images = {role: np.asarray(sample.images[role]) for role in roles}
        # Dataset timestamps are recording-time values, not current wall clock.
        # The policy wire schema receives only image/state/prompt; deterministic
        # ordering metadata remains in the saved report/artifacts.
        return build_observation_snapshot(
            images,
            state=np.asarray(sample.state, dtype=np.float32),
            prompt=prompt,
            resize_size=self.config.resize_size,
            image_masks=image_masks,
            captured_at_ns=time.monotonic_ns(),
        )

    def _episodes(self, source: RecordedEpisodeSource) -> tuple[int, ...]:
        available = tuple(item.episode_index for item in source.list_episodes())
        requested = self.config.episode_indices or available
        missing = set(requested) - set(available)
        if missing:
            raise OpenLoopInputError("requested episodes are absent: " + ", ".join(map(str, sorted(missing))))
        return tuple(requested)

    def _loop_config(self, fps: float) -> InferenceLoopConfig:
        return InferenceLoopConfig(
            fps=fps,
            replan_steps=self.config.replan_steps,
            max_steps=None,
            use_rtc=False,
            rtc_replan_stride=0,
            rtc_prefetch_steps=0,
            rtc_exp_weight=0.0,
            hold_last_action=False,
            infer_only=True,
            infer_only_chunks=1,
            infer_only_output=None,
            prompt=self.config.prompt_override or "",
            log_timing=False,
        )

    def _config_payload(self, source: RecordedEpisodeSource, episodes: Sequence[int]) -> dict[str, Any]:
        metadata = source.get_dataset_metadata()
        return {
            "schema_version": 1,
            "evaluation_id": self.config.evaluation_id,
            "teacher_forced": True,
            "config": config_json(self.config),
            "source_dataset": str(metadata.root),
            "source_dataset_info": _json_safe(metadata.info),
            "source_action_spec": _action_spec_json(source.get_action_spec()),
            "state_schema": list(source.get_state_schema()),
            "dataset_fps": metadata.info.get("fps"),
            "episodes": list(episodes),
            "git_commit": _git_commit(),
            "report_interpretation": _interpretation(),
        }

    def _summary(
        self,
        source: RecordedEpisodeSource,
        completed: Sequence[int],
        failed: Sequence[int],
        policy_metadata: Mapping[str, Any],
        startup_metrics: Any,
        fallback_connect_latency_ms: float | None,
        status: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evaluation_id": self.config.evaluation_id,
            "status": status,
            "teacher_forced": True,
            "source_dataset": str(source.get_dataset_metadata().root),
            "policy": {
                "label": self.config.policy_label,
                "url": self.config.policy_url,
                "metadata": _json_safe(policy_metadata),
            },
            "target_source": self.config.target_source.value,
            "alignment_mode": self.config.alignment_mode.value,
            "completed_episodes": list(completed),
            "failed_episodes": list(failed),
            "performance": {
                "connection_latency_ms": getattr(startup_metrics, "connection_latency_ms", None)
                or self._connection_latency_ms
                or fallback_connect_latency_ms,
                "metadata_latency_ms": self._metadata_latency_ms,
                "warmup_latency_ms": getattr(startup_metrics, "warmup_latency_ms", None),
                "first_live_inference_latency_ms": self._latencies_ms[0] if self._latencies_ms else None,
                "steady_inference_latency_ms": _latency_summary(np.asarray(self._latencies_ms[1:])),
                "timeout_count": self._timeout_count,
                "protocol_error_count": self._protocol_error_count,
            },
            "interpretation": _interpretation(),
        }

    def _failed_report(self, episode_index: int, exc: BaseException) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evaluation_id": self.config.evaluation_id,
            "episode_index": episode_index,
            "status": "error",
            "teacher_forced": True,
            "incomplete_observation": isinstance(exc, OpenLoopInputError) and "incomplete_observation=true" in str(exc),
            "target_source": self.config.target_source.value,
            "alignment_mode": self.config.alignment_mode.value,
            "completed_samples": 0,
            "valid_prediction_count": 0,
            "metrics": {},
            "errors": [{"error_type": type(exc).__name__, "message": str(exc)}],
            "interpretation": _interpretation(),
        }

    def _record_policy_error(self, exc: BaseException) -> None:
        if isinstance(exc, TimeoutError):
            self._timeout_count += 1
        if type(exc).__name__ in {"PolicyProtocolError", "ActionDecodeError"}:
            self._protocol_error_count += 1

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise OpenLoopEvaluationCancelled("open-loop evaluation was stopped")

    def _progress(self, phase: str, **values: Any) -> None:
        if self._progress_callback is not None:
            self._progress_callback({"phase": phase, "evaluation_id": self.config.evaluation_id, **values})


class _NoopHooks:
    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def _episode_arrays(requests: int, horizon: int, action_dim: int) -> dict[str, np.ndarray]:
    values = _empty_arrays(action_dim, requests=requests, horizon=horizon)
    values["predicted_chunks"] = np.full((requests, horizon, action_dim), np.nan, dtype=np.float32)
    values["targets"] = np.full((requests, horizon, action_dim), np.nan, dtype=np.float32)
    values["valid_mask"] = np.zeros((requests, horizon), dtype=np.bool_)
    values["target_sample_index"] = np.full((requests, horizon), -1, dtype=np.int64)
    values["target_timestamp"] = np.full((requests, horizon), np.nan, dtype=np.float64)
    values["alignment_error_s"] = np.full((requests, horizon), np.nan, dtype=np.float64)
    values["chunk_length"] = np.zeros(requests, dtype=np.int64)
    values["latency_ns"] = np.full(requests, -1, dtype=np.int64)
    return values


def _empty_arrays(action_dim: int, *, requests: int = 0, horizon: int = 0) -> dict[str, np.ndarray]:
    return {
        "predicted_chunks": np.empty((requests, horizon, action_dim), dtype=np.float32),
        "targets": np.empty((requests, horizon, action_dim), dtype=np.float32),
        "valid_mask": np.empty((requests, horizon), dtype=np.bool_),
        "target_sample_index": np.empty((requests, horizon), dtype=np.int64),
        "target_timestamp": np.empty((requests, horizon), dtype=np.float64),
        "alignment_error_s": np.empty((requests, horizon), dtype=np.float64),
        "source_sample_index": np.empty(requests, dtype=np.int64),
        "frame_index": np.empty(requests, dtype=np.int64),
        "source_timestamp": np.empty(requests, dtype=np.float64),
        "chunk_length": np.empty(requests, dtype=np.int64),
        "latency_ns": np.empty(requests, dtype=np.int64),
    }


def _top_errors(
    arrays: Mapping[str, np.ndarray], spec: ActionSpec, samples: Sequence[RecordedSample], count: int
) -> list[dict[str, Any]]:
    valid = arrays["valid_mask"]
    error = np.where(valid[..., None], np.abs(arrays["predicted_chunks"] - arrays["targets"]), np.nan)
    positions = np.argwhere(np.isfinite(error))
    ranked = sorted(positions.tolist(), key=lambda value: float(error[tuple(value)]), reverse=True)[:count]
    names = spec.action_field_names or tuple(f"dim_{index}" for index in range(spec.action_dim))
    return [
        {
            "error": float(error[request, horizon, dimension]),
            "request_index": request,
            "source_sample_index": int(samples[request].index),
            "source_frame_index": int(samples[request].frame_index),
            "horizon": horizon,
            "target_sample_index": int(arrays["target_sample_index"][request, horizon]),
            "dimension": dimension,
            "dimension_name": names[dimension],
        }
        for request, horizon, dimension in ranked
    ]


def _latency_summary(latency_ns: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(latency_ns, dtype=np.float64)
    values = values[values >= 0]
    if not len(values):
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None}
    milliseconds = values / 1e6
    return {
        "count": int(len(milliseconds)),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "mean_ms": float(np.mean(milliseconds)),
    }


def _action_spec_json(spec: ActionSpec) -> dict[str, Any]:
    return spec.to_dict()


def _lerobot_source(path: Path) -> RecordedEpisodeSource:
    from mp_real.data.lerobot_v21 import LeRobotV21EpisodeSource

    return LeRobotV21EpisodeSource(path)


def _interpretation() -> list[str]:
    return [
        "Open-loop action error is not real-robot task success rate.",
        "For multimodal tasks, a prediction different from the recorded expert action may still be effective.",
        "teacher_forced=true means every request uses recorded real observations.",
        "Open-loop evaluation cannot reveal every closed-loop error accumulation effect.",
        "Gripper event timing can matter more than pointwise MAE.",
        "Results with different ActionSpec values cannot be merged directly.",
        "Results with different target sources cannot be merged directly.",
    ]


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value
