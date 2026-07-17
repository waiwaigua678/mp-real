"""Application service for Baseline creation, lineage, attachment, and runs."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mp_real.evaluation.baseline.comparison import compare_baselines
from mp_real.evaluation.baseline.diff import BaselineDiff, diff_baselines
from mp_real.evaluation.baseline.models import (
    Baseline,
    BaselineOpenLoopReference,
    BaselineRunReference,
)
from mp_real.evaluation.baseline.store import BaselineStore


class BaselineConfigurationConflict(RuntimeError):
    """The connected runtime differs from an immutable Baseline."""

    def __init__(self, diff: BaselineDiff) -> None:
        self.diff = diff
        super().__init__("Current runtime differs from Baseline; create an explicit derived Baseline")


class BaselineService:
    def __init__(self, store: BaselineStore) -> None:
        self.store = store

    def list(self) -> tuple[Baseline, ...]:
        return self.store.list()

    def get(self, baseline_id: str) -> Baseline:
        return self.store.get(baseline_id)

    def create_from_runtime(
        self, payload: Mapping[str, Any], *, runtime_config: Mapping[str, Any], git_commit: str | None
    ) -> Baseline:
        if not git_commit:
            raise ValueError("Cannot create a reproducible Baseline without a Git commit")
        baseline = Baseline.from_runtime_snapshot(payload, runtime_config=runtime_config, git_commit=git_commit)
        return self.store.create(baseline)

    def create_from_evaluation(self, snapshot: Mapping[str, Any], *, name: str | None = None) -> Baseline:
        config = snapshot.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("evaluation snapshot has no configuration")
        runtime = config.get("runtime_config_snapshot")
        if not isinstance(runtime, Mapping):
            raise ValueError("evaluation snapshot has no runtime configuration snapshot")
        action_spec = config.get("action_spec_snapshot")
        if not isinstance(action_spec, Mapping):
            raise ValueError("evaluation snapshot has no ActionSpec snapshot")
        payload = {
            "name": name or config.get("name", "evaluation-baseline"),
            "robot_name": config.get("robot_name"),
            "task_name": config.get("task_name"),
            "prompt": config.get("prompt"),
            "policy_label": config.get("policy_label") or "unlabeled-policy",
            "planned_episodes": config.get("planned_episodes"),
            "max_episode_duration_s": config.get("max_episode_seconds"),
            "operator": config.get("operator"),
            "tags": config.get("tags", ()),
            "notes": config.get("notes", ""),
            "action_spec": action_spec,
        }
        baseline = self.create_from_runtime(
            payload,
            runtime_config=runtime,
            git_commit=_text(runtime.get("git_commit")),
        )
        return self.attach_evaluation(baseline.baseline_id, snapshot)

    def clone(self, baseline_id: str, patch: Mapping[str, Any], *, derived_reason: str) -> Baseline:
        if not derived_reason.strip():
            raise ValueError("derived_reason is required when cloning a Baseline")
        original = self.get(baseline_id)
        payload = original.without_references()
        for forbidden in ("baseline_id", "schema_version", "created_at", "parent_baseline_id", "derived_reason"):
            if forbidden in patch:
                raise ValueError(f"{forbidden} cannot be supplied while cloning")
        _merge(payload, patch)
        payload["baseline_id"] = uuid.uuid4().hex
        payload["parent_baseline_id"] = original.baseline_id
        payload["derived_reason"] = derived_reason
        payload["created_at"] = None
        payload["created_at"] = _created_now()
        clone = Baseline.from_dict(payload)
        return self.store.create(clone)

    def diff(self, baseline_a: str, baseline_b: str) -> BaselineDiff:
        return diff_baselines(self.get(baseline_a), self.get(baseline_b))

    def diff_current(
        self, baseline_id: str, *, runtime_config: Mapping[str, Any], git_commit: str | None
    ) -> BaselineDiff:
        baseline = self.get(baseline_id)
        candidate = Baseline.from_runtime_snapshot(
            {
                "name": baseline.name,
                "robot_name": baseline.robot_name,
                "task_name": baseline.task_name,
                "prompt": runtime_config.get("prompt", baseline.prompt),
                "policy_server_url": runtime_config.get("server_url", baseline.policy_server_url),
                "policy_label": baseline.policy_label,
                "checkpoint_hash": baseline.checkpoint_hash,
                "dataset_format": baseline.dataset_format,
                "planned_episodes": baseline.evaluation_protocol["planned_episodes"],
                "max_episode_duration_s": baseline.evaluation_protocol["max_episode_duration_s"],
                "initial_position_protocol": baseline.initial_position_protocol,
                "source_dataset": baseline.source.get("dataset"),
                "source_episode": baseline.source.get("episode"),
                "source_sample": baseline.source.get("sample"),
                "operator": baseline.operator,
                "tags": baseline.tags,
                "notes": baseline.notes,
                "action_spec": runtime_config.get("action_spec"),
            },
            runtime_config=runtime_config,
            git_commit=git_commit or baseline.git_commit,
            baseline_id="current-runtime",
        )
        return diff_baselines(baseline, candidate)

    def prepare_evaluation_run(
        self, baseline_id: str, *, runtime_config: Mapping[str, Any], git_commit: str | None
    ) -> dict[str, Any]:
        baseline = self.get(baseline_id)
        diff = self.diff_current(baseline_id, runtime_config=runtime_config, git_commit=git_commit)
        if not diff.compatible_for_run:
            raise BaselineConfigurationConflict(diff)
        protocol = baseline.evaluation_protocol
        return {
            "name": baseline.name,
            "robot_name": baseline.robot_name,
            "task_name": baseline.task_name,
            "prompt": baseline.prompt,
            "planned_episodes": protocol["planned_episodes"],
            "max_episode_seconds": protocol["max_episode_duration_s"],
            "policy_label": baseline.policy_label,
            "operator": baseline.operator,
            "tags": list(baseline.tags),
            "notes": baseline.notes,
            "baseline_id": baseline.baseline_id,
        }

    def attach_evaluation(self, baseline_id: str, snapshot: Mapping[str, Any]) -> Baseline:
        baseline = self.get(baseline_id)
        recording = snapshot.get("recording", {})
        reference = BaselineRunReference(
            evaluation_id=str(snapshot.get("evaluation_id") or snapshot.get("session_id") or ""),
            state=str(snapshot.get("state", "UNKNOWN")),
            summary=_compact_evaluation_summary(snapshot),
            dataset_path=_text(recording.get("dataset_root")) if isinstance(recording, Mapping) else None,
        )
        if any(item.evaluation_id == reference.evaluation_id for item in baseline.evaluation_runs):
            return baseline
        updated = dataclasses_replace(baseline, evaluation_runs=(*baseline.evaluation_runs, reference))
        return self.store.replace(updated)

    def attach_open_loop(self, baseline_id: str, result_dir: Path | str) -> Baseline:
        root = Path(result_dir).expanduser().resolve()
        summary_path = root / "summary.json"
        config_path = root / "config.json"
        if not summary_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("open-loop result directory requires summary.json and config.json")
        summary = _read_object(summary_path)
        config = _read_object(config_path)
        summary["comparison_contract"] = _open_loop_contract(config, summary)
        reference = BaselineOpenLoopReference(
            evaluation_id=str(summary.get("evaluation_id") or config.get("evaluation_id") or root.name),
            output_dir=str(root),
            config_fingerprint=_text(config.get("config_fingerprint")),
            status=str(summary.get("status", "unknown")),
            summary=summary,
        )
        baseline = self.get(baseline_id)
        if any(item.evaluation_id == reference.evaluation_id for item in baseline.open_loop_runs):
            return baseline
        return self.store.replace(dataclasses_replace(baseline, open_loop_runs=(*baseline.open_loop_runs, reference)))

    def compare(self, baseline_ids: Iterable[str]) -> dict[str, Any]:
        return compare_baselines(self.get(item) for item in baseline_ids)


def dataclasses_replace(baseline: Baseline, **changes: Any) -> Baseline:
    import dataclasses

    return dataclasses.replace(baseline, **changes)


def _merge(destination: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), Mapping):
            nested = copy.deepcopy(dict(destination[key]))
            _merge(nested, value)
            destination[key] = nested
        else:
            destination[str(key)] = copy.deepcopy(value)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _created_now() -> str:
    from mp_real.evaluation.baseline.models import utc_now

    return utc_now()


def _compact_evaluation_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Keep aggregate evaluation metrics, never per-episode Baseline copies."""
    summary = dict(snapshot.get("summary", {}))
    durations_s: list[float] = []
    successful_durations_s: list[float] = []
    failed_durations_s: list[float] = []
    episodes = snapshot.get("episodes", ())
    if isinstance(episodes, list | tuple):
        for episode in episodes:
            if not isinstance(episode, Mapping):
                continue
            started = episode.get("started_at_monotonic_ns")
            stopped = episode.get("stopped_at_monotonic_ns")
            try:
                duration = max(0.0, (int(stopped) - int(started)) / 1e9)
            except (TypeError, ValueError):
                continue
            durations_s.append(duration)
            result = episode.get("result")
            if result == "SUCCESS":
                successful_durations_s.append(duration)
            elif result in {"FAILURE", "TIMEOUT", "SAFETY_ABORT"}:
                failed_durations_s.append(duration)
    summary["durations_s"] = durations_s
    summary["successful_durations_s"] = successful_durations_s
    summary["failed_durations_s"] = failed_durations_s
    recording = snapshot.get("recording")
    if isinstance(recording, Mapping):
        summary["dropped"] = {
            key: recording.get(key, 0)
            for key in ("dropped_event_count", "dropped_frame_count", "queue_high_watermark")
        }
    return summary


def _open_loop_contract(config: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the exact offline target/alignment contract for safe A/B review."""
    settings = config.get("config")
    settings = settings if isinstance(settings, Mapping) else {}
    return {
        "source_dataset": config.get("source_dataset", summary.get("source_dataset")),
        "episodes": config.get("episodes"),
        "target_source": settings.get("target_source", summary.get("target_source")),
        "alignment_mode": settings.get("alignment_mode", summary.get("alignment_mode")),
        "max_timestamp_error_s": settings.get("max_timestamp_error_s"),
        "state_derived": settings.get("state_derived"),
        "action_spec": config.get("source_action_spec"),
        "state_schema": config.get("state_schema"),
        "camera_roles": settings.get("selected_camera_roles"),
    }
