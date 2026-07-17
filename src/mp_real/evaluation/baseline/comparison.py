"""A/B reports that preserve numerators, denominators, and invalid handling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from mp_real.evaluation.baseline.models import Baseline

_EXCLUDED_RESULTS = frozenset({"INVALID", "SYSTEM_ERROR", "OPERATOR_ABORT"})
_VALID_RESULTS = frozenset({"SUCCESS", "FAILURE", "TIMEOUT", "SAFETY_ABORT"})


def compare_baselines(baselines: Iterable[Baseline]) -> dict[str, Any]:
    selected = tuple(baselines)
    if len(selected) < 2:
        raise ValueError("compare requires at least two baselines")
    contracts = {_contract_key(item) for item in selected}
    rows = [_row(item) for item in selected]
    open_loop = _open_loop_comparability(selected)
    return {
        "baseline_ids": [item.baseline_id for item in selected],
        "comparable": len(contracts) == 1,
        "incompatibilities": [] if len(contracts) == 1 else _contract_reasons(selected),
        "invalid_handling": {
            "excluded_from_success_rate": sorted(_EXCLUDED_RESULTS),
            "included_in_success_rate_denominator": sorted(_VALID_RESULTS),
            "rule": "INVALID, SYSTEM_ERROR, and OPERATOR_ABORT remain visible but do not enter success-rate N.",
        },
        "results": rows,
        "open_loop": open_loop,
        "small_sample_warning": any(row["live"]["success_rate"]["denominator"] < 10 for row in rows),
    }


def _row(baseline: Baseline) -> dict[str, Any]:
    result_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    completed = 0
    durations: list[float] = []
    successful_durations: list[float] = []
    failed_durations: list[float] = []
    for reference in baseline.evaluation_runs:
        summary = reference.summary
        result_counts.update({str(key): int(value) for key, value in dict(summary.get("result_counts", {})).items()})
        failure_reasons.update(
            {str(key): int(value) for key, value in dict(summary.get("failure_reason_counts", {})).items()}
        )
        completed += int(summary.get("completed_episodes", 0))
        _extend_numbers(durations, summary.get("durations_s"))
        _extend_numbers(successful_durations, summary.get("successful_durations_s"))
        _extend_numbers(failed_durations, summary.get("failed_durations_s"))
    denominator = sum(result_counts[result] for result in _VALID_RESULTS)
    successes = result_counts["SUCCESS"]
    live = {
        "completed_episode_count": completed,
        "valid_episode_count": denominator,
        "success_count": successes,
        "failure_count": result_counts["FAILURE"],
        "invalid_count": result_counts["INVALID"],
        "timeout_count": result_counts["TIMEOUT"],
        "safety_abort_count": result_counts["SAFETY_ABORT"],
        "system_error_count": result_counts["SYSTEM_ERROR"],
        "operator_abort_count": result_counts["OPERATOR_ABORT"],
        "success_rate": _rate(successes, denominator, completed),
        "failure_reason_distribution": dict(sorted(failure_reasons.items())),
        "average_duration_s": _mean(durations),
        "successful_duration_s": _mean(successful_durations),
        "failure_duration_s": _mean(failed_durations),
        "inference": _aggregate_metric(baseline.evaluation_runs, "inference"),
        "actions": _aggregate_metric(baseline.evaluation_runs, "actions"),
        "tracking": _aggregate_metric(baseline.evaluation_runs, "tracking"),
        "dropped": _aggregate_metric(baseline.evaluation_runs, "dropped"),
    }
    return {
        "baseline_id": baseline.baseline_id,
        "name": baseline.name,
        "robot_name": baseline.robot_name,
        "policy_label": baseline.policy_label,
        "planned_episodes": baseline.evaluation_protocol["planned_episodes"],
        "live": live,
        "open_loop": [_open_loop_row(reference) for reference in baseline.open_loop_runs],
    }


def _rate(numerator: int, denominator: int, sample_size: int) -> dict[str, Any]:
    return {
        "percentage": (100.0 * numerator / denominator) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "sample_size": sample_size,
    }


def _aggregate_metric(references: Iterable[Any], name: str) -> dict[str, Any]:
    values: list[Mapping[str, Any]] = []
    for reference in references:
        candidate = reference.summary.get(name)
        if isinstance(candidate, Mapping):
            values.append(candidate)
    return {"available": bool(values), "samples": [dict(value) for value in values]} if values else {"available": False}


def _open_loop_row(reference: Any) -> dict[str, Any]:
    summary = reference.summary
    return {
        "evaluation_id": reference.evaluation_id,
        "status": reference.status,
        "valid_prediction_count": summary.get("valid_prediction_count"),
        "metrics": summary.get("metrics"),
        "comparison_contract": summary.get("comparison_contract"),
        "config_fingerprint": reference.config_fingerprint,
    }


def _open_loop_comparability(baselines: tuple[Baseline, ...]) -> dict[str, Any]:
    references = [reference for baseline in baselines for reference in baseline.open_loop_runs]
    if not references:
        return {"available": False, "comparable": False, "reason": "no attached open-loop results"}
    if any(not baseline.open_loop_runs for baseline in baselines):
        return {
            "available": True,
            "comparable": False,
            "reason": "Every selected Baseline needs an attached open-loop result before open-loop comparison.",
            "reference_count": len(references),
        }
    contracts = [_canonical(reference.summary.get("comparison_contract", {})) for reference in references]
    distinct = set(contracts)
    return {
        "available": True,
        "comparable": len(distinct) == 1,
        "reason": (
            None
            if len(distinct) == 1
            else "Open-loop target source, alignment, dataset, ActionSpec, state schema, or camera roles differ."
        ),
        "reference_count": len(references),
    }


def _contract_key(baseline: Baseline) -> tuple[Any, ...]:
    return (
        baseline.robot_name,
        baseline.dataset_format,
        _canonical(baseline.action_spec),
        baseline.state_schema,
        baseline.camera_roles,
    )


def _contract_reasons(baselines: tuple[Baseline, ...]) -> list[str]:
    first = baselines[0]
    values: list[str] = []
    for candidate in baselines[1:]:
        if candidate.robot_name != first.robot_name:
            values.append("robot_name differs")
        if candidate.dataset_format != first.dataset_format:
            values.append("dataset_format differs")
        if candidate.action_spec != first.action_spec:
            values.append("ActionSpec differs")
        if candidate.state_schema != first.state_schema:
            values.append("state schema differs")
        if candidate.camera_roles != first.camera_roles:
            values.append("camera roles differ")
    return list(dict.fromkeys(values))


def _canonical(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _extend_numbers(destination: list[float], value: Any) -> None:
    if isinstance(value, list | tuple):
        for item in value:
            try:
                destination.append(float(item))
            except (TypeError, ValueError):
                continue


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
