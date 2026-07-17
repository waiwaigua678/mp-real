"""Action-chunk metrics with explicit valid masks and ActionSpec semantics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from mp_real.runtime.models import ActionSpec


def compute_open_loop_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    *,
    target_indices: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    action_spec: ActionSpec,
) -> dict[str, Any]:
    """Compute metrics without treating invalid tail/timestamp cells as zero."""

    if predicted.shape != targets.shape or predicted.ndim != 3:
        raise ValueError("predicted and targets must have matching [request, horizon, dimension] shapes")
    if valid_mask.shape != predicted.shape[:2]:
        raise ValueError("valid_mask must have shape [request, horizon]")
    if predicted.shape[2] != action_spec.action_dim:
        raise ValueError("prediction action dimension must match ActionSpec")
    valid_values = valid_mask[..., None]
    error = predicted - targets
    flattened_error = error[valid_values.repeat(predicted.shape[2], axis=2)].reshape(-1, predicted.shape[2])
    flattened_prediction = predicted[valid_values.repeat(predicted.shape[2], axis=2)].reshape(-1, predicted.shape[2])
    flattened_target = targets[valid_values.repeat(predicted.shape[2], axis=2)].reshape(-1, predicted.shape[2])
    values: dict[str, Any] = {
        "valid_prediction_count": int(valid_mask.sum()),
        "valid_action_value_count": int(valid_mask.sum() * action_spec.action_dim),
        "per_dimension": _per_dimension(flattened_error, action_spec),
        "overall_mae": _mean_abs(flattened_error),
        "overall_rmse": _rmse(flattened_error),
        "joint_l2_error": _joint_l2(flattened_error, action_spec),
        "first_action_error": _first_action(error, valid_mask, action_spec),
        "max_action_error": _max_abs(error, valid_mask, action_spec),
        "action_direction_cosine_similarity": _cosine(flattened_prediction, flattened_target),
        "horizons": _horizon_metrics(error, valid_mask, action_spec),
    }
    values["temporal"] = _temporal_metrics(
        predicted[:, 0],
        targets[:, 0],
        valid_mask[:, 0],
        target_indices[:, 0],
        source_timestamps,
        target_timestamps[:, 0],
    )
    values["gripper"] = _gripper_metrics(
        predicted[:, 0],
        targets[:, 0],
        valid_mask[:, 0],
        target_timestamps[:, 0],
        action_spec,
    )
    values["chunks"] = _chunk_metrics(predicted, valid_mask, target_indices)
    return values


def _per_dimension(error: np.ndarray, spec: ActionSpec) -> list[dict[str, Any]]:
    names = spec.action_field_names or tuple(f"dim_{index}" for index in range(spec.action_dim))
    return [
        {
            "name": names[index],
            "mae": _mean_abs(error[:, index]),
            "rmse": _rmse(error[:, index]),
            "valid_count": int(len(error)),
        }
        for index in range(spec.action_dim)
    ]


def _horizon_metrics(error: np.ndarray, valid: np.ndarray, spec: ActionSpec) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    joint_indices = _joint_indices(spec)
    for horizon in range(error.shape[1]):
        values = error[valid[:, horizon], horizon]
        result.append(
            {
                "horizon": horizon,
                "valid_sample_count": int(len(values)),
                "mae": _mean_abs(values),
                "rmse": _rmse(values),
                "l2": _l2(values, joint_indices),
            }
        )
    return result


def _first_action(error: np.ndarray, valid: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    values = error[valid[:, 0], 0]
    return {
        "valid_sample_count": int(len(values)),
        "mae": _mean_abs(values),
        "rmse": _rmse(values),
        "joint_l2": _l2(values, _joint_indices(spec)),
    }


def _joint_l2(error: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    indices = _joint_indices(spec)
    return {"indices": indices, "mean": _l2(error, indices), "max": _l2_max(error, indices)}


def _joint_indices(spec: ActionSpec) -> list[int]:
    if not spec.action_fields:
        return []
    return [index for index, field in enumerate(spec.action_fields) if field.semantics == "joint_position"]


def _max_abs(error: np.ndarray, valid: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    masked = np.where(valid[..., None], np.abs(error), np.nan)
    if not np.any(valid):
        return {"value": None, "request_index": None, "horizon": None, "dimension": None}
    flat = int(np.nanargmax(masked))
    request, horizon, dimension = np.unravel_index(flat, masked.shape)
    name = spec.action_field_names[dimension] if spec.action_fields else f"dim_{dimension}"
    return {
        "value": _float(masked[request, horizon, dimension]),
        "request_index": int(request),
        "horizon": int(horizon),
        "dimension": int(dimension),
        "dimension_name": name,
    }


def _cosine(predicted: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if len(predicted) == 0:
        return {"mean": None, "valid_count": 0, "skipped_zero_norm_count": 0}
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(targets, axis=1)
    use = denominator > 0
    scores = np.sum(predicted[use] * targets[use], axis=1) / denominator[use]
    return {
        "mean": _float(np.mean(scores)) if len(scores) else None,
        "valid_count": int(len(scores)),
        "skipped_zero_norm_count": int((~use).sum()),
    }


def _temporal_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_indices: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> dict[str, Any]:
    pairs: list[tuple[int, int, float]] = []
    for index in range(1, len(valid)):
        if not valid[index - 1] or not valid[index]:
            continue
        if target_indices[index] != target_indices[index - 1] + 1:
            continue
        dt = float(target_timestamps[index] - target_timestamps[index - 1])
        if not np.isfinite(dt) or dt <= 0:
            continue
        pairs.append((index - 1, index, dt))
    if not pairs:
        return _unavailable_temporal()
    predicted_velocity = np.asarray([(predicted[right] - predicted[left]) / dt for left, right, dt in pairs])
    target_velocity = np.asarray([(target[right] - target[left]) / dt for left, right, dt in pairs])
    velocity_error = predicted_velocity - target_velocity
    result: dict[str, Any] = {
        "pair_count": len(pairs),
        "velocity_error_mae": _mean_abs(velocity_error),
        "action_jump": _float(
            np.mean([np.linalg.norm(predicted[right] - predicted[left]) for left, right, _ in pairs])
        ),
        "predicted_smoothness": _float(np.mean(np.linalg.norm(predicted_velocity, axis=1))),
        "target_smoothness": _float(np.mean(np.linalg.norm(target_velocity, axis=1))),
    }
    if len(velocity_error) < 2:
        result.update({"acceleration_error_mae": None, "jerk_difference_mae": None})
        return result
    velocity_dt = np.asarray([pairs[index + 1][2] for index in range(len(pairs) - 1)], dtype=np.float32)
    predicted_acceleration = np.diff(predicted_velocity, axis=0) / velocity_dt[:, None]
    target_acceleration = np.diff(target_velocity, axis=0) / velocity_dt[:, None]
    acceleration_error = predicted_acceleration - target_acceleration
    result["acceleration_error_mae"] = _mean_abs(acceleration_error)
    if len(acceleration_error) < 2:
        result["jerk_difference_mae"] = None
        return result
    acceleration_dt = velocity_dt[1:]
    jerk_difference = np.diff(acceleration_error, axis=0) / acceleration_dt[:, None]
    result["jerk_difference_mae"] = _mean_abs(jerk_difference)
    del source_timestamps
    return result


def _unavailable_temporal() -> dict[str, Any]:
    return {
        "pair_count": 0,
        "velocity_error_mae": None,
        "acceleration_error_mae": None,
        "jerk_difference_mae": None,
        "action_jump": None,
        "predicted_smoothness": None,
        "target_smoothness": None,
    }


def _gripper_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    spec: ActionSpec,
) -> dict[str, Any]:
    indices = _gripper_indices(spec)
    if not indices:
        return {"available": False, "reason": "ActionSpec has no gripper semantics"}
    values = valid.nonzero()[0]
    if not len(values):
        return {"available": True, "valid_sample_count": 0, "mae": None, "classification_accuracy": None}
    pred = predicted[values][:, indices]
    actual = target[values][:, indices]
    opened_when_high = np.asarray([_open_when_high(spec, index) for index in indices])
    predicted_open = np.where(opened_when_high[None, :], pred >= 0.5, pred < 0.5)
    target_open = np.where(opened_when_high[None, :], actual >= 0.5, actual < 0.5)
    close = _event_summary(predicted_open, target_open, timestamps[values], event_is_open=False)
    opened = _event_summary(predicted_open, target_open, timestamps[values], event_is_open=True)
    return {
        "available": True,
        "indices": indices,
        "valid_sample_count": int(len(values)),
        "mae": _mean_abs(pred - actual),
        "classification_accuracy": _float(np.mean(predicted_open == target_open)),
        "close_event_timing_error_s": close["timing_error_s"],
        "open_event_timing_error_s": opened["timing_error_s"],
        "premature_close_count": close["premature_count"],
        "delayed_close_count": close["delayed_count"],
        "missed_close_count": close["missed_count"],
        "missed_open_count": opened["missed_count"],
    }


def _gripper_indices(spec: ActionSpec) -> list[int]:
    return [
        index
        for index, field in enumerate(spec.action_fields)
        if "gripper" in field.semantics.lower()
    ]


def _open_when_high(spec: ActionSpec, index: int) -> bool:
    unit = spec.action_fields[index].unit.lower()
    if "0_closed_1_open" in unit:
        return True
    if "0_open_1" in unit:
        return False
    # The semantics is an explicit openness fraction, not a positional guess.
    return spec.action_fields[index].semantics == "gripper_open_fraction"


def _event_summary(
    predicted_open: np.ndarray, target_open: np.ndarray, timestamps: np.ndarray, *, event_is_open: bool
) -> dict[str, Any]:
    predicted_events = _events(predicted_open, timestamps, event_is_open)
    target_events = _events(target_open, timestamps, event_is_open)
    count = min(len(predicted_events), len(target_events))
    differences = [predicted_events[index] - target_events[index] for index in range(count)]
    premature = sum(value < 0 for value in differences) if not event_is_open else 0
    delayed = sum(value > 0 for value in differences) if not event_is_open else 0
    return {
        "timing_error_s": _float(np.mean(np.abs(differences))) if differences else None,
        "premature_count": int(premature),
        "delayed_count": int(delayed),
        "missed_count": int(abs(len(predicted_events) - len(target_events))),
    }


def _events(open_values: np.ndarray, timestamps: np.ndarray, event_is_open: bool) -> list[float]:
    if len(open_values) < 2:
        return []
    events: list[float] = []
    for row in range(1, len(open_values)):
        changed = np.any(open_values[row] != open_values[row - 1])
        direction = np.any(open_values[row]) if event_is_open else not np.all(open_values[row])
        if changed and direction:
            events.append(float(timestamps[row]))
    return events


def _chunk_metrics(predicted: np.ndarray, valid: np.ndarray, target_indices: np.ndarray) -> dict[str, Any]:
    by_target: dict[int, list[np.ndarray]] = defaultdict(list)
    neighbor_differences: list[float] = []
    boundaries: list[float] = []
    first_differences: list[float] = []
    for request in range(predicted.shape[0]):
        for horizon in range(predicted.shape[1]):
            if valid[request, horizon] and target_indices[request, horizon] >= 0:
                by_target[int(target_indices[request, horizon])].append(predicted[request, horizon])
        if request + 1 >= predicted.shape[0]:
            continue
        if valid[request, 0] and valid[request + 1, 0]:
            first_differences.append(float(np.linalg.norm(predicted[request + 1, 0] - predicted[request, 0])))
        for horizon in range(1, predicted.shape[1]):
            if valid[request, horizon] and valid[request + 1, horizon - 1]:
                if target_indices[request, horizon] == target_indices[request + 1, horizon - 1]:
                    neighbor_differences.append(
                        float(np.linalg.norm(predicted[request, horizon] - predicted[request + 1, horizon - 1]))
                    )
        last = max((horizon for horizon in range(predicted.shape[1]) if valid[request, horizon]), default=None)
        if last is not None and valid[request + 1, 0]:
            boundaries.append(float(np.linalg.norm(predicted[request, last] - predicted[request + 1, 0])))
    variances = [float(np.mean(np.var(np.stack(items), axis=0))) for items in by_target.values() if len(items) > 1]
    return {
        "overlap_pair_count": len(neighbor_differences),
        "neighboring_chunk_same_target_l2": _float(np.mean(neighbor_differences)) if neighbor_differences else None,
        "rtc_compatible_overlap_consistency": _float(np.mean(neighbor_differences)) if neighbor_differences else None,
        "chunk_boundary_jump": _float(np.mean(boundaries)) if boundaries else None,
        "chunk_variance": _float(np.mean(variances)) if variances else None,
        "first_horizon_consistency": _float(np.mean(first_differences)) if first_differences else None,
    }


def _mean_abs(values: np.ndarray) -> float | None:
    return _float(np.mean(np.abs(values))) if values.size else None


def _rmse(values: np.ndarray) -> float | None:
    return _float(np.sqrt(np.mean(np.square(values)))) if values.size else None


def _l2(values: np.ndarray, indices: Iterable[int]) -> float | None:
    selected = list(indices)
    if not selected or not len(values):
        return None
    return _float(np.mean(np.linalg.norm(values[:, selected], axis=1)))


def _l2_max(values: np.ndarray, indices: Iterable[int]) -> float | None:
    selected = list(indices)
    if not selected or not len(values):
        return None
    return _float(np.max(np.linalg.norm(values[:, selected], axis=1)))


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
