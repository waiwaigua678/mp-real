"""Command-line entry point for hardware-free teacher-forced evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mp_real.evaluation.open_loop.evaluator import OpenLoopEvaluator
from mp_real.evaluation.open_loop.models import (
    AlignmentMode,
    EvaluationRequestMode,
    OpenLoopEvaluationConfig,
    OpenLoopWarmupConfig,
    PredictionResultSource,
    StateDerivedTargetConfig,
)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-forced LeRobot v2.1 open-loop policy evaluation (never creates Robot)"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--episode", action="append", default=[], help="Episode index, range such as 3-7, or all; may repeat"
    )
    parser.add_argument("--session", action="store_true", help="Evaluate every episode in the selected dataset")
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--policy-label", required=True)
    parser.add_argument("--policy-api-key")
    parser.add_argument("--connection-timeout", type=float, default=10.0)
    parser.add_argument("--metadata-timeout", type=float, default=10.0)
    parser.add_argument(
        "--policy-config", action="append", type=Path, default=[], help="JSON policy override; may repeat"
    )
    parser.add_argument("--target-source", choices=[item.value for item in PredictionResultSource], default="action")
    parser.add_argument("--alignment", choices=[item.value for item in AlignmentMode], default="sample_index")
    parser.add_argument("--max-timestamp-error", type=float, default=0.05)
    parser.add_argument("--allow-frame-index-as-control-step", action="store_true")
    parser.add_argument("--prompt-override")
    parser.add_argument("--camera-role", action="append", default=[])
    parser.add_argument("--image-mask", action="append", default=[], metavar="ROLE=BOOL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-id", help="Stable ID required when resuming an API-created result directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mode", choices=[item.value for item in EvaluationRequestMode], default="sequential")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-timeout", type=float, default=60.0)
    parser.add_argument("--inference-timeout", type=float, default=3.0)
    parser.add_argument("--state-derived-config", type=Path)
    args = parser.parse_args()
    episodes = None if args.session else _parse_episodes(args.episode)
    derived = _load_state_derived(args.state_derived_config) if args.state_derived_config else None
    base: dict[str, Any] = {
        "dataset": args.dataset,
        "episode_indices": episodes,
        "policy_url": args.policy_url,
        "policy_label": args.policy_label,
        "prompt_override": args.prompt_override,
        "policy_api_key": args.policy_api_key,
        "connection_timeout_s": args.connection_timeout,
        "metadata_timeout_s": args.metadata_timeout,
        "warmup": OpenLoopWarmupConfig(
            enabled=not args.no_warmup,
            requests=args.warmup_requests,
            timeout_s=args.warmup_timeout,
            inference_timeout_s=args.inference_timeout,
        ),
        "target_source": PredictionResultSource(args.target_source),
        "alignment_mode": AlignmentMode(args.alignment),
        "max_timestamp_error_s": args.max_timestamp_error,
        "selected_camera_roles": tuple(args.camera_role) or None,
        "image_masks": _parse_masks(args.image_mask),
        "request_mode": EvaluationRequestMode(args.mode),
        "batch_size": args.batch_size,
        "deterministic_seed": args.seed,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "allow_frame_index_as_control_step": args.allow_frame_index_as_control_step,
        "state_derived": derived,
        "resume": args.resume,
        "limit": args.limit,
    }
    overrides = _load_policy_overrides(args.policy_config)
    if not overrides:
        overrides = [{}]
    if args.output is not None and len(overrides) > 1 and args.resume:
        parser.error("--resume with multiple --policy-config values requires separate output directories")
    for index, override in enumerate(overrides):
        values = {**base, **override}
        if "target_source" in values and not isinstance(values["target_source"], PredictionResultSource):
            values["target_source"] = PredictionResultSource(str(values["target_source"]))
        if "alignment_mode" in values and not isinstance(values["alignment_mode"], AlignmentMode):
            values["alignment_mode"] = AlignmentMode(str(values["alignment_mode"]))
        label = str(values["policy_label"])
        output = args.output
        if output is None:
            output = Path("open_loop_results") / f"{_safe_label(label)}-{index:02d}"
        elif len(overrides) > 1:
            output = output / f"{_safe_label(label)}-{index:02d}"
        values["output_dir"] = output
        values["evaluation_id"] = args.evaluation_id or output.name
        config = OpenLoopEvaluationConfig(**values)
        result = OpenLoopEvaluator(config).run()
        print(
            json.dumps(
                {"evaluation_id": result.evaluation_id, "status": result.status, "output": str(result.output_dir)}
            )
        )


def _parse_episodes(values: list[str]) -> tuple[int, ...] | None:
    if not values or "all" in values:
        return None
    result: set[int] = set()
    for value in values:
        if re.fullmatch(r"\d+", value):
            result.add(int(value))
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", value)
        if not match:
            raise ValueError(f"invalid --episode value {value!r}; use INDEX, START-END, or all")
        start, end = map(int, match.groups())
        if end < start:
            raise ValueError("episode range end must be >= start")
        result.update(range(start, end + 1))
    return tuple(sorted(result))


def _parse_masks(values: list[str]) -> dict[str, bool] | None:
    if not values:
        return None
    masks: dict[str, bool] = {}
    for value in values:
        role, separator, raw = value.partition("=")
        if not separator or raw.lower() not in {"true", "false", "1", "0"}:
            raise ValueError("--image-mask must be ROLE=true or ROLE=false")
        masks[role] = raw.lower() in {"true", "1"}
    return masks


def _load_state_derived(path: Path) -> StateDerivedTargetConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StateDerivedTargetConfig(
        converter_id=str(payload["converter_id"]),
        state_indices=tuple(int(value) for value in payload["state_indices"]),
        scale=tuple(float(value) for value in payload["scale"]),
        offset=tuple(float(value) for value in payload["offset"]),
    )


def _load_policy_overrides(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{path} must contain an object or list of objects")
            allowed = {
                "policy_url",
                "policy_label",
                "policy_api_key",
                "connection_timeout_s",
                "metadata_timeout_s",
                "replan_steps",
            }
            unknown = set(entry) - allowed
            if unknown:
                raise ValueError(f"{path} has unsupported policy override fields: {', '.join(sorted(unknown))}")
            result.append(dict(entry))
    return result


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "policy"
