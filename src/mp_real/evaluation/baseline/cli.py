"""The ``mp-baseline`` command-line interface."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mp_real.evaluation.baseline.service import BaselineService
from mp_real.evaluation.baseline.store import BaselineStore


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create, compare, and attach reproducible evaluation Baselines")
    parser.add_argument("--store-root", type=Path, default=Path("recordings/baselines"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("baseline_id")
    create = commands.add_parser("create")
    create.add_argument("--config", type=Path, help="Baseline metadata JSON")
    create.add_argument("--runtime-config", type=Path, help="Sanitized Web runtime config JSON")
    create.add_argument("--git-commit")
    create.add_argument("--from-evaluation-snapshot", type=Path, help="Existing EvaluationSession snapshot JSON")
    clone = commands.add_parser("clone")
    clone.add_argument("baseline_id")
    clone.add_argument("--patch", required=True, type=Path)
    clone.add_argument("--reason", required=True)
    diff = commands.add_parser("diff")
    diff.add_argument("baseline_a")
    diff.add_argument("baseline_b")
    compare = commands.add_parser("compare")
    compare.add_argument("baseline_ids", nargs="+")
    run = commands.add_parser("run")
    run.add_argument("baseline_id")
    run.add_argument("--web-url", required=True, help="Robot Web base URL; creates but never starts an evaluation")
    run.add_argument("--access-key")
    attach_evaluation = commands.add_parser("attach-evaluation")
    attach_evaluation.add_argument("baseline_id")
    attach_evaluation.add_argument("--snapshot", required=True, type=Path)
    attach_open_loop = commands.add_parser("attach-open-loop")
    attach_open_loop.add_argument("baseline_id")
    attach_open_loop.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    service = BaselineService(BaselineStore(args.store_root))

    if args.command == "list":
        _print(
            [
                {
                    "baseline_id": item.baseline_id,
                    "name": item.name,
                    "robot_name": item.robot_name,
                    "task_name": item.task_name,
                    "policy_label": item.policy_label,
                    "created_at": item.created_at,
                    "evaluation_runs": len(item.evaluation_runs),
                    "open_loop_runs": len(item.open_loop_runs),
                }
                for item in service.list()
            ]
        )
    elif args.command == "show":
        _print(service.get(args.baseline_id).to_dict())
    elif args.command == "create":
        if args.from_evaluation_snapshot is not None:
            if args.config is not None or args.runtime_config is not None or args.git_commit is not None:
                parser.error(
                    "--from-evaluation-snapshot cannot be combined with --config, --runtime-config, or --git-commit"
                )
            _print(service.create_from_evaluation(_object(args.from_evaluation_snapshot)).to_dict())
        else:
            if args.config is None or args.runtime_config is None or args.git_commit is None:
                parser.error("create requires --config, --runtime-config, and --git-commit")
            _print(
                service.create_from_runtime(
                    _object(args.config), runtime_config=_object(args.runtime_config), git_commit=args.git_commit
                ).to_dict()
            )
    elif args.command == "clone":
        _print(service.clone(args.baseline_id, _object(args.patch), derived_reason=args.reason).to_dict())
    elif args.command == "diff":
        _print(service.diff(args.baseline_a, args.baseline_b).to_dict())
    elif args.command == "compare":
        _print(service.compare(args.baseline_ids))
    elif args.command == "run":
        url = args.web_url.rstrip("/") + f"/api/baselines/{args.baseline_id}/run"
        headers = {"Content-Type": "application/json"}
        if args.access_key:
            headers["X-Motrix-Key"] = args.access_key
        request = urllib.request.Request(url, data=b"{}", headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                _print(json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Baseline run request failed: {exc.code} {detail}") from exc
    elif args.command == "attach-evaluation":
        _print(service.attach_evaluation(args.baseline_id, _object(args.snapshot)).to_dict())
    elif args.command == "attach-open-loop":
        _print(service.attach_open_loop(args.baseline_id, args.result_dir).to_dict())
    else:  # pragma: no cover - argparse makes this unreachable.
        parser.error(f"unsupported command {args.command}")
    return 0


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
