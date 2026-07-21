#!/usr/bin/env python3
"""Clamp inactive LeRobot arm joints to their modal values.

The RM2 right-arm datasets keep the left arm physically stationary, but small
feedback noise in the left joint dimensions can dominate quantile normalization.
This script replaces selected joint dimensions in both observation.state and
action with the per-joint mode, making the inactive arm a true no-op.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_COLUMNS = ("observation.state", "action")
DEFAULT_DIMS = (0, 1, 2, 3, 4, 5)
LEFT_GRIPPER_DIM = 12


def parse_dims(raw: str) -> tuple[int, ...]:
    dims: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_raw, end_raw = part.split(":", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid dim range {part!r}")
            dims.extend(range(start, end))
        else:
            dims.append(int(part))
    if not dims:
        raise argparse.ArgumentTypeError("at least one dimension is required")
    if len(set(dims)) != len(dims):
        raise argparse.ArgumentTypeError(f"duplicate dimensions in {raw!r}")
    return tuple(dims)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def parquet_paths(root: Path, total_episodes: int, chunks_size: int) -> list[Path]:
    return [
        root / f"data/chunk-{episode // chunks_size:03d}/episode_{episode:06d}.parquet"
        for episode in range(total_episodes)
    ]


def load_column(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float64)


def rounded_values(values: np.ndarray, decimals: int | None) -> np.ndarray:
    if decimals is None:
        return values
    return np.round(values, decimals=decimals)


def compute_modes(
    paths: list[Path],
    columns: tuple[str, ...],
    dims: tuple[int, ...],
    *,
    decimals: int | None,
    combined: bool,
) -> dict[str, dict[int, float]] | dict[int, float]:
    if combined:
        counters = {dim: Counter() for dim in dims}
        for path in paths:
            table = pq.read_table(path, columns=list(columns))
            for column in columns:
                values = rounded_values(load_column(table, column), decimals)
                for dim in dims:
                    counters[dim].update(values[:, dim].tolist())
        return {dim: float(counters[dim].most_common(1)[0][0]) for dim in dims}

    modes: dict[str, dict[int, float]] = {}
    for column in columns:
        counters = {dim: Counter() for dim in dims}
        for path in paths:
            values = rounded_values(load_column(pq.read_table(path, columns=[column]), column), decimals)
            for dim in dims:
                counters[dim].update(values[:, dim].tolist())
        modes[column] = {dim: float(counters[dim].most_common(1)[0][0]) for dim in dims}
    return modes


def describe_modes(
    paths: list[Path],
    columns: tuple[str, ...],
    dims: tuple[int, ...],
    modes: dict[str, dict[int, float]] | dict[int, float],
    *,
    decimals: int | None,
    combined: bool,
) -> None:
    total_rows = 0
    counters = {(column, dim): Counter() for column in columns for dim in dims}
    for path in paths:
        table = pq.read_table(path, columns=list(columns))
        total_rows += table.num_rows
        for column in columns:
            values = rounded_values(load_column(table, column), decimals)
            for dim in dims:
                counters[(column, dim)].update(values[:, dim].tolist())

    print(f"rows={total_rows}")
    for dim in dims:
        target = modes[dim] if combined else None  # type: ignore[index]
        if combined:
            print(f"dim {dim}: replacement={target!r}")
        for column in columns:
            mode_value, mode_count = counters[(column, dim)].most_common(1)[0]
            replacement = target if combined else modes[column][dim]  # type: ignore[index]
            replacement_count = counters[(column, dim)][replacement]
            ratio = replacement_count / total_rows if total_rows else 0.0
            print(
                f"  {column}: current_mode={mode_value!r} current_mode_count={mode_count} "
                f"replacement_count={replacement_count} replacement_ratio={ratio:.6f}"
            )


def replace_columns(
    paths: list[Path],
    columns: tuple[str, ...],
    dims: tuple[int, ...],
    modes: dict[str, dict[int, float]] | dict[int, float],
    *,
    combined: bool,
) -> None:
    for path in paths:
        table = pq.read_table(path)
        updated = table
        for column in columns:
            values = load_column(updated, column)
            for dim in dims:
                replacement = modes[dim] if combined else modes[column][dim]  # type: ignore[index]
                values[:, dim] = replacement
            field = updated.schema.field(column)
            array = pa.array(values.tolist(), type=field.type)
            updated = updated.set_column(updated.schema.get_field_index(column), field, array)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(updated, tmp)
        tmp.replace(path)


def column_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def refresh_stats(root: Path, paths: list[Path], columns: tuple[str, ...]) -> None:
    stats_path = root / "meta/stats.json"
    episodes_stats_path = root / "meta/episodes_stats.jsonl"

    global_values = {column: [] for column in columns}
    episode_rows = read_jsonl(episodes_stats_path)
    rows_by_episode = {int(row["episode_index"]): row for row in episode_rows}

    for path in paths:
        table = pq.read_table(path, columns=list(columns) + ["episode_index"])
        episode_index = int(table["episode_index"][0].as_py())
        row = rows_by_episode[episode_index]
        row.setdefault("stats", {})
        for column in columns:
            values = load_column(table, column)
            row["stats"][column] = column_stats(values)
            global_values[column].append(values)

    stats = json.loads(stats_path.read_text())
    for column in columns:
        stats[column] = column_stats(np.vstack(global_values[column]))
    write_json(stats_path, stats)
    write_jsonl(episodes_stats_path, episode_rows)


def backup_inputs(root: Path, stamp: str) -> None:
    data_backup = root / f"data.backup_before_static_left_arm_{stamp}"
    meta_backup = root / f"meta.backup_before_static_left_arm_{stamp}"
    if data_backup.exists() or meta_backup.exists():
        raise FileExistsError(f"backup already exists: {data_backup} or {meta_backup}")
    shutil.copytree(root / "data", data_backup)
    meta_backup.mkdir()
    for name in ("stats.json", "episodes_stats.jsonl"):
        shutil.copy2(root / "meta" / name, meta_backup / name)
    print(f"backup_data={data_backup}")
    print(f"backup_meta={meta_backup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the LeRobot dataset root, e.g. ~/.cache/huggingface/lerobot/local/rm2_stack_cupv01.",
    )
    parser.add_argument(
        "--dims",
        type=parse_dims,
        default=DEFAULT_DIMS,
        help="Comma/range dimensions to replace. Default: 0:6 for RM2 left arm joints.",
    )
    parser.add_argument(
        "--include-left-gripper",
        action="store_true",
        help="Also replace dim 12, the RM2 left gripper.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated list columns to modify. Default: observation.state,action.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=None,
        help="Round values before computing modes. By default exact float values are used.",
    )
    parser.add_argument(
        "--per-column-mode",
        action="store_true",
        help="Compute a separate mode per column. Default computes one combined mode per joint across all columns.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually modify parquet and metadata files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create data/meta backups before --apply.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    info_path = root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing LeRobot info file: {info_path}")

    info = json.loads(info_path.read_text())
    total_episodes = int(info["total_episodes"])
    chunks_size = int(info.get("chunks_size", 1000))
    paths = parquet_paths(root, total_episodes, chunks_size)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing parquet files: {missing[:3]}")

    dims = tuple(args.dims) + ((LEFT_GRIPPER_DIM,) if args.include_left_gripper else ())
    columns = tuple(column.strip() for column in args.columns.split(",") if column.strip())
    combined = not args.per_column_mode

    print(f"dataset_root={root}")
    print(f"episodes={total_episodes}")
    print(f"columns={columns}")
    print(f"dims={dims}")
    print(f"mode_source={'combined' if combined else 'per-column'}")
    print(f"round_decimals={args.round_decimals}")

    modes = compute_modes(paths, columns, dims, decimals=args.round_decimals, combined=combined)
    describe_modes(paths, columns, dims, modes, decimals=args.round_decimals, combined=combined)

    if not args.apply:
        print("dry_run=true; add --apply to modify files")
        return 0

    if not args.no_backup:
        backup_inputs(root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    replace_columns(paths, columns, dims, modes, combined=combined)
    refresh_stats(root, paths, columns)
    print("updated=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
