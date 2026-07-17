"""Durable, isolated artifact writer for open-loop evaluation results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class PredictionResultWriter:
    """Write only beneath one result root; source LeRobot data is never opened writable."""

    def __init__(self, root: Path | str, *, resume: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self._resume = resume
        self._config_fingerprint: str | None = None
        self._prepared = False

    def prepare(self, config: Mapping[str, Any]) -> None:
        fingerprint = _fingerprint(config)
        config_path = self.root / "config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != fingerprint:
                raise ValueError("existing result directory has a different evaluation configuration")
            if not self._resume:
                raise FileExistsError("result directory already exists; pass --resume to continue it")
        else:
            if self._resume:
                raise FileNotFoundError("cannot resume: config.json does not exist")
            self.root.mkdir(parents=True, exist_ok=False)
            self._atomic_json(config_path, {**config, "config_fingerprint": fingerprint})
        (self.root / "predictions").mkdir(exist_ok=True)
        (self.root / "reports").mkdir(exist_ok=True)
        self._config_fingerprint = fingerprint
        self._prepared = True

    def has_completed_episode(self, episode_index: int) -> bool:
        path = self.root / "reports" / f"episode_{episode_index:06d}.json"
        if not path.is_file():
            return False
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return report.get("status") == "complete" and report.get("config_fingerprint") == self._config_fingerprint

    def write_episode(
        self,
        episode_index: int,
        *,
        arrays: Mapping[str, np.ndarray],
        report: Mapping[str, Any],
    ) -> None:
        self._require_prepared()
        prediction_path = self.root / "predictions" / f"episode_{episode_index:06d}.npz"
        report_path = self.root / "reports" / f"episode_{episode_index:06d}.json"
        self._atomic_npz(prediction_path, arrays)
        payload = {**report, "config_fingerprint": self._config_fingerprint}
        self._atomic_json(report_path, payload)
        line = {
            "episode_index": episode_index,
            "status": payload.get("status"),
            "valid_prediction_count": payload.get("valid_prediction_count", 0),
            "report": str(report_path.relative_to(self.root)),
            "prediction": str(prediction_path.relative_to(self.root)),
        }
        with (self.root / "per_episode.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self._require_prepared()
        self._atomic_json(self.root / "summary.json", summary)

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("PredictionResultWriter.prepare() must be called first")

    def _atomic_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default).encode()
        self._atomic_write(path, data)

    def _atomic_npz(self, path: Path, arrays: Mapping[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)


def _fingerprint(value: Mapping[str, Any]) -> str:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))
    # Resume changes only operational behavior, never the evaluated inputs.
    if isinstance(normalized.get("config"), dict):
        normalized["config"]["resume"] = False
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, default=_json_default, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
