"""Atomic filesystem store for small Baseline JSON documents."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path

from mp_real.evaluation.baseline.models import Baseline


class BaselineNotFoundError(KeyError):
    pass


class BaselineStore:
    """One Baseline per atomic JSON file; no database or browser state."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self._lock = threading.RLock()

    def list(self) -> tuple[Baseline, ...]:
        with self._lock:
            if not self.root.is_dir():
                return ()
            items = [self._read_path(path) for path in self.root.glob("*.json")]
        return tuple(sorted(items, key=lambda item: (item.created_at, item.baseline_id), reverse=True))

    def get(self, baseline_id: str) -> Baseline:
        with self._lock:
            path = self._path_for(baseline_id)
            if not path.is_file():
                raise BaselineNotFoundError(f"unknown baseline {baseline_id}")
            return self._read_path(path)

    def create(self, baseline: Baseline) -> Baseline:
        with self._lock:
            path = self._path_for(baseline.baseline_id)
            if path.exists():
                raise FileExistsError(f"baseline already exists: {baseline.baseline_id}")
            self._atomic_write(path, baseline)
        return baseline

    def replace(self, baseline: Baseline) -> Baseline:
        """Replace references only after the caller has loaded a valid Baseline."""
        with self._lock:
            path = self._path_for(baseline.baseline_id)
            if not path.is_file():
                raise BaselineNotFoundError(f"unknown baseline {baseline.baseline_id}")
            self._atomic_write(path, baseline)
        return baseline

    def _path_for(self, baseline_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", baseline_id):
            raise ValueError("baseline_id must contain only letters, digits, '.', '_' or '-'")
        return self.root / f"{baseline_id}.json"

    @staticmethod
    def _read_path(path: Path) -> Baseline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Baseline JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Baseline JSON must be an object: {path}")
        return Baseline.from_dict(payload)

    def _atomic_write(self, path: Path, baseline: Baseline) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode()
        with tempfile.NamedTemporaryFile(dir=self.root, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, path)
