from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mp_real.data.lerobot_v21 import CODEBASE_VERSION, validate_lerobot_v21_dataset
from mp_real.data.models import EpisodeStatus


@dataclasses.dataclass(frozen=True)
class CatalogEpisode:
    dataset_root: Path
    dataset_name: str
    episode_index: int
    length: int
    robot_name: str
    tasks: tuple[str, ...]
    status: EpisodeStatus
    is_mp_real: bool
    result: str | None


@dataclasses.dataclass(frozen=True)
class CatalogDataset:
    root: Path
    name: str
    robot_name: str
    status: EpisodeStatus
    is_mp_real: bool
    episode_count: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class RecordedDataCatalog:
    """Cached, path-safe catalog for local LeRobot v2.1 storage roots."""

    def __init__(self, storage_roots: Sequence[Path | str]) -> None:
        if not storage_roots:
            raise ValueError("storage_roots cannot be empty")
        self._roots = tuple(Path(root).expanduser().resolve(strict=False) for root in storage_roots)
        self._lock = threading.RLock()
        self._cache: dict[Path, tuple[tuple[int, int], CatalogDataset, tuple[CatalogEpisode, ...]]] = {}

    @property
    def storage_roots(self) -> tuple[Path, ...]:
        return self._roots

    def resolve_dataset_path(self, root: Path | str, relative_path: Path | str) -> Path:
        storage_root = Path(root).expanduser().resolve(strict=False)
        if storage_root not in self._roots:
            raise ValueError("Unknown storage root")
        candidate = (storage_root / relative_path).resolve(strict=False)
        if not candidate.is_relative_to(storage_root):
            raise ValueError("Path traversal outside storage root is not allowed")
        return candidate

    def scan(self, *, force: bool = False) -> tuple[CatalogDataset, ...]:
        datasets: list[CatalogDataset] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
                info_path = candidate / "meta" / "info.json"
                if not info_path.is_file():
                    continue
                dataset, _ = self._load_dataset(candidate, force=force)
                datasets.append(dataset)
        return tuple(datasets)

    def list_episodes(
        self,
        *,
        robot_name: str | None = None,
        task: str | None = None,
        result: str | None = None,
        status: EpisodeStatus | None = None,
    ) -> tuple[CatalogEpisode, ...]:
        episodes: list[CatalogEpisode] = []
        for dataset in self.scan():
            _, candidates = self._load_dataset(dataset.root, force=False)
            episodes.extend(candidates)
        filtered = [
            episode
            for episode in episodes
            if (robot_name is None or episode.robot_name == robot_name)
            and (task is None or any(task.casefold() in item.casefold() for item in episode.tasks))
            and (result is None or episode.result == result)
            and (status is None or episode.status is status)
        ]
        return tuple(filtered)

    def _load_dataset(self, candidate: Path, *, force: bool) -> tuple[CatalogDataset, tuple[CatalogEpisode, ...]]:
        info_path = candidate / "meta" / "info.json"
        fingerprint = (info_path.stat().st_mtime_ns, info_path.stat().st_size)
        with self._lock:
            cached = self._cache.get(candidate)
            if cached is not None and cached[0] == fingerprint and not force:
                return cached[1], cached[2]
        info = _read_json(info_path)
        incomplete = candidate.name.endswith(".inprogress") or (
            candidate / "meta" / "mp_real" / "recovery.json"
        ).exists()
        if info.get("codebase_version") != CODEBASE_VERSION:
            state = EpisodeStatus.CORRUPTED
        elif incomplete:
            state = EpisodeStatus.INCOMPLETE
        else:
            report = validate_lerobot_v21_dataset(candidate, check_videos=False)
            state = EpisodeStatus.COMPLETE if report.valid else EpisodeStatus.CORRUPTED
        is_mp_real = (candidate / "meta" / "mp_real" / "schema.json").is_file()
        episodes_json = _read_jsonl(candidate / "meta" / "episodes.jsonl")
        labels = {
            int(item["episode_index"]): item
            for item in _read_jsonl(candidate / "meta" / "mp_real" / "episode_labels.jsonl")
        }
        dataset = CatalogDataset(
            root=candidate,
            name=candidate.name.removesuffix(".inprogress"),
            robot_name=str(info.get("robot_type", "unknown")),
            status=state,
            is_mp_real=is_mp_real,
            episode_count=len(episodes_json),
        )
        episodes = tuple(
            CatalogEpisode(
                dataset_root=candidate,
                dataset_name=dataset.name,
                episode_index=int(item["episode_index"]),
                length=int(item.get("length", 0)),
                robot_name=dataset.robot_name,
                tasks=tuple(str(task_item) for task_item in item.get("tasks", ())),
                status=state,
                is_mp_real=is_mp_real,
                result=labels.get(int(item["episode_index"]), {}).get("result"),
            )
            for item in episodes_json
        )
        with self._lock:
            self._cache[candidate] = (fingerprint, dataset, episodes)
        return dataset, episodes
