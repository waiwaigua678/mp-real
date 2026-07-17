"""Versioned recorded-state mapping configuration loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mp_real.pose.models import MappingEntry, PoseMappingConfig


def load_pose_mapping_config(path: Path) -> PoseMappingConfig:
    """Load one explicit, versioned state mapping from a local JSON file."""
    with path.open(encoding="utf-8") as stream:
        payload: Any = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("pose mapping config must be a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("pose mapping config entries must be a JSON list")
    return PoseMappingConfig(
        version=int(payload["version"]),
        entries=tuple(MappingEntry(**item) for item in entries),
        source_robot_name=payload.get("source_robot_name"),
        target_robot_name=payload.get("target_robot_name"),
        metadata=dict(payload.get("metadata", {})),
    )
