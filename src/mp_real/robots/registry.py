from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any

from mp_real.robots.base import Robot

RobotFactory = Callable[[Any], Robot]
_FACTORIES: dict[str, RobotFactory] = {}
_BUILTIN_MODULES = {
    "piper": "mp_real.robots.piper.infer",
    "rm2": "mp_real.robots.rm2.infer",
}


def register_robot(name: str, factory: RobotFactory) -> None:
    if name in _FACTORIES:
        raise ValueError(f"Robot factory already registered: {name}")
    _FACTORIES[name] = factory


def create_robot(name: str, config: Any) -> Robot:
    if name not in _FACTORIES and name in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[name])
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_FACTORIES)) or "none"
        raise ValueError(f"Unknown robot {name!r}; available: {available}") from exc
    return factory(config)


def available_robots() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))
