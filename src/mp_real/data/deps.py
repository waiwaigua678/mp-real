from __future__ import annotations

import importlib
from typing import Any


class MissingOptionalDependencyError(ImportError):
    """Raised when a data/recording feature is used without its optional extra."""


def install_hint(*, feature: str, package: str, extra: str) -> str:
    return (
        f"{feature} requires optional dependency {package!r}. "
        f"Install it with `uv sync --extra {extra}` or `pip install 'mp-real[{extra}]'`."
    )


def import_optional(module_name: str, *, package: str | None = None, feature: str, extra: str) -> Any:
    package_name = package or module_name.split(".", 1)[0]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {package_name, module_name}:
            raise MissingOptionalDependencyError(
                install_hint(feature=feature, package=package_name, extra=extra)
            ) from exc
        raise


class OptionalModule:
    """Lazy module proxy that keeps imports cheap until a feature is actually used."""

    def __init__(self, module_name: str, *, package: str | None = None, feature: str, extra: str) -> None:
        self._module_name = module_name
        self._package = package
        self._feature = feature
        self._extra = extra
        self._module: Any | None = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = import_optional(
                self._module_name,
                package=self._package,
                feature=self._feature,
                extra=self._extra,
            )
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


def require_pyarrow(feature: str = "LeRobot data access") -> tuple[Any, Any]:
    pa = import_optional("pyarrow", feature=feature, extra="recording")
    pq = import_optional("pyarrow.parquet", package="pyarrow", feature=feature, extra="recording")
    return pa, pq


def require_av(feature: str = "LeRobot video access") -> Any:
    return import_optional("av", feature=feature, extra="recording")

