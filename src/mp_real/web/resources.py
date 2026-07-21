from __future__ import annotations

import dataclasses
import enum
import threading
import uuid
from collections.abc import Iterable


class ResourceType(enum.StrEnum):
    """Exclusive resources coordinated by one Web-process lifecycle."""

    ROBOT_CONTROL = "robot_control"
    CAMERAS = "cameras"
    POLICY_CLIENT = "policy_client"
    RECORDED_DATA = "recorded_data"


@dataclasses.dataclass(frozen=True)
class ResourceRequest:
    """A resource type plus its concrete scope (robot, endpoint, or session)."""

    resource_type: ResourceType
    scope: str

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("resource scope cannot be empty")


class ResourceLeaseConflict(RuntimeError):
    """A requested resource is held by a different lifecycle owner."""


class ResourceLease:
    """Idempotent handle returned from :class:`ResourceLeaseManager`."""

    def __init__(self, manager: ResourceLeaseManager, owner_id: str, requests: tuple[ResourceRequest, ...]) -> None:
        self._manager = manager
        self.owner_id = owner_id
        self.lease_id = uuid.uuid4().hex
        self.requests = requests
        self._released = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._manager._release(self.owner_id, self.requests)

    def replace(self, requests: Iterable[ResourceRequest]) -> ResourceLease:
        """Atomically change this owner's resource set without a release gap."""
        with self._lock:
            if self._released:
                raise RuntimeError("cannot replace a released resource lease")
            replacement = self._manager._replace(self.owner_id, self.requests, requests)
            self._released = True
            return replacement

    def __enter__(self) -> ResourceLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class ResourceLeaseManager:
    """Atomically own resource scopes for Web runtime sessions.

    This manager deliberately coordinates in-process Web lifecycles.  It owns
    no robot SDK, policy client, or camera implementation; callers must retain
    and release the returned lease during their resource lifecycle.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: dict[ResourceRequest, str] = {}

    def acquire(self, owner_id: str, requests: Iterable[ResourceRequest]) -> ResourceLease:
        if not owner_id:
            raise ValueError("resource owner_id cannot be empty")
        normalized = tuple(dict.fromkeys(requests))
        if not normalized:
            raise ValueError("at least one resource request is required")
        with self._lock:
            conflicts = [
                (request, owner)
                for request in normalized
                if (owner := self._owners.get(request)) is not None and owner != owner_id
            ]
            if conflicts:
                request, owner = conflicts[0]
                raise ResourceLeaseConflict(
                    f"{request.resource_type.value}:{request.scope} is already owned by {owner}"
                )
            for request in normalized:
                self._owners[request] = owner_id
        return ResourceLease(self, owner_id, normalized)

    def owner_of(self, request: ResourceRequest) -> str | None:
        with self._lock:
            return self._owners.get(request)

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {
                f"{request.resource_type.value}:{request.scope}": owner
                for request, owner in sorted(
                    self._owners.items(), key=lambda item: (item[0].resource_type.value, item[0].scope)
                )
            }

    def _release(self, owner_id: str, requests: tuple[ResourceRequest, ...]) -> None:
        with self._lock:
            for request in requests:
                if self._owners.get(request) == owner_id:
                    del self._owners[request]

    def _replace(
        self,
        owner_id: str,
        previous: tuple[ResourceRequest, ...],
        requested: Iterable[ResourceRequest],
    ) -> ResourceLease:
        normalized = tuple(dict.fromkeys(requested))
        if not normalized:
            raise ValueError("at least one resource request is required")
        with self._lock:
            # A stale handle must never delete a resource that another
            # lifecycle acquired after the original owner released it.
            if any(self._owners.get(request) != owner_id for request in previous):
                raise ResourceLeaseConflict("resource lease is no longer owned by this lifecycle")
            conflicts = [
                (request, owner)
                for request in normalized
                if (owner := self._owners.get(request)) is not None and owner != owner_id
            ]
            if conflicts:
                request, owner = conflicts[0]
                raise ResourceLeaseConflict(
                    f"{request.resource_type.value}:{request.scope} is already owned by {owner}"
                )
            for request in previous:
                if request not in normalized and self._owners.get(request) == owner_id:
                    del self._owners[request]
            for request in normalized:
                self._owners[request] = owner_id
        return ResourceLease(self, owner_id, normalized)
