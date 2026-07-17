"""Synchronous websocket policy client with no robot or Web-server dependency."""

from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Any

import websockets.sync.client

from mp_real.policy_client import msgpack_numpy


class PolicyClient:
    """One-request-at-a-time OpenPI websocket client.

    Construction performs the connection and metadata handshake so callers can
    report those latencies separately from warmup and live inference.  It is
    intentionally hardware-neutral and is shared by deployment and offline
    evaluation code.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str | None,
        *,
        timeout: float,
        metadata_timeout: float | None = None,
    ) -> None:
        self.uri = self._normalize_uri(server_url)
        self.timeout = timeout
        self.connect_latency_ms: float | None = None
        self.metadata_latency_ms: float | None = None
        self._packer = msgpack_numpy.Packer()
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        connect_kwargs = {
            "compression": None,
            "max_size": None,
            "additional_headers": headers,
        }
        connect_started_ns = time.monotonic_ns()
        try:
            self._ws = websockets.sync.client.connect(self.uri, open_timeout=timeout, **connect_kwargs)
        except TypeError:
            # Compatibility with older websockets releases that lack
            # ``open_timeout``.  In supported deployments the first branch is
            # used and preserves the configured connection timeout.
            self._ws = websockets.sync.client.connect(self.uri, **connect_kwargs)
        self.connect_latency_ms = (time.monotonic_ns() - connect_started_ns) / 1e6

        metadata_started_ns = time.monotonic_ns()
        previous_timeout = self.timeout
        self.timeout = metadata_timeout if metadata_timeout is not None else timeout
        try:
            self.metadata = msgpack_numpy.unpackb(self._recv())
        finally:
            self.timeout = previous_timeout
        self.metadata_latency_ms = (time.monotonic_ns() - metadata_started_ns) / 1e6

    @staticmethod
    def _normalize_uri(server_url: str) -> str:
        server_url = server_url.strip()
        if not server_url:
            raise ValueError("server_url cannot be empty")
        parsed = urllib.parse.urlparse(server_url)
        return server_url if parsed.scheme else f"ws://{server_url}"

    def _recv(self) -> bytes | str:
        try:
            return self._ws.recv(timeout=self.timeout)
        except TypeError:
            return self._ws.recv()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(observation))
        response = self._recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def set_timeout(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("Policy timeout must be positive")
        self.timeout = timeout_s

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            logging.exception("Failed to close policy websocket")
