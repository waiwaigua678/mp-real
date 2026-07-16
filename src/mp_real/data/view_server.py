"""Standalone HTTP server for the read-only offline episode viewer.

This module intentionally imports only the data-view stack.  In particular it
does not import a robot profile, robot registry, camera backend, or policy
client, so ``mp-data-view`` is safe to run on a recording workstation.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pathlib
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

from mp_real.data.view import DataViewError, DataViewSession


class OfflineDataViewHandler(BaseHTTPRequestHandler):
    server: OfflineDataViewServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path in {"/", "/data-view"}:
                self._send_file(self.server.static_dir / "data_view.html")
                return
            if parsed.path.startswith("/static/"):
                self._send_file(self.server.static_dir / parsed.path.removeprefix("/static/"))
                return
            if not parsed.path.startswith("/api/data-view/"):
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            self._handle_api_get(parsed.path, urllib.parse.parse_qs(parsed.query))
        except DataViewError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            return
        except Exception as exc:
            logging.exception("offline data view GET failed")
            self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path != "/api/data-view/selection":
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            payload = self._read_json()
            self._send_json(
                {
                    "ok": True,
                    **self.server.viewer.select(
                        str(payload["dataset_id"]),
                        int(payload["episode_index"]),
                        int(payload["sample_index"]),
                        playing=bool(payload.get("playing", False)),
                        playback_rate=float(payload.get("playback_rate", 1.0)),
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
        except DataViewError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logging.exception("offline data view POST failed")
            self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/data-view/datasets":
            self._send_json({"ok": True, "datasets": self.server.viewer.datasets()})
            return
        if path == "/api/data-view/selection":
            self._send_json({"ok": True, **self.server.viewer.selection()})
            return
        parts = path.split("/")
        # /api/data-view/datasets/{id}/episodes[/index/{operation}]
        if len(parts) < 6 or parts[3] != "datasets" or parts[5] != "episodes":
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        dataset_id = parts[4]
        if len(parts) == 6:
            self._send_json({"ok": True, "episodes": self.server.viewer.episodes(dataset_id)})
            return
        if len(parts) < 8:
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        episode_index = int(parts[6])
        operation = parts[7]
        if operation == "metadata":
            self._send_json({"ok": True, **self.server.viewer.episode_metadata(dataset_id, episode_index)})
        elif operation == "sample":
            self._send_json(
                {
                    "ok": True,
                    **self.server.viewer.sample(dataset_id, episode_index, self._query_int(query, "sample_index")),
                }
            )
        elif operation == "sample-at":
            self._send_json(
                {
                    "ok": True,
                    **self.server.viewer.sample_at_timestamp(
                        dataset_id, episode_index, self._query_float(query, "timestamp")
                    ),
                }
            )
        elif operation == "frame":
            frame, metadata = self.server.viewer.camera_frame(
                dataset_id,
                episode_index,
                self._query_int(query, "sample_index"),
                self._query_text(query, "role"),
            )
            self._send_jpeg(frame, metadata)
        elif operation == "curves":
            requested = tuple(item for item in self._query_text(query, "series", "action").split(",") if item)
            max_points = int(self._query_text(query, "max_points", "600"))
            self._send_json(
                {
                    "ok": True,
                    **self.server.viewer.curves(
                        dataset_id, episode_index, series=requested, max_points=max_points
                    ),
                }
            )
        elif operation == "events":
            limit = int(self._query_text(query, "limit", "2000"))
            self._send_json({"ok": True, **self.server.viewer.runtime_events(dataset_id, episode_index, limit=limit)})
        elif operation == "metrics":
            self._send_json({"ok": True, **self.server.viewer.metrics(dataset_id, episode_index)})
        else:
            self._error("not found", HTTPStatus.NOT_FOUND)

    @staticmethod
    def _query_text(query: dict[str, list[str]], name: str, default: str | None = None) -> str:
        values = query.get(name)
        if not values:
            if default is None:
                raise DataViewError(f"missing query parameter: {name}")
            return default
        return values[0]

    def _query_int(self, query: dict[str, list[str]], name: str) -> int:
        return int(self._query_text(query, name))

    def _query_float(self, query: dict[str, list[str]], name: str) -> float:
        return float(self._query_text(query, name))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_jpeg(self, frame: np.ndarray, metadata: dict[str, Any]) -> None:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
        payload = io.BytesIO()
        image.save(payload, format="JPEG", quality=90, optimize=True)
        data = payload.getvalue()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Index", str(metadata["frame_index"]))
        self.send_header("X-Frame-Id", str(metadata["frame_id"]))
        self.send_header("X-Rendered-Frame-Index", str(metadata.get("rendered_frame_index", metadata["frame_index"])))
        self.send_header("X-Frame-Reused", "true" if metadata.get("frame_reused") else "false")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: pathlib.Path) -> None:
        root = self.server.static_dir.resolve()
        resolved = path.resolve()
        if root not in resolved.parents or not resolved.is_file():
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(resolved.suffix, "application/octet-stream")
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message: str, status: HTTPStatus) -> None:
        self._send_json({"ok": False, "error": message}, status)


class OfflineDataViewServer(ThreadingHTTPServer):
    """Threaded local server whose only resource is read-only recorded data."""

    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], viewer: DataViewSession) -> None:
        super().__init__(server_address, OfflineDataViewHandler)
        self.viewer = viewer
        packaged_static = pathlib.Path(__file__).resolve().parent.parent / "web" / "static"
        source_static = pathlib.Path(__file__).resolve().parents[3] / "static"
        self.static_dir = packaged_static if packaged_static.is_dir() else source_static

    def server_close(self) -> None:
        try:
            self.viewer.close()
        finally:
            super().server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only LeRobot v2.1 offline episode viewer (no robot or policy access)"
    )
    parser.add_argument(
        "--storage-root",
        action="append",
        type=pathlib.Path,
        dest="storage_roots",
        help="Dataset storage root, or a dataset directory itself. May be repeated.",
    )
    parser.add_argument("--dataset", help="Optional dataset name or catalog dataset ID to select on first load.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to select on first load.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("--episode must be non-negative")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    viewer = DataViewSession(args.storage_roots or [pathlib.Path("recordings")])
    if args.dataset:
        candidates = [
            item for item in viewer.datasets() if args.dataset in {item["dataset_id"], item["name"]}
        ]
        if not candidates:
            parser.error(f"dataset {args.dataset!r} was not found under the selected storage roots")
        try:
            viewer.select(candidates[0]["dataset_id"], args.episode, 0)
        except (DataViewError, KeyError, IndexError) as exc:
            parser.error(str(exc))
    server = OfflineDataViewServer((args.host, args.port), viewer)
    logging.info("Offline data viewer: http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping offline data viewer")
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
