"""Standalone HTTP server for the offline episode viewer.

This module never imports a robot profile, registry or camera backend.  An
explicitly requested open-loop job may create a hardware-neutral PolicyClient
inside its background worker, but startup and ordinary browsing create none.
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

from mp_real.data.view import DataViewError, DataViewSession, downsample_series
from mp_real.evaluation.open_loop.jobs import OpenLoopEvaluationJobManager
from mp_real.evaluation.open_loop.models import (
    AlignmentMode,
    EvaluationRequestMode,
    OpenLoopEvaluationConfig,
    PredictionResultSource,
)


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
            if parsed.path == "/api/data-view/selection":
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
                return
            if parsed.path == "/api/data-view/open-loop-evaluations":
                payload = self._read_json()
                self._send_json({"ok": True, "job": self.server.submit_open_loop_job(payload)}, HTTPStatus.ACCEPTED)
                return
            parts = parsed.path.split("/")
            is_stop = (
                len(parts) == 6
                and parts[:4] == ["", "api", "data-view", "open-loop-evaluations"]
                and parts[5] == "stop"
            )
            if is_stop:
                self._send_json({"ok": True, "job": self.server.open_loop_jobs.stop(parts[4])})
                return
            else:
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
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
        if path == "/api/data-view/open-loop-evaluations":
            self._send_json({"ok": True, "jobs": self.server.open_loop_jobs.list_status()})
            return
        open_loop_prefix = "/api/data-view/open-loop-evaluations/"
        if path.startswith(open_loop_prefix):
            parts = path.split("/")
            if len(parts) == 5:
                self._send_json({"ok": True, "job": self.server.open_loop_jobs.status(parts[4])})
                return
            if len(parts) == 7 and parts[5] == "reports":
                report_path = self.server.open_loop_jobs.report_path(parts[4], int(parts[6]))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if self._query_text(query, "curves", "0") == "1":
                    report["curves"] = self.server.open_loop_curves(parts[4], int(parts[6]))
                self._send_json({"ok": True, "report": report})
                return
            self._error("not found", HTTPStatus.NOT_FOUND)
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
    """Threaded local server with read-only data and explicit policy jobs only."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        viewer: DataViewSession,
        *,
        open_loop_output_root: pathlib.Path | str = "open_loop_results",
    ) -> None:
        super().__init__(server_address, OfflineDataViewHandler)
        self.viewer = viewer
        self.open_loop_jobs = OpenLoopEvaluationJobManager(open_loop_output_root)
        packaged_static = pathlib.Path(__file__).resolve().parent.parent / "web" / "static"
        source_static = pathlib.Path(__file__).resolve().parents[3] / "static"
        self.static_dir = packaged_static if packaged_static.is_dir() else source_static

    def server_close(self) -> None:
        try:
            self.open_loop_jobs.close()
        finally:
            try:
                self.viewer.close()
            finally:
                super().server_close()

    def submit_open_loop_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(payload["dataset_id"])
        episode = int(payload["episode_index"])
        source = self.viewer.replay_source(dataset_id)
        config = OpenLoopEvaluationConfig(
            dataset=source.get_dataset_metadata().root,
            episode_indices=(episode,),
            policy_url=str(payload["policy_url"]),
            policy_label=str(payload["policy_label"]),
            output_dir=self.open_loop_jobs.output_root / "pending",
            prompt_override=_optional_text(payload.get("prompt_override")),
            connection_timeout_s=float(payload.get("connection_timeout", 10.0)),
            metadata_timeout_s=float(payload.get("metadata_timeout", 10.0)),
            target_source=PredictionResultSource(str(payload.get("target_source", "action"))),
            alignment_mode=AlignmentMode(str(payload.get("alignment", "sample_index"))),
            max_timestamp_error_s=float(payload.get("max_timestamp_error", 0.05)),
            selected_camera_roles=tuple(str(role) for role in payload.get("camera_roles", ())) or None,
            request_mode=EvaluationRequestMode(str(payload.get("mode", "sequential"))),
            allow_frame_index_as_control_step=bool(payload.get("allow_frame_index_as_control_step", False)),
            limit=_optional_positive_int(payload.get("limit")),
        )
        return self.open_loop_jobs.submit(config)

    def open_loop_curves(self, job_id: str, episode_index: int) -> list[dict[str, Any]]:
        prediction_path = self.open_loop_jobs.prediction_path(job_id, episode_index)
        with np.load(prediction_path, allow_pickle=False) as archive:
            predicted = np.asarray(archive["predicted_chunks"], dtype=np.float32)
            target = np.asarray(archive["targets"], dtype=np.float32)
            valid = np.asarray(archive["valid_mask"], dtype=np.bool_)
        curves: list[dict[str, Any]] = []
        for dimension in range(predicted.shape[2]):
            points = valid[:, 0]
            predicted_values = np.where(points, predicted[:, 0, dimension], np.nan)
            target_values = np.where(points, target[:, 0, dimension], np.nan)
            curves.append(
                {
                    "label": f"prediction dim {dimension}",
                    "points": downsample_series(predicted_values, max_points=600),
                    "kind": "prediction",
                }
            )
            curves.append(
                {
                    "label": f"target dim {dimension}",
                    "points": downsample_series(target_values, max_points=600),
                    "kind": "target",
                }
            )
        return curves


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
    parser.add_argument(
        "--open-loop-output-root",
        type=pathlib.Path,
        default=pathlib.Path("open_loop_results"),
        help="Root for isolated teacher-forced evaluation artifacts",
    )
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
    server = OfflineDataViewServer(
        (args.host, args.port), viewer, open_loop_output_root=args.open_loop_output_root
    )
    logging.info("Offline data viewer: http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping offline data viewer")
    finally:
        server.server_close()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result.strip() else None


def _optional_positive_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("limit must be positive")
    return result


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
