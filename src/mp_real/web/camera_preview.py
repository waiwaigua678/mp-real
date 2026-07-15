from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from mp_real.common.camera import Camera
from mp_real.common.image import preprocess_image
from mp_real.web.profiles import RobotWebProfile, clone_default_args, close_profile_cameras, get_web_profile


@dataclasses.dataclass(frozen=True)
class PreviewFrame:
    image: np.ndarray | None = None
    sequence: int = 0
    timestamp_monotonic_ns: int = 0
    error: str | None = None


class CameraPreviewSession:
    """Camera-only lifecycle shared by the standalone preview command.

    It owns only cameras created by a ``RobotWebProfile``.  In particular it
    has no Robot, PolicyClient, or robot-SDK construction path.
    """

    def __init__(self, profile: RobotWebProfile, args: Any, *, stream_fps: float = 10.0) -> None:
        if stream_fps <= 0:
            raise ValueError("stream_fps must be positive")
        self.profile = profile
        self.args = args
        self.stream_fps = stream_fps
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._cameras: dict[str, Camera] = {}
        self._frames = {role: PreviewFrame() for role in profile.camera_roles_for_args(args)}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
        cameras = self.profile.create_cameras(self.args)
        with self._lock:
            self._cameras = cameras
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"{self.profile.robot_name}-camera-preview",
                daemon=False,
            )
            self._thread.start()

    def stop(self, *, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        stopped = thread is None or not thread.is_alive()
        if not stopped:
            return False
        with self._lock:
            cameras = self._cameras
            self._cameras = {}
            self._thread = None
        if cameras:
            close_profile_cameras(cameras)
        return True

    def wait_for_frames(self, *, timeout: float) -> Mapping[str, PreviewFrame]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if all(frame.image is not None or frame.error is not None for frame in self._frames.values()):
                    return dict(self._frames)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return dict(self._frames)
                self._condition.wait(min(0.1, remaining))

    def frames(self) -> Mapping[str, PreviewFrame]:
        with self._lock:
            return dict(self._frames)

    def save_preview(self, directory: pathlib.Path) -> list[pathlib.Path]:
        from PIL import Image

        directory.mkdir(parents=True, exist_ok=True)
        saved: list[pathlib.Path] = []
        for role, frame in self.frames().items():
            if frame.image is None:
                continue
            target = directory / f"{role}.jpg"
            Image.fromarray(frame.image).save(target, format="JPEG")
            saved.append(target)
        return saved

    def _run(self) -> None:
        period = 1.0 / self.stream_fps
        while not self._stop_event.is_set():
            started = time.monotonic()
            with self._lock:
                cameras = dict(self._cameras)
            for role, camera in cameras.items():
                if self._stop_event.is_set():
                    break
                try:
                    frame = camera.read_frame(timeout=float(self.args.camera_timeout))
                    image = preprocess_image(frame.image, int(self.args.resize_size))
                    with self._condition:
                        previous = self._frames.get(role, PreviewFrame())
                        self._frames[role] = PreviewFrame(
                            image=image,
                            sequence=previous.sequence + 1,
                            timestamp_monotonic_ns=frame.timestamp_monotonic_ns,
                        )
                        self._condition.notify_all()
                except BaseException as exc:
                    with self._condition:
                        previous = self._frames.get(role, PreviewFrame())
                        self._frames[role] = dataclasses.replace(previous, error=f"{type(exc).__name__}: {exc}")
                        self._condition.notify_all()
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))


def _parse_role_values(values: list[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        role, separator, selected = value.partition("=")
        if not separator or not role or not selected:
            raise ValueError(f"{option} must use ROLE=VALUE, got {value!r}")
        parsed[role] = selected
    return parsed


def _apply_camera_overrides(
    profile: RobotWebProfile,
    args: Any,
    backends: Mapping[str, str],
    selectors: Mapping[str, str],
) -> None:
    roles = set(profile.camera_roles_for_args(args))
    unknown = (set(backends) | set(selectors)) - roles
    if unknown:
        raise ValueError(f"Unknown {profile.robot_name} camera role(s): {', '.join(sorted(unknown))}")
    if profile.robot_name == "piper":
        for role, backend in backends.items():
            setattr(args, f"{role}_backend", backend)
        for role, selector in selectors.items():
            setattr(args, role, selector)
    elif profile.robot_name == "rm2":
        selected_backends = set(backends.values())
        if len(selected_backends) > 1:
            raise ValueError("RM2 uses one camera backend for all of its real camera roles")
        if selected_backends:
            args.camera_backend = selected_backends.pop()
        selector_fields = {
            "left_color": "cam_left_topic" if args.camera_backend == "ros" else "cam_left_serial",
            "right_color": "cam_right_topic" if args.camera_backend == "ros" else "cam_right_serial",
            "head_color": "cam_head_topic" if args.camera_backend == "ros" else "cam_head_serial",
        }
        for role, selector in selectors.items():
            setattr(args, selector_fields[role], selector)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Start a camera-only preview without creating a robot or policy client"
    )
    parser.add_argument("--robot", choices=("piper", "rm2"), default="piper")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds.")
    parser.add_argument("--no-web", action="store_true", help="Only open and check configured cameras.")
    parser.add_argument("--save-preview", type=pathlib.Path, default=None, metavar="DIR")
    parser.add_argument("--camera-backend", action="append", default=[], metavar="ROLE=BACKEND")
    parser.add_argument("--camera-selector", action="append", default=[], metavar="ROLE=SELECTOR")
    parser.add_argument("--camera-width", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--camera-timeout", type=float, default=None)
    parser.add_argument("--camera-stream-fps", type=float, default=10.0)
    cli = parser.parse_args(argv)
    if cli.duration is not None and cli.duration <= 0:
        parser.error("--duration must be positive")

    profile = get_web_profile(cli.robot)
    args = clone_default_args(profile)
    for field in ("camera_width", "camera_height", "camera_fps", "camera_timeout"):
        value = getattr(cli, field)
        if value is not None:
            setattr(args, field, value)
    try:
        _apply_camera_overrides(
            profile,
            args,
            _parse_role_values(cli.camera_backend, option="--camera-backend"),
            _parse_role_values(cli.camera_selector, option="--camera-selector"),
        )
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not cli.no_web:
        # Import lazily so --help and --no-web do not need the HTTP runtime.
        from mp_real.web.server import PiperWebHandler, PiperWebRuntime, PiperWebServer

        runtime = PiperWebRuntime(args, profile=profile)
        runtime.update_config({"runtime_mode": "camera_preview"})
        server = PiperWebServer((cli.host, cli.port), PiperWebHandler, runtime)
        logging.info("Camera preview UI: http://%s:%s", cli.host, cli.port)
        timer: threading.Timer | None = None
        if cli.duration is not None:
            timer = threading.Timer(cli.duration, server.shutdown)
            timer.daemon = False
            timer.start()
        try:
            runtime.start()
            if cli.save_preview is not None:
                from PIL import Image

                cli.save_preview.mkdir(parents=True, exist_ok=True)
                for role in profile.camera_roles_for_args(args):
                    frame = runtime.wait_frame(role, -1, timeout=float(args.camera_timeout) + 1.0)
                    if frame.image is not None:
                        Image.fromarray(frame.image).save(cli.save_preview / f"{role}.jpg", format="JPEG")
            server.serve_forever()
        except KeyboardInterrupt:
            logging.info("Stopping camera preview")
        finally:
            if timer is not None:
                timer.cancel()
                timer.join(timeout=1.0)
            runtime.disconnect()
            server.server_close()
        return

    session = CameraPreviewSession(profile, args, stream_fps=cli.camera_stream_fps)
    try:
        session.start()
        frames = session.wait_for_frames(timeout=float(args.camera_timeout) + 1.0)
        for role, frame in frames.items():
            if frame.error:
                logging.error("%s: %s", role, frame.error)
            else:
                logging.info("%s: frame %d", role, frame.sequence)
        if cli.save_preview is not None:
            saved = session.save_preview(cli.save_preview)
            logging.info("Saved %d preview images to %s", len(saved), cli.save_preview)
        if cli.duration is not None:
            session._stop_event.wait(cli.duration)
        elif cli.save_preview is None:
            while True:
                session._stop_event.wait(1.0)
    except KeyboardInterrupt:
        logging.info("Stopping camera preview")
    finally:
        if not session.stop(timeout=float(args.camera_timeout) + 1.0):
            raise TimeoutError("Camera preview worker did not stop")


if __name__ == "__main__":
    main()
