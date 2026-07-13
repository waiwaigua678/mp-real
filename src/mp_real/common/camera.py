from __future__ import annotations

import dataclasses
import fcntl
import importlib
import logging
import mmap
import os
import select
import threading
import time
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import numpy as np

try:
    import cv2

    _CV2_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc

try:
    import v4l2
except ImportError:
    v4l2 = None


CameraBackend = Literal["realsense", "v4l2", "black"]


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    timestamp_monotonic: float
    camera_timestamp: float | None = None
    info: dict[str, Any] | None = None
    frame_id: int = 0
    timestamp_monotonic_ns: int = 0
    source_sequence: int | None = None
    capture_latency_ns: int | None = None

    def __post_init__(self) -> None:
        """Retain the legacy float timestamp while preferring canonical nanoseconds."""
        if self.timestamp_monotonic_ns <= 0:
            object.__setattr__(self, "timestamp_monotonic_ns", int(self.timestamp_monotonic * 1e9))


class Camera(Protocol):
    def read(self, *, timeout: float = 2.0) -> np.ndarray: ...

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame: ...

    def camera_info(self) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


class BlackCamera:
    """Camera placeholder that returns a black RGB frame."""

    def __init__(self, name: str, *, width: int = 640, height: int = 480) -> None:
        self.name = name
        self.width = width
        self.height = height
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._frame_id = 0
        self._frame_lock = threading.Lock()
        logging.info("Using black placeholder for %s", self.name)

    def read(self, *, timeout: float = 2.0) -> np.ndarray:
        return self.read_frame(timeout=timeout).image

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame:
        del timeout
        with self._frame_lock:
            self._frame_id += 1
            frame_id = self._frame_id
        captured_ns = time.monotonic_ns()
        return CameraFrame(
            image=self._frame.copy(),
            timestamp_monotonic=captured_ns / 1e9,
            timestamp_monotonic_ns=captured_ns,
            frame_id=frame_id,
            source_sequence=frame_id,
        )

    def camera_info(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        pass


class RealSenseCamera:
    """RealSense color camera reader returning RGB uint8 frames."""

    def __init__(
        self,
        name: str,
        serial: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fallback_backends: str = "black",
        multiple_devices_hint: str | None = None,
    ) -> None:
        self.name = name
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.fallback_backends = fallback_backends
        self.multiple_devices_hint = multiple_devices_hint
        self.rs = import_realsense(fallback_backends=fallback_backends)
        self.pipeline: Any | None = None
        self._frame_id = 0
        self._open()

    def _open(self) -> None:
        rs = self.rs
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("No RealSense devices found")

        serial = self.serial
        if serial:
            serials = {dev.get_info(rs.camera_info.serial_number) for dev in devices}
            if serial not in serials:
                raise RuntimeError(f"Could not find RealSense camera with serial number {serial}")
        else:
            serial = devices[0].get_info(rs.camera_info.serial_number)
            if len(devices) > 1:
                hint = self.multiple_devices_hint or "Pass an explicit serial number to choose a camera."
                logging.warning("Multiple RealSense devices found; %s selected %s. %s", self.name, serial, hint)

        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        logging.info("Opened %s RealSense color stream serial=%s", self.name, self.serial)

    def read(self, *, timeout: float = 2.0) -> np.ndarray:
        return self.read_frame(timeout=timeout).image

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} RealSense pipeline is not open")

        capture_started_ns = time.monotonic_ns()
        frames = self.pipeline.wait_for_frames(round(timeout * 1000))
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self.name} failed to get a RealSense color frame")
        bgr = np.asanyarray(color_frame.get_data())
        camera_timestamp = float(color_frame.get_timestamp()) if hasattr(color_frame, "get_timestamp") else None
        frame_number = color_frame.get_frame_number() if hasattr(color_frame, "get_frame_number") else None
        if frame_number is None:
            self._frame_id += 1
            frame_id = self._frame_id
        else:
            frame_id = int(frame_number)
            self._frame_id = max(self._frame_id, frame_id)
        captured_ns = time.monotonic_ns()
        return CameraFrame(
            image=np.ascontiguousarray(bgr[:, :, ::-1]),
            timestamp_monotonic=captured_ns / 1e9,
            camera_timestamp=camera_timestamp,
            frame_id=frame_id,
            timestamp_monotonic_ns=captured_ns,
            source_sequence=frame_id,
            capture_latency_ns=captured_ns - capture_started_ns,
        )

    def camera_info(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.stop()
        self.pipeline = None


class V4L2MJPEGCamera:
    """Minimal V4L2 MJPEG camera reader returning RGB uint8 frames."""

    def __init__(self, name: str, device: str, *, width: int = 640, height: int = 480) -> None:
        if _CV2_IMPORT_ERROR is not None:
            raise RuntimeError("opencv-python is required for v4l2 camera capture") from _CV2_IMPORT_ERROR

        self.name = name
        self.device = device
        self.width = width
        self.height = height
        self.fd: int | None = None
        self.cap: Any | None = None
        self.buffers: list[mmap.mmap] = []
        self._frame_id = 0
        self._frame_lock = threading.Lock()
        self._open()

    def _next_frame_id(self) -> int:
        with self._frame_lock:
            self._frame_id += 1
            return self._frame_id

    def _open(self) -> None:
        if v4l2 is None:
            self._open_opencv()
            return

        assert v4l2 is not None

        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)

        fmt = v4l2.v4l2_format()
        fmt.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_MJPEG
        fmt.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, v4l2.VIDIOC_S_FMT, fmt)

        req = v4l2.v4l2_requestbuffers()
        req.count = 4
        req.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_REQBUFS, req)

        self.buffers = []
        for i in range(req.count):
            buf = v4l2.v4l2_buffer()
            buf.type = req.type
            buf.memory = v4l2.V4L2_MEMORY_MMAP
            buf.index = i
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QUERYBUF, buf)

            mm = mmap.mmap(
                self.fd,
                buf.length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED,
                offset=buf.m.offset,
            )
            self.buffers.append(mm)
            fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMON, buf_type)
        logging.info("Opened %s on %s", self.name, self.device)

    def _open_opencv(self) -> None:
        assert cv2 is not None

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open {self.name} on {self.device} with OpenCV V4L2")

        self.cap = cap
        logging.info("Opened %s on %s with OpenCV V4L2 fallback", self.name, self.device)

    def read(self, *, timeout: float = 2.0) -> np.ndarray:
        return self.read_frame(timeout=timeout).image

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame:
        capture_started_ns = time.monotonic_ns()
        if self.cap is not None:
            return self._read_opencv_frame(timeout=timeout, capture_started_ns=capture_started_ns)

        assert self.fd is not None
        assert cv2 is not None
        assert v4l2 is not None

        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            raise TimeoutError(f"{self.name} timed out after {timeout:.1f}s")

        buf = v4l2.v4l2_buffer()
        buf.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = v4l2.V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, v4l2.VIDIOC_DQBUF, buf)
        data = self.buffers[buf.index][: buf.bytesused]
        fcntl.ioctl(self.fd, v4l2.VIDIOC_QBUF, buf)

        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"{self.name} returned an undecodable MJPEG frame")
        captured_ns = time.monotonic_ns()
        frame_id = self._next_frame_id()
        return CameraFrame(
            image=np.ascontiguousarray(bgr[:, :, ::-1]),
            timestamp_monotonic=captured_ns / 1e9,
            frame_id=frame_id,
            timestamp_monotonic_ns=captured_ns,
            source_sequence=frame_id,
            capture_latency_ns=captured_ns - capture_started_ns,
        )

    def _read_opencv_frame(self, *, timeout: float = 2.0, capture_started_ns: int | None = None) -> CameraFrame:
        assert self.cap is not None
        assert cv2 is not None

        deadline = time.monotonic() + timeout
        while True:
            ok, bgr = self.cap.read()
            if ok and bgr is not None:
                if bgr.ndim == 2:
                    image = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
                else:
                    image = np.ascontiguousarray(bgr[:, :, ::-1])
                captured_ns = time.monotonic_ns()
                frame_id = self._next_frame_id()
                return CameraFrame(
                    image=image,
                    timestamp_monotonic=captured_ns / 1e9,
                    frame_id=frame_id,
                    timestamp_monotonic_ns=captured_ns,
                    source_sequence=frame_id,
                    capture_latency_ns=(captured_ns - capture_started_ns) if capture_started_ns is not None else None,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{self.name} timed out after {timeout:.1f}s")
            time.sleep(0.01)

    def camera_info(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            return

        if self.fd is None:
            return

        assert v4l2 is not None
        try:
            buf_type = v4l2.v4l2_buf_type(v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE)
            fcntl.ioctl(self.fd, v4l2.VIDIOC_STREAMOFF, buf_type)
        except OSError as exc:
            logging.warning("%s STREAMOFF failed: %s", self.name, exc)

        for mm in self.buffers:
            mm.close()
        self.buffers.clear()
        os.close(self.fd)
        self.fd = None


class ROSImageCamera:
    def __init__(self, name: str, image_topic: str, info_topic: str | None) -> None:
        self.name = name
        self.image_topic = image_topic
        self.info_topic = info_topic or image_topic.replace("/image_raw", "/camera_info")
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: CameraFrame | None = None
        self._frame_id = 0
        self._info: dict[str, Any] | None = None
        self._sub_img = None
        self._sub_info = None
        self._open()

    def _open(self) -> None:
        rospy, Image, CameraInfo = import_ros_image_types()
        self._sub_img = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)
        self._sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self._info_cb, queue_size=1)
        logging.info("Subscribed %s image=%s info=%s", self.name, self.image_topic, self.info_topic)

    def _image_cb(self, msg: Any) -> None:
        capture_started_ns = time.monotonic_ns()
        frame = ros_image_to_rgb(msg)
        if frame is None:
            return
        info = self.camera_info()
        camera_timestamp = None
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None:
            try:
                camera_timestamp = float(stamp.to_sec())
            except Exception:
                camera_timestamp = None
        with self._cond:
            self._frame_id += 1
            captured_ns = time.monotonic_ns()
            source_sequence = getattr(getattr(msg, "header", None), "seq", None)
            self._frame = CameraFrame(
                image=frame,
                timestamp_monotonic=captured_ns / 1e9,
                camera_timestamp=camera_timestamp,
                info=info,
                frame_id=self._frame_id,
                timestamp_monotonic_ns=captured_ns,
                source_sequence=int(source_sequence) if source_sequence is not None else None,
                capture_latency_ns=captured_ns - capture_started_ns,
            )
            self._cond.notify_all()

    def _info_cb(self, msg: Any) -> None:
        info = {
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": str(msg.distortion_model),
            "D": list(msg.D),
            "K": list(msg.K),
            "R": list(msg.R),
            "P": list(msg.P),
        }
        with self._lock:
            self._info = info

    def read(self, *, timeout: float = 2.0) -> np.ndarray:
        return self.read_frame(timeout=timeout).image

    def read_frame(self, *, timeout: float = 2.0) -> CameraFrame:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self.name} timed out waiting for {self.image_topic}")
                self._cond.wait(remaining)
            frame = self._frame
            return dataclasses.replace(frame, image=frame.image.copy())

    def camera_info(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._info is None else dict(self._info)

    def close(self) -> None:
        for sub in (self._sub_img, self._sub_info):
            if sub is not None:
                try:
                    sub.unregister()
                except Exception:
                    pass


def import_realsense(*, fallback_backends: str = "black") -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for RealSense cameras. Install it in this environment or switch the camera "
            f"backend to {fallback_backends}."
        ) from exc


def import_ros_image_types() -> tuple[Any, Any, Any]:
    try:
        import rospy
        from sensor_msgs.msg import CameraInfo, Image
    except Exception as exc:
        raise RuntimeError(
            "ROS image topics require a sourced ROS environment. Source /opt/ros/noetic/setup.bash and the "
            "catkin workspace, or run with --camera-backend black."
        ) from exc
    return rospy, Image, CameraInfo


def init_ros_node(*, node_name: str = "rm2_vla_infer") -> None:
    rospy, _, _ = import_ros_image_types()
    if not rospy.core.is_initialized():
        logging.info("Initializing ROS node")
        rospy.init_node(node_name, anonymous=True, disable_signals=True, disable_rosout=True)
        logging.basicConfig(level=logging.INFO, force=True)
        logging.info("Initialized ROS node")


def ros_image_to_rgb(msg: Any) -> np.ndarray | None:
    h, w = int(msg.height), int(msg.width)
    enc = str(msg.encoding).lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if enc in ("rgb8", "bgr8"):
            img = raw.reshape(h, w, 3)
            if enc == "bgr8":
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        if enc in ("mono8", "8uc1"):
            return np.repeat(raw.reshape(h, w)[:, :, None], 3, axis=2)
        if enc in ("rgba8", "bgra8"):
            img = raw.reshape(h, w, 4)[:, :, :3]
            if enc == "bgra8":
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        logging.warning("Unsupported ROS image encoding %s; trying HxWx3 uint8", enc)
        return np.ascontiguousarray(raw.reshape(h, w, 3))
    except Exception as exc:
        logging.warning("Failed to convert ROS image %s: %s", enc, exc)
        return None


def make_camera(
    name: str,
    backend: CameraBackend,
    selector: str = "",
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fallback_backends: str = "black",
    multiple_devices_hint: str | None = None,
) -> Camera:
    match backend:
        case "realsense":
            return RealSenseCamera(
                name,
                selector,
                width=width,
                height=height,
                fps=fps,
                fallback_backends=fallback_backends,
                multiple_devices_hint=multiple_devices_hint,
            )
        case "v4l2":
            if not selector:
                raise ValueError(f"{name} backend is v4l2, but no device path was provided")
            return V4L2MJPEGCamera(name, selector, width=width, height=height)
        case "black":
            return BlackCamera(name, width=width, height=height)
        case _:
            raise ValueError(f"Unsupported camera backend: {backend}")


def make_realsense_cameras(
    serials: Mapping[str, str],
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    fallback_backends: str = "black",
    require_serials: bool = False,
    missing_message: str | None = None,
) -> dict[str, Camera]:
    if require_serials and not all(serials.values()):
        if missing_message is not None:
            raise ValueError(missing_message)
        missing = ", ".join(name for name, serial in serials.items() if not serial)
        raise ValueError(f"RealSense backend needs explicit serial numbers. Missing: {missing}.")
    return {
        name: RealSenseCamera(
            name,
            serial,
            width=width,
            height=height,
            fps=fps,
            fallback_backends=fallback_backends,
        )
        for name, serial in serials.items()
    }


def close_cameras(cameras: Mapping[str, Camera]) -> None:
    for camera in cameras.values():
        camera.close()
