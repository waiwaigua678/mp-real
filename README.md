# mp-real

Lightweight deployment repository for Motrix real-robot inference and the Piper web control panel. It intentionally does not include the OpenPI training or model-serving stack; it connects to a separately deployed websocket policy server.

The source package lives in `src/mp_real`. Robot-specific entrypoints live under `mp_real.robots`; the old `mp_ex`, `mp_web`, and `openpi_client` module names are not part of this deployment package.

## Install

```bash
cd mp-real
uv sync
```

Install the command profile and only the hardware dependencies used on the robot:

```bash
./scripts/bootstrap-piper.sh     # installs ../pyAgxArm as an editable Piper dependency
uv sync --extra rm2              # vendor RM2 SDK is configured separately; see configs/rm2.env.example
uv sync --extra realsense         # RealSense cameras
uv sync --extra v4l2              # V4L2 MJPEG cameras
uv sync --extra web               # faster OpenCV JPEG encoding for the Web UI
uv sync --extra dev               # local lint tooling
```

Piper uses a fixed sibling layout, so deployment does not need a per-machine SDK path:

```bash
parent/
  mp-real/
  pyAgxArm/
```

The ROS camera backend uses the system ROS installation (`rospy`, `sensor_msgs`) and is deliberately not listed as a PyPI dependency. For RM2, source a copy of `configs/rm2.env.example` with the vendor SDK path before starting inference.

## Adding A Robot

Implement the `Robot` boundary in `mp_real.robots.<name>`: publish an `ActionSpec`, read normalized policy state, execute normalized actions, reset, and close. Register its factory with `register_robot`. The shared runtime owns policy requests, action-chunk scheduling, RTC fusion, infer-only persistence, and timestamped camera/state observations.

For the Piper SDK, install it from its local checkout or point to it at runtime:

```bash
uv pip install -e /path/to/pyAgxArm
export PYAGXARM_ROOT=/path/to/pyAgxArm
```

`PYAGXARM_ROOT` is optional after the editable install. It is useful when the SDK is kept outside this repository.

## Piper CLI

```bash
uv run mp-piper-infer --help
```

Minimal no-motion policy and camera wiring check:

```bash
uv run mp-piper-infer \
  --server-url ws://127.0.0.1:8000 \
  --cam-head-backend black \
  --cam-left-wrist-backend black \
  --cam-right-wrist-backend black \
  --dry-run \
  --no-reset-on-start \
  --no-use-rtc \
  --max-steps 1
```

This still reads the connected robot state. `--infer-only` also reads joint feedback, so it is not an offline simulator mode.

## Piper Web

The normal web process retains the existing hardware-camera defaults:

```bash
uv run mp-piper-web --host 0.0.0.0 --port 8765
```

The Web server supports Piper and RM2. Select the robot before connecting, or set the initial runtime from the command line. To protect control requests, set a key with `--access-key` or `MOTRIX_WEB_ACCESS_KEY`; the browser stores the entered key only for its current session and sends it as `X-Motrix-Key`.

```bash
MOTRIX_WEB_ACCESS_KEY=change-me \
uv run mp-piper-web --host 0.0.0.0 --port 8765 --robot rm2
```

For a deployment or policy connection check on a computer without RealSense/V4L2 dependencies, start it with black frames and prevent startup reset/enabling:

```bash
uv run mp-piper-web \
  --host 0.0.0.0 \
  --port 8765 \
  --camera-profile black \
  --no-enable-on-start \
  --no-reset-on-start \
  --dry-run
```

Open `http://<robot-computer-ip>:8765`. Connection parameters can also be set in the Settings page before clicking Connect.

## What runs elsewhere

- The OpenPI websocket policy server and its model/checkpoint remain on the inference server.
- Robot SDKs and Linux hardware access remain on the robot controller.
- `mp-real` contains only the client, camera acquisition, robot invocation, and web control layers.
