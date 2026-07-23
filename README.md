# mp-real

`mp-real` is a lightweight deployment and evaluation client for Motrix real
robots. It connects to an externally deployed WebSocket policy server; the
OpenPI training and model-serving stack is not included here.

English documentation is in [`docs/en`](docs/en/README.md); Chinese
documentation is in [`docs/zh`](docs/zh/README.md).

## Install

```bash
uv sync
```

Choose every required optional dependency in the same command. A later
`uv sync --extra ...` selects a new environment and can remove extras chosen
previously.

```bash
# Piper with RealSense, faster Web JPEG encoding, and development tools.
uv sync --extra piper --extra realsense --extra web --extra dev
uv pip install -e /path/to/pyAgxArm

# RM2 with RealSense and Web support.
uv sync --extra rm2 --extra realsense --extra web --extra dev
```

Use `--extra recording`, `--extra data`, or `--extra evaluation` for LeRobot
recording and offline data tools. The ROS camera backend uses the system ROS
installation. For RM2, source a configuration based on
[`configs/rm2.env.example`](configs/rm2.env.example) before starting.

## Entry points

```bash
uv run mp-piper-infer --help
uv run mp-rm2-infer --help
uv run mp-piper-web --help
uv run mp-camera-preview --help
uv run mp-data-inspect --help
uv run mp-data-validate --help
uv run mp-data-view --help
uv run mp-open-loop-eval --help
uv run mp-baseline --help
uv run mp-move-to-recorded-state --help
uv run mp-robot-replay --help
```

`mp-real-web` is a robot-neutral alias for `mp-piper-web`. The Web server has
four resource modes: `DEPLOYMENT` creates robot, cameras, and policy client;
`CAMERA_PREVIEW` creates cameras only; `OFFLINE_REPLAY` creates none of those
resources; `DATA_VIEW` embeds the data viewer in the Run page and opens only
recorded data. Give it `--recorded-data-root` options at startup or import a
server-local dataset/storage path from the DATA_VIEW page; paths imported in
the page exist only for the current Web process. Its explicit teacher-forced
open-loop action creates a policy client only in a
background worker and never creates or commands a robot. Its robot replay
action remains separately safety-planned, low-speed, and confirmation-gated.
Policy warmup actions are discarded and a fresh chunk is prepared before the
deployment control loop starts.

## Safety and status

Do not treat `--dry-run` or `--infer-only` as an offline simulator: both can
read connected robot feedback. Any command that can move hardware requires an
operator-approved safety procedure. Recorded-state movement and trajectory
replay are experimental and blocked by default safety validation.

RM2 CLI deployment defaults have an operator-recorded real-hardware validation
scope. This does **not** validate RM2 Web deployment, recorded-state movement,
or replay. Piper real-hardware movement/replay validation is not claimed.
See the [capability matrix](docs/en/robot_capability_matrix.md),
[known limitations](docs/en/known_limitations.md). Hardware-safety runbooks
are maintained outside this repository.

## Architecture

Implement new robots beneath `src/mp_real/robots/<name>/` and register their
factory. Shared runtime, data, evaluation, pose, and replay code must remain
vendor-SDK-free and use `Robot`, `ActionSpec`, and camera roles rather than a
fixed action layout.

## External components

- OpenPI policy server and checkpoints run outside this repository.
- Vendor SDKs and Linux hardware access run on the robot controller.
- This repository provides the robot client, camera acquisition, runtime,
  recording/evaluation tools, and Web control layer.
