# mp-real

Lightweight deployment repository for Motrix real-robot inference and the Piper web control panel. It intentionally does not include the OpenPI training or model-serving stack; it connects to a separately deployed websocket policy server.

The source package lives in `src/mp_real`. Robot-specific entrypoints live under `mp_real.robots`; the old `mp_ex`, `mp_web`, and `openpi_client` module names are not part of this deployment package.

## Install

```bash
cd mp-real
uv sync
```

The core install is for deployment, robot control, Web startup and camera
preview. It intentionally does not install LeRobot Parquet/video dependencies.
For recording, data inspection, data audit, offline viewing or open-loop
evaluation, include the matching extra:

```bash
uv sync --extra recording
uv sync --extra data
uv sync --extra evaluation
```

Install every required extra in one `uv sync` command. Running separate
`uv sync --extra ...` commands selects a new exact environment each time, so a
later command can remove extras selected by an earlier one.

For the standard Piper deployment, first select every needed extra in one
sync, then install the deployed `pyAgxArm` checkout as an editable package:

```bash
# Piper + RealSense + faster Web JPEG encoding + local lint tools
uv sync --extra piper --extra realsense --extra web --extra dev
uv pip install -e /home/server/prj/pyAgxArm
uv pip install   --python /home/server/prj/mp-real/.venv/bin/python   --editable /home/server/prj/pyAgxArm
```

If the controller also needs RM2 and V4L2 support, include those extras in
the same `uv sync` invocation:

```bash
# Add RM2 support and V4L2 cameras when this controller needs them too.
# The RM2 vendor SDK is configured separately; see configs/rm2.env.example.
uv sync --extra piper --extra rm2 --extra realsense --extra v4l2 --extra web --extra dev
uv pip install -e /home/server/prj/pyAgxArm
```

`./scripts/bootstrap-piper.sh --extra ...` remains available when `pyAgxArm`
is checked out as the sibling `../pyAgxArm` directory.

For a non-Piper deployment, use the same repeated-flag pattern directly:

```bash
uv sync --extra rm2 --extra realsense --extra web --extra dev
```

If this machine needs every optional Python dependency, use
`./scripts/bootstrap-piper.sh --all-extras` for Piper, or `uv sync --all-extras`
without Piper.

The optional bootstrap script expects this sibling layout:

```bash
parent/
  mp-real/
  pyAgxArm/
```

When `pyAgxArm` is deployed elsewhere, use its absolute path with `uv pip install -e` as shown above. The ROS camera backend uses the system ROS installation (`rospy`, `sensor_msgs`) and is deliberately not listed as a PyPI dependency. For RM2, source a copy of `configs/rm2.env.example` with the vendor SDK path before starting inference.

## Capability Status

Software paths with Mock/Fake coverage include the shared sync/RTC/infer-only
runtime, Web resource lifecycle, EvaluationSession, LeRobot v2.1 recorder and
reader, offline data view, open-loop evaluation, Baseline, plan integrity and
delayed FakeRobot replay feedback.

The RM2 CLI defaults match the operator-provided command that has been verified
on real hardware with ROS cameras, `follow` command mode, `fps=10`,
`replan_steps=10`, `speed_percent=35`, raw gripper units, static left-state and
left-arm command disabled. RM2 gripper commands are coalesced on a dedicated
non-blocking worker by default (`--async-gripper`), at 10 Hz with a 2% deadband;
use `--no-async-gripper` to retain synchronous command behavior. The worker is
Mock-covered; concurrent RM SDK joint/gripper calls still require hardware
validation. H6 does not claim that RM2 Web replay, move-to-state or robot
trajectory replay are hardware validated.

H6 does not record a Piper hardware-validation pass. Piper deployment code is
present and covered by software tests, but real move-to-state and replay remain
behind hardening gates.

Experimental and hardware-blocked capabilities are listed in
[`robot_capability_matrix.md`](docs/robot_capability_matrix.md) and
[`known_limitations.md`](docs/known_limitations.md).

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

## Robot Web

The normal web process retains the existing hardware-camera defaults:

```bash
uv run mp-piper-web --host 0.0.0.0 --port 8765
```

`mp-real-web` is the robot-neutral alias for the same entry point; existing
`mp-piper-web` deployment scripts remain supported.

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

### Web runtime modes

The Settings page has an explicit runtime mode. Choose it before connecting;
changing it requires a disconnect.

- `DEPLOYMENT` creates the robot, configured cameras and policy client. Sync,
  RTC and infer-only are policy execution choices within this mode.
- `CAMERA_PREVIEW` creates only the configured cameras. It never opens CAN,
  creates a robot or policy client, resets/enables an arm, or reads robot
  state. The camera page remains usable even when one configured camera has a
  read error; that error is displayed on the affected stream.
- `OFFLINE_REPLAY` intentionally creates no robot, camera or policy resource.
  It shows the stage-7 replay placeholder until recorded-session playback is
  implemented.

## Camera-only preview

Use the standalone command to open configured cameras without creating a
Robot, connecting CAN, or creating a policy client:

```bash
uv run mp-camera-preview --robot piper --no-web --camera-backend cam_head=black
uv run mp-camera-preview --robot rm2 --no-web --camera-backend left_color=black
```

Use repeated `--camera-backend ROLE=BACKEND` and
`--camera-selector ROLE=SELECTOR` options for real cameras. Omit `--no-web`
to serve the same camera-only Web preview lifecycle; `--duration` and
`--save-preview DIR` are optional.

## Recorded LeRobot v2.1 data

Evaluation sessions with `save_data=true` write a self-contained LeRobot v2.1
dataset under `recordings/` by default. The recording worker writes Parquet,
MP4 and telemetry off the control thread, records executed actions as the
standard `action`, and finalizes the session atomically after its final label.

Inspect or validate a local dataset without creating a robot, camera or policy
client:

```bash
uv run mp-data-inspect recordings/<dataset>
uv run mp-data-validate recordings/<dataset>
```

`mp-data-inspect` also accepts ordinary LeRobot v2.1 datasets that do not have
the optional `meta/mp_real/` or `telemetry/` extensions. `mp-data-validate`
returns a non-zero status for schema, timestamp, Parquet, metadata or video
alignment errors.

### Safe recorded-state pose planning

Inspect a recorded `observation.state` without creating a Robot, camera, or
PolicyClient. Dry-run is the default; `--execute` additionally requires a
freshly revalidated plan hash and is intended only after the Stage 9 hardware
gates have been approved.

```bash
uv run mp-move-to-recorded-state \
  --robot piper --dataset recordings/<dataset> --episode-index 0 --sample-index 0
```

For a different but explicitly approved state schema, pass the same versioned
JSON mapping to the CLI with `--config mapping.json`, or to the Web server
with `--pose-mapping-config mapping.json`. Mapping is total and records every
unit conversion; positional or implicit unit conversion is rejected.

The Web server can be given an allow-listed recording root with repeated
`--recorded-data-root PATH` options. The recorded-state panel submits only a
dataset/episode/sample reference; the server rereads `observation.state`,
performs schema preflight, then requires a plan-hash confirmation before a
low-speed move. See `docs/hardware_validation_stage_9.md` before any real
motion test.

### Offline data viewer

Use the dedicated read-only Episode Viewer for synchronized LeRobot v2.1
videos, state/action curves, runtime events, metrics, and a draggable sample
timeline. It is intentionally separate from real-robot trajectory replay:
the process imports no robot SDK and never creates a Robot or Camera. Browsing
does not create a PolicyClient; an explicitly submitted Stage-11 open-loop job
may create one in its background worker only.

```bash
uv run mp-data-view \
  --storage-root /home/pc4/.cache/huggingface/lerobot/local/piper_1armblowv01 \
  --dataset piper_1armblowv01 \
  --episode 0
```

Open `http://127.0.0.1:8766`. A storage root may instead contain multiple
dataset directories; the UI exposes only catalog-generated dataset IDs, never
arbitrary frontend file paths. Standard LeRobot datasets remain viewable when
mp-real telemetry is absent; those unavailable fields are displayed as
unrecorded rather than inferred.

### Teacher-forced open-loop policy evaluation

`mp-open-loop-eval` evaluates full policy action chunks against real LeRobot
v2.1 observations. It creates no `Robot`, does not import a robot SDK, and
never sends an action. Warmup chunks are discarded; fresh per-sample
predictions are written beneath an isolated result directory.

```bash
uv run mp-open-loop-eval \
  --dataset recordings/<dataset> \
  --episode 0 \
  --policy-url ws://127.0.0.1:8000 \
  --policy-label pi05-checkpoint-a \
  --target-source action \
  --alignment sample_index \
  --output open_loop_results/pi05-checkpoint-a
```

For absolute-control-step alignment, the dataset must have valid
`mp_real.chunk_cursor` telemetry and the operator must explicitly declare that
its `frame_index` values are control steps:

```bash
uv run mp-open-loop-eval ... \
  --alignment absolute_control_step \
  --allow-frame-index-as-control-step
```

For an episode with multiple tasks, `--prompt-override` is required. Results
with different ActionSpec or target sources must not be combined. See
`docs/validation_stage_11.md` before comparing real policy checkpoints.

### Reproducible Baselines and A/B comparison

`mp-baseline` stores small, versioned experiment definitions under
`recordings/baselines/` by default. A Baseline captures the Git commit, policy
identity, ActionSpec, camera/robot/runtime/RTC/safety settings, evaluation
protocol and compact links to real-robot and open-loop results. It never stores
an API key or episode/video payload.

```bash
uv run mp-baseline list
uv run mp-baseline show <id>
uv run mp-baseline diff <id-a> <id-b>
uv run mp-baseline compare <id-a> <id-b>
```

The Robot Web **Baseline** page is the authoritative UI storage path; create
and clone operations are queued to a bounded background writer. Starting from
a Baseline only creates a manual evaluation session. It does not warm up a
policy, start an episode, or send an action. A changed runtime is rejected with
a categorized configuration diff until the operator explicitly creates a
derived Baseline. See the [Baseline workflow](docs/en/baseline_workflow.md), the Piper/RM2
[`robot_capability_matrix.md`](docs/robot_capability_matrix.md), and the
[`piper_rm2_generality_audit.md`](docs/piper_rm2_generality_audit.md).

### Safe robot trajectory replay

`mp-robot-replay` is distinct from offline viewing: it is policy-free and
defaults to a completely offline plan/report.  It accepts only explicitly
declared standard actions for command replay; state trajectory following is a
separate, visibly labelled mode.

```bash
uv run mp-robot-replay \
  --robot piper --dataset recordings/<dataset> --episode-index 0 \
  --mode command --timing recorded --speed-scale 0.1
```

`--execute` additionally requires the exact reviewed `--confirm-plan-hash`.
It creates no `PolicyClient` or camera, moves to the recorded start state at
low speed, then waits for confirmation before sending trajectory targets.
Review [`robot_replay_safety.md`](docs/robot_replay_safety.md) and
[`hardware_validation_h5.md`](docs/hardware_validation_h5.md) before any
physical test.

### Policy warmup and first action

Deployment exposes separate connection, metadata, warmup and steady-inference
timeouts. By default it sends one real observation as a warmup request with a
60-second warmup timeout, validates and discards its action chunk, then
prefetches one fresh live action chunk before control begins. No action is
executed during warmup, and RTC starts with the prefetched chunk instead of an
empty buffer. `检查服务` only confirms a WebSocket connection and metadata; it
does not claim that the model is warmed up.

### Canonical runtime time and in-memory events

The legacy float `timestamp_monotonic` fields remain available for existing
callers. New capture and runtime records also carry canonical
`timestamp_monotonic_ns` values from `time.monotonic_ns()`; ordering decisions
must use the integer nanosecond field. Wall-clock ISO timestamps belong only to
event display and future filenames.

Each `CameraFrame`/`CameraSample` has a trusted `frame_id`. Black and V4L2
increment it only when they obtain a new frame, RealSense uses the SDK frame
number when available, and ROS increments it only in an image callback. A
repeat read of the ROS cache retains the prior id. Source-native sequence and
capture-latency fields are optional metadata, not substitutions for the local
frame id.

`ObservationSnapshot.max_camera_skew_ns` is the difference between the latest
and earliest included camera timestamps. `observation_age_ns` is measured at
snapshot completion relative to the oldest included camera/state source. The
shared runtime translates its existing hooks into bounded in-memory events;
actions retain distinct raw-chunk, selected, stabilized, and executed payloads.
Event ndarray payloads are copied before dispatch. When an evaluation enables
`save_data`, the dedicated Stage 6 recorder consumes these copied events and
writes its files asynchronously.

## What runs elsewhere

- The OpenPI websocket policy server and its model/checkpoint remain on the inference server.
- Robot SDKs and Linux hardware access remain on the robot controller.
- `mp-real` contains only the client, camera acquisition, robot invocation, and web control layers.
