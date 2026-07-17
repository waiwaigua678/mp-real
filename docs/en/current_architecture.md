# Current architecture (stage 0 baseline)

This document is descriptive only. It records the architecture inspected for the
evaluation-platform refactor; it does not prescribe new behavior.

## Scope and stable boundaries

- Robot is the hardware boundary. Piper and RM2 SDK calls remain in their
  respective robots/<name>/infer.py modules.
- ActionSpec is the policy-facing shape contract. The shared runtime does not
  assume Piper's 14-element action layout.
- InferenceAdapter supplies observation, decoding, stabilization, execution and
  infer-only metadata to runtime/inference.py.
- There are two synchronous WebSocket policy clients today:
  WebsocketClientPolicy for CLI and web.server.PolicyClient for Web.
- capture_observation retains camera and robot-state timestamps in an
  ObservationSnapshot; to_policy_observation() omits them from the policy wire
  payload.

## A. mp-piper-infer CLI call chain

~~~mermaid
flowchart TD
  A[mp-piper-infer entry point] --> B[piper.infer.cli / tyro Args]
  B --> C[piper.infer.main]
  C --> D[InferenceLoopConfig.from_args + validate]
  C --> E[create_robot Args]
  E --> F[create_piper_arm left/right; Piper SDK connect/configure/optional enable]
  F --> G[PiperRobot]
  C --> H{reset_only?}
  H -- yes --> I[robot.reset] --> Z[finally close robot]
  H -- no --> J{infer_only?}
  J -- no --> K[robot.reset]
  J -- yes --> L[skip reset action]
  K --> M[WebsocketClientPolicy]
  L --> M
  M --> N[make_cameras]
  N --> O[PiperInferenceAdapter: PiperRobot + cameras + Args]
  O --> P{loop mode}
  P -- infer_only --> Q[shared run_infer_only]
  P -- use_rtc --> R[shared run_rtc_loop]
  P -- otherwise --> S[shared run_sync_loop]
  Q --> T[finally close_cameras + robot.close]
  R --> T
  S --> T
~~~

main() validates the generic loop configuration and command_rate_hz, then owns
construction and cleanup. It creates both Piper arms before the policy
connection and creates cameras after it. A partial second-arm creation closes
the first arm; finally closes all created cameras and the PiperRobot. The CLI
WebSocket client has no explicit close call on this path.

PiperInferenceAdapter.observe() calls prepare_observation(), then
capture_observation() to read the cameras and robot state. The generic loop
then does one of the following:

- sync: fetch a chunk when its local deque is empty, stabilize one selected raw
  action, execute one transition, then sleep to fps;
- RTC: start a producer, fuse eligible overlapping chunks in
  RealTimeChunkingBuffer, stabilize and execute; optionally repeat the last
  executed action while waiting;
- infer-only: fetch and print fresh chunks without stabilization or execution.

## B. Web startup-to-Piper-inference call chain

~~~mermaid
flowchart TD
  A[mp-piper-web] --> B[web.server.main]
  B --> C[PiperWebServer + PiperWebRuntime]
  C --> D[HTTP POST /api/start]
  D --> E[PiperWebRuntime.start]
  E --> F{already connected?}
  F -- no --> G[_connect]
  G --> H[create_robot piper]
  H --> I[direct PiperRobot/PiperArm access]
  I --> J[optional infer_piper.maybe_reset_arms]
  J --> K[web.PolicyClient]
  K --> L[infer_piper.make_cameras]
  L --> M[start daemon camera thread]
  F -- yes --> N[apply direct Piper arm settings]
  M --> N
  N --> O[start daemon run thread]
  O --> P[_run_loop]
  P --> Q{infer_only / RTC / sync}
  Q --> R[_run_infer_only_loop]
  Q --> S[_run_rtc_control_loop + producer]
  Q --> T[_run_sync_control_loop]
  R --> U[_prepare_observation + _infer_action_chunk]
  S --> U
  T --> U
  U --> V[latest preview frames + direct infer_piper.read_state]
  V --> W[web.PolicyClient.infer]
  W --> X[direct infer_piper decode/stabilize/execute]
~~~

The request handler calls PiperWebRuntime.start(); inference itself runs on
_run_thread. start() can run _connect() synchronously first, so arm and camera
creation, reset and policy connection can occur on the HTTP request thread.
disconnect() and reset_arms() also do lifecycle or robot work from a request
thread. This is a baseline fact, not an endorsement.

Piper Web's camera worker reads and JPEG-encodes images.
_prepare_observation() copies the latest images and calls infer_piper.read_state()
directly. FrameSnapshot retains only Web receipt time (updated_at), not the
source camera timestamp used by the CLI path.

## Shared runtime and duplicated Piper Web control logic

runtime/inference.py owns _fetch_chunk, infer-only collection, sync planning,
the RTC producer, RealTimeChunkingBuffer use, hold-last behavior and
producer-error propagation. Both CLI adapters call it.

| Generic runtime behavior | Piper Web duplicate |
| --- | --- |
| run_infer_only | _run_infer_only_loop |
| run_sync_loop | _run_sync_control_loop |
| _rtc_producer and run_rtc_loop | _rtc_action_producer_loop and _run_rtc_control_loop |
| _fetch_chunk | _prepare_observation plus _infer_action_chunk |
| raise_rtc_producer_error | _raise_producer_error |

The Web copies add UI metrics and use preview frames, but they also diverge:
Web infer-only does not use the shared .npz writer, counts max_steps as chunks,
and cannot use adapter infer-only metadata. Its RTC producer has an extra
producer_stop in addition to the runtime stop event.

## Piper Web direct dependency on infer_piper

web/server.py imports mp_real.robots.piper.infer at module import time and uses
it throughout PiperWebRuntime:

- defaults, camera masks and config serialization use infer_piper.Args and
  Piper-specific fields;
- _connect() creates registry name "piper", asserts PiperRobot, extracts
  left/right, and directly calls reset, camera and arm close helpers;
- observation, action decode, arm speed updates, all three loops and reset
  directly call Piper module functions;
- robot selection instantiates either PiperWebRuntime or Rm2WebRuntime.

The Web layer therefore knows Piper arm handles, camera wiring, decoding,
stabilization and transition execution. This is the dependency surface an
incremental migration needs to reduce without changing the browser API.

## Web selection of RM2

At startup --robot rm2 creates Rm2WebRuntime. At runtime, POST /api/robot calls
PiperWebServer.select_robot(), which only permits a switch when the current
runtime is neither connected nor running; it then replaces the runtime object
with a new Rm2WebRuntime.

Rm2WebRuntime.connect() runs in the HTTP request thread:

1. create the Web PolicyClient;
2. call infer_rm2.make_cameras();
3. call create_robot("rm2", args) and require an Rm2Robot;
4. call robot.reset() (which checks reset_on_start);
5. retain the resources and start the preview camera worker.

start() creates the daemon run thread. _run_loop() creates _Rm2WebAdapter and
invokes shared run_policy_loop() with the Web stop event and a step callback.
The adapter supplies latest preview images, camera parameters and
Rm2Robot.read_state(), and delegates decode/stabilize/execute to RM2 code.
RM2 Web therefore already shares the inference loop, although its Web
lifecycle, preview worker and policy client remain Web-specific.

## Resource ownership, threads and failure behavior

| Path | Creates | Stops/closes | Thread and error behavior |
| --- | --- | --- | --- |
| Piper CLI | PiperRobot, policy client, cameras in main() | finally: close_cameras, robot.close | sync has no thread; generic RTC starts a daemon producer, sets the event in finally, joins for 2 s, then re-raises queued producer errors. |
| Piper Web | arms/robot, Web client, cameras; camera worker on connect and run worker on start | stop() signals run; disconnect() joins run for 5 s and camera thread for camera_timeout + 1, then closes resources | all workers, including RTC producer, are daemon. Loop errors change Web error state; camera errors are stored per frame; RTC producer errors are queued then re-raised by consumer. |
| RM2 Web | Web client, cameras, Rm2Robot; camera worker on connect and run worker on start | disconnect() signals run, joins run for 5 s and camera for 3 s, then closes resources | run/camera and generic RTC producer are daemon. Generic producer errors reach run_policy_loop and are caught by _run_loop. |

Both Web stop() calls are idempotent when idle: Piper returns current status;
RM2 sets an already-set event. Both have bounded joins but do not verify that a
timed-out worker actually exited before broader lifecycle cleanup. The generic
RTC error queue is unbounded; RealTimeChunkingBuffer retains chunks in a dict
until a read removes expired chunks.

Shared run_policy_loop() gives its stop event to sync and RTC modes. Its
infer-only branch calls run_infer_only() without one, so RM2 Web infer-only is
bounded only by configured chunks and blocking camera/policy calls; Piper Web
has its own infer-only stop check.

## Infer-only persistence format

Shared infer-only mode writes a compressed NumPy .npz to infer_only_output when
configured:

| Key | Value |
| --- | --- |
| actions | float32, [infer_only_chunks, T, action_dim]; decoded raw policy chunks after robot validation/replan slicing, before stabilization or execution. |
| states | float32, [infer_only_chunks, state_dim]; state in the policy observation. |
| prompt | scalar NumPy array/string from config.prompt. |
| extra keys | object arrays from infer_only_metadata; RM2 writes camera_params, Piper writes none. |

It creates parents but writes directly: there is no .inprogress directory or
atomic rename. It does not save images, source timestamps, raw server
responses, selected actions, stabilized targets or executed actions. Piper Web
infer-only only fetches chunks and updates metrics; it has no output-path
setting.

## Web API inventory

PiperWebHandler provides these routes. Every POST requires X-Motrix-Key when
the server has an access key. JSON errors are {"ok": false, "error": "..."} with
the relevant HTTP status.

| Method | Path | Request | Response | Runtime method / behavior |
| --- | --- | --- | --- | --- |
| GET | / | none | index.html | static file |
| GET | /static/<path> | none | static file | static file constrained to static root |
| GET | /api/status | none | status object | runtime.status() |
| GET | /api/config | none | config object | runtime.get_config() |
| GET | /snapshot/<camera>.jpg | named camera | JPEG | runtime.wait_frame(camera, -1, timeout=0.1) |
| GET | /stream/<camera>.mjpg | named camera | multipart MJPEG | repeated runtime.wait_frame(camera, sequence) |
| POST | /api/config | JSON partial config | {"ok": true, "config": ...} | runtime.update_config(payload) |
| POST | /api/connect | empty JSON accepted | {"ok": true, "status": ...} | runtime.connect() |
| POST | /api/start | empty JSON accepted | {"ok": true, "status": ...} | runtime.start() |
| POST | /api/stop | empty JSON accepted | {"ok": true, "status": ...} | runtime.stop(wait=False) |
| POST | /api/disconnect | empty JSON accepted | {"ok": true, "status": ...} | runtime.disconnect() |
| POST | /api/reset | empty JSON accepted | {"ok": true, "status": ...} | runtime.reset_arms() |
| POST | /api/ping_policy | empty JSON accepted | {ok, connected, metadata, latency_ms?} or error | runtime.ping_policy() |
| POST | /api/robot | {"robot": "piper" or "rm2"} | {"ok": true, "config": ...} | server.select_robot() |

Status contains phase/connectivity/capability flags, task/counters, policy
metadata/error, metrics, frame metadata and logs. Piper additionally has
stop_requested, infer-only and detailed metrics; RM2 returns empty metrics and
metadata today. Unknown paths return JSON 404; other methods are not explicitly
implemented.

## Stage-0 characterization protection line

tests/test_runtime.py locks this documented behavior without hardware, network,
or a policy server: sync chunk consumption, RTC overlap fusion and generation
rejection, hold-last execution, timestamp retention, registry behavior, Piper
Web config round-trip coercion and idempotent stop. They are behavior guards,
not product features.

