# Evaluation-platform refactor plan

## Purpose and non-goals

This is an incremental migration from the architecture recorded in
current_architecture.md. It protects Piper CLI and Web behavior while making
evaluation and recording possible on the existing Robot, ActionSpec,
InferenceAdapter and shared runtime boundaries.

It does not propose a framework migration, a database, a replacement policy
client, a second inference loop, a fixed 14-dimensional shared action schema,
or a big-bang rewrite of web/server.py.

## Compatibility and hardware guardrails

- Preserve existing CLI entry points and browser endpoints at every stage.
- Keep vendor SDK calls under robots/piper and robots/rm2; shared runtime code
  remains vendor-SDK-free.
- Use ActionSpec dimensions and camera roles at shared boundaries.
- Never use development tests to enable, reset, move, replay or otherwise
  command hardware. Tests use fakes, mock policies and black cameras only.
- Give delayed results a session_id, generation_id, request_id and chunk_id as
  applicable. Order runtime events with time.monotonic_ns().
- Make recording background workers bounded, explicitly stopped and joined,
  with reported failures and temporary/atomic session finalization.

## Stage 0 — baseline and protection line (this change)

Files: docs/current_architecture.md, this plan, and tests/test_runtime.py.

- Record both Piper call chains, RM2 Web's actual shared-runtime path, duplicate
  Piper Web loops, lifecycles, infer-only data and the full API contract.
- Add hardware-free unittest characterization coverage for shared runtime,
  observation, registry and Web config/stop behavior.
- Do not add evaluation product behavior, move existing large files, or change
  production control behavior.

Exit criterion: the documented behavior and characterization tests pass the
repository unittest and Ruff checks.

## Stage 1 — lifecycle identity and shared runtime seam

Primary files: small focused additions under runtime/, targeted edits to
web/server.py, and tests. Do not move server.py wholesale.

1. Define small runtime-owned session and request identities, using monotonic
   timestamps. Thread them through queued work and reject stale results at each
   handoff.
2. Refactor Piper Web one loop at a time to call existing run_policy_loop()
   rather than its private infer-only/sync/RTC loops. Implement a Piper Web
   InferenceAdapter that reads the existing preview-frame source and delegates
   robot work through PiperRobot. It is not a new Robot, Camera or Policy
   abstraction.
3. Preserve Web metrics through on_step and narrowly scoped profiling hooks.
   Characterize intentional UI differences before aligning them with generic
   runtime behavior.
4. Replace daemon lifecycle masking only where a worker has an explicit stop
   signal, bounded wait, join outcome and propagated exception. Do not change
   hardware motion semantics in this stage.

Exit criterion: Piper CLI, Piper Web and RM2 Web select the same shared
inference loop for each mode, while normal API response shapes and lifecycle
behavior remain compatible.

## Stage 2 — observation and action telemetry contracts

Primary files: runtime/models.py, runtime/observation.py, shared runtime and
robot adapters.

1. Retain ObservationSnapshot timestamps through the in-process evaluation
   pipeline while leaving the policy wire schema unchanged.
2. Record separate raw policy chunk, selected raw action, stabilized target and
   robot-returned executed action. No shared field assumes Piper's shape.
3. Add explicit event identities and monotonic timestamps to observations,
   policy requests/results, chunk selection and execution.
4. Extend Web frame snapshots to retain original camera time and frame identity,
   not merely JPEG update time.

Exit criterion: fake-robot tests prove timestamp retention and stale-result
rejection across request, chunk and session changes.

## Stage 3 — recording worker and versioned session format

Primary files: a focused recording module plus narrow runtime integration. Do
not perform disk writes in HTTP handlers or the robot loop.

1. Define a versioned manifest and telemetry schema with ActionSpec,
   policy/server metadata, dropped-frame/telemetry counters and failure state.
2. Create a bounded producer queue and non-daemon writer worker with explicit
   stop, join result and exception channel.
3. Write to a .inprogress session directory, then atomically finalize where
   practical. Preserve incomplete/failure metadata if finalization fails.
4. Reuse Stage-2 telemetry contracts and encode video off the control path.

Exit criterion: overloaded fake recording reports drops and writer failures; no
control-loop call stack writes data to disk.

## Stage 4 — evaluation orchestration and Web exposure

Primary files: focused evaluation/session modules and narrow handler additions.

1. Put evaluation commands onto a runtime-owned command queue; handlers only
   validate/enqueue and return IDs/status.
2. Enforce state, initial-condition and recording-schema validation before any
   replay. Keep motion-capable tests disabled unless explicitly approved.
3. Add backward-compatible status/API fields for session identity, queue
   position, error and final artifact metadata.
4. Extend the existing UI incrementally without changing its framework.

Exit criterion: fake end-to-end runs cover start, stop, stale-work rejection,
recording finalization, errors and repeated lifecycle requests.

## Stage 5 — hardware verification and rollout

Unit-test success is not real-hardware verification. With explicit approval for
each motion-capable command, verify:

- Piper and RM2 connection/preview with the scoped camera setup;
- reset, speed, units and joint-order preservation;
- stop latency and shutdown during sync and RTC execution;
- policy delay, reconnect and stale-result handling;
- bounded recording under frame and disk pressure.

For every hardware command, record the expected motion, stop procedure and
result. Keep a rollback path to the previous behavior until parity is shown.

## File-level migration order

1. Keep robots/*/infer.py as SDK-owning implementations; make only small
   adapter/lifecycle call-site changes.
2. Consolidate loops through runtime/inference.py before adding evaluation
   features.
3. Add telemetry models before recording them.
4. Add a recording worker before exposing recording controls.
5. Add evaluation API/UI after the worker and lifecycle contracts are tested.

At every stage run:

~~~
uv run python -m unittest discover -s tests -v
uv run ruff check .
~~~

Then review the diff for unrelated changes and state what still needs
real-hardware verification.

