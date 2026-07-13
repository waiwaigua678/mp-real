# mp-real repository instructions

## Project scope

mp-real is a lightweight real-robot deployment, control, evaluation,
recording, replay, and visualization project.

The OpenPI training and model-serving stack is external.
mp-real connects to a separately deployed WebSocket policy server.

The current implementation supports:
- Piper
- RM2
- CLI deployment
- Web deployment and control
- Camera preview
- Synchronous action-chunk execution
- RTC action-chunk execution
- Infer-only execution

Read README.md and the relevant source files before modifying behavior.

## Existing architecture

Do not create parallel replacements for existing abstractions.

Existing boundaries include:
- Robot Protocol
- ActionSpec
- Robot registry
- PiperRobot
- Rm2Robot
- InferenceAdapter
- PolicyClient
- Shared synchronous and RTC inference runtime
- RealSense, V4L2, ROS, and black camera backends

Robot-specific SDK calls must stay under:

- src/mp_real/robots/piper/
- src/mp_real/robots/rm2/

Shared runtime code must not import vendor robot SDKs.

Prefer extending existing abstractions over introducing:
- RobotAdapter
- CameraAdapter
- another PolicyClient
- another independent inference loop

## Important files

Read these before architectural changes:

- README.md
- pyproject.toml
- src/mp_real/robots/base.py
- src/mp_real/robots/registry.py
- src/mp_real/runtime/models.py
- src/mp_real/runtime/inference.py
- src/mp_real/runtime/observation.py
- src/mp_real/common/camera.py
- src/mp_real/robots/piper/infer.py
- src/mp_real/robots/rm2/infer.py
- src/mp_real/web/server.py
- static/index.html
- static/app.js
- tests/test_runtime.py

For the evaluation-platform roadmap, also read:

- docs/plans/evaluation-platform.md

## Architectural constraints

- Reuse the shared runtime from runtime/inference.py.
- Do not continue duplicating control loops inside the Web layer.
- Web request handlers must not directly execute robot actions.
- Web request handlers must not perform blocking inference or disk writes.
- Do not rewrite server.py in one large change.
- Do not migrate the existing frontend framework.
- Do not add databases, pandas, PyArrow, React, Vue, or FastAPI unless
  explicitly approved.
- Preserve existing Piper and RM2 CLI and Web behavior.
- New data structures must use ActionSpec and must not assume a fixed
  action dimension.
- Do not hard-code Piper's 14-dimensional action layout in shared modules.

## Runtime and concurrency rules

Every background worker must have:
- an explicit stop mechanism
- bounded queues where appropriate
- defined join behavior
- exception propagation
- deterministic cleanup

Do not use daemon threads to hide lifecycle problems.

Every delayed result must carry sufficient identity to reject stale results:
- session_id
- generation_id
- request_id
- chunk_id, where relevant

Use time.monotonic_ns() for ordering runtime events.
Wall-clock time is only for display and filenames.

Distinguish these values whenever available:
- raw policy action chunk
- selected raw action
- stabilized target action
- executed action returned by the robot boundary

Do not silently merge them into a single action field.

## Recording requirements

Recording and video encoding must not block the robot control loop.

Use:
- bounded producer queues
- background writer workers
- versioned recording schemas
- temporary .inprogress directories
- atomic finalization where practical

Always record dropped frames or dropped telemetry.
Never silently discard recording failures.

## Hardware safety

Never run a command that can move a real robot unless the user explicitly
authorizes a real-robot motion test in the current task.

Before any motion-capable test:
1. Explain which command will run.
2. Explain why it can cause motion.
3. State the expected motion.
4. State the stop and rollback procedure.
5. Wait for explicit user approval.

Default automated tests must use:
- fake robots
- mock policies
- black cameras
- dry-run behavior

A dry-run is not necessarily an offline simulation. Do not assume hardware
is absent merely because dry-run is enabled.

Never:
- bypass joint or action safety checks
- increase robot speed during refactoring
- automatically enable or reset hardware in tests
- replay a trajectory without schema and initial-state validation
- guess action units, joint ordering, or gripper semantics

## Development workflow

For non-trivial work:

1. Inspect the current implementation.
2. Summarize the existing call path.
3. Propose a file-level implementation plan.
4. Identify compatibility and hardware risks.
5. Implement the smallest coherent change.
6. Add or update automated tests.
7. Run validation commands.
8. Review the diff for unrelated changes.
9. Report what still requires real-hardware verification.

Do not make large unrelated formatting changes.
Do not rename public APIs without a migration requirement.
Do not delete stable behavior merely to simplify an abstraction.

Do not commit or push Git changes unless explicitly requested.

## Validation commands

Run the relevant checks before reporting completion:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

When dependency changes are made, also verify:

```bash
uv sync
```

Check that these entrypoints remain importable:

```bash
uv run mp-piper-infer --help
uv run mp-rm2-infer --help
uv run mp-piper-web --help
```

If a command cannot run because hardware or an external policy server is
unavailable, report that clearly. Do not claim it passed.

## Completion report

At the end of every task, report:

1. Files changed.
2. Important design decisions.
3. Tests executed and exact results.
4. Tests not executed.
5. Hardware behavior that still needs verification.
6. Known risks or limitations.
7. Suggested next step.

For staged roadmap work, implement only the requested stage.
Do not begin the next stage automatically.
