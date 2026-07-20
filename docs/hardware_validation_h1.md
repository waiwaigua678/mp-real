# Hardware Validation H1

H1 is a data-semantics hardening stage. It does not require real robot motion, and no real-robot motion, enable, reset, or control command was executed for this stage.

## Offline Validation Required

- Run the full fake-hardware unittest suite.
- Run Ruff and record any unrelated pre-existing lint failures.
- Confirm infer-only and policy warmup emit no `ControlStepRecorded` events.
- Confirm sync runtime records one row per executed control step.
- Confirm RTC runtime records control-step observations for robot-bound actions.
- Confirm repeated camera frames are represented as reuse telemetry, not as automatic dataset invalidation.

## Before Any Future Real Motion

Do not run these checks unless the operator explicitly authorizes real-robot motion testing in the current task.

- Verify the exact robot, SDK backend, workspace, e-stop, and stop procedure.
- Verify the configured `ActionSpec` matches the live robot state and action dimensions.
- Verify cameras are connected and their frame ids/timestamps are visible in telemetry.
- Start with a non-motion camera/state capture check.
- Execute only a bounded, low-risk action sequence that has been reviewed by the operator.
- Stop immediately on missing telemetry, stale camera frames beyond configured limits, shape mismatch, SDK error, or unexpected robot motion.

## H1 Acceptance Signals

- A short real episode records `control_step_aligned=true`.
- Consecutive rows from the same policy chunk have distinct `control_step_id` and control-step `observation_id`.
- The same rows may share `policy_observation_id`.
- `observation.state` matches the state captured before each action, not the policy inference observation for the chunk.
- `action` matches the executed action returned by the robot boundary.

## Remaining Hardware Risks

H1 does not validate Piper/RM2 workspace limits, joint limits, health codes, replay plan immutability, replay acknowledgement semantics, or gripper-specific replay limits. Those remain H2, H3, and H5 blockers for move-to-state and robot replay.
