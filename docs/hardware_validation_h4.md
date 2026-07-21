# Hardware Validation H4

H4 implements auditable safety profiles and health reporting. It does not
authorize real motion by itself. Until all gates below pass, real
move-to-state and robot replay remain `hardware blocked`.

## Gate 0: Configuration Only

No robot connection.

- Validate user supplied Piper and RM2 safety profile files.
- Confirm profile source and version.
- Confirm joint names match the current `ActionSpec`.
- Confirm joint limits, gripper limits, velocity, acceleration and single-step
  limits come from vendor SDK, vendor documentation, repository configuration
  or explicit user supplied configuration.
- Confirm `DEVELOPMENT_OVERRIDE` is absent for formal Baseline runs.

## Gate 1: Read-Only Health

Connect robot only for read-only status.

- Do not enable arms.
- Do not reset.
- Do not send motion or gripper commands.
- Read `RobotState.health`.
- Confirm left/right health details are shown.
- Confirm raw SDK status is preserved.
- Confirm unavailable checks are visible and are not displayed as passed.

## Gate 2: Freshness And Error Mapping

Still no motion.

- Confirm configured communication timeout.
- Confirm feedback freshness changes when feedback stops or becomes stale.
- Confirm vendor error codes map to unhealthy status.
- Manually disconnect communication and verify health becomes unhealthy.
- Confirm stop capability is reported from the SDK, not from Python
  `stop_event` alone.

## Gate 3: Target Equals Current State

Still no execution.

- Capture current robot state.
- Build a target equal to current state.
- Run validation and plan generation only.
- Confirm plan hash includes safety profile hash and policy.
- Change the safety profile and confirm the old plan is rejected.

## Gate 4: Minimal Motion

Requires explicit user approval in the current task before any command runs.

- Empty workspace.
- Low speed.
- One arm at a time if the hardware procedure requires it.
- Very small joint-space target within reviewed soft limits.
- Operator at emergency stop.
- Stop procedure rehearsed before starting.
- Abort on any health warning, stale feedback, tracking error or unexpected
  SDK status.

## H4 No-Go Status

- Current data for training: unchanged by H4.
- Current move-to-state on real hardware: No-Go until Gates 0-3 pass and Gate 4
  is explicitly authorized.
- Current robot replay on real hardware: No-Go. Replay still needs H5 control
  semantics for sent, observed and acknowledged state before real replay can be
  considered.

