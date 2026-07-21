# Piper Safety Profile

Hardening H4 adds an auditable safety profile and health report for Piper.
It does not make real Piper motion automatically available. In the default
configuration, real move-to-state and replay remain hardware blocked.

## Safety Profile

The shared profile type is `RobotSafetyProfile` in
`src/mp_real/safety/models.py`. Piper creates a default profile from its
existing `ActionSpec` and runtime args:

- robot name and model: `piper`
- joint names: from Piper `ActionSpec.state_fields`
- gripper indices and normalized range: from existing repository action
  semantics, `[0, 1]`
- stop capability: observed from both SDK arm objects exposing `stop()`
- policy: `STRICT` by default
- hardware motion: disabled by default

The default profile intentionally does not invent:

- joint min/max
- workspace or collision validation
- vendor health error-code mapping
- SDK feedback freshness timestamp
- vendor documented velocity or acceleration limits

Those fields must come from a user supplied profile or a verified SDK/vendor
source before they can be marked as passed.

## Health Snapshot

Piper health is reported as a structured `RobotHealthSnapshot` stored in
`RobotState.health`.

Per arm, the adapter tries to read:

- `get_arm_status()`
- `get_joint_angles()`
- `stop()` availability
- common status attributes if the SDK object exposes them, such as enabled,
  healthy, error or error_code
- monotonic feedback timestamp only if the SDK feedback object exposes a
  monotonic timestamp attribute

Raw SDK/status objects are preserved as short raw strings in `raw_status`.
Unknown fields are reported as unavailable, not passed.

## Validation Semantics

`PiperRobot.validate_pose_target()` now runs profile-aware validation:

- state dimension, schema, units and finite values
- profile robot/model match
- joint names
- configured soft joint limits
- configured gripper range
- structured robot health
- feedback freshness when a timeout and SDK timestamp are available
- stop capability
- workspace validation capability

Unavailable checks appear in `unavailable_checks`. In `STRICT`, configured
critical unavailable checks are also errors and block execution. In
`JOINT_SPACE_RECORDED_TRAJECTORY_ONLY`, workspace validation may remain
unavailable only for the restricted recorded-trajectory workflow. In
`DEVELOPMENT_OVERRIDE`, unavailable checks can be non-blocking only when an
operator and reason are explicitly configured.

## Current Status

Piper real motion remains hardware blocked until a reviewed profile supplies
real joint limits, health error mapping, freshness semantics and all required
operator safety gates pass.

