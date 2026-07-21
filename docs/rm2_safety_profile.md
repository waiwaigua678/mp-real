# RM2 Safety Profile

Hardening H4 adds an auditable safety profile and health report for RM2.
It does not make real RM2 motion automatically available. In the default
configuration, real move-to-state and replay remain hardware blocked.

## Safety Profile

RM2 creates a default `RobotSafetyProfile` from its existing `ActionSpec` and
runtime args:

- robot name and model: `rm2`
- joint names: from RM2 `ActionSpec.state_fields`
- policy joint unit: from existing `policy_joint_unit`
- gripper indices and normalized range: from existing repository action
  semantics, `[0, 1]`
- stop capability: observed from the arm object exposing `stop()`
- policy: `STRICT` by default
- hardware motion: disabled by default

The default profile intentionally does not invent:

- RM joint min/max
- workspace or collision validation
- full RM health error-code mapping
- SDK feedback freshness timestamp
- vendor documented velocity or acceleration limits

Those fields must come from a user supplied profile or a verified SDK/vendor
source before they can be marked as passed.

## Health Snapshot

RM2 health is reported as a structured `RobotHealthSnapshot` stored in
`RobotState.health`.

For real RM SDK arms, the adapter currently reads the APIs already bound in
`src/mp_real/robots/rm2/infer.py`:

- valid arm handle
- `rm_get_current_arm_state()`
- `RmCurrentArmState.err`
- `rm_get_gripper_state()`
- `rm_set_arm_stop()` availability through the existing stop method

Left and right arm health are stored separately, and the robot health snapshot
contains an aggregate connected/enabled/healthy status. The current RM binding
does not expose a reliable SDK feedback timestamp, so feedback freshness remains
unavailable unless a future binding provides one or a reviewed source is added.

MockArm exposes configurable health fields for automated tests only.

## Validation Semantics

`Rm2Robot.validate_pose_target()` now runs profile-aware validation:

- state dimension, schema, units and finite values
- profile robot/model match
- joint names
- configured soft joint limits
- configured gripper range
- structured left/right health
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

RM2 real motion remains hardware blocked until a reviewed profile supplies real
joint limits, health error mapping, freshness semantics and all required
operator safety gates pass.

Any change to RM2 `ActionSpec` metadata, safety policy or safety profile
requires a fresh pose/replay plan. H6 metadata fixes do not grandfather older
unknown-schema recordings into hardware-safe data.
