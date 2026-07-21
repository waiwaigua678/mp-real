# Plan Integrity

Hardening H3 makes reviewed motion plans reject mutation between planning,
confirmation and execution.

## Scope

Covered plan objects:

- `ReplayPlan`, `ReplayStep`, `ReplaySafetyReport`
- `RecordedPoseTarget`, `PoseWaypoint`, `MoveToRecordedStatePlan`
- `PoseMappingConfig`, `PoseSafetyLimits`, `ValidatedPoseTarget`
- replay and pose constraints stored inside those plans

The change does not alter replay acknowledgement semantics, gripper kinematic
semantics, Piper/RM2 workspace validation, or hardware motion capability.
Those remain H4/H5 work.

## Immutability

Plan-owned NumPy arrays are copied on construction and marked read-only with
`array.setflags(write=False)`. This includes replay targets and expected
states, pose target states, current-state snapshots, waypoints, safety-limit
arrays and validation outputs.

Mutating the caller's original ndarray after plan creation does not affect the
plan. Mutating JSON returned to Web/UI code also cannot mutate the plan because
JSON payloads are independently serialized lists and dicts.

## Canonical Hash

The shared implementation is `src/mp_real/common/plan_integrity.py`.

Canonical encoding rules:

- dict keys are converted to strings and serialized with sorted keys
- Python floats are encoded with `float.hex()`
- NumPy arrays are encoded as dtype, shape and C-contiguous raw bytes hex
- NumPy scalar values are converted through the same scalar path
- dataclasses are encoded as module-qualified type plus canonical field values
- enums are encoded by value
- object ids, Python repr strings and insertion order are never used

Replay plan hashes cover schema version, plan/session/generation identity,
dataset identity and hash, episode and sample range, robot name, replay and
timing mode, speed scale, ActionSpec, state/action schemas, source contract,
every target, every expected state, timestamps, target offsets, replay
constraints, joint limits, gripper metadata, safety flags, resource owner/lease
identity when available, creation time and expiration.

Move-to-recorded-state hashes cover schema version, plan/session/generation
identity, recorded target identity, dataset/sample/source metadata, ActionSpec,
state schema, current robot state snapshot, target state, per-dimension delta,
mapped joint names, unit conversions, gripper indices, every waypoint and
timestamp, pose constraints, required confirmations, safety warnings, mapping
fingerprint, resource owner/lease identity when available, creation time and
expiration.

## Revalidation Points

`require_integrity()` recomputes the full payload hash from the current object.
If it differs from `plan_hash`, a `PlanIntegrityError` subtype is raised:

- `ReplayPlanIntegrityError`
- `PosePlanIntegrityError`

Execution paths revalidate at these points:

- replay plan creation
- Web replay plan JSON serialization
- Web replay connect after lease/generation binding
- replay prepare before move-to-start
- replay confirm/start before spawning the run thread
- replay run thread before sending trajectory commands
- replay resume before continuing after pause
- pose plan creation
- Web pose plan JSON serialization
- Web pose execute
- Web pose deployment handoff
- Web pose deployment start
- pose controller start and worker entry
- pose and replay CLI execute paths

Expiration is controlled by `ReplayConstraints.plan_expiration_s` and
`PoseMotionConstraints.plan_expiration_s`. Passing `None` disables timeout for
that plan, but execution paths still recompute the canonical hash.

## Stale Identity

Replay plans are bound to `plan_id`, `session_id`, `generation_id`,
`dataset_id`, `dataset_hash`, resource owner and resource lease id when a Web
lease exists. Web replay rebases the plan hash after acquiring the replay
lease, so the hash displayed in the armed state is the one the operator must
confirm.

Replay controllers also reject stale resource/generation callbacks and a
changed connected `ActionSpec` before starting or resuming execution.

Pose plans are built only after a robot connection and include the current
robot state snapshot hash. The pose controller still performs the live drift
check before execution.

## Validation

H3 automated tests use FakeRobot/FakePoseRobot only. They verify:

- plan-owned arrays are read-only
- original input arrays do not affect generated plans
- tampering with target, expected state, timing, ActionSpec, dataset identity
  or gripper constraints changes the canonical hash
- arm, execute and resume paths reject tampered plans
- stale generation, robot reconnect/action-spec mismatch and plan expiration
  are rejected before commands are sent
- JSON round trips do not alter the canonical hash
- Piper and RM2 style fake plans both pass integrity checks
