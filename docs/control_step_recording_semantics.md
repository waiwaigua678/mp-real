# Control Step Recording Semantics

Hardening H1 defines the standard recording row as:

`observation_t + action_t`

where `observation_t` is the robot state and camera snapshot captured immediately before one actual robot-bound action, and `action_t` is the action returned by the robot boundary after that command is sent.

## Identities

- `control_step_id`: the runtime control step that produced one standard LeRobot row.
- `observation_id`: the control-step observation id for that row.
- `policy_observation_id`: the policy inference observation that produced the selected raw action chunk. Multiple rows may share this id when they consume different cursors from the same chunk.
- `chunk_id` and `chunk_cursor`: the raw policy chunk identity and the cursor selected for the control step.
- `policy_request_id`: the policy request that produced the chunk when that identity is available.

Policy observations are telemetry. They are not used to reconstruct standard per-action LeRobot rows.

## Runtime Flow

For each robot-bound control action, the shared runtime now:

1. Selects a raw action from the current plan or RTC buffer.
2. Captures a fresh control-step observation through the existing robot/camera adapter.
3. Applies the existing stabilization and safety boundary.
4. Sends the action through `execute_transition()`.
5. Builds one `ControlStepRecord`.
6. Emits one immutable `ControlStepRecorded` event.
7. Emits the existing action events for compatibility and live telemetry.

Warmup, first-live prefetch, and infer-only requests do not emit `ControlStepRecorded` because no robot-bound action is executed.

## LeRobot Rows

The standard LeRobot fields are written from `ControlStepRecorded`:

- `observation.state`: `robot_state_before_action`
- `action`: `executed_action`
- `timestamp`: frame index divided by dataset fps

The mp-real extension fields include:

- `mp_real.control_step_id`
- `mp_real.observation_id`
- `mp_real.policy_observation_id`
- `mp_real.policy_request_id`
- `mp_real.chunk_id`
- `mp_real.chunk_cursor`
- `mp_real.action_sent_timestamp_ns`
- `mp_real.selected_raw_action`
- `mp_real.stabilized_action`
- `mp_real.control_cycle_ns`
- `mp_real.camera_skew_ns`

Raw policy chunks remain in episode telemetry and are matched through `policy_observation_id` plus `chunk_cursor`.

## Camera Reuse

Camera frame reuse is allowed and is recorded as telemetry, not treated as automatic data corruption. The recorder stores camera frame ids, timestamps, reuse flags, and camera ages so downstream audits can detect risk signals without assuming every repeated frame is wrong.

## Metadata

Finalized H1-aligned recordings set:

- `mp_real.schema_version = 2`
- `mp_real.recorder_version = h1-control-step-v2`
- `mp_real.recording_semantics = control_step_observation_action`
- `mp_real.control_step_aligned = true`
- `mp_real.policy_observation_reuse_possible = false`

Datasets produced through the legacy observation/action event assembler are finalized with unknown control-step semantics and must be audited before training.

## Non-Goals

H1 does not add a new robot adapter, policy client, recorder, or inference loop. It does not change the recorder data model beyond metadata and extension fields needed to express the corrected semantics.
