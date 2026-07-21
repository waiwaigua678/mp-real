# Robot Replay Safety

Robot trajectory replay is separate from offline data viewing and policy
deployment. Planning is offline by default. Physical execution requires an
exclusive robot-control lease, a reviewed plan hash and a second confirmation.

## Required Data Metadata

Command replay requires:

- complete dataset and episode status
- matching robot name
- matching `ActionSpec`
- known action source
- known action mode
- matching joint unit
- declared arm count
- declared gripper indices and semantics
- `recording_semantics=control_step_observation_action`
- `control_step_aligned=true`

Legacy or unknown datasets are not silently upgraded into replay-safe data.

## Execution Semantics

H5 separates:

- planned sample
- sent command
- matched feedback
- acknowledged sample
- displayed sample

The replay manifest stores the final tracking cursors so an operator can audit
whether the robot merely received commands or actually reached the targets.

## Hardware Status

Piper and RM2 replay remain hardware blocked until their H4/H5 gates are
completed. A Python stop event is not a substitute for a verified robot stop
capability.

See `docs/replay_feedback_semantics.md` and `docs/hardware_validation_h5.md`.

