# Replay Feedback Semantics

Hardening H5 separates command submission from robot feedback and state-arrival
acknowledgement. A replay step is no longer treated as acknowledged merely
because `Robot.execute_transition()` returned.

## Cursors

`RobotReplayCursor` publishes these independent sample indices:

- `planned_sample_index`: latest source sample selected by the replay schedule.
- `sent_sample_index`: latest source sample whose command returned from the
  robot interface without raising.
- `feedback_sample_index`: latest source sample matched to a robot feedback
  sample taken after a command timestamp.
- `acknowledged_sample_index`: latest source sample whose feedback is within
  the configured tracking threshold or follower policy.
- `displayed_sample_index`: UI-facing cursor. For real replay this follows the
  acknowledged cursor.

The top-level `progress_ratio` follows `acknowledged_progress_ratio`. Separate
`sent_progress_ratio` and `feedback_progress_ratio` are exposed so operators can
see lag instead of seeing a completed trajectory before the robot has caught up.

## Command And Feedback Records

Each sent command carries:

- `command_id`
- `source_sample_index`
- `sent_timestamp_ns`
- `target`
- `expected_state`
- `acknowledgement_deadline_ns`
- joint and gripper tracking thresholds

Each feedback sample carries:

- `feedback_timestamp_ns`
- `robot_state`
- `feedback_age_s`
- `matched_command_id`
- instantaneous and lag-adjusted tracking error
- whether it acknowledged a command

Replay record events include these records plus the cursor snapshot. The replay
record manifest schema is version `2` and stores the acknowledgement strategy
plus final `tracking_cursors` for sent, feedback, acknowledged and displayed
sample indices.

## Acknowledgement Strategies

`ReplayConstraints.acknowledgement_strategy` supports:

- `feedback_threshold`: acknowledge the next pending command only when feedback
  is inside the joint and gripper thresholds.
- `follower_window`: compare feedback with a bounded window of pending expected
  states and acknowledge the best matching in-window command.
- `state_trajectory_settle`: require a target to remain inside threshold for
  `state_trajectory_settle_cycles` feedback cycles.
- `immediate_interface_ack`: record that the robot interface accepted commands.
  This is not state-arrival acknowledgement and is rejected when a connected
  hardware-enabled safety profile is present.

Defaults are conservative and use `feedback_threshold`.

## Tracking Error

The controller records instantaneous error against the latest sent expected
state and lag-adjusted error against the configured acknowledgement candidate.
Small transient errors produce `replay_tracking_warning` events. Sustained error
aborts after `sustained_tracking_error_limit` feedback cycles. Error above
`extreme_tracking_error` aborts immediately. Stale feedback and robot health
errors also abort.

Thresholds come from `ReplayConstraints` first. If joint tracking threshold is
not configured and the robot exposes a `RobotSafetyProfile` with
`tracking_error_threshold`, that profile value is used.

## Joint And Gripper Constraints

Planner kinematic checks split dimensions by `ActionSpec` field semantics and
declared gripper indices:

- Joint dimensions use `joint_max_step`, `joint_max_velocity`,
  `joint_max_acceleration`, and joint limit vectors.
- Gripper dimensions use `gripper_min`, `gripper_max`, `gripper_max_step`,
  `gripper_command_mode`, `gripper_settle_timeout_s`, and
  `gripper_tracking_threshold`.

Legacy `max_step`, `max_velocity`, `max_acceleration`, `tracking_tolerance`,
`lower_limits`, and `upper_limits` remain accepted for compatibility. H5 applies
the legacy kinematic values only to joint dimensions unless explicit gripper
constraints are supplied. A normal gripper transition from `1.0` to `0.0` is not
rejected by a joint `max_step`.

## Pause And Resume

Pause stops consuming new trajectory commands and calls the robot stop/hold
capability when available. The controller preserves both sent and acknowledged
cursors. Resume rereads robot state, checks health and feedback freshness, and
compares against the acknowledged sample or start state. It does not blindly
continue from the sent cursor.

If the robot has moved away from the acknowledged replay state, resume is
rejected. A future operator flow may generate an explicit move-to-state plan,
but H5 does not automatically move hardware from a Web handler.
