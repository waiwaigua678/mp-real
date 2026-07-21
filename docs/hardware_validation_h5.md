# Hardware Validation H5

H5 does not authorize automatic hardware motion. Piper and RM2 replay remain
`hardware blocked` until every gate below is completed for the specific robot,
installation, safety profile, dataset source, and operator procedure.

Before any motion-capable command, record the exact command, expected motion,
stop procedure, rollback procedure, operator, and result.

## Gate 0: Delayed FakeRobot

- Run the automated delayed-feedback replay tests.
- Confirm sent, feedback, and acknowledged cursors are monotonic and separate.
- Confirm normal gripper open/close is not rejected by joint max-step limits.
- Confirm stale feedback, health error, and sustained tracking error abort.

## Gate 1: Read-Only Feedback

- Connect Piper or RM2 in a read-only mode.
- Do not enable, reset, or send commands.
- Measure feedback frequency and timestamp freshness for each arm.
- Confirm health snapshot reports connected/enabled/healthy/error fields.

## Gate 2: Single Tiny Target

- User must explicitly allow this motion test.
- Use the configured safety profile and lowest practical speed.
- Send one tiny target from the current state.
- Measure command return time and first feedback change time.
- Confirm `sent_sample_index` advances before `acknowledged_sample_index`.

## Gate 3: 3 To 5 Points

- Empty workspace, no object, low speed.
- Replay 3 to 5 points.
- Confirm lag remains inside configured follower or threshold policy.
- Stop immediately on stale feedback, health error, or extreme error.

## Gate 4: 10 To 20 Points And Stop

- Replay 10 to 20 low-speed points.
- Trigger normal stop and emergency stop paths separately.
- Confirm command consumption stops and no further commands are sent.
- Confirm replay record contains command, feedback, ack, and cursor fields.

## Gate 5: Pause And Resume

- Pause while commands are in flight.
- Confirm no new trajectory commands are consumed while paused.
- Resume only when the robot remains close to the acknowledged sample.
- Deliberately move away from the acknowledged sample and confirm resume is
  rejected.

## Gate 6: Gripper Short Trajectory

- User must explicitly allow this motion test.
- No object in the gripper.
- Replay a short trajectory containing normal open and close transitions.
- Confirm gripper thresholds and settle timeout, not joint max-step, govern the
  gripper acknowledgement.

## Gate 7: Full Empty Trajectory

- Empty workspace.
- Replay the complete trajectory at the approved speed scale.
- Confirm sent, feedback, and acknowledged progress reach 100%.
- Confirm replay writer finalizes and the robot can be stopped at any point.

## Gate 8: Known Safe Object Trajectory

- Only use a dataset already reviewed for the same robot model, installation,
  ActionSpec, safety profile, and object setup.
- Stop at the first unexpected lag, health warning, or contact condition.
- Do not continue to Gate 8 if any earlier gate failed.

## No-Go Conditions

- Immediate interface acknowledgement is the only available strategy.
- Stop capability is unavailable.
- Feedback timestamp freshness is unavailable or stale.
- Robot health reports disconnected, disabled, unhealthy, or unmapped errors.
- Safety profile, ActionSpec, dataset identity, or plan hash changed after
  planning.

