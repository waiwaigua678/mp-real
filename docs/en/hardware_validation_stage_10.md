# Stage 10 hardware validation: robot trajectory replay

This document applies to `mp-robot-replay` and the Web **真机轨迹回放** panel.
It is separate from offline data viewing and from policy inference.  Never run
a gate for Piper and RM2 concurrently.  A failure at any gate blocks all later
gates for that robot.

## Preconditions

- The workspace is clear, an operator owns the emergency stop, and a second
  observer can see the robot.
- Confirm the dataset is complete, belongs to the selected robot, and its
  action/state names, units, semantics, and action mode are reviewed.
- Command replay requires `mp_real.replay.action_source` and
  `mp_real.replay.action_mode=joint_position_target`.  Do not substitute raw
  chunks or model telemetry.
- Verify the vendor deployment reports working stop, health, workspace, and
  joint-limit validation.  The current default Piper/RM2 adapters deliberately
  reject execution until those checks are configured.
- Keep speed scale at `0.1` initially; never exceed `1.0`.

## Piper gates

| Gate | Procedure | Pass criterion |
| --- | --- | --- |
| 0 | Run `mp-robot-replay` without `--execute`. | A valid plan/report; no Robot is created. |
| 1 | Connect from Web, do not confirm start. | Current state is read; no motion, policy, or camera connection. |
| 2 | Confirm move-to-start at 0.1 speed. | Low-speed verified pose with available emergency stop. |
| 3 | Replay only 3–5 samples at 0.1. | Tracking error and control timing remain within plan limits. |
| 4 | Replay 10–20 samples at 0.1–0.2. | Stable trajectory and complete replay record. |
| 5 | Full unloaded replay; issue stop. | Stop enters `ABORTED`, releases lease, and records termination. |
| 6 | Full unloaded replay; pause/resume. | Resume rejects any excess pose drift. |
| 7 | Known-safe object trajectory at scale ≤1.0. | Result and replay record are reviewed. |

## RM2 gates

Repeat gates 0–7 independently with the RM2 SDK, its own stop/health/limit
checks, and RM2-specific workspace clearance.  Do not treat a Piper pass as
evidence for RM2 joint order, units, gripper semantics, or stop latency.

## Stop and rollback

The expected motion is only the reviewed `ReplayPlan` at the displayed speed
scale.  Use the Web emergency-stop control or the vendor physical stop if any
unexpected movement, tracking error, health error, communication loss, or
operator concern occurs.  Disconnect after the worker reaches a terminal
state; this closes the robot and releases `ROBOT_CONTROL`.  Resume only after
the controller accepts the measured state against the recorded pause state.
